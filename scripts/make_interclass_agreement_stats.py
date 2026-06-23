"""Measure CLIP+DINO agreement for broad inter-class retrieval.

Each generated anchor is compared with one same-class ImageNet candidate and a
set of different-class ImageNet candidates.  The script reports both:

  - all-candidate behavior, which includes the class-discrimination question;
  - negatives-only behavior, which asks how metrics rank candidates from other
    classes once the same-class positive is removed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_clip_dino_agreement import (  # noqa: E402
    clip_embeddings,
    dino_embeddings,
    pairwise_agreement,
    spearman,
)
from scripts.make_metric_geometry_figures import (  # noqa: E402
    build_feature_fn,
    decode_latents_to_pil,
    dist_many,
    extract_descriptors,
    generate_latents,
    index_latent_cache,
    load_generator,
    load_latents,
    standardize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/0428_2340_gen_latent_ablation")
    parser.add_argument("--cfg-scale", type=float, default=1.1)
    parser.add_argument("--mae-path", default="hf://mae_latent_256")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino-model", default="facebook/dinov2-small")
    parser.add_argument("--out", default="figures/interclass_clip_dino_agreement.json")
    parser.add_argument("--seed", type=int, default=9876)
    parser.add_argument("--num-anchors", type=int, default=128)
    parser.add_argument("--num-negatives", type=int, default=31)
    parser.add_argument("--min-class-items", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=96)
    parser.add_argument("--gen-batch-size", type=int, default=32)
    parser.add_argument("--decode-batch-size", type=int, default=64)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def topk_overlap(metric_row: np.ndarray, ref_row: np.ndarray, k: int) -> float:
    k = min(k, metric_row.shape[0])
    metric_top = set(np.argsort(metric_row)[:k].tolist())
    ref_top = set(np.argsort(ref_row)[:k].tolist())
    return len(metric_top & ref_top) / k


def mean_ci(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))


def summarize_agreement(metric_distances: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    pairwise = []
    correlations = []
    top1 = []
    top3 = []
    top5 = []
    for metric_row, ref_row in zip(metric_distances, reference):
        pairwise.append(pairwise_agreement(metric_row, ref_row))
        correlations.append(spearman(metric_row, ref_row))
        top1.append(float(np.argmin(metric_row) == np.argmin(ref_row)))
        top3.append(topk_overlap(metric_row, ref_row, 3))
        top5.append(topk_overlap(metric_row, ref_row, 5))

    out: dict[str, float] = {}
    for key, values in [
        ("pairwise_agreement", pairwise),
        ("spearman", correlations),
        ("top1_match", top1),
        ("top3_overlap", top3),
        ("top5_overlap", top5),
    ]:
        mean, ci = mean_ci(values)
        out[key] = mean
        out[f"{key}_ci95"] = ci
    return out


def summarize_positive(metric_distances: np.ndarray, reference: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, distances in [("reference", reference), ("metric", metric_distances)]:
        pos = distances[:, 0]
        neg = distances[:, 1:]
        nearest_neg = neg.min(axis=1)
        ranks = 1 + (neg < pos[:, None]).sum(axis=1)
        raw_margin = nearest_neg - pos
        relative_margin = raw_margin / (np.abs(nearest_neg) + np.abs(pos) + 1e-8)
        out[name] = {
            "same_class_top1": float((ranks == 1).mean()),
            "same_class_top3": float((ranks <= 3).mean()),
            "same_class_top5": float((ranks <= 5).mean()),
            "same_class_mean_rank": float(ranks.mean()),
            "same_class_median_rank": float(np.median(ranks)),
            "same_vs_nearest_diff_median_raw_margin": float(np.median(raw_margin)),
            "same_vs_nearest_diff_q25_raw_margin": float(np.quantile(raw_margin, 0.25)),
            "same_vs_nearest_diff_q75_raw_margin": float(np.quantile(raw_margin, 0.75)),
            "same_vs_nearest_diff_median_relative_margin": float(np.median(relative_margin)),
            "same_vs_nearest_diff_q25_relative_margin": float(np.quantile(relative_margin, 0.25)),
            "same_vs_nearest_diff_q75_relative_margin": float(np.quantile(relative_margin, 0.75)),
        }
    return out


def sample_interclass_layout(
    by_label: dict[int, list[Path]],
    *,
    num_anchors: int,
    num_negatives: int,
    min_class_items: int,
    rng: random.Random,
) -> tuple[np.ndarray, list[list[Path]], list[list[int]]]:
    eligible = [label for label, paths in by_label.items() if len(paths) >= min_class_items]
    if num_anchors > len(eligible):
        raise ValueError(f"Asked for {num_anchors} anchors, but only {len(eligible)} classes are eligible.")
    if num_negatives >= len(eligible):
        raise ValueError(f"Asked for {num_negatives} negatives, but only {len(eligible)} classes are eligible.")

    labels = np.asarray(rng.sample(eligible, num_anchors), dtype=np.int64)
    candidate_paths: list[list[Path]] = []
    candidate_labels: list[list[int]] = []
    for label in labels.tolist():
        positive = rng.choice(by_label[label])
        negative_labels = rng.sample([x for x in eligible if x != label], num_negatives)
        paths = [positive] + [rng.choice(by_label[neg]) for neg in negative_labels]
        candidate_paths.append(paths)
        candidate_labels.append([label] + negative_labels)
    return labels, candidate_paths, candidate_labels


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    rng = random.Random(args.seed)
    by_label, label_to_wnid = index_latent_cache("train")
    anchor_labels, candidate_paths, candidate_labels = sample_interclass_layout(
        by_label,
        num_anchors=args.num_anchors,
        num_negatives=args.num_negatives,
        min_class_items=args.min_class_items,
        rng=rng,
    )

    model = load_generator(args.checkpoint, device)
    anchors = generate_latents(
        model,
        torch.from_numpy(anchor_labels),
        args.cfg_scale,
        args.gen_batch_size,
        args.seed + 11,
        device,
    )
    flat_candidate_paths = [path for row in candidate_paths for path in row]
    candidates = load_latents(flat_candidate_paths).reshape(args.num_anchors, args.num_negatives + 1, 32, 32, 4)

    activation_fn = build_feature_fn(args.mae_path, device)
    all_latents = torch.cat([anchors, candidates.reshape(-1, 32, 32, 4)], dim=0)
    desc = extract_descriptors(all_latents, activation_fn, device, args.feature_batch_size)
    n = args.num_anchors
    k = args.num_negatives + 1

    e_all = standardize(desc.euc)
    m_all = standardize(desc.ms)
    e_anchor, e_candidates = e_all[:n], e_all[n:].reshape(n, k, -1)
    m_anchor, m_candidates = m_all[:n], m_all[n:].reshape(n, k, -1)
    euc_distances = dist_many(e_anchor, e_candidates, "euc").numpy()
    ms_distances = dist_many(m_anchor, m_candidates, "ms").numpy()

    decode_latents = []
    for i in range(n):
        decode_latents.append(anchors[i])
        decode_latents.extend(candidates[i])
    images = decode_latents_to_pil(torch.stack(decode_latents), args.decode_batch_size)
    clip = clip_embeddings(images, device, args.clip_model, args.embed_batch_size)
    dino = dino_embeddings(images, device, args.dino_model, args.embed_batch_size)

    reference_rows = []
    stride = k + 1
    for i in range(n):
        base = stride * i
        candidate_slice = slice(base + 1, base + stride)
        d_clip = (1 - (clip[base : base + 1] @ clip[candidate_slice].T)).squeeze(0).numpy()
        d_dino = (1 - (dino[base : base + 1] @ dino[candidate_slice].T)).squeeze(0).numpy()
        z_clip = (d_clip - d_clip.mean()) / (d_clip.std() + 1e-8)
        z_dino = (d_dino - d_dino.mean()) / (d_dino.std() + 1e-8)
        reference_rows.append(0.5 * (z_clip + z_dino))
    reference = np.stack(reference_rows)

    labels_as_wnids = [
        {
            "anchor": label_to_wnid[int(anchor_labels[i])],
            "candidates": [label_to_wnid[int(label)] for label in candidate_labels[i]],
        }
        for i in range(min(8, n))
    ]

    return {
        "checkpoint": args.checkpoint,
        "cfg_scale": args.cfg_scale,
        "clip_model": args.clip_model,
        "dino_model": args.dino_model,
        "num_anchors": args.num_anchors,
        "num_candidates_per_anchor": k,
        "candidate_order": ["same_class"] + ["different_class"] * args.num_negatives,
        "sampled_wnids_preview": labels_as_wnids,
        "all_candidates": {
            "reference_positive_behavior": summarize_positive(reference, reference)["reference"],
            "euclidean": {
                **summarize_agreement(euc_distances, reference),
                **summarize_positive(euc_distances, reference)["metric"],
            },
            "ms_stat": {
                **summarize_agreement(ms_distances, reference),
                **summarize_positive(ms_distances, reference)["metric"],
            },
        },
        "different_class_only": {
            "euclidean": summarize_agreement(euc_distances[:, 1:], reference[:, 1:]),
            "ms_stat": summarize_agreement(ms_distances[:, 1:], reference[:, 1:]),
        },
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    metrics = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
