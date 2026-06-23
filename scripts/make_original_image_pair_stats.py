"""Measure CLIP+DINO agreement on original ImageNet image pairs.

This is the JPEG counterpart to the generated-anchor inter-class probe:
anchors and candidates are all original ImageNet training images.  CLIP/DINO
see the original JPEGs, while Euclidean/MS-stat are computed on the matching
cached VAE latents used by the drifting-model pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_clip_dino_agreement import clip_embeddings, dino_embeddings  # noqa: E402
from scripts.make_interclass_agreement_stats import (  # noqa: E402
    summarize_agreement,
    summarize_positive,
)
from scripts.make_metric_geometry_figures import (  # noqa: E402
    build_feature_fn,
    dist_many,
    extract_descriptors,
    index_latent_cache,
    load_latents,
    standardize,
)
from utils.env import IMAGENET_PATH  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--mae-path", default="hf://mae_latent_256")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino-model", default="facebook/dinov2-small")
    parser.add_argument("--out", default="figures/original_image_pair_clip_dino_agreement.json")
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--num-anchors", type=int, default=128)
    parser.add_argument("--num-negatives", type=int, default=31)
    parser.add_argument("--min-class-items", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=96)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def image_path_for_latent(latent_path: Path, split: str) -> Path:
    return Path(IMAGENET_PATH) / split / latent_path.parent.name / f"{latent_path.stem}.JPEG"


def index_original_pairs(split: str) -> tuple[dict[int, list[tuple[Path, Path]]], dict[int, str]]:
    by_label, label_to_wnid = index_latent_cache(split)
    paired: dict[int, list[tuple[Path, Path]]] = {}
    for label, latent_paths in by_label.items():
        # The latent cache mirrors ImageNet class/stem names, so avoid an
        # expensive exists() sweep over the full cache on shared filesystems.
        paired[label] = [(latent_path, image_path_for_latent(latent_path, split)) for latent_path in latent_paths]
    return paired, label_to_wnid


def sample_layout(
    by_label: dict[int, list[tuple[Path, Path]]],
    *,
    num_anchors: int,
    num_negatives: int,
    min_class_items: int,
    rng: random.Random,
) -> tuple[list[tuple[Path, Path]], list[list[tuple[Path, Path]]], list[list[int]], np.ndarray]:
    eligible = [label for label, items in by_label.items() if len(items) >= min_class_items]
    if num_anchors > len(eligible):
        raise ValueError(f"Asked for {num_anchors} anchors, but only {len(eligible)} classes are eligible.")
    if num_negatives >= len(eligible):
        raise ValueError(f"Asked for {num_negatives} negatives, but only {len(eligible)} classes are eligible.")

    anchor_labels = np.asarray(rng.sample(eligible, num_anchors), dtype=np.int64)
    anchors: list[tuple[Path, Path]] = []
    candidate_rows: list[list[tuple[Path, Path]]] = []
    candidate_label_rows: list[list[int]] = []
    for label in anchor_labels.tolist():
        anchor, positive = rng.sample(by_label[label], 2)
        negative_labels = rng.sample([x for x in eligible if x != label], num_negatives)
        negatives = [rng.choice(by_label[neg]) for neg in negative_labels]
        anchors.append(anchor)
        candidate_rows.append([positive] + negatives)
        candidate_label_rows.append([label] + negative_labels)
    return anchors, candidate_rows, candidate_label_rows, anchor_labels


def load_original_images(items: list[tuple[Path, Path]]) -> list[Image.Image]:
    images = []
    for _, image_path in items:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return images


def reference_distances(
    images: list[Image.Image],
    *,
    num_anchors: int,
    num_candidates: int,
    device: torch.device,
    clip_model: str,
    dino_model: str,
    batch_size: int,
) -> np.ndarray:
    clip = clip_embeddings(images, device, clip_model, batch_size)
    dino = dino_embeddings(images, device, dino_model, batch_size)
    rows = []
    stride = num_candidates + 1
    for i in range(num_anchors):
        base = stride * i
        candidate_slice = slice(base + 1, base + stride)
        d_clip = (1 - (clip[base : base + 1] @ clip[candidate_slice].T)).squeeze(0).numpy()
        d_dino = (1 - (dino[base : base + 1] @ dino[candidate_slice].T)).squeeze(0).numpy()
        z_clip = (d_clip - d_clip.mean()) / (d_clip.std() + 1e-8)
        z_dino = (d_dino - d_dino.mean()) / (d_dino.std() + 1e-8)
        rows.append(0.5 * (z_clip + z_dino))
    return np.stack(rows)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    rng = random.Random(args.seed)
    by_label, label_to_wnid = index_original_pairs(args.split)
    anchors, candidate_rows, candidate_label_rows, anchor_labels = sample_layout(
        by_label,
        num_anchors=args.num_anchors,
        num_negatives=args.num_negatives,
        min_class_items=args.min_class_items,
        rng=rng,
    )
    k = args.num_negatives + 1

    flat_candidates = [item for row in candidate_rows for item in row]
    anchor_latents = load_latents([latent for latent, _ in anchors])
    candidate_latents = load_latents([latent for latent, _ in flat_candidates]).reshape(args.num_anchors, k, 32, 32, 4)

    activation_fn = build_feature_fn(args.mae_path, device)
    all_latents = torch.cat([anchor_latents, candidate_latents.reshape(-1, 32, 32, 4)], dim=0)
    desc = extract_descriptors(all_latents, activation_fn, device, args.feature_batch_size)
    n = args.num_anchors
    e_all = standardize(desc.euc)
    m_all = standardize(desc.ms)
    e_anchor, e_candidates = e_all[:n], e_all[n:].reshape(n, k, -1)
    m_anchor, m_candidates = m_all[:n], m_all[n:].reshape(n, k, -1)
    euc_distances = dist_many(e_anchor, e_candidates, "euc").numpy()
    ms_distances = dist_many(m_anchor, m_candidates, "ms").numpy()

    image_items = []
    for anchor, row in zip(anchors, candidate_rows):
        image_items.append(anchor)
        image_items.extend(row)
    reference = reference_distances(
        load_original_images(image_items),
        num_anchors=n,
        num_candidates=k,
        device=device,
        clip_model=args.clip_model,
        dino_model=args.dino_model,
        batch_size=args.embed_batch_size,
    )

    labels_as_wnids = [
        {
            "anchor": label_to_wnid[int(anchor_labels[i])],
            "candidates": [label_to_wnid[int(label)] for label in candidate_label_rows[i]],
        }
        for i in range(min(8, n))
    ]

    return {
        "split": args.split,
        "clip_model": args.clip_model,
        "dino_model": args.dino_model,
        "num_anchors": n,
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
