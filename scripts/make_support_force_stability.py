"""Support-force stability diagnostic for the proposed metric.

The metric is used inside a finite-support drift estimator, not as a generic
semantic retrieval model.  This diagnostic therefore builds noisy support sets
with known local positives: weak views of a generated anchor.  It mixes those
positives with layout impostors and asks which metric yields a stable attraction
direction toward the local positives under repeated support resampling.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_metric_geometry_figures import (  # noqa: E402
    BLUE,
    DARK,
    GRAY,
    GREEN,
    GRID,
    ORANGE,
    PURPLE,
    build_feature_fn,
    dist_many,
    draw_text,
    extract_descriptors,
    font,
    generate_latents,
    index_latent_cache,
    load_generator,
    save_png_pdf,
    standardize,
    synthetic_impostor_bank,
    text_size,
    weak_view,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/0428_2340_gen_latent_ablation")
    parser.add_argument("--cfg-scale", type=float, default=1.1)
    parser.add_argument("--mae-path", default="hf://mae_latent_256")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--seed", type=int, default=1357)
    parser.add_argument("--num-anchors", type=int, default=128)
    parser.add_argument("--num-weak", type=int, default=8)
    parser.add_argument("--num-impostors", type=int, default=32)
    parser.add_argument("--hard-impostor-candidate-pool", type=int, default=96)
    parser.add_argument("--resamples", type=int, default=24)
    parser.add_argument("--sample-weak", type=int, default=3)
    parser.add_argument("--sample-impostors", type=int, default=9)
    parser.add_argument("--force-topk", type=int, default=3)
    parser.add_argument("--feature-batch-size", type=int, default=96)
    parser.add_argument("--gen-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.flatten(1).float()
    b = b.flatten(1).float()
    return (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))


def summarize_metric(
    distances: torch.Tensor,
    pool_latents: torch.Tensor,
    anchors: torch.Tensor,
    *,
    num_weak: int,
    resamples: int,
    sample_weak: int,
    sample_impostors: int,
    force_topk: int,
    seed: int,
) -> dict:
    n = distances.shape[0]
    labels = torch.zeros_like(distances, dtype=torch.bool)
    labels[:, :num_weak] = True

    top1 = torch.argsort(distances, dim=1)[:, 0]
    top3 = torch.argsort(distances, dim=1)[:, :3]
    top5 = torch.argsort(distances, dim=1)[:, :5]
    top1_weak = labels.gather(1, top1[:, None]).float().mean().item()
    top3_weak_mass = labels.gather(1, top3).float().mean(dim=1).numpy()
    top5_weak_mass = labels.gather(1, top5).float().mean(dim=1).numpy()

    oracle = pool_latents[:, :num_weak].mean(dim=1) - anchors
    force_alignment = []
    force_stability = []
    sampled_weak_mass = []
    rng = np.random.default_rng(seed)
    for i in range(n):
        forces = []
        weak_masses = []
        for _ in range(resamples):
            weak_idx = rng.choice(num_weak, size=sample_weak, replace=False)
            impostor_idx = rng.choice(
                np.arange(num_weak, pool_latents.shape[1]),
                size=sample_impostors,
                replace=False,
            )
            subset_idx = np.concatenate([weak_idx, impostor_idx])
            local_dist = distances[i, subset_idx]
            chosen_local = torch.argsort(local_dist)[:force_topk].cpu().numpy()
            chosen = subset_idx[chosen_local]
            chosen_latents = pool_latents[i, chosen]
            force = chosen_latents.mean(dim=0) - anchors[i]
            forces.append(force.flatten())
            weak_masses.append(float((chosen < num_weak).mean()))

        force_mat = torch.stack(forces)
        oracle_i = oracle[i].flatten()[None, :]
        align = (force_mat * oracle_i).sum(dim=1) / (force_mat.norm(dim=1) * oracle_i.norm(dim=1)).clamp_min(1e-8)
        normed = force_mat / force_mat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        sim = normed @ normed.T
        mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
        force_alignment.append(float(align.mean().item()))
        force_stability.append(float(sim[mask].mean().item()))
        sampled_weak_mass.append(float(np.mean(weak_masses)))

    align_arr = np.asarray(force_alignment, dtype=np.float64)
    stability_arr = np.asarray(force_stability, dtype=np.float64)
    sampled_mass_arr = np.asarray(sampled_weak_mass, dtype=np.float64)
    top3_arr = np.asarray(top3_weak_mass, dtype=np.float64)
    top5_arr = np.asarray(top5_weak_mass, dtype=np.float64)

    def with_ci(key: str, arr: np.ndarray) -> dict[str, float]:
        mean, ci = mean_ci(arr)
        return {key: mean, f"{key}_ci95": ci, f"{key}_median": float(np.median(arr))}

    return {
        "top1_weak": float(top1_weak),
        **with_ci("top3_weak_mass", top3_arr),
        **with_ci("top5_weak_mass", top5_arr),
        **with_ci("sampled_force_weak_mass", sampled_mass_arr),
        **with_ci("oracle_force_cosine", align_arr),
        **with_ci("resample_force_stability", stability_arr),
        "oracle_force_cosine_values": align_arr.tolist(),
        "resample_force_stability_values": stability_arr.tolist(),
    }


def draw_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    value: float,
    color: tuple[int, int, int],
    label: str,
    scale: tuple[float, float] = (0.0, 1.0),
) -> None:
    lo, hi = scale
    frac = max(0.0, min(1.0, (value - lo) / max(1e-8, hi - lo)))
    draw.rounded_rectangle((x, y, x + width, y + height), radius=8, fill=(243, 244, 246))
    draw.rounded_rectangle((x, y, x + int(frac * width), y + height), radius=8, fill=color)
    draw_text(draw, (x + width + 15, y - 1), label, font(23, True), color)


def draw_box(
    draw: ImageDraw.ImageDraw,
    cx: int,
    rect: tuple[int, int, int, int],
    values: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    seed: int,
    value_range: tuple[float, float],
) -> None:
    _, y0, _, y1 = rect
    lo, hi = value_range

    def yy(v: float) -> int:
        frac = (float(v) - lo) / max(1e-8, hi - lo)
        return y1 - int(max(0.0, min(1.0, frac)) * (y1 - y0))

    q10, q25, q50, q75, q90 = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
    rng = np.random.default_rng(seed)
    sample = values if len(values) <= 120 else rng.choice(values, 120, replace=False)
    for v in sample:
        jitter = int(rng.normal(0, 10))
        draw.ellipse((cx + jitter - 3, yy(v) - 3, cx + jitter + 3, yy(v) + 3), fill=color + (65,))
    draw.line((cx, yy(q10), cx, yy(q90)), fill=color, width=5)
    draw.rounded_rectangle((cx - 38, yy(q75), cx + 38, yy(q25)), radius=7, fill=(255, 255, 255), outline=color, width=5)
    draw.line((cx - 45, yy(q50), cx + 45, yy(q50)), fill=color, width=6)
    tw, _ = text_size(draw, label, font(22, True))
    draw_text(draw, (cx - tw // 2, y1 + 18), label, font(22, True), color)
    draw_text(draw, (cx - 34, yy(q50) - 34), f"{q50:.2f}", font(20, True), color)


def make_figure(path_png: Path, path_pdf: Path, metrics: dict) -> Image.Image:
    canvas = Image.new("RGB", (2300, 1040), "white")
    draw = ImageDraw.Draw(canvas, "RGBA")
    left = 70
    draw_text(draw, (left, 45), "Metric turns noisy supports into coherent drift directions", font(43, True), DARK)
    draw_text(
        draw,
        (left, 103),
        "Known local positives are weak anchor views; false supports are Euclidean-mined layout impostors.",
        font(26),
        GRAY,
    )

    draw_text(draw, (left, 178), "A  Nearest-support purity", font(31, True), DARK)
    draw_text(
        draw,
        (left, 218),
        f"{metrics['num_anchors']} generated anchors; each pool has {metrics['num_weak']} weak positives and "
        f"{metrics['num_impostors']} Euclidean-mined layout impostors.",
        font(22),
        GRAY,
    )

    metric_cols = [
        ("Top-1 weak", "top1_weak", (0.0, 1.0), True),
        ("Top-3 weak mass", "top3_weak_mass", (0.0, 1.0), True),
        ("Sampled force weak mass", "sampled_force_weak_mass", (0.0, 1.0), True),
        ("Oracle force cosine", "oracle_force_cosine", (-0.2, 1.0), False),
    ]
    row_names = [("Euclidean", "euclidean", BLUE), ("MS-stat", "ms_stat", ORANGE)]
    table_x = left
    table_y = 310
    col_w = 435
    for c, (title, _, _, _) in enumerate(metric_cols):
        draw_text(draw, (table_x + 220 + c * col_w, table_y - 45), title, font(22, True), DARK)
    for r, (label, key, color) in enumerate(row_names):
        y = table_y + r * 112
        draw_text(draw, (table_x, y + 20), label, font(26, True), color)
        for c, (_, metric_key, scale, percent) in enumerate(metric_cols):
            x = table_x + 220 + c * col_w
            value = metrics["results"][key][metric_key]
            txt = f"{value * 100:.1f}%" if percent else f"{value:.2f}"
            draw_bar(draw, x, y, 275, 46, value, color, txt, scale)

    draw_text(draw, (left, 600), "B  Resampled drift-force distributions", font(31, True), DARK)
    plot = (left + 65, 675, 1120, 925)
    x0, y0, x1, y1 = plot
    draw.rectangle(plot, outline=(229, 231, 235), width=2)
    value_range = (-0.2, 1.0)
    for tick in [-0.2, 0.0, 0.4, 0.8, 1.0]:
        y = y1 - int((tick - value_range[0]) / (value_range[1] - value_range[0]) * (y1 - y0))
        draw.line((x0 - 7, y, x1, y), fill=GRID, width=1)
        draw_text(draw, (x0 - 62, y - 12), f"{tick:.1f}", font(20), GRAY)
    groups = [
        ("Oracle\ncosine", "oracle_force_cosine_values", x0 + 245),
        ("Resample\nstability", "resample_force_stability_values", x0 + 705),
    ]
    for gi, (label, key, cx) in enumerate(groups):
        e_vals = np.asarray(metrics["results"]["euclidean"][key], dtype=np.float64)
        m_vals = np.asarray(metrics["results"]["ms_stat"][key], dtype=np.float64)
        draw_box(draw, cx - 50, plot, e_vals, BLUE, "Euc", 11 + gi, value_range)
        draw_box(draw, cx + 50, plot, m_vals, ORANGE, "MS", 23 + gi, value_range)
        for j, line in enumerate(label.split("\n")):
            tw, _ = text_size(draw, line, font(23, True))
            draw_text(draw, (cx - tw // 2, y1 + 57 + 28 * j), line, font(23, True), DARK)

    note_x = 1265
    draw_text(draw, (note_x, 600), "C  Drift-estimator read", font(31, True), DARK)
    draw.rounded_rectangle((note_x, 670, 2190, 905), radius=14, fill=(249, 250, 251), outline=(229, 231, 235))
    e = metrics["results"]["euclidean"]
    m = metrics["results"]["ms_stat"]
    gain_align = m["oracle_force_cosine"] - e["oracle_force_cosine"]
    gain_stab = m["resample_force_stability"] - e["resample_force_stability"]
    gain_mass = m["sampled_force_weak_mass"] - e["sampled_force_weak_mass"]
    draw_text(draw, (note_x + 34, 705), f"MS-stat adds {gain_mass * 100:.1f} points weak-support mass.", font(24, True), ORANGE)
    draw_text(draw, (note_x + 34, 750), f"Oracle-force cosine increases by {gain_align:.2f}.", font(24, True), ORANGE)
    if abs(gain_stab) < 0.03:
        stability_text = (
            "Resampling stability stays comparable "
            f"({e['resample_force_stability']:.2f} vs {m['resample_force_stability']:.2f})."
        )
    else:
        stability_text = f"Resampling stability changes by {gain_stab:.2f}."
    draw_text(draw, (note_x + 34, 795), stability_text, font(24, True), DARK)
    draw_text(draw, (note_x + 34, 850), "This tests the metric where it is used: finite supports", font(22), GRAY)
    draw_text(draw, (note_x + 34, 881), "must create a coherent attraction direction.", font(22), GRAY)

    save_png_pdf(canvas, path_png, path_pdf)
    return canvas


def run_experiment(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    rng = random.Random(args.seed)
    by_label, _ = index_latent_cache("train")
    eligible = [label for label, paths in by_label.items() if len(paths) >= 8]
    labels = torch.tensor(rng.sample(eligible, args.num_anchors), dtype=torch.long)

    model = load_generator(args.checkpoint, device)
    anchors = generate_latents(model, labels, args.cfg_scale, args.gen_batch_size, args.seed + 7, device)
    weak_pools = [weak_view(anchors, args.seed + 101 + 17 * i) for i in range(args.num_weak)]
    weak_pool = torch.stack(weak_pools, dim=1)
    candidate_count = max(args.num_impostors, args.hard_impostor_candidate_pool)
    impostor_candidates = synthetic_impostor_bank(anchors, candidate_count, args.seed + 809)

    activation_fn = build_feature_fn(args.mae_path, device)
    all_latents = torch.cat([anchors, weak_pool.reshape(-1, 32, 32, 4), impostor_candidates.reshape(-1, 32, 32, 4)], dim=0)
    desc = extract_descriptors(all_latents, activation_fn, device, args.feature_batch_size)
    n = args.num_anchors
    k = args.num_weak + args.num_impostors

    e_all = standardize(desc.euc)
    m_all = standardize(desc.ms)
    e_anchor = e_all[:n]
    m_anchor = m_all[:n]
    e_weak = e_all[n : n + n * args.num_weak].reshape(n, args.num_weak, -1)
    m_weak = m_all[n : n + n * args.num_weak].reshape(n, args.num_weak, -1)
    e_candidate = e_all[n + n * args.num_weak :].reshape(n, candidate_count, -1)
    m_candidate = m_all[n + n * args.num_weak :].reshape(n, candidate_count, -1)
    hard_idx = torch.argsort(dist_many(e_anchor, e_candidate, "euc"), dim=1)[:, : args.num_impostors]

    def gather(pool_tensor: torch.Tensor) -> torch.Tensor:
        idx = hard_idx[:, :, None].expand(-1, -1, pool_tensor.shape[-1])
        return torch.gather(pool_tensor, dim=1, index=idx)

    e_impostor = gather(e_candidate)
    m_impostor = gather(m_candidate)
    flat_hard = hard_idx[:, :, None, None, None].expand(-1, -1, 32, 32, 4)
    impostor_pool = torch.gather(impostor_candidates, dim=1, index=flat_hard)
    pool = torch.cat([weak_pool, impostor_pool], dim=1)
    e_pool = torch.cat([e_weak, e_impostor], dim=1)
    m_pool = torch.cat([m_weak, m_impostor], dim=1)
    e_dist = dist_many(e_anchor, e_pool, "euc")
    m_dist = dist_many(m_anchor, m_pool, "ms")

    return {
        "checkpoint": args.checkpoint,
        "cfg_scale": args.cfg_scale,
        "mae_path": args.mae_path,
        "seed": args.seed,
        "num_anchors": args.num_anchors,
        "num_weak": args.num_weak,
        "num_impostors": args.num_impostors,
        "hard_impostor_candidate_pool": candidate_count,
        "resamples": args.resamples,
        "sample_weak": args.sample_weak,
        "sample_impostors": args.sample_impostors,
        "force_topk": args.force_topk,
        "results": {
            "euclidean": summarize_metric(
                e_dist,
                pool,
                anchors,
                num_weak=args.num_weak,
                resamples=args.resamples,
                sample_weak=args.sample_weak,
                sample_impostors=args.sample_impostors,
                force_topk=args.force_topk,
                seed=args.seed + 3001,
            ),
            "ms_stat": summarize_metric(
                m_dist,
                pool,
                anchors,
                num_weak=args.num_weak,
                resamples=args.resamples,
                sample_weak=args.sample_weak,
                sample_impostors=args.sample_impostors,
                force_topk=args.force_topk,
                seed=args.seed + 4001,
            ),
        },
    }


def json_ready(metrics: dict) -> dict:
    out = json.loads(json.dumps(metrics))
    for method in ("euclidean", "ms_stat"):
        out["results"][method].pop("oracle_force_cosine_values", None)
        out["results"][method].pop("resample_force_stability_values", None)
    return out


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_experiment(args)
    (out_dir / "support_force_stability.json").write_text(json.dumps(json_ready(metrics), indent=2) + "\n", encoding="utf-8")
    make_figure(out_dir / "support_force_stability.png", out_dir / "support_force_stability.pdf", metrics)
    print(json.dumps(json_ready(metrics), indent=2))


if __name__ == "__main__":
    main()
