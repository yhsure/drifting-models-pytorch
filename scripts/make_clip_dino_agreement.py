"""Measure agreement with a CLIP+DINO visual-consensus metric.

This experiment is deliberately external to the proposed MAE multiscale-stat
metric. It decodes generated anchors and controlled candidate views to RGB,
builds a reference distance by averaging z-scored CLIP and DINOv2 cosine
distances, and asks whether Euclidean or MS-stat better matches that reference.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

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
    decode_latents_to_pil,
    dist_many,
    dist_one,
    draw_text,
    extract_descriptors,
    font,
    generate_latents,
    index_latent_cache,
    load_generator,
    patch_shuffle_corruption,
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
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino-model", default="facebook/dinov2-small")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--num-anchors", type=int, default=128)
    parser.add_argument("--hard-candidate-pool", type=int, default=96)
    parser.add_argument("--feature-batch-size", type=int, default=96)
    parser.add_argument("--gen-batch-size", type=int, default=24)
    parser.add_argument("--decode-batch-size", type=int, default=64)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = rankdata(a)
    rb = rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def pairwise_agreement(a: np.ndarray, b: np.ndarray) -> float:
    aa = a[:, None] - a[None, :]
    bb = b[:, None] - b[None, :]
    mask = np.triu(np.ones_like(aa, dtype=bool), 1)
    return float(((aa[mask] * bb[mask]) > 0).mean())


def clip_embeddings(images: list[Image.Image], device: torch.device, model_name: str, batch_size: int) -> torch.Tensor:
    from transformers import CLIPImageProcessor, CLIPModel

    processor = CLIPImageProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    parts: list[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        pixel_values = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.inference_mode():
            emb = model.get_image_features(pixel_values=pixel_values).float()
        parts.append(F.normalize(emb, dim=-1).cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(parts, dim=0)


def dino_embeddings(images: list[Image.Image], device: torch.device, model_name: str, batch_size: int) -> torch.Tensor:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    parts: list[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        pixel_values = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.inference_mode():
            outputs = model(pixel_values=pixel_values)
            emb = getattr(outputs, "pooler_output", None)
            if emb is None:
                emb = outputs.last_hidden_state[:, 0]
            emb = emb.float()
        parts.append(F.normalize(emb, dim=-1).cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(parts, dim=0)


def mean_ci(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))


def summarize(metric_distances: np.ndarray, reference: np.ndarray) -> dict:
    pairwise = []
    correlations = []
    top1 = []
    weak_first = []
    for metric_row, ref_row in zip(metric_distances, reference):
        pairwise.append(pairwise_agreement(metric_row, ref_row))
        correlations.append(spearman(metric_row, ref_row))
        top1.append(float(np.argmin(metric_row) == np.argmin(ref_row)))
        weak_first.append(float(np.argmin(metric_row) == 0))
    pair_mean, pair_ci = mean_ci(pairwise)
    corr_mean, corr_ci = mean_ci(correlations)
    top1_mean, top1_ci = mean_ci(top1)
    weak_mean, weak_ci = mean_ci(weak_first)
    return {
        "pairwise_agreement": pair_mean,
        "pairwise_agreement_ci95": pair_ci,
        "spearman": corr_mean,
        "spearman_ci95": corr_ci,
        "top1_match": top1_mean,
        "top1_match_ci95": top1_ci,
        "weak_first": weak_mean,
        "weak_first_ci95": weak_ci,
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
    draw.rounded_rectangle((x, y, x + width, y + height), radius=9, fill=(243, 244, 246))
    draw.rounded_rectangle((x, y, x + int(frac * width), y + height), radius=9, fill=color)
    draw_text(draw, (x + width + 16, y - 1), label, font(23, True), color)


def make_figure(path_png: Path, path_pdf: Path, metrics: dict) -> Image.Image:
    image = Image.new("RGB", (2200, 980), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    left = 70
    draw_text(draw, (left, 45), "Agreement with CLIP+DINO visual consensus", font(44, True), DARK)
    draw_text(
        draw,
        (left, 104),
        "Reference distance is the average of z-scored CLIP and DINOv2 cosine distances over controlled candidates.",
        font(27),
        GRAY,
    )

    draw_text(draw, (left, 175), "A  Agreement scores", font(31, True), DARK)
    table_x = left
    table_y = 245
    metric_cols = [
        ("Pairwise agreement", "pairwise_agreement", (0.50, 0.95)),
        ("Spearman rank", "spearman", (0.35, 0.90)),
        ("Top-1 match", "top1_match", (0.00, 1.00)),
        ("Weak view top-1", "weak_first", (0.00, 1.00)),
    ]
    row_names = [("Euclidean", "euclidean", BLUE), ("MS-stat", "ms_stat", ORANGE)]
    col_w = 410
    row_h = 115
    draw_text(draw, (table_x, table_y - 45), f"{metrics['num_anchors']} generated anchors; candidates are weak view, roll mix, patch shuffle, and Euc-mined impostor.", font(22), GRAY)
    for c, (title, _, _) in enumerate(metric_cols):
        x = table_x + 240 + c * col_w
        draw_text(draw, (x, table_y), title, font(21, True), DARK)
    for r, (label, key, color) in enumerate(row_names):
        y = table_y + 55 + r * row_h
        draw_text(draw, (table_x, y + 21), label, font(25, True), color)
        for c, (_, metric_key, scale) in enumerate(metric_cols):
            x = table_x + 240 + c * col_w
            value = metrics["results"][key][metric_key]
            if metric_key == "spearman":
                txt = f"{value:.2f}"
            else:
                txt = f"{value * 100:.1f}%"
            draw_bar(draw, x, y, 260, 44, value, color, txt, scale)

    draw_text(draw, (left, 575), "B  CLIP+DINO reference top-1 distribution", font(31, True), DARK)
    dist = metrics["reference_top1_distribution"]
    labels = ["Weak", "Roll", "Patch", "Euc-mined"]
    colors = [GREEN, PURPLE, PURPLE, PURPLE]
    total = max(1, sum(dist))
    chart_x = left
    chart_y = 650
    bar_w = 115
    gap = 75
    chart_h = 210
    draw.line((chart_x, chart_y + chart_h, chart_x + 4 * (bar_w + gap), chart_y + chart_h), fill=GRID, width=2)
    for i, (label, count, color) in enumerate(zip(labels, dist, colors)):
        x = chart_x + i * (bar_w + gap)
        pct = count / total
        h = int(chart_h * pct)
        draw.rounded_rectangle((x, chart_y + chart_h - h, x + bar_w, chart_y + chart_h), radius=10, fill=color)
        pct_text = f"{pct * 100:.0f}%"
        tw, _ = text_size(draw, pct_text, font(25, True))
        draw_text(draw, (x + bar_w // 2 - tw // 2, chart_y + chart_h - h - 36), pct_text, font(25, True), color)
        lw, _ = text_size(draw, label, font(22))
        draw_text(draw, (x + bar_w // 2 - lw // 2, chart_y + chart_h + 18), label, font(22), GRAY)

    note_x = 930
    draw.rounded_rectangle((note_x, 620, 2100, 875), radius=16, fill=(249, 250, 251), outline=(229, 231, 235))
    draw_text(draw, (note_x + 35, 655), "Interpretation", font(29, True), DARK)
    gain_pair = metrics["results"]["ms_stat"]["pairwise_agreement"] - metrics["results"]["euclidean"]["pairwise_agreement"]
    gain_top1 = metrics["results"]["ms_stat"]["top1_match"] - metrics["results"]["euclidean"]["top1_match"]
    draw_text(draw, (note_x + 35, 705), f"MS-stat improves pairwise agreement by {gain_pair * 100:.1f} points.", font(24, True), ORANGE)
    draw_text(draw, (note_x + 35, 745), f"MS-stat improves top-1 reference match by {gain_top1 * 100:.1f} points.", font(24, True), ORANGE)
    draw_text(draw, (note_x + 35, 800), "This supports the metric only in the controlled layout-trap regime;", font(22), GRAY)
    draw_text(draw, (note_x + 35, 832), "broad same-class retrieval should be reported separately.", font(22), GRAY)

    save_png_pdf(image, path_png, path_pdf)
    return image


def run_experiment(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    rng = random.Random(args.seed)
    by_label, _ = index_latent_cache("train")
    eligible = [label for label, paths in by_label.items() if len(paths) >= 64]
    labels_np = np.asarray(rng.sample(eligible, args.num_anchors), dtype=np.int64)
    labels = torch.from_numpy(labels_np)

    model = load_generator(args.checkpoint, device)
    generated = generate_latents(model, labels, args.cfg_scale, args.gen_batch_size, args.seed + 11, device)
    weak = weak_view(generated, args.seed + 23)
    roll = 0.55 * generated + 0.45 * torch.roll(generated, shifts=8, dims=2)
    patch = 0.50 * generated + 0.50 * patch_shuffle_corruption(generated, args.seed + 31, grid=4)
    hard_bank = synthetic_impostor_bank(generated, args.hard_candidate_pool, args.seed + 37)

    activation_fn = build_feature_fn(args.mae_path, device)
    all_latents = torch.cat([generated, weak, roll, patch, hard_bank.reshape(-1, 32, 32, 4)], dim=0)
    desc = extract_descriptors(all_latents, activation_fn, device, args.feature_batch_size)
    n = args.num_anchors

    def split(pack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pack = standardize(pack)
        return pack[:n], pack[n : 2 * n], pack[2 * n : 3 * n], pack[3 * n : 4 * n], pack[4 * n :].reshape(n, args.hard_candidate_pool, -1)

    e_a, e_w, e_r, e_p, e_h_pool = split(desc.euc)
    m_a, m_w, m_r, m_p, m_h_pool = split(desc.ms)
    hard_idx = torch.argsort(dist_many(e_a, e_h_pool, "euc"), dim=1)[:, 0]
    e_h = torch.stack([e_h_pool[i, int(hard_idx[i])] for i in range(n)])
    m_h = torch.stack([m_h_pool[i, int(hard_idx[i])] for i in range(n)])
    hard_latents = torch.stack([hard_bank[i, int(hard_idx[i])] for i in range(n)])

    decode_latents = []
    for i in range(n):
        decode_latents.extend([generated[i], weak[i], roll[i], patch[i], hard_latents[i]])
    images = decode_latents_to_pil(torch.stack(decode_latents), args.decode_batch_size)
    clip = clip_embeddings(images, device, args.clip_model, args.embed_batch_size)
    dino = dino_embeddings(images, device, args.dino_model, args.embed_batch_size)

    reference_rows = []
    for i in range(n):
        base = 5 * i
        candidate_slice = slice(base + 1, base + 5)
        d_clip = (1 - (clip[base : base + 1] @ clip[candidate_slice].T)).squeeze(0).numpy()
        d_dino = (1 - (dino[base : base + 1] @ dino[candidate_slice].T)).squeeze(0).numpy()
        z_clip = (d_clip - d_clip.mean()) / (d_clip.std() + 1e-8)
        z_dino = (d_dino - d_dino.mean()) / (d_dino.std() + 1e-8)
        reference_rows.append(0.5 * (z_clip + z_dino))
    reference = np.stack(reference_rows)

    euclidean_distances = np.stack(
        [
            dist_one(e_a, e_w, "euc").numpy(),
            dist_one(e_a, e_r, "euc").numpy(),
            dist_one(e_a, e_p, "euc").numpy(),
            dist_one(e_a, e_h, "euc").numpy(),
        ],
        axis=1,
    )
    ms_distances = np.stack(
        [
            dist_one(m_a, m_w, "ms").numpy(),
            dist_one(m_a, m_r, "ms").numpy(),
            dist_one(m_a, m_p, "ms").numpy(),
            dist_one(m_a, m_h, "ms").numpy(),
        ],
        axis=1,
    )

    return {
        "checkpoint": args.checkpoint,
        "cfg_scale": args.cfg_scale,
        "clip_model": args.clip_model,
        "dino_model": args.dino_model,
        "num_anchors": args.num_anchors,
        "candidate_order": ["weak", "roll", "patch", "euc_mined"],
        "reference_top1_distribution": np.bincount(np.argmin(reference, axis=1), minlength=4).astype(int).tolist(),
        "results": {
            "euclidean": summarize(euclidean_distances, reference),
            "ms_stat": summarize(ms_distances, reference),
        },
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_experiment(args)
    (out_dir / "clip_dino_metric_agreement.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    make_figure(out_dir / "clip_dino_metric_agreement.png", out_dir / "clip_dino_metric_agreement.pdf", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
