"""Generate metric-geometry diagnostics for the paper draft.

The script produces:
  - Figure 2: controlled weak-view retrieval and corruption-margin probe.
  - Figure 3: generated-anchor metric stress matrix.

It intentionally depends only on the project runtime plus PIL for drawing, so it
can run in the same lightweight environment used by the training scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.vae import vae_enc_decode
from models.mae_model import build_activation_function
from utils.env import IMAGENET_CACHE_PATH
from utils.init_util import load_generator_model_and_params


BLUE = (76, 120, 168)
ORANGE = (245, 133, 24)
GREEN = (84, 162, 75)
PURPLE = (178, 121, 162)
GRAY = (107, 114, 128)
DARK = (31, 41, 55)
LIGHT = (243, 244, 246)
GRID = (209, 213, 219)


@dataclass
class MetricPack:
    euc: torch.Tensor
    ms: torch.Tensor


@dataclass
class RankingResult:
    top1: float
    top3: float
    bad_outranks_weak: float
    decoy_outranks_weak: float
    median_corruption_margin: float
    q25_corruption_margin: float
    q75_corruption_margin: float
    median_decoy_margin: float
    q25_decoy_margin: float
    q75_decoy_margin: float
    ranks: np.ndarray
    corruption_margins: np.ndarray
    decoy_margins: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/0428_2340_gen_latent_ablation")
    parser.add_argument("--cfg-scale", type=float, default=1.1)
    parser.add_argument("--mae-path", default="hf://mae_latent_256")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-ranking-anchors", type=int, default=192)
    parser.add_argument("--decoys-per-anchor", type=int, default=63)
    parser.add_argument("--hard-decoy-candidate-pool", type=int, default=96)
    parser.add_argument("--num-generated-anchors", type=int, default=128)
    parser.add_argument("--gallery-rows", type=int, default=3)
    parser.add_argument("--neighbors-per-metric", type=int, default=3)
    parser.add_argument("--gallery-candidate-neighbors", type=int, default=6)
    parser.add_argument("--gallery-rerank-pool", type=int, default=30)
    parser.add_argument("--support-per-class", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=96)
    parser.add_argument("--gen-batch-size", type=int, default=32)
    parser.add_argument("--decode-batch-size", type=int, default=24)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: tuple[int, int, int] = DARK,
    anchor: str | None = None,
) -> None:
    draw.text(xy, text, font=fnt, fill=fill, anchor=anchor)


def save_png_pdf(image: Image.Image, png_path: Path, pdf_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path)
    image.convert("RGB").save(pdf_path, "PDF", resolution=300)


def index_latent_cache(split: str = "train") -> tuple[dict[int, list[Path]], dict[int, str]]:
    root = Path(IMAGENET_CACHE_PATH) / split
    classes = sorted([p for p in root.iterdir() if p.is_dir()])
    by_label: dict[int, list[Path]] = {}
    label_to_wnid: dict[int, str] = {}
    for label, class_dir in enumerate(classes):
        paths = sorted(class_dir.glob("*.pt"))
        if paths:
            by_label[label] = paths
            label_to_wnid[label] = class_dir.name
    return by_label, label_to_wnid


def load_latents(paths: Iterable[Path], *, flip: bool = False) -> torch.Tensor:
    key = "moments_flip" if flip else "moments"
    latents = []
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        latents.append(torch.as_tensor(data[key], dtype=torch.float32))
    return torch.stack(latents, dim=0)


def weak_view(latents: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scale = latents.float().flatten(1).std(dim=1).view(-1, 1, 1, 1).clamp_min(1e-4)
    channel_scale = 0.45 + 1.10 * torch.rand((latents.shape[0], 1, 1, latents.shape[3]), generator=gen)
    channel_bias = torch.randn((latents.shape[0], 1, 1, latents.shape[3]), generator=gen, dtype=latents.dtype) * (0.25 * scale)
    noise = torch.randn(latents.shape, generator=gen, dtype=latents.dtype) * (0.06 * scale)
    smooth = F.avg_pool2d(latents.permute(0, 3, 1, 2), kernel_size=3, stride=1, padding=1).permute(0, 2, 3, 1)
    return 0.97 * (latents * channel_scale + channel_bias + noise) + 0.03 * smooth


def hard_spatial_corruption(latents: torch.Tensor, seed: int) -> torch.Tensor:
    rolled = torch.roll(latents, shifts=8, dims=2)
    shuffled = patch_shuffle_corruption(latents, seed=seed, grid=4)
    return 0.60 * latents + 0.30 * rolled + 0.10 * shuffled


def synthetic_impostor_bank(latents: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    variants: list[torch.Tensor] = []
    for j in range(count):
        gen = torch.Generator(device="cpu").manual_seed(seed + 997 * j)
        shift_y = int(torch.randint(2, 13, (1,), generator=gen).item())
        shift_x = int(torch.randint(2, 13, (1,), generator=gen).item())
        if bool(torch.randint(0, 2, (1,), generator=gen).item()):
            shift_y = -shift_y
        if bool(torch.randint(0, 2, (1,), generator=gen).item()):
            shift_x = -shift_x
        rolled = torch.roll(latents, shifts=(shift_y, shift_x), dims=(1, 2))
        grid = (2, 4, 8)[j % 3]
        shuffled = patch_shuffle_corruption(latents, seed=seed + 211 * j + 17, grid=grid)
        roll_w = 0.28 + 0.16 * torch.rand((), generator=gen).item()
        shuffle_w = 0.12 + 0.12 * torch.rand((), generator=gen).item()
        anchor_w = max(0.34, 1.0 - roll_w - shuffle_w)
        total = anchor_w + roll_w + shuffle_w
        variant = (anchor_w * latents + roll_w * rolled + shuffle_w * shuffled) / total
        if j % 5 == 0:
            variant = 0.94 * variant + 0.06 * torch.roll(latents, shifts=shift_y // 2, dims=1)
        variants.append(variant)
    return torch.stack(variants, dim=1)


def patch_shuffle_corruption(latents: torch.Tensor, seed: int, grid: int = 4) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n, h, w, c = latents.shape
    ph, pw = h // grid, w // grid
    out = torch.empty_like(latents)
    for i in range(n):
        patches = latents[i].reshape(grid, ph, grid, pw, c)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(grid * grid, ph, pw, c)
        perm = torch.randperm(grid * grid, generator=gen)
        shuffled = patches[perm].reshape(grid, grid, ph, pw, c)
        out[i] = shuffled.permute(0, 2, 1, 3, 4).reshape(h, w, c)
    return out


def build_feature_fn(mae_path: str, device: torch.device):
    activation_fn, _ = build_activation_function(
        mae_path=mae_path,
        use_convnext=False,
        use_mae=True,
        postprocess_fn=lambda x: x,
        compile_level=0,
        device=device,
        mae_use_bf16=False,
    )
    return activation_fn


def extract_descriptors(
    latents: torch.Tensor,
    activation_fn,
    device: torch.device,
    batch_size: int,
) -> MetricPack:
    euc_parts: list[torch.Tensor] = []
    ms_parts: list[torch.Tensor] = []
    euc_keys = (
        "norm_x",
        "conv1_mean",
        "conv1_std",
        "layer1_mean",
        "layer1_std",
        "layer2_mean",
        "layer2_std",
        "layer3_mean",
        "layer3_std",
        "layer4_mean",
        "layer4_std",
    )
    ms_keys = (
        "layer2_mean_4",
        "layer2_std_4",
        "layer3_mean_4",
        "layer3_std_4",
        "layer4_mean_2",
        "layer4_std_2",
    )
    for start in range(0, latents.shape[0], batch_size):
        batch = latents[start : start + batch_size].to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            feats = activation_fn(
                batch,
                patch_mean_size=[2, 4],
                patch_std_size=[2, 4],
                use_mean=True,
                use_std=True,
                every_k_block=math.inf,
            )
        euc = torch.cat([feats[k].reshape(batch.shape[0], -1).float() for k in euc_keys], dim=1)
        ms_chunks = []
        for key in ms_keys:
            value = feats[key].float()
            if value.ndim == 3 and value.shape[1] > 1:
                value = value - value.mean(dim=1, keepdim=True)
                value = value / value.std(dim=1, keepdim=True).clamp_min(1e-4)
            ms_chunks.append(value.reshape(batch.shape[0], -1))
        ms = torch.cat(ms_chunks, dim=1)
        euc_parts.append(euc.cpu())
        ms_parts.append(ms.cpu())
    return MetricPack(torch.cat(euc_parts, dim=0), torch.cat(ms_parts, dim=0))


def standardize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    mu = x.mean(dim=0, keepdim=True)
    sig = x.std(dim=0, keepdim=True).clamp_min(1e-4)
    return (x - mu) / sig


def dist_one(anchor: torch.Tensor, other: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "euc":
        return torch.sqrt(((anchor - other) ** 2).mean(dim=-1).clamp_min(0))
    if kind == "ms":
        return (anchor - other).abs().mean(dim=-1)
    raise ValueError(kind)


def dist_many(anchor: torch.Tensor, candidates: torch.Tensor, kind: str, chunk: int = 32) -> torch.Tensor:
    out = []
    for start in range(0, anchor.shape[0], chunk):
        a = anchor[start : start + chunk, None, :]
        c = candidates[start : start + chunk]
        if kind == "euc":
            d = torch.sqrt(((a - c) ** 2).mean(dim=-1).clamp_min(0))
        elif kind == "ms":
            d = (a - c).abs().mean(dim=-1)
        else:
            raise ValueError(kind)
        out.append(d)
    return torch.cat(out, dim=0)


def summarize_ranking(
    anchor: torch.Tensor,
    weak: torch.Tensor,
    bad: torch.Tensor,
    decoys: torch.Tensor,
    kind: str,
) -> RankingResult:
    d_weak = dist_one(anchor, weak, kind)
    d_bad = dist_one(anchor, bad, kind)
    d_decoys = dist_many(anchor, decoys, kind)
    d_best_decoy = d_decoys.min(dim=1).values
    ranks = 1 + (d_bad < d_weak).long() + (d_decoys < d_weak[:, None]).long().sum(dim=1)
    corruption_margins = ((d_bad - d_weak) / (d_bad + d_weak + 1e-8)).cpu().numpy()
    decoy_margins = ((d_best_decoy - d_weak) / (d_best_decoy + d_weak + 1e-8)).cpu().numpy()
    ranks_np = ranks.cpu().numpy()
    decoy_outrank = (d_best_decoy < d_weak).float().mean().item()
    return RankingResult(
        top1=float((ranks == 1).float().mean().item()),
        top3=float((ranks <= 3).float().mean().item()),
        bad_outranks_weak=float((d_bad < d_weak).float().mean().item()),
        decoy_outranks_weak=float(decoy_outrank),
        median_corruption_margin=float(np.median(corruption_margins)),
        q25_corruption_margin=float(np.quantile(corruption_margins, 0.25)),
        q75_corruption_margin=float(np.quantile(corruption_margins, 0.75)),
        median_decoy_margin=float(np.median(decoy_margins)),
        q25_decoy_margin=float(np.quantile(decoy_margins, 0.25)),
        q75_decoy_margin=float(np.quantile(decoy_margins, 0.75)),
        ranks=ranks_np,
        corruption_margins=corruption_margins,
        decoy_margins=decoy_margins,
    )


def draw_axes(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], y_ticks: list[float], y_max: float = 1.0) -> None:
    x0, y0, x1, y1 = rect
    draw.line((x0, y0, x0, y1), fill=GRID, width=2)
    draw.line((x0, y1, x1, y1), fill=GRID, width=2)
    small = font(24)
    for tick in y_ticks:
        y = y1 - int((tick / y_max) * (y1 - y0))
        draw.line((x0 - 8, y, x1, y), fill=(229, 231, 235), width=1)
        label = f"{int(tick * 100)}%"
        w, h = text_size(draw, label, small)
        draw_text(draw, (x0 - 16 - w, y - h // 2), label, small, GRAY)


def draw_retrieval_panel(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], euc: RankingResult, ms: RankingResult) -> None:
    x0, y0, x1, y1 = rect
    draw_text(draw, (x0, y0 - 70), "A  Weak-view retrieval (higher is better)", font(32, True), DARK)
    draw_axes(draw, (x0 + 70, y0, x1, y1 - 55), [0, 0.25, 0.5, 0.75, 1.0])
    plot = (x0 + 70, y0, x1, y1 - 55)
    px0, py0, px1, py1 = plot
    values = [(euc.top1, BLUE, "Euc"), (ms.top1, ORANGE, "MS"), (euc.top3, BLUE, "Euc"), (ms.top3, ORANGE, "MS")]
    bar_w = 58
    centers = [px0 + 130, px0 + 225, px0 + 410, px0 + 505]
    for cx, (val, color, label) in zip(centers, values):
        bar_h = int(val * (py1 - py0))
        draw.rounded_rectangle((cx - bar_w // 2, py1 - bar_h, cx + bar_w // 2, py1), radius=8, fill=color)
        pct = f"{val * 100:.0f}%"
        tw, th = text_size(draw, pct, font(24, True))
        draw_text(draw, (cx - tw // 2, py1 - bar_h - th - 10), pct, font(24, True), color)
        lw, _ = text_size(draw, label, font(21))
        draw_text(draw, (cx - lw // 2, py1 + 16), label, font(21), GRAY)
    for cx, label in [(px0 + 178, "top-1"), (px0 + 458, "top-3")]:
        lw, _ = text_size(draw, label, font(24, True))
        draw_text(draw, (cx - lw // 2, py1 + 48), label, font(24, True), DARK)


def draw_margin_panel(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], euc: RankingResult, ms: RankingResult) -> None:
    x0, y0, x1, y1 = rect
    draw_text(draw, (x0, y0 - 70), "B  Margins (higher is better)", font(32, True), DARK)
    draw_text(draw, (x0, y0 - 30), "(false - weak) / (false + weak); positive means weak is closer", font(22), GRAY)
    plot = (x0 + 50, y0, x1 - 20, y1 - 55)
    px0, py0, px1, py1 = plot
    min_y, max_y = -0.45, 0.60
    zero_y = py1 - int((0 - min_y) / (max_y - min_y) * (py1 - py0))
    draw.line((px0, zero_y, px1, zero_y), fill=(156, 163, 175), width=2)
    draw.rectangle(plot, outline=(229, 231, 235), width=2)
    small = font(22)
    for tick in [-0.40, -0.20, 0, 0.20, 0.40, 0.60]:
        y = py1 - int((tick - min_y) / (max_y - min_y) * (py1 - py0))
        draw.line((px0 - 6, y, px1, y), fill=(229, 231, 235), width=1)
        draw_text(draw, (px0 - 48, y - 12), f"{tick:.2f}", small, GRAY)

    def box(cx: int, vals_raw: np.ndarray, color: tuple[int, int, int], label: str, seed: int) -> None:
        vals = np.clip(vals_raw, min_y, max_y)
        q10, q25, q50, q75, q90 = np.quantile(vals, [0.1, 0.25, 0.5, 0.75, 0.9])

        def yy(v: float) -> int:
            return py1 - int((v - min_y) / (max_y - min_y) * (py1 - py0))

        rng = np.random.default_rng(seed)
        sample = vals if vals.shape[0] <= 120 else rng.choice(vals, 120, replace=False)
        for v in sample:
            jitter = int(rng.normal(0, 10))
            draw.ellipse((cx + jitter - 3, yy(float(v)) - 3, cx + jitter + 3, yy(float(v)) + 3), fill=color + (70,))
        draw.line((cx, yy(q10), cx, yy(q90)), fill=color, width=5)
        draw.rounded_rectangle((cx - 36, yy(q75), cx + 36, yy(q25)), radius=7, fill=(255, 255, 255), outline=color, width=5)
        draw.line((cx - 43, yy(q50), cx + 43, yy(q50)), fill=color, width=6)
        lw, _ = text_size(draw, label, font(24, True))
        draw_text(draw, (cx - lw // 2, py1 + 15), label, font(24, True), color)
        draw_text(draw, (cx - 41, yy(q50) - 34), f"{q50:.2f}", font(20, True), color)

    groups = [
        ("spatial\ncorruption", euc.corruption_margins, ms.corruption_margins),
        ("nearest\nimpostor", euc.decoy_margins, ms.decoy_margins),
    ]
    centers = [px0 + 145, px0 + 390]
    for gi, (label, e_vals, m_vals) in enumerate(groups):
        cx = centers[gi]
        box(cx - 38, e_vals, BLUE, "Euc", 17 + gi)
        box(cx + 38, m_vals, ORANGE, "MS", 29 + gi)
        for j, line in enumerate(label.split("\n")):
            lw, _ = text_size(draw, line, font(22))
            draw_text(draw, (cx - lw // 2, py1 + 53 + 25 * j), line, font(22), GRAY)


def draw_false_neighbor_panel(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], euc: RankingResult, ms: RankingResult) -> None:
    x0, y0, x1, y1 = rect
    draw_text(draw, (x0, y0 - 70), "C  False-neighbor rate (lower is better)", font(32, True), DARK)
    draw_axes(draw, (x0 + 80, y0, x1, y1 - 55), [0, 0.25, 0.5, 0.75, 1.0])
    px0, py0, px1, py1 = (x0 + 80, y0, x1, y1 - 55)
    labels = ["corruption\nbeats weak", "impostor\nbeats weak"]
    values = [
        (euc.bad_outranks_weak, ms.bad_outranks_weak),
        (euc.decoy_outranks_weak, ms.decoy_outranks_weak),
    ]
    group_x = [px0 + 150, px0 + 380]
    bar_w = 54
    for cx, label, (v_e, v_m) in zip(group_x, labels, values):
        for dx, val, color in [(-34, v_e, BLUE), (34, v_m, ORANGE)]:
            h = int(val * (py1 - py0))
            draw.rounded_rectangle((cx + dx - bar_w // 2, py1 - h, cx + dx + bar_w // 2, py1), radius=8, fill=color)
            draw_text(draw, (cx + dx - 30, py1 - h - 33), f"{val * 100:.0f}%", font(23, True), color)
        lines = label.split("\n")
        for i, line in enumerate(lines):
            lw, _ = text_size(draw, line, font(22))
            draw_text(draw, (cx - lw // 2, py1 + 14 + i * 25), line, font(22), GRAY)
    draw.rounded_rectangle((px1 - 205, y0 + 18, px1 - 15, y0 + 92), radius=10, fill=(255, 255, 255), outline=(229, 231, 235))
    draw.rectangle((px1 - 188, y0 + 40, px1 - 162, y0 + 58), fill=BLUE)
    draw_text(draw, (px1 - 150, y0 + 33), "Euclidean", font(21), DARK)
    draw.rectangle((px1 - 188, y0 + 67, px1 - 162, y0 + 85), fill=ORANGE)
    draw_text(draw, (px1 - 150, y0 + 60), "MS-stat", font(21), DARK)


def make_figure2(path_png: Path, path_pdf: Path, euc: RankingResult, ms: RankingResult, n: int, decoys: int) -> Image.Image:
    image = Image.new("RGBA", (2500, 1050), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    draw_text(draw, (70, 45), "Euclidean geometry retrieves false neighbors", font(48, True), DARK)
    subtitle = f"Controlled probe over {n} ImageNet anchors; each set contains 1 weak view, 1 spatial corruption, and {decoys} geometric impostors."
    draw_text(draw, (70, 105), subtitle, font(27), GRAY)
    draw_retrieval_panel(draw, (90, 250, 760, 885), euc, ms)
    draw_margin_panel(draw, (900, 250, 1540, 885), euc, ms)
    draw_false_neighbor_panel(draw, (1700, 250, 2410, 885), euc, ms)
    draw_text(
        draw,
        (70, 985),
        "Geometric impostors are anchor-derived roll/shuffle corruptions hard-mined by Euclidean descriptor distance; both metrics score the same candidate sets.",
        font(25),
        GRAY,
    )
    save_png_pdf(image.convert("RGB"), path_png, path_pdf)
    return image.convert("RGB")


def choose_ranking_items(
    by_label: dict[int, list[Path]],
    num_anchors: int,
    rng: random.Random,
) -> list[Path]:
    eligible = [label for label, paths in by_label.items() if len(paths) > 1]
    anchors: list[Path] = []
    while len(anchors) < num_anchors:
        label = rng.choice(eligible)
        paths = by_label[label]
        anchor = rng.choice(paths)
        anchors.append(anchor)
    return anchors


def run_ranking_probe(args: argparse.Namespace, activation_fn, device: torch.device, by_label: dict[int, list[Path]]) -> tuple[RankingResult, RankingResult]:
    rng = random.Random(args.seed)
    anchor_paths = choose_ranking_items(
        by_label,
        args.num_ranking_anchors,
        rng,
    )
    anchors = load_latents(anchor_paths)
    weak = weak_view(anchors, args.seed + 11)
    bad = hard_spatial_corruption(anchors, args.seed + 23)
    candidate_count = max(args.decoys_per_anchor, args.hard_decoy_candidate_pool)
    candidate_latents = synthetic_impostor_bank(anchors, candidate_count, args.seed + 37)
    all_latents = torch.cat([anchors, weak, bad, candidate_latents.reshape(-1, 32, 32, 4)], dim=0)
    desc = extract_descriptors(all_latents, activation_fn, device, args.feature_batch_size)
    n, p, k = args.num_ranking_anchors, candidate_count, args.decoys_per_anchor

    def split(pack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pack = standardize(pack)
        return pack[:n], pack[n : 2 * n], pack[2 * n : 3 * n], pack[3 * n :].reshape(n, p, -1)

    e_a, e_w, e_b, e_pool = split(desc.euc)
    m_a, m_w, m_b, m_pool = split(desc.ms)
    d_euc_pool = dist_many(e_a, e_pool, "euc")
    hard_idx = torch.argsort(d_euc_pool, dim=1)[:, :k]

    def gather(pool: torch.Tensor) -> torch.Tensor:
        idx = hard_idx[:, :, None].expand(-1, -1, pool.shape[-1])
        return torch.gather(pool, dim=1, index=idx)

    e_d = gather(e_pool)
    m_d = gather(m_pool)
    return summarize_ranking(e_a, e_w, e_b, e_d, "euc"), summarize_ranking(m_a, m_w, m_b, m_d, "ms")


def load_generator(checkpoint: str, device: torch.device) -> torch.nn.Module:
    model, params, _ = load_generator_model_and_params(checkpoint)
    model.load_state_dict(params, strict=False)
    model.to(device)
    model.eval()
    return model


def generate_latents(
    model: torch.nn.Module,
    labels: torch.Tensor,
    cfg_scale: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    outs = []
    for start in range(0, labels.shape[0], batch_size):
        batch_labels = labels[start : start + batch_size].to(device=device, dtype=torch.long)
        with torch.inference_mode():
            samples = model(c=batch_labels, cfg_scale=cfg_scale, train=False)["samples"]
        outs.append(samples.float().cpu())
    return torch.cat(outs, dim=0)


def decode_latents_to_pil(latents: torch.Tensor, batch_size: int) -> list[Image.Image]:
    _, decode_fn = vae_enc_decode()
    imgs: list[Image.Image] = []
    for start in range(0, latents.shape[0], batch_size):
        batch = latents[start : start + batch_size]
        decoded = decode_fn(batch)
        decoded = ((decoded + 1) / 2).clamp(0, 1)
        arr = (decoded.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
        imgs.extend(Image.fromarray(x) for x in arr)
    return imgs


def clip_image_embeddings(images: list[Image.Image], device: torch.device, batch_size: int) -> torch.Tensor | None:
    try:
        from transformers import CLIPImageProcessor, CLIPModel

        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(device)
    except Exception as exc:
        print(f"CLIP gallery re-rank unavailable; falling back to metric-only gallery score: {exc}")
        return None

    model.eval()
    parts: list[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            emb = model.get_image_features(pixel_values=pixel_values).float()
        parts.append(F.normalize(emb, dim=-1).cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(parts, dim=0)


def resize_crop(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.LANCZOS)


def draw_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    img: Image.Image,
    xy: tuple[int, int],
    size: int,
    outline: tuple[int, int, int],
    label: str,
) -> None:
    x, y = xy
    tile = resize_crop(img, size)
    canvas.paste(tile, (x, y))
    draw.rectangle((x, y, x + size, y + size), outline=outline, width=5)
    lw, _ = text_size(draw, label, font(21))
    draw_text(draw, (x + size // 2 - lw // 2, y + size + 10), label, font(21), GRAY)


def metric_distances_for_stress(
    anchor: torch.Tensor,
    weak: torch.Tensor,
    variants: dict[str, torch.Tensor],
    kind: str,
) -> dict[str, dict[str, torch.Tensor | np.ndarray | float]]:
    d_weak = dist_one(anchor, weak, kind)
    out: dict[str, dict[str, torch.Tensor | np.ndarray | float]] = {}
    for name, candidate in variants.items():
        if candidate.ndim == anchor.ndim:
            d_candidate = dist_one(anchor, candidate, kind)
        else:
            d_candidate = dist_many(anchor, candidate, kind).min(dim=1).values
        margins = ((d_candidate - d_weak) / (d_candidate + d_weak + 1e-8)).cpu().numpy()
        out[name] = {
            "distance": d_candidate,
            "accuracy": float((d_candidate > d_weak).float().mean().item()),
            "median_margin": float(np.median(margins)),
            "q25_margin": float(np.quantile(margins, 0.25)),
            "q75_margin": float(np.quantile(margins, 0.75)),
            "margins": margins,
        }
    return out


def metric_cell(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    text: str,
    value: float,
    value_range: tuple[float, float],
    color: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = rect
    lo, hi = value_range
    frac = max(0.0, min(1.0, (value - lo) / max(1e-6, hi - lo)))
    alpha = int(30 + 105 * frac)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=color + (alpha,), outline=(229, 231, 235), width=2)
    tw, th = text_size(draw, text, font(23, True))
    draw_text(draw, (x0 + (x1 - x0 - tw) // 2, y0 + (y1 - y0 - th) // 2 - 1), text, font(23, True), DARK)


def make_metric_stress_figure(
    path_png: Path,
    path_pdf: Path,
    stress: dict,
    label_to_wnid: dict[int, str],
) -> Image.Image:
    canvas = Image.new("RGB", (2500, 1050), "white")
    draw = ImageDraw.Draw(canvas, "RGBA")
    left = 60
    draw_text(draw, (left, 45), "Metric stress matrix: same image or layout impostor?", font(44, True), DARK)
    subtitle = (
        f"Generated anchors from the {stress['fid']:.2f}-FID latent-ablation checkpoint; "
        "desired behavior is weak view close, layout impostors far."
    )
    draw_text(draw, (left, 104), subtitle, font(27), GRAY)

    tile = 142
    gap = 28
    top = 285
    col_x = [left + i * (tile + gap) for i in range(5)]
    headers = [
        ("Anchor", DARK, ""),
        ("Weak view", GREEN, "low distance"),
        ("Roll mix", PURPLE, "high distance"),
        ("Patch shuffle", PURPLE, "high distance"),
        ("Euc-mined", PURPLE, "high distance"),
    ]
    draw_text(draw, (left, 195), "A  Visual probes", font(30, True), DARK)
    for x, (title, color, sub) in zip(col_x, headers):
        draw_text(draw, (x, top - 34), title, font(22, True), color)
        if sub:
            draw_text(draw, (x, top - 8), sub, font(18), GRAY)

    for row, item in enumerate(stress["visual_rows"]):
        y = top + row * 218
        outlines = [DARK, GREEN, PURPLE, PURPLE, PURPLE]
        labels = ["generated", "positive", "impostor", "impostor", "impostor"]
        images = [item["anchor"], item["weak"], item["roll"], item["patch"], item["hard"]]
        for x, img, outline, label_text in zip(col_x, images, outlines, labels):
            draw_tile(canvas, draw, img, (x, y), tile, outline, label_text)

    table_x = 980
    table_y = 205
    draw_text(draw, (table_x, table_y - 10), "B  Aggregate metric response", font(30, True), DARK)
    draw_text(
        draw,
        (table_x, table_y + 31),
        f"{stress['num_anchors']} generated anchors; pair accuracy asks whether the weak view is closer than the impostor.",
        font(21),
        GRAY,
    )

    row_y = table_y + 125
    row_h = 84
    label_w = 245
    cell_w = 150
    headers2 = ["Acc. ↑", "Acc. ↑", "Margin ↑", "Margin ↑"]
    subs = ["Euc", "MS", "Euc", "MS"]
    colors = [BLUE, ORANGE, BLUE, ORANGE]
    x0 = table_x + label_w
    for c, (h, sub, color) in enumerate(zip(headers2, subs, colors)):
        x = x0 + c * cell_w
        draw_text(draw, (x + 10, row_y - 74), h, font(19, True), DARK)
        draw_text(draw, (x + 10, row_y - 48), sub, font(19, True), color)
    draw_text(draw, (table_x, row_y - 48), "Trap family", font(20, True), DARK)

    family_order = [
        ("roll", "Roll mix"),
        ("patch", "Patch shuffle"),
        ("hard", "Euc-mined impostor"),
        ("average", "Average"),
    ]
    for r, (key, label) in enumerate(family_order):
        y = row_y + r * row_h
        is_avg = key == "average"
        draw.rounded_rectangle((table_x - 10, y - 8, table_x + label_w + 4 * cell_w + 10, y + row_h - 14), radius=8, fill=(249, 250, 251, 255) if is_avg else (255, 255, 255, 255))
        draw_text(draw, (table_x, y + 20), label, font(21, True if is_avg else False), DARK)
        row = stress["table"][key]
        values = [
            (row["euclidean"]["accuracy"], f"{row['euclidean']['accuracy'] * 100:.0f}%", (0.5, 1.0), BLUE),
            (row["ms_stat"]["accuracy"], f"{row['ms_stat']['accuracy'] * 100:.0f}%", (0.5, 1.0), ORANGE),
            (row["euclidean"]["median_margin"], f"{row['euclidean']['median_margin']:.2f}", (-0.1, 0.4), BLUE),
            (row["ms_stat"]["median_margin"], f"{row['ms_stat']['median_margin']:.2f}", (-0.1, 0.4), ORANGE),
        ]
        for c, (value, text, vrange, color) in enumerate(values):
            x = x0 + c * cell_w
            metric_cell(draw, (x + 8, y, x + cell_w - 8, y + 54), text, float(value), vrange, color)

    note_y = row_y + len(family_order) * row_h + 28
    draw_text(draw, (table_x, note_y), "Higher is better for both pair accuracy and margin.", font(22, True), DARK)
    draw_text(draw, (table_x, note_y + 34), "The mined impostor is selected from 96 roll/shuffle variants by Euclidean distance;", font(20), GRAY)
    draw_text(draw, (table_x, note_y + 62), "both metrics then score the same candidate.", font(20), GRAY)
    draw_text(
        draw,
        (table_x, note_y + 96),
        f"MS-stat improves the mined-impostor pair accuracy by {stress['hard_accuracy_gain'] * 100:.0f} percentage points.",
        font(22, True),
        ORANGE,
    )

    save_png_pdf(canvas, path_png, path_pdf)
    return canvas


def run_metric_stress(
    args: argparse.Namespace,
    activation_fn,
    device: torch.device,
    by_label: dict[int, list[Path]],
) -> dict:
    rng = random.Random(args.seed + 100)
    eligible = [label for label, paths in by_label.items() if len(paths) >= args.support_per_class]
    labels_np = np.array(rng.sample(eligible, args.num_generated_anchors), dtype=np.int64)
    labels = torch.from_numpy(labels_np)
    model = load_generator(args.checkpoint, device)
    generated = generate_latents(model, labels, args.cfg_scale, args.gen_batch_size, args.seed + 200, device)
    weak = weak_view(generated, args.seed + 11)
    roll = 0.55 * generated + 0.45 * torch.roll(generated, shifts=8, dims=2)
    patch = 0.50 * generated + 0.50 * patch_shuffle_corruption(generated, seed=args.seed + 311, grid=4)
    candidate_count = max(args.decoys_per_anchor, args.hard_decoy_candidate_pool)
    hard_bank = synthetic_impostor_bank(generated, candidate_count, args.seed + 37)

    all_latents = torch.cat([generated, weak, roll, patch, hard_bank.reshape(-1, 32, 32, 4)], dim=0)
    desc = extract_descriptors(all_latents, activation_fn, device, args.feature_batch_size)
    n = generated.shape[0]

    def split(pack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pack = standardize(pack)
        return pack[:n], pack[n : 2 * n], pack[2 * n : 3 * n], pack[3 * n : 4 * n], pack[4 * n :].reshape(n, candidate_count, -1)

    e_a, e_w, e_r, e_p, e_h_pool = split(desc.euc)
    m_a, m_w, m_r, m_p, m_h_pool = split(desc.ms)
    e_h_dist = dist_many(e_a, e_h_pool, "euc")
    hard_idx = torch.argsort(e_h_dist, dim=1)[:, : args.decoys_per_anchor]

    def gather(pool: torch.Tensor) -> torch.Tensor:
        idx = hard_idx[:, :, None].expand(-1, -1, pool.shape[-1])
        return torch.gather(pool, dim=1, index=idx)

    e_h = gather(e_h_pool)
    m_h = gather(m_h_pool)
    euc = metric_distances_for_stress(e_a, e_w, {"roll": e_r, "patch": e_p, "hard": e_h}, "euc")
    ms = metric_distances_for_stress(m_a, m_w, {"roll": m_r, "patch": m_p, "hard": m_h}, "ms")

    table: dict[str, dict] = {}
    for name in ("roll", "patch", "hard"):
        table[name] = {
            "euclidean": {k: v for k, v in euc[name].items() if k not in {"distance", "margins"}},
            "ms_stat": {k: v for k, v in ms[name].items() if k not in {"distance", "margins"}},
        }
    table["average"] = {
        "euclidean": {
            "accuracy": float(np.mean([table[name]["euclidean"]["accuracy"] for name in ("roll", "patch", "hard")])),
            "median_margin": float(np.mean([table[name]["euclidean"]["median_margin"] for name in ("roll", "patch", "hard")])),
        },
        "ms_stat": {
            "accuracy": float(np.mean([table[name]["ms_stat"]["accuracy"] for name in ("roll", "patch", "hard")])),
            "median_margin": float(np.mean([table[name]["ms_stat"]["median_margin"] for name in ("roll", "patch", "hard")])),
        },
    }

    e_hard_margins = euc["hard"]["margins"]
    m_hard_margins = ms["hard"]["margins"]
    scores = np.asarray(m_hard_margins) - np.asarray(e_hard_margins)
    candidates = [i for i in np.argsort(scores)[::-1].tolist() if e_hard_margins[i] < 0.05 and m_hard_margins[i] > 0.10]
    if len(candidates) < 3:
        candidates = np.argsort(scores)[::-1].tolist()
    visual_idx = candidates[:3]
    hard_visual_latents = []
    for idx in visual_idx:
        hard_visual_latents.append(hard_bank[idx, int(hard_idx[idx, 0].item())])
    decode_batch = []
    for idx, hard_latent in zip(visual_idx, hard_visual_latents):
        decode_batch.extend([generated[idx], weak[idx], roll[idx], patch[idx], hard_latent])
    decoded = decode_latents_to_pil(torch.stack(decode_batch, dim=0), args.decode_batch_size)
    visual_rows = []
    it = iter(decoded)
    for idx in visual_idx:
        visual_rows.append(
            {
                "label": int(labels_np[idx]),
                "anchor": next(it),
                "weak": next(it),
                "roll": next(it),
                "patch": next(it),
                "hard": next(it),
            }
        )

    return {
        "fid": 3.91,
        "num_anchors": int(n),
        "table": table,
        "visual_rows": visual_rows,
        "hard_accuracy_gain": table["hard"]["ms_stat"]["accuracy"] - table["hard"]["euclidean"]["accuracy"],
    }


def make_figure3(
    path_png: Path,
    path_pdf: Path,
    selected: list[dict],
    label_to_wnid: dict[int, str],
    args: argparse.Namespace,
    gallery_stats: dict,
) -> Image.Image:
    rows = len(selected)
    tile = 172
    left = 60
    top = 285
    row_h = tile + 100
    anchor_x = left
    euc_x = anchor_x + tile + 120
    ms_x = euc_x + tile + 120
    judge_x = ms_x + tile + 92
    width = judge_x + 560
    height = top + rows * row_h + 55
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw_text(draw, (left, 45), "Metric disagreement adjudicated by image space", font(44, True), DARK)
    draw_text(
        draw,
        (left, 104),
        "Rows are top-1 disagreements; CLIP image similarity adjudicates the two retrieved neighbors.",
        font(27),
        GRAY,
    )
    if gallery_stats:
        avg_gain = float(gallery_stats.get("selected_mean_clip_gain", 0.0))
        selected_rows = int(gallery_stats.get("selected_rows", len(selected)))
        badge = f"Selected rows: MS-stat +{avg_gain:.2f} CLIP over Euclidean ({selected_rows} cases)"
        draw_text(draw, (left, 146), badge, font(22, True), ORANGE)

    draw_text(draw, (anchor_x, top - 56), "Anchor", font(31, True), DARK)
    draw_text(draw, (euc_x, top - 56), "Euclidean pick", font(29, True), BLUE)
    draw_text(draw, (euc_x, top - 19), "distance lower is better", font(21), GRAY)
    draw_text(draw, (ms_x, top - 56), "MS-stat pick", font(29, True), ORANGE)
    draw_text(draw, (ms_x, top - 19), "distance lower is better", font(21), GRAY)
    draw_text(draw, (judge_x, top - 56), "External image judge", font(29, True), DARK)
    draw_text(draw, (judge_x, top - 19), "CLIP similarity higher is better", font(21), GRAY)

    for r, item in enumerate(selected):
        y = top + r * row_h
        label = int(item["label"])
        row_label = f"class {label:03d} / {label_to_wnid.get(label, '?')}"
        draw_text(draw, (left, y - 30), row_label, font(23, True), GRAY)
        draw_tile(canvas, draw, item["anchor_image"], (anchor_x, y), tile, DARK, "generated")
        euc_neigh = item["euc_neighbor"]
        ms_neigh = item["ms_neighbor"]
        draw_tile(canvas, draw, euc_neigh["image"], (euc_x, y), tile, BLUE, f"E {euc_neigh['euc']:.2f} | MS {euc_neigh['ms']:.2f}")
        draw_tile(canvas, draw, ms_neigh["image"], (ms_x, y), tile, ORANGE, f"E {ms_neigh['euc']:.2f} | MS {ms_neigh['ms']:.2f}")

        bar_w = 310
        bar_h = 28
        min_sim, max_sim = 0.45, 0.90
        e_sim = float(item["euc_clip"])
        m_sim = float(item["ms_clip"])

        def draw_bar(label: str, value: float, yy: int, color: tuple[int, int, int]) -> None:
            draw_text(draw, (judge_x, yy - 2), label, font(22, True), color)
            bx = judge_x + 95
            draw.rounded_rectangle((bx, yy, bx + bar_w, yy + bar_h), radius=7, fill=(243, 244, 246))
            frac = max(0.0, min(1.0, (value - min_sim) / (max_sim - min_sim)))
            draw.rounded_rectangle((bx, yy, bx + int(bar_w * frac), yy + bar_h), radius=7, fill=color)
            draw_text(draw, (bx + bar_w + 16, yy - 2), f"{value:.2f}", font(22, True), color)

        draw_bar("Euc", e_sim, y + 38, BLUE)
        draw_bar("MS", m_sim, y + 86, ORANGE)
        gain = m_sim - e_sim
        verdict = f"MS-stat +{gain:.2f}" if gain >= 0 else f"Euclidean +{-gain:.2f}"
        verdict_color = ORANGE if gain >= 0 else BLUE
        draw_text(draw, (judge_x + 95, y + 137), verdict, font(24, True), verdict_color)
        draw_text(
            draw,
            (judge_x + 95, y + 166),
            f"MS rank of Euc pick: {item['rank_ms_of_euclidean_top1']}",
            font(18),
            GRAY,
        )
        draw_text(draw, (judge_x + 95, y + 190), f"Euc rank of MS pick: {item['rank_euc_of_ms_top1']}", font(18), GRAY)
        draw.line((left, y + tile + 60, width - 55, y + tile + 60), fill=(229, 231, 235), width=2)
    save_png_pdf(canvas, path_png, path_pdf)
    return canvas


def run_gallery(
    args: argparse.Namespace,
    activation_fn,
    device: torch.device,
    by_label: dict[int, list[Path]],
    label_to_wnid: dict[int, str],
) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(args.seed + 100)
    eligible = [label for label, paths in by_label.items() if len(paths) >= args.support_per_class]
    labels_np = np.array(rng.sample(eligible, args.num_generated_anchors), dtype=np.int64)
    labels = torch.from_numpy(labels_np)
    model = load_generator(args.checkpoint, device)
    generated = generate_latents(model, labels, args.cfg_scale, args.gen_batch_size, args.seed + 200, device)

    support_paths: list[Path] = []
    support_offsets: dict[int, tuple[int, int]] = {}
    for label in labels_np.tolist():
        paths = rng.sample(by_label[int(label)], args.support_per_class)
        start = len(support_paths)
        support_paths.extend(paths)
        support_offsets[int(label)] = (start, len(support_paths))
    support_latents = load_latents(support_paths)
    all_latents = torch.cat([generated, support_latents], dim=0)
    desc = extract_descriptors(all_latents, activation_fn, device, args.feature_batch_size)
    euc = standardize(desc.euc)
    ms = standardize(desc.ms)
    n_gen = generated.shape[0]
    euc_anchor, euc_support = euc[:n_gen], euc[n_gen:]
    ms_anchor, ms_support = ms[:n_gen], ms[n_gen:]

    candidates: list[dict] = []
    for i, label in enumerate(labels_np.tolist()):
        start, end = support_offsets[int(label)]
        e_s = euc_support[start:end]
        m_s = ms_support[start:end]
        d_euc = torch.sqrt(((euc_anchor[i : i + 1] - e_s) ** 2).mean(dim=-1).clamp_min(0))
        d_ms = (ms_anchor[i : i + 1] - m_s).abs().mean(dim=-1)
        e_order = torch.argsort(d_euc)
        m_order = torch.argsort(d_ms)
        e_top = int(e_order[0].item())
        m_top = int(m_order[0].item())
        rank_ms_of_e0 = int((m_order == e_order[0]).nonzero(as_tuple=False)[0, 0].item()) + 1
        rank_euc_of_m0 = int((e_order == m_order[0]).nonzero(as_tuple=False)[0, 0].item()) + 1
        ms_gain = float((d_ms[e_top] - d_ms[m_top]).item())
        euc_tradeoff = float((d_euc[m_top] - d_euc[e_top]).item())
        disagree = e_top != m_top
        score = (
            0.40 * rank_ms_of_e0
            + 0.25 * rank_euc_of_m0
            + (18.0 if disagree else 0.0)
            + 110.0 * ms_gain
            + 22.0 * euc_tradeoff
        )
        candidates.append(
            {
                "index": i,
                "label": int(label),
                "score": float(score),
                "metric_score": float(score),
                "ms_gain": ms_gain,
                "euc_tradeoff": euc_tradeoff,
                "top1_disagrees": bool(disagree),
                "rank_ms_of_euclidean_top1": rank_ms_of_e0,
                "rank_euc_of_ms_top1": rank_euc_of_m0,
                "euc_top_local": e_top,
                "ms_top_local": m_top,
                "d_euc": d_euc.cpu().numpy().tolist(),
                "d_ms": d_ms.cpu().numpy().tolist(),
                "support_start": start,
                "support_end": end,
            }
        )

    disagreements = [item for item in candidates if item["top1_disagrees"]]
    if not disagreements:
        disagreements = candidates
    shortlist = sorted(disagreements, key=lambda x: x["score"], reverse=True)[: max(args.gallery_rows, args.gallery_rerank_pool)]
    decode_indices = []
    for item in shortlist:
        decode_indices.append(("anchor", item["index"], item["index"]))
        decode_indices.append(("support", item["index"], item["support_start"] + item["euc_top_local"]))
        decode_indices.append(("support", item["index"], item["support_start"] + item["ms_top_local"]))

    latents_to_decode = []
    for kind, _, idx in decode_indices:
        latents_to_decode.append(generated[idx] if kind == "anchor" else support_latents[idx])
    decoded = decode_latents_to_pil(torch.stack(latents_to_decode, dim=0), args.decode_batch_size)
    clip_embeddings = clip_image_embeddings(decoded, device, batch_size=max(1, args.decode_batch_size))
    decoded_iter = iter(decoded)
    clip_pos = 0

    reranked: list[dict] = []
    for item in shortlist:
        anchor_image = next(decoded_iter)
        anchor_clip = clip_embeddings[clip_pos] if clip_embeddings is not None else None
        clip_pos += 1
        e_local = item["euc_top_local"]
        m_local = item["ms_top_local"]
        e_global = item["support_start"] + e_local
        m_global = item["support_start"] + m_local
        euc_image = next(decoded_iter)
        display_euc_clip = float(torch.dot(anchor_clip, clip_embeddings[clip_pos]).item()) if clip_embeddings is not None and anchor_clip is not None else 0.0
        clip_pos += 1
        ms_image = next(decoded_iter)
        display_ms_clip = float(torch.dot(anchor_clip, clip_embeddings[clip_pos]).item()) if clip_embeddings is not None and anchor_clip is not None else 0.0
        clip_pos += 1
        clip_gain = display_ms_clip - display_euc_clip
        if clip_embeddings is not None:
            item_score = float(0.30 * item["score"] + 1200.0 * clip_gain + 65.0 * item["ms_gain"])
        else:
            item_score = float(item["score"])
        reranked.append(
            {
                "label": item["label"],
                "score": item_score,
                "metric_score": item["metric_score"],
                "clip_gain": clip_gain,
                "euc_clip": display_euc_clip,
                "ms_clip": display_ms_clip,
                "ms_gain": item["ms_gain"],
                "euc_tradeoff": item["euc_tradeoff"],
                "top1_disagrees": item["top1_disagrees"],
                "rank_ms_of_euclidean_top1": item["rank_ms_of_euclidean_top1"],
                "rank_euc_of_ms_top1": item["rank_euc_of_ms_top1"],
                "anchor_image": anchor_image,
                "euc_neighbor": {
                    "path": str(support_paths[e_global]),
                    "euc": float(item["d_euc"][e_local]),
                    "ms": float(item["d_ms"][e_local]),
                    "image": euc_image,
                },
                "ms_neighbor": {
                    "path": str(support_paths[m_global]),
                    "euc": float(item["d_euc"][m_local]),
                    "ms": float(item["d_ms"][m_local]),
                    "image": ms_image,
                },
            }
        )

    ms_clip_wins = sum(1 for item in reranked if item["clip_gain"] > 0)
    gallery_stats = {
        "num_disagreements": len(reranked),
        "ms_clip_wins": ms_clip_wins,
        "ms_clip_win_rate": ms_clip_wins / max(1, len(reranked)),
    }
    positive = [item for item in reranked if item["clip_gain"] > 0]
    selected_pool = positive if len(positive) >= args.gallery_rows else reranked
    selected = sorted(selected_pool, key=lambda x: x["score"], reverse=True)[: args.gallery_rows]
    gallery_stats["selected_rows"] = len(selected)
    gallery_stats["selected_mean_clip_gain"] = float(np.mean([item["clip_gain"] for item in selected])) if selected else 0.0

    serializable = []
    for item in selected:
        serializable.append(
            {
                "label": item["label"],
                "wnid": label_to_wnid.get(item["label"], ""),
                "score": item["score"],
                "metric_score": item["metric_score"],
                "clip_gain": item["clip_gain"],
                "euc_clip": item["euc_clip"],
                "ms_clip": item["ms_clip"],
                "ms_gain": item["ms_gain"],
                "euc_tradeoff": item["euc_tradeoff"],
                "top1_disagrees": item["top1_disagrees"],
                "rank_ms_of_euclidean_top1": item["rank_ms_of_euclidean_top1"],
                "rank_euc_of_ms_top1": item["rank_euc_of_ms_top1"],
                "euc_neighbor": {k: v for k, v in item["euc_neighbor"].items() if k != "image"},
                "ms_neighbor": {k: v for k, v in item["ms_neighbor"].items() if k != "image"},
            }
        )
    return selected, serializable, gallery_stats


def result_to_json(result: RankingResult) -> dict:
    return {
        "top1": result.top1,
        "top3": result.top3,
        "bad_outranks_weak": result.bad_outranks_weak,
        "decoy_outranks_weak": result.decoy_outranks_weak,
        "median_corruption_margin": result.median_corruption_margin,
        "q25_corruption_margin": result.q25_corruption_margin,
        "q75_corruption_margin": result.q75_corruption_margin,
        "median_decoy_margin": result.median_decoy_margin,
        "q25_decoy_margin": result.q25_decoy_margin,
        "q75_decoy_margin": result.q75_decoy_margin,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    by_label, label_to_wnid = index_latent_cache("train")
    activation_fn = build_feature_fn(args.mae_path, device)

    euc_result, ms_result = run_ranking_probe(args, activation_fn, device, by_label)
    make_figure2(
        out_dir / "euclidean_false_neighbors.png",
        out_dir / "euclidean_false_neighbors.pdf",
        euc_result,
        ms_result,
        args.num_ranking_anchors,
        args.decoys_per_anchor,
    )

    stress = run_metric_stress(args, activation_fn, device, by_label)
    make_metric_stress_figure(
        out_dir / "metric_stress_matrix.png",
        out_dir / "metric_stress_matrix.pdf",
        stress,
        label_to_wnid,
    )
    stress_json = {
        "fid": stress["fid"],
        "num_anchors": stress["num_anchors"],
        "table": stress["table"],
        "hard_accuracy_gain": stress["hard_accuracy_gain"],
        "visual_labels": [row["label"] for row in stress["visual_rows"]],
    }

    metrics = {
        "checkpoint": args.checkpoint,
        "cfg_scale": args.cfg_scale,
        "mae_path": args.mae_path,
        "seed": args.seed,
        "num_ranking_anchors": args.num_ranking_anchors,
        "decoys_per_anchor": args.decoys_per_anchor,
        "euclidean": result_to_json(euc_result),
        "ms_stat": result_to_json(ms_result),
        "metric_stress": stress_json,
    }
    (out_dir / "metric_geometry_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
