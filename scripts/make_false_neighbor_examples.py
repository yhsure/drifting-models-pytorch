"""Visual examples for the false-neighbor ranking probe."""

from __future__ import annotations

import argparse
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
    PURPLE,
    build_feature_fn,
    decode_latents_to_pil,
    dist_many,
    dist_one,
    draw_text,
    extract_descriptors,
    font,
    hard_spatial_corruption,
    index_latent_cache,
    load_latents,
    resize_crop,
    save_png_pdf,
    standardize,
    synthetic_impostor_bank,
    text_size,
    weak_view,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mae-path", default="hf://mae_latent_256")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-anchors", type=int, default=192)
    parser.add_argument("--decoys-per-anchor", type=int, default=63)
    parser.add_argument("--hard-decoy-candidate-pool", type=int, default=96)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--rank-offset", type=int, default=0)
    parser.add_argument("--feature-batch-size", type=int, default=96)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def choose_anchor_paths(by_label: dict[int, list[Path]], num_anchors: int, rng: random.Random) -> list[Path]:
    eligible = [label for label, paths in by_label.items() if len(paths) > 1]
    return [rng.choice(by_label[rng.choice(eligible)]) for _ in range(num_anchors)]


def make_gallery(path_png: Path, path_pdf: Path, rows: list[dict]) -> Image.Image:
    tile = 150
    x0 = 68
    y0 = 225
    gap = 24
    row_gap = 76
    columns = [
        ("Anchor", DARK),
        ("Weak view", GREEN),
        ("Spatial corrupt.", BLUE),
        ("Mined impostor", PURPLE),
        ("Mined impostor", PURPLE),
        ("Mined impostor", PURPLE),
    ]
    width = max(1440, x0 * 2 + len(columns) * tile + (len(columns) - 1) * gap)
    height = y0 + len(rows) * (tile + row_gap) + 70
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw_text(draw, (x0, 45), "False-neighbor examples", font(40, True), DARK)
    draw_text(
        draw,
        (x0, 100),
        "Weak views should be nearest; spatial corruptions and Euclidean-mined roll/shuffle impostors should be farther.",
        font(23),
        GRAY,
    )
    for c, (title, color) in enumerate(columns):
        x = x0 + c * (tile + gap)
        draw_text(draw, (x, y0 - 42), title, font(18, True), color)

    for r, row in enumerate(rows):
        y = y0 + r * (tile + row_gap)
        images = [row["anchor"], row["weak"], row["bad"], *row["impostors"]]
        colors = [DARK, GREEN, BLUE, PURPLE, PURPLE, PURPLE]
        for c, (img, color) in enumerate(zip(images, colors)):
            x = x0 + c * (tile + gap)
            canvas.paste(resize_crop(img, tile), (x, y))
            draw.rectangle((x, y, x + tile, y + tile), outline=color, width=5)
        label = (
            f"row {r + 1}: weak rank Euc {row['euc_rank']}, MS {row['ms_rank']}; "
            f"best-impostor margin Euc {row['euc_decoy_margin']:.2f}, MS {row['ms_decoy_margin']:.2f}"
        )
        draw_text(draw, (x0, y + tile + 16), label, font(19), GRAY)

    save_png_pdf(canvas, path_png, path_pdf)
    return canvas


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rng = random.Random(args.seed)

    by_label, _ = index_latent_cache("train")
    anchors = load_latents(choose_anchor_paths(by_label, args.num_anchors, rng))
    weak = weak_view(anchors, args.seed + 11)
    bad = hard_spatial_corruption(anchors, args.seed + 23)
    candidate_count = max(args.decoys_per_anchor, args.hard_decoy_candidate_pool)
    candidates = synthetic_impostor_bank(anchors, candidate_count, args.seed + 37)

    activation_fn = build_feature_fn(args.mae_path, device)
    all_latents = torch.cat([anchors, weak, bad, candidates.reshape(-1, 32, 32, 4)], dim=0)
    desc = extract_descriptors(all_latents, activation_fn, device, args.feature_batch_size)
    n = args.num_anchors

    def split(pack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pack = standardize(pack)
        return pack[:n], pack[n : 2 * n], pack[2 * n : 3 * n], pack[3 * n :].reshape(n, candidate_count, -1)

    e_a, e_w, e_b, e_pool_all = split(desc.euc)
    m_a, m_w, m_b, m_pool_all = split(desc.ms)
    hard_idx = torch.argsort(dist_many(e_a, e_pool_all, "euc"), dim=1)[:, : args.decoys_per_anchor]

    def gather_desc(pool: torch.Tensor) -> torch.Tensor:
        idx = hard_idx[:, :, None].expand(-1, -1, pool.shape[-1])
        return torch.gather(pool, dim=1, index=idx)

    e_pool = gather_desc(e_pool_all)
    m_pool = gather_desc(m_pool_all)
    flat_idx = hard_idx[:, :, None, None, None].expand(-1, -1, 32, 32, 4)
    decoys = torch.gather(candidates, dim=1, index=flat_idx)

    e_d_weak = dist_one(e_a, e_w, "euc")
    e_d_bad = dist_one(e_a, e_b, "euc")
    e_d_decoys = dist_many(e_a, e_pool, "euc")
    m_d_weak = dist_one(m_a, m_w, "ms")
    m_d_bad = dist_one(m_a, m_b, "ms")
    m_d_decoys = dist_many(m_a, m_pool, "ms")

    e_rank = 1 + (e_d_bad < e_d_weak).long() + (e_d_decoys < e_d_weak[:, None]).long().sum(dim=1)
    m_rank = 1 + (m_d_bad < m_d_weak).long() + (m_d_decoys < m_d_weak[:, None]).long().sum(dim=1)
    e_decoy_margin = (e_d_decoys.min(dim=1).values - e_d_weak) / (e_d_decoys.min(dim=1).values + e_d_weak + 1e-8)
    m_decoy_margin = (m_d_decoys.min(dim=1).values - m_d_weak) / (m_d_decoys.min(dim=1).values + m_d_weak + 1e-8)
    score = (e_rank.float() - m_rank.float()) + 4.0 * (m_decoy_margin - e_decoy_margin)
    ranked = torch.argsort(score, descending=True).tolist()
    chosen = ranked[args.rank_offset : args.rank_offset + args.rows]

    decode_latents = []
    row_specs = []
    for i in chosen:
        decoy_order = torch.argsort(e_d_decoys[i])[:3].tolist()
        row_specs.append((i, decoy_order))
        decode_latents.extend([anchors[i], weak[i], bad[i]])
        decode_latents.extend(decoys[i, decoy_order])
    images = decode_latents_to_pil(torch.stack(decode_latents), args.decode_batch_size)

    rows = []
    cursor = 0
    for i, _ in row_specs:
        rows.append(
            {
                "anchor": images[cursor],
                "weak": images[cursor + 1],
                "bad": images[cursor + 2],
                "impostors": images[cursor + 3 : cursor + 6],
                "euc_rank": int(e_rank[i].item()),
                "ms_rank": int(m_rank[i].item()),
                "euc_decoy_margin": float(e_decoy_margin[i].item()),
                "ms_decoy_margin": float(m_decoy_margin[i].item()),
            }
        )
        cursor += 6

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    make_gallery(out_dir / "false_neighbor_examples.png", out_dir / "false_neighbor_examples.pdf", rows)
    print(f"wrote {out_dir / 'false_neighbor_examples.png'}")


if __name__ == "__main__":
    main()
