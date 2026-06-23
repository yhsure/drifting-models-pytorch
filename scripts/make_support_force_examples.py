"""Visual examples for the support-force stability diagnostic."""

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
    DARK,
    GRAY,
    GREEN,
    PURPLE,
    build_feature_fn,
    decode_latents_to_pil,
    dist_many,
    draw_text,
    extract_descriptors,
    font,
    generate_latents,
    index_latent_cache,
    load_generator,
    resize_crop,
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
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--rank-offset", type=int, default=0)
    parser.add_argument("--feature-batch-size", type=int, default=96)
    parser.add_argument("--gen-batch-size", type=int, default=32)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_gallery(path_png: Path, path_pdf: Path, rows: list[dict]) -> Image.Image:
    tile = 154
    x0 = 70
    y0 = 220
    gap = 22
    row_gap = 70
    columns = [
        ("Anchor", DARK),
        ("Weak view", GREEN),
        ("Weak view", GREEN),
        ("Weak view", GREEN),
        ("Mined impostor", PURPLE),
        ("Mined impostor", PURPLE),
        ("Mined impostor", PURPLE),
    ]
    width = max(1620, x0 * 2 + len(columns) * tile + (len(columns) - 1) * gap)
    height = y0 + len(rows) * (tile + row_gap) + 70
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw_text(draw, (x0, 45), "Support-force examples: weak views vs mined impostors", font(39, True), DARK)
    draw_text(
        draw,
        (x0, 100),
        "Weak views preserve local content; impostors are Euclidean-nearest roll/shuffle variants.",
        font(24),
        GRAY,
    )
    for c, (title, color) in enumerate(columns):
        x = x0 + c * (tile + gap)
        draw_text(draw, (x, y0 - 42), title, font(18, True), color)

    for r, row in enumerate(rows):
        y = y0 + r * (tile + row_gap)
        images = [row["anchor"], *row["weak"], *row["impostor"]]
        colors = [DARK, GREEN, GREEN, GREEN, PURPLE, PURPLE, PURPLE]
        for c, (img, color) in enumerate(zip(images, colors)):
            x = x0 + c * (tile + gap)
            canvas.paste(resize_crop(img, tile), (x, y))
            draw.rectangle((x, y, x + tile, y + tile), outline=color, width=5)
        label = f"row {r + 1}: Euclidean top-5 weak mass {row['euc_top5_mass']:.1f}; MS-stat {row['ms_top5_mass']:.1f}"
        draw_text(draw, (x0, y + tile + 16), label, font(20), GRAY)

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
    eligible = [label for label, paths in by_label.items() if len(paths) >= 8]
    labels = torch.tensor(rng.sample(eligible, args.num_anchors), dtype=torch.long)

    model = load_generator(args.checkpoint, device)
    anchors = generate_latents(model, labels, args.cfg_scale, args.gen_batch_size, args.seed + 7, device)
    weak_pool = torch.stack([weak_view(anchors, args.seed + 101 + 17 * i) for i in range(args.num_weak)], dim=1)
    candidate_count = max(args.num_impostors, args.hard_impostor_candidate_pool)
    impostor_candidates = synthetic_impostor_bank(anchors, candidate_count, args.seed + 809)

    activation_fn = build_feature_fn(args.mae_path, device)
    all_latents = torch.cat([anchors, weak_pool.reshape(-1, 32, 32, 4), impostor_candidates.reshape(-1, 32, 32, 4)], dim=0)
    desc = extract_descriptors(all_latents, activation_fn, device, args.feature_batch_size)
    n = args.num_anchors
    e_all = standardize(desc.euc)
    m_all = standardize(desc.ms)
    e_anchor = e_all[:n]
    m_anchor = m_all[:n]
    e_weak = e_all[n : n + n * args.num_weak].reshape(n, args.num_weak, -1)
    m_weak = m_all[n : n + n * args.num_weak].reshape(n, args.num_weak, -1)
    e_candidate = e_all[n + n * args.num_weak :].reshape(n, candidate_count, -1)
    m_candidate = m_all[n + n * args.num_weak :].reshape(n, candidate_count, -1)
    hard_idx = torch.argsort(dist_many(e_anchor, e_candidate, "euc"), dim=1)[:, : args.num_impostors]

    def gather_desc(pool_tensor: torch.Tensor) -> torch.Tensor:
        idx = hard_idx[:, :, None].expand(-1, -1, pool_tensor.shape[-1])
        return torch.gather(pool_tensor, dim=1, index=idx)

    flat_hard = hard_idx[:, :, None, None, None].expand(-1, -1, 32, 32, 4)
    impostor_pool = torch.gather(impostor_candidates, dim=1, index=flat_hard)
    e_pool = torch.cat([e_weak, gather_desc(e_candidate)], dim=1)
    m_pool = torch.cat([m_weak, gather_desc(m_candidate)], dim=1)
    e_dist = dist_many(e_anchor, e_pool, "euc")
    m_dist = dist_many(m_anchor, m_pool, "ms")
    labels_weak = torch.zeros(e_dist.shape, dtype=torch.bool)
    labels_weak[:, : args.num_weak] = True
    e_top5_mass = labels_weak.gather(1, torch.argsort(e_dist, dim=1)[:, :5]).float().mean(dim=1)
    m_top5_mass = labels_weak.gather(1, torch.argsort(m_dist, dim=1)[:, :5]).float().mean(dim=1)
    ranked = torch.argsort((m_top5_mass - e_top5_mass), descending=True).tolist()
    chosen = ranked[args.rank_offset : args.rank_offset + args.rows]

    decode_latents = []
    row_specs = []
    for i in chosen:
        weak_order = torch.argsort(m_dist[i, : args.num_weak])[:3].tolist()
        impostor_order = torch.argsort(e_dist[i, args.num_weak :])[:3].tolist()
        row_specs.append((i, weak_order, impostor_order))
        decode_latents.append(anchors[i])
        decode_latents.extend(weak_pool[i, weak_order])
        decode_latents.extend(impostor_pool[i, impostor_order])
    images = decode_latents_to_pil(torch.stack(decode_latents), args.decode_batch_size)

    rows = []
    cursor = 0
    for i, _, _ in row_specs:
        rows.append(
            {
                "anchor": images[cursor],
                "weak": images[cursor + 1 : cursor + 4],
                "impostor": images[cursor + 4 : cursor + 7],
                "euc_top5_mass": float(e_top5_mass[i].item()),
                "ms_top5_mass": float(m_top5_mass[i].item()),
            }
        )
        cursor += 7

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    make_gallery(out_dir / "support_force_examples.png", out_dir / "support_force_examples.pdf", rows)
    print(f"wrote {out_dir / 'support_force_examples.png'}")


if __name__ == "__main__":
    main()
