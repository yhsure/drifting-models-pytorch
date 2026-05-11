from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.dataset import center_crop_arr, get_postprocess_fn
from utils.env import HF_ROOT, IMAGENET_PATH
from utils.init_util import load_generator_model_and_params


def _is_latent(metadata: dict[str, Any]) -> bool:
    model_cfg = metadata.get("model_config", {})
    return model_cfg.get("in_channels", 3) == 4


def _load_generator(init_from: str):
    model, params, metadata = load_generator_model_and_params(init_from, hf_cache_dir=HF_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if hasattr(model, "resize_likelihood_prior_from_state_dict"):
        model.resize_likelihood_prior_from_state_dict(params)
    model.load_state_dict(params, strict=False)
    model.eval()
    postprocess_fn = get_postprocess_fn(use_aug=False, use_latent=False, use_cache=_is_latent(metadata))
    return model, postprocess_fn, metadata, device


@torch.no_grad()
def _generate_images(
    model,
    postprocess_fn,
    labels: torch.Tensor,
    *,
    cfg_scale: float,
    likelihood_conditioning: str,
    likelihood_topk: int,
    likelihood_temperature: float,
) -> torch.Tensor:
    device = next(model.parameters()).device
    labels = labels.to(device=device, dtype=torch.long)
    components = None
    weights = None
    if hasattr(model, "has_likelihood_prior") and model.has_likelihood_prior():
        components, weights = model.sample_likelihood_components(
            labels,
            conditioning=likelihood_conditioning,
            topk=likelihood_topk,
            temperature=likelihood_temperature,
        )
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        samples = model(
            c=labels,
            cfg_scale=float(cfg_scale),
            likelihood_components=components,
            likelihood_weights=weights,
            likelihood_conditioning=likelihood_conditioning,
            likelihood_topk=likelihood_topk,
            likelihood_temperature=likelihood_temperature,
        )["samples"]
    return postprocess_fn(samples).detach().cpu().float()


def _tensor_to_pil(images: torch.Tensor) -> list[Image.Image]:
    if images.ndim != 4:
        raise ValueError(f"Expected image tensor BCHW or BHWC, got {tuple(images.shape)}")
    if images.shape[1] in (1, 3):
        images = images.permute(0, 2, 3, 1)
    images = images.clamp(0, 1)
    arr = (images.numpy() * 255.0).round().clip(0, 255).astype("uint8")
    return [Image.fromarray(x).convert("RGB") for x in arr]


def _load_clip(model_name: str, *, local_files_only: bool, device: torch.device):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError("transformers is required for the nearest-train CLIP panel.") from exc
    model = CLIPModel.from_pretrained(model_name, local_files_only=local_files_only).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    return model, processor


@torch.no_grad()
def _clip_features(model, processor, images: list[Image.Image], *, device: torch.device, batch_size: int) -> torch.Tensor:
    features = []
    for start in range(0, len(images), max(1, int(batch_size))):
        batch = images[start : start + max(1, int(batch_size))]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feat = model.get_image_features(**inputs).float()
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        features.append(feat.cpu())
    return torch.cat(features, dim=0)


def _candidate_paths(labels: list[int], *, candidates_per_sample: int, seed: int) -> tuple[list[str], list[list[int]]]:
    dataset = ImageFolder(str(Path(IMAGENET_PATH) / "train"))
    by_class: dict[int, list[str]] = defaultdict(list)
    for path, label in dataset.samples:
        by_class[int(label)].append(path)

    rng = random.Random(int(seed))
    all_paths: list[str] = []
    per_sample_indices: list[list[int]] = []
    for label in labels:
        pool = by_class[int(label)]
        if not pool:
            raise RuntimeError(f"No ImageNet train candidates found for class {label}.")
        if len(pool) >= candidates_per_sample:
            chosen = rng.sample(pool, candidates_per_sample)
        else:
            chosen = [rng.choice(pool) for _ in range(candidates_per_sample)]
        indices = []
        for path in chosen:
            indices.append(len(all_paths))
            all_paths.append(path)
        per_sample_indices.append(indices)
    return all_paths, per_sample_indices


def _load_candidate_images(paths: list[str], *, resolution: int) -> list[Image.Image]:
    out = []
    for path in paths:
        with Image.open(path) as img:
            out.append(center_crop_arr(img.convert("RGB"), resolution))
    return out


def _panel_image(
    generated: list[Image.Image],
    candidate_images: list[Image.Image],
    rows: list[dict[str, Any]],
    *,
    topk: int,
    tile_size: int,
) -> Image.Image:
    cols = int(topk) + 1
    label_h = 24
    grid = Image.new("RGB", (cols * tile_size, len(rows) * (tile_size + label_h)), "white")
    draw = ImageDraw.Draw(grid)
    for row_idx, row in enumerate(rows):
        y0 = row_idx * (tile_size + label_h)
        gen = ImageOps.fit(generated[row_idx], (tile_size, tile_size))
        gen = ImageOps.expand(gen, border=3, fill=(255, 128, 0))
        grid.paste(gen, (0, y0 + label_h))
        draw.text((4, y0 + 4), f"gen c{row['class_index']}", fill=(0, 0, 0))
        for j, item in enumerate(row["nearest"][:topk]):
            img = ImageOps.fit(candidate_images[int(item["candidate_index"])], (tile_size, tile_size))
            img = ImageOps.expand(img, border=3, fill=(60, 120, 255))
            x0 = (j + 1) * tile_size
            grid.paste(img, (x0, y0 + label_h))
            draw.text((x0 + 4, y0 + 4), f"{float(item['similarity']):.3f}", fill=(0, 0, 0))
    return grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a generated-vs-nearest-train CLIP panel for an ImageNet likelihood run.")
    parser.add_argument("--init-from", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--class-ids", default="", help="Comma-separated class ids. Random classes are used when empty.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--candidates-per-sample", type=int, default=128)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=160)
    parser.add_argument("--likelihood-conditioning", default="sparse_soft", choices=("none", "hard", "top1", "sparse_soft", "soft"))
    parser.add_argument("--likelihood-topk", type=int, default=8)
    parser.add_argument("--likelihood-temperature", type=float, default=1.0)
    return parser


def _labels_from_args(args: argparse.Namespace) -> torch.Tensor:
    if args.class_ids.strip():
        labels = [int(x.strip()) for x in args.class_ids.split(",") if x.strip()]
        if not labels:
            raise ValueError("--class-ids did not contain any labels.")
        return torch.tensor(labels[: int(args.num_samples)], dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    return torch.randint(0, 1000, (int(args.num_samples),), generator=generator, dtype=torch.long)


def main() -> None:
    args = build_parser().parse_args()
    labels = _labels_from_args(args)
    model, postprocess_fn, metadata, device = _load_generator(args.init_from)
    del metadata
    generated_t = _generate_images(
        model,
        postprocess_fn,
        labels,
        cfg_scale=float(args.cfg_scale),
        likelihood_conditioning=str(args.likelihood_conditioning),
        likelihood_topk=int(args.likelihood_topk),
        likelihood_temperature=float(args.likelihood_temperature),
    )
    generated = _tensor_to_pil(generated_t)
    candidate_paths, per_sample_indices = _candidate_paths(
        labels.detach().cpu().tolist(),
        candidates_per_sample=int(args.candidates_per_sample),
        seed=int(args.seed) + 1,
    )
    candidate_images = _load_candidate_images(candidate_paths, resolution=int(args.resolution))

    clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, clip_processor = _load_clip(
        str(args.clip_model),
        local_files_only=bool(args.local_files_only),
        device=clip_device,
    )
    gen_features = _clip_features(
        clip_model,
        clip_processor,
        generated,
        device=clip_device,
        batch_size=int(args.clip_batch_size),
    )
    cand_features = _clip_features(
        clip_model,
        clip_processor,
        candidate_images,
        device=clip_device,
        batch_size=int(args.clip_batch_size),
    )

    rows: list[dict[str, Any]] = []
    topk = max(1, int(args.topk))
    for idx, candidate_indices in enumerate(per_sample_indices):
        cand_idx_t = torch.tensor(candidate_indices, dtype=torch.long)
        sims = cand_features[cand_idx_t] @ gen_features[idx]
        values, order = torch.topk(sims, k=min(topk, sims.numel()))
        nearest = []
        for value, local_idx in zip(values.tolist(), order.tolist()):
            global_idx = int(candidate_indices[int(local_idx)])
            nearest.append(
                {
                    "candidate_index": global_idx,
                    "path": candidate_paths[global_idx],
                    "similarity": float(value),
                }
            )
        rows.append({"generated_index": idx, "class_index": int(labels[idx].item()), "nearest": nearest})

    panel = _panel_image(
        generated,
        candidate_images,
        rows,
        topk=topk,
        tile_size=int(args.tile_size),
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)
    json_out = Path(args.json_out).expanduser().resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(
            {
                "init_from": args.init_from,
                "output": str(output),
                "cfg_scale": float(args.cfg_scale),
                "num_samples": int(labels.numel()),
                "clip_model": str(args.clip_model),
                "class_ids": labels.detach().cpu().tolist(),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
