"""Make a crude FID/MIND sample-count sweep for a generator checkpoint.

This uses the repository's existing Inception feature/FID convention, and uses
torch-fidelity's upstream MIND implementation when available. MIND is the
sliced 2-Wasserstein metric from torch-fidelity master / MIND paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import create_imagenet_split, epoch0_sampler, get_postprocess_fn
from utils.env import HF_ROOT
from utils.fid_util import _load_ref_stats
from utils.init_util import load_generator_model_and_params
from utils.misc import run_init

try:
    from torch_fidelity.metric_mind import KEY_METRIC_MIND, mind_features_to_metric
except Exception:  # pragma: no cover - fallback for PyPI torch-fidelity 0.4.0
    KEY_METRIC_MIND = "monge_inception_distance"
    mind_features_to_metric = None

run_init()


def _is_latent(metadata: dict) -> bool:
    model_cfg = metadata.get("model_config", {})
    return model_cfg.get("in_channels", 3) == 4


def _sample_counts(value: str) -> list[int]:
    counts = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not counts or any(x < 2 for x in counts):
        raise ValueError("sample counts must be comma-separated integers >= 2")
    return counts


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _to_uint8_nchw(images: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = images.detach().cpu().float().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    if arr.shape[-1] == 3:
        arr = arr.transpose(0, 3, 1, 2)
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


class InceptionFeatures:
    def __init__(self, device: torch.device, batch_size: int):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx]).to(device).eval()
        self.device = device
        self.batch_size = int(batch_size)

    @torch.inference_mode()
    def __call__(self, images_uint8: np.ndarray) -> np.ndarray:
        if images_uint8.shape[-1] == 3:
            images_uint8 = images_uint8.transpose(0, 3, 1, 2)
        feats = []
        for start in range(0, len(images_uint8), self.batch_size):
            batch = torch.from_numpy(images_uint8[start : start + self.batch_size]).to(
                device=self.device,
                dtype=torch.float32,
            )
            batch = batch / 255.0
            pred = self.model(batch)[0].squeeze(-1).squeeze(-1)
            feats.append(pred.cpu().numpy())
        return np.concatenate(feats, axis=0)


def _load_model(init_from: str, *, compile_model: bool):
    init_path = Path(init_from)
    if init_path.is_file():
        run_dir = init_path.parent.parent if init_path.parent.name == "checkpoints" else init_path.parent
        _, _, metadata = load_generator_model_and_params(str(run_dir), hf_cache_dir=HF_ROOT)
        model_cfg = dict(metadata.get("model_config", {}) or {})
        if not model_cfg:
            raise ValueError(f"Cannot find model_config metadata via {run_dir}")
        from models.generator import build_generator_from_config

        model = build_generator_from_config(model_cfg)
        payload = torch.load(init_path, map_location="cpu", weights_only=False)
        params = payload.get("ema_model", payload.get("ema_params", payload.get("model", payload)))
        if isinstance(params, dict) and any(k.startswith("_orig_mod.") for k in params):
            params = {(k.removeprefix("_orig_mod.")): v for k, v in params.items()}
        metadata = dict(metadata)
        metadata["step"] = int(payload.get("step", metadata.get("step", 0) or 0)) if isinstance(payload, dict) else 0
    else:
        model, params, metadata = load_generator_model_and_params(init_from, hf_cache_dir=HF_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(params, strict=False)
    model.eval()
    if compile_model:
        model = torch.compile(model, dynamic=False, fullgraph=True)
    postprocess_fn = get_postprocess_fn(use_aug=False, use_latent=False, use_cache=_is_latent(metadata))
    return model, postprocess_fn, metadata, device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _collect_features(
    *,
    init_from: str,
    cfg_scale: float,
    max_samples: int,
    eval_batch_size: int,
    feature_batch_size: int,
    compile_model: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    torch.manual_seed(0)
    np.random.seed(0)
    model, postprocess_fn, metadata, device = _load_model(init_from, compile_model=compile_model)
    feature_model = InceptionFeatures(device=device, batch_size=feature_batch_size)
    loader, _, _ = create_imagenet_split(
        resolution=256,
        split="val",
        batch_size=eval_batch_size,
        num_workers=0,
    )

    gen_features: list[np.ndarray] = []
    real_features: list[np.ndarray] = []
    timings = {
        "generation_seconds": 0.0,
        "gen_feature_seconds": 0.0,
        "real_feature_seconds": 0.0,
    }
    num_done = 0

    for real_images, labels in epoch0_sampler(loader):
        if num_done >= max_samples:
            break
        remaining = max_samples - num_done
        real_images = real_images[:remaining]
        labels = labels[:remaining].to(device=device, dtype=torch.long)

        _synchronize(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            latent_samples = model(c=labels, cfg_scale=cfg_scale)["samples"]
            gen_images = postprocess_fn(latent_samples)
        _synchronize(device)
        timings["generation_seconds"] += time.perf_counter() - t0

        real_images = torch.clamp((real_images + 1.0) / 2.0, 0.0, 1.0)
        gen_uint8 = _to_uint8_nchw(gen_images)
        real_uint8 = _to_uint8_nchw(real_images)

        _synchronize(device)
        t0 = time.perf_counter()
        gen_features.append(feature_model(gen_uint8))
        _synchronize(device)
        timings["gen_feature_seconds"] += time.perf_counter() - t0

        _synchronize(device)
        t0 = time.perf_counter()
        real_features.append(feature_model(real_uint8))
        _synchronize(device)
        timings["real_feature_seconds"] += time.perf_counter() - t0

        num_done += len(gen_uint8)
        print(f"[features] {num_done}/{max_samples}", flush=True)

    meta = {
        "init_from": init_from,
        "cfg_scale": float(cfg_scale),
        "max_samples": int(max_samples),
        "device": str(device),
        "compile_model": bool(compile_model),
        "metadata_step": int(metadata.get("step", 0) or 0),
        **timings,
    }
    return np.concatenate(gen_features, axis=0)[:max_samples], np.concatenate(real_features, axis=0)[:max_samples], meta


def _feature_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feats64 = features.astype(np.float64)
    return np.mean(feats64, axis=0), np.cov(feats64, rowvar=False)


def _fid(features: np.ndarray, ref_stats: dict[str, np.ndarray]) -> float:
    mu, sigma = _feature_stats(features)
    return float(calculate_frechet_distance(ref_stats["mu"], ref_stats["sigma"], mu, sigma))


def _fallback_mind(features_1: np.ndarray, features_2: np.ndarray, *, num_projections: int, seed: int) -> float:
    x = torch.as_tensor(features_1, dtype=torch.float32)
    y = torch.as_tensor(features_2, dtype=torch.float32)
    n = min(x.shape[0], y.shape[0])
    d = x.shape[1]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if x.shape[0] != n:
        x = x[torch.randperm(x.shape[0], generator=generator)[:n]]
    if y.shape[0] != n:
        y = y[torch.randperm(y.shape[0], generator=generator)[:n]]
    directions = torch.randn(num_projections, d, generator=generator, dtype=torch.float32)
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    y = y.to(device)
    directions = directions.to(device)
    proj_x = (x @ directions.t()).sort(dim=0).values
    proj_y = (y @ directions.t()).sort(dim=0).values
    return float((3.0 * d) * ((proj_x - proj_y) ** 2).mean().item())


def _mind(features_1: np.ndarray, features_2: np.ndarray, *, num_projections: int, seed: int) -> float:
    if mind_features_to_metric is None:
        return _fallback_mind(features_1, features_2, num_projections=num_projections, seed=seed)
    out = mind_features_to_metric(
        torch.as_tensor(features_1),
        torch.as_tensor(features_2),
        mind_num_projections=num_projections,
        cuda=torch.cuda.is_available(),
        rng_seed=seed,
        verbose=False,
    )
    return float(out[KEY_METRIC_MIND])


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], rows: list[dict], key: str, title: str, color):
    x0, y0, x1, y1 = box
    left, right, top, bottom = x0 + 58, x1 - 16, y0 + 34, y1 - 46
    draw.rectangle([left, top, right, bottom], outline=(180, 180, 180))
    xs = [math.log2(float(row["samples"])) for row in rows]
    ys = [float(row[key]) for row in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if math.isclose(ymin, ymax):
        ymin -= 1.0
        ymax += 1.0
    pad = 0.08 * (ymax - ymin)
    ymin = max(0.0, ymin - pad)
    ymax += pad

    def sx(x):
        return left + (x - xmin) / max(xmax - xmin, 1e-12) * (right - left)

    def sy(y):
        return bottom - (y - ymin) / max(ymax - ymin, 1e-12) * (bottom - top)

    points = [(sx(x), sy(y)) for x, y in zip(xs, ys, strict=True)]
    if len(points) > 1:
        draw.line(points, fill=color, width=3)
    for (px, py), row in zip(points, rows, strict=True):
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=color)
        draw.text((px - 18, bottom + 10), str(row["samples"]), fill=(70, 70, 70))
    for frac in (0.0, 0.5, 1.0):
        value = ymin + frac * (ymax - ymin)
        py = sy(value)
        draw.line([left - 5, py, left, py], fill=(150, 150, 150))
        draw.text((x0 + 4, py - 7), f"{value:.3g}", fill=(70, 70, 70))
    draw.text((x0 + 8, y0 + 8), title, fill=(20, 20, 20))


def _plot_metric_seconds_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], rows: list[dict]) -> None:
    x0, y0, x1, y1 = box
    left, right, top, bottom = x0 + 58, x1 - 16, y0 + 34, y1 - 46
    draw.rectangle([left, top, right, bottom], outline=(180, 180, 180))
    keys = [
        ("fid_metric_seconds", (35, 140, 90), "FID"),
        ("mind_metric_seconds", (125, 70, 160), "MIND"),
    ]
    xs = [math.log2(float(row["samples"])) for row in rows]
    xmin, xmax = min(xs), max(xs)
    all_ys = [float(row[key]) for key, _, _ in keys for row in rows]
    ymin, ymax = min(all_ys), max(all_ys)
    if math.isclose(ymin, ymax):
        ymin -= 1.0
        ymax += 1.0
    pad = 0.08 * (ymax - ymin)
    ymin = max(0.0, ymin - pad)
    ymax += pad

    def sx(x):
        return left + (x - xmin) / max(xmax - xmin, 1e-12) * (right - left)

    def sy(y):
        return bottom - (y - ymin) / max(ymax - ymin, 1e-12) * (bottom - top)

    for frac in (0.0, 0.5, 1.0):
        value = ymin + frac * (ymax - ymin)
        py = sy(value)
        draw.line([left - 5, py, left, py], fill=(150, 150, 150))
        draw.text((x0 + 4, py - 7), f"{value:.3g}", fill=(70, 70, 70))
    for key, color, label in keys:
        points = [(sx(x), sy(float(row[key]))) for x, row in zip(xs, rows, strict=True)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for px, py in points:
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=color)
        lx = x0 + 300 if label == "MIND" else x0 + 230
        draw.text((lx, y0 + 10), label, fill=color)
    for x, row in zip(xs, rows, strict=True):
        draw.text((sx(x) - 18, bottom + 10), str(row["samples"]), fill=(70, 70, 70))
    draw.text((x0 + 8, y0 + 8), "Metric seconds", fill=(20, 20, 20))


def _plot(path: Path, rows: list[dict], title: str) -> None:
    image = Image.new("RGB", (1500, 520), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 14), title, fill=(20, 20, 20))
    _plot_panel(draw, (20, 55, 490, 500), rows, "fid", "FID vs samples", (36, 100, 180))
    _plot_panel(draw, (515, 55, 985, 500), rows, "mind", "MIND vs samples", (180, 80, 40))
    _plot_metric_seconds_panel(draw, (1010, 55, 1480, 500), rows)
    image.save(path)


def _plot_two_metric_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    rows: list[dict],
    values: dict[str, list[float]],
    title: str,
) -> None:
    x0, y0, x1, y1 = box
    left, right, top, bottom = x0 + 68, x1 - 22, y0 + 38, y1 - 48
    draw.rectangle([left, top, right, bottom], outline=(180, 180, 180))
    colors = {"FID": (36, 100, 180), "MIND": (180, 80, 40)}
    xs = [math.log2(float(row["samples"])) for row in rows]
    xmin, xmax = min(xs), max(xs)
    all_ys = [value for series in values.values() for value in series]
    ymin, ymax = min(all_ys), max(all_ys)
    if math.isclose(ymin, ymax):
        ymin -= 1.0
        ymax += 1.0
    pad = 0.08 * (ymax - ymin)
    ymin = max(0.0, ymin - pad)
    ymax += pad

    def sx(x):
        return left + (x - xmin) / max(xmax - xmin, 1e-12) * (right - left)

    def sy(y):
        return bottom - (y - ymin) / max(ymax - ymin, 1e-12) * (bottom - top)

    for frac in (0.0, 0.5, 1.0):
        value = ymin + frac * (ymax - ymin)
        py = sy(value)
        draw.line([left - 5, py, left, py], fill=(150, 150, 150))
        draw.text((x0 + 4, py - 7), f"{value:.3g}", fill=(70, 70, 70))
    for i, (label, ys) in enumerate(values.items()):
        points = [(sx(x), sy(y)) for x, y in zip(xs, ys, strict=True)]
        if len(points) > 1:
            draw.line(points, fill=colors[label], width=3)
        for px, py in points:
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=colors[label])
        draw.text((x0 + 250 + 72 * i, y0 + 12), label, fill=colors[label])
    for x, row in zip(xs, rows, strict=True):
        draw.text((sx(x) - 18, bottom + 10), str(row["samples"]), fill=(70, 70, 70))
    draw.text((x0 + 8, y0 + 8), title, fill=(20, 20, 20))


def _plot_convergence(path: Path, rows: list[dict], title: str) -> None:
    final_fid = float(rows[-1]["fid"])
    final_mind = float(rows[-1]["mind"])
    ratios = {
        "FID": [float(row["fid"]) / final_fid for row in rows],
        "MIND": [float(row["mind"]) / final_mind for row in rows],
    }
    excess = {
        "FID": [float(row["fid"]) - final_fid for row in rows],
        "MIND": [float(row["mind"]) - final_mind for row in rows],
    }
    image = Image.new("RGB", (1080, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 14), title, fill=(20, 20, 20))
    _plot_two_metric_panel(draw, (20, 55, 530, 475), rows, ratios, "Score / 50k score")
    _plot_two_metric_panel(draw, (550, 55, 1060, 475), rows, excess, "Excess over 50k score")
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-from", required=True)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--sample-counts", default="512,1024,2048,4096")
    parser.add_argument("--mind-num-projections", type=int, default=1000)
    parser.add_argument("--rng-seed", type=int, default=2020)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--workdir", default="runs/fid_mind_sweeps")
    parser.add_argument("--feature-cache", default="")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    sample_counts = _sample_counts(args.sample_counts)
    max_samples = max(sample_counts)
    outdir = Path(args.workdir)
    outdir.mkdir(parents=True, exist_ok=True)
    init_path = Path(args.init_from)
    if init_path.is_file() and init_path.parent.name == "checkpoints":
        run_name = _safe_name(f"{init_path.parent.parent.name}_{init_path.stem}")
    else:
        run_name = _safe_name(init_path.name or "checkpoint")
    cfg_name = f"{args.cfg_scale:g}".replace(".", "p")
    stem = outdir / f"{run_name}_cfg{cfg_name}_n{max_samples}_fid_mind"
    feature_cache = Path(args.feature_cache) if args.feature_cache else stem.with_suffix(".features.npz")

    if feature_cache.exists() and not args.overwrite_cache:
        cached = np.load(feature_cache, allow_pickle=False)
        gen_features = cached["gen_features"]
        real_features = cached["real_features"]
        if len(gen_features) < max_samples or len(real_features) < max_samples:
            raise ValueError(
                f"Feature cache {feature_cache} has "
                f"{min(len(gen_features), len(real_features))} samples, "
                f"but this run needs {max_samples}. Use a larger cache or --overwrite-cache."
            )
        meta = json.loads(str(cached["meta"]))
        print(f"[cache] loaded {feature_cache}", flush=True)
    else:
        gen_features, real_features, meta = _collect_features(
            init_from=args.init_from,
            cfg_scale=args.cfg_scale,
            max_samples=max_samples,
            eval_batch_size=args.eval_batch_size,
            feature_batch_size=args.feature_batch_size,
            compile_model=not args.no_compile,
        )
        np.savez_compressed(
            feature_cache,
            gen_features=gen_features,
            real_features=real_features,
            meta=json.dumps(meta),
        )
        print(f"[cache] wrote {feature_cache}", flush=True)

    ref_stats = _load_ref_stats("imagenet256")
    rows = []
    for n in sample_counts:
        t0 = time.perf_counter()
        fid = _fid(gen_features[:n], ref_stats)
        fid_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        mind = _mind(
            gen_features[:n],
            real_features[:n],
            num_projections=args.mind_num_projections,
            seed=args.rng_seed,
        )
        mind_seconds = time.perf_counter() - t0

        row = {
            "samples": int(n),
            "fid": fid,
            "mind": mind,
            "fid_metric_seconds": fid_seconds,
            "mind_metric_seconds": mind_seconds,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    result = {
        "meta": meta
        | {
            "sample_counts": sample_counts,
            "mind_num_projections": args.mind_num_projections,
            "rng_seed": args.rng_seed,
            "feature_cache": str(feature_cache),
            "mind_source": "torch_fidelity.metric_mind" if mind_features_to_metric is not None else "local_fallback",
        },
        "rows": rows,
    }
    json_path = stem.with_suffix(".json")
    csv_path = stem.with_suffix(".csv")
    png_path = stem.with_suffix(".png")
    convergence_png_path = stem.with_name(f"{stem.name}_convergence").with_suffix(".png")
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_path, rows)
    _plot(png_path, rows, f"{run_name}, cfg={args.cfg_scale:g}, MIND projections={args.mind_num_projections}")
    _plot_convergence(
        convergence_png_path,
        rows,
        f"{run_name}, cfg={args.cfg_scale:g}, normalized to {max_samples} samples",
    )
    print(f"[out] {json_path}", flush=True)
    print(f"[out] {csv_path}", flush=True)
    print(f"[out] {png_path}", flush=True)
    print(f"[out] {convergence_png_path}", flush=True)


if __name__ == "__main__":
    main()
