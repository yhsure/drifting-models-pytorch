from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
from tqdm import tqdm

from dataset.dataset import create_imagenet_split, get_postprocess_fn
from dataset.latent import LatentDataset
from models.mae_model import build_activation_function
from utils.env import HF_ROOT, IMAGENET_CACHE_PATH
from utils.fid_util import evaluate_fid
from utils.logging import WandbLogger


DEFAULT_DESCRIPTOR_KEYS = ("layer4_mean", "layer4_std")


@dataclass
class DecoderPriorConfig:
    component_count: int = 8192
    num_classes: int = 1000
    descriptor_mode: str = "mae"
    descriptor_keys: tuple[str, ...] = DEFAULT_DESCRIPTOR_KEYS
    mae_path: str = "hf://mae_latent_640"
    feature_stats_from: str = ""
    kmeans_iters: int = 8
    center_mode: str = "mean"
    use_flips: bool = False
    max_classes: int = 0
    max_per_class: int = 0
    seed: int = 42
    min_scale: float = 0.0
    max_scale: float = 2.0


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def _is_rank_zero() -> bool:
    return _rank() == 0


def maybe_init_distributed() -> None:
    if dist.is_available() and not dist.is_initialized() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def _device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _class_component_counts(component_count: int, num_classes: int) -> torch.Tensor:
    if component_count < num_classes:
        raise ValueError(f"component_count={component_count} must be >= num_classes={num_classes}")
    base = component_count // num_classes
    extra = component_count - base * num_classes
    counts = torch.full((num_classes,), base, dtype=torch.long)
    if extra > 0:
        counts[:extra] += 1
    return counts


def _class_offsets(counts: torch.Tensor) -> torch.Tensor:
    offsets = torch.zeros_like(counts)
    if counts.numel() > 1:
        offsets[1:] = torch.cumsum(counts[:-1], dim=0)
    return offsets


def _load_latent_file(path: str | Path, *, use_flip: bool = False) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False)
    key = "moments_flip" if use_flip else "moments"
    if key not in data:
        key = "moments"
    return torch.as_tensor(data[key], dtype=torch.float32)


def _build_class_paths(cache_root: str | Path, num_classes: int) -> list[list[str]]:
    ds = LatentDataset(str(Path(cache_root) / "train"))
    by_class: list[list[str]] = [[] for _ in range(num_classes)]
    for path, label in ds.samples:
        if 0 <= int(label) < num_classes:
            by_class[int(label)].append(path)
    return by_class


def _load_feature_stats(path: str, descriptor_dim: int | None = None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not path:
        return None, None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mean = payload.get("feature_mean")
    std = payload.get("feature_std")
    if mean is None or std is None:
        return None, None
    mean_t = torch.as_tensor(mean, dtype=torch.float32)
    std_t = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6)
    if descriptor_dim is not None and mean_t.numel() != descriptor_dim:
        raise ValueError(
            f"Feature stats at {path} have dim {mean_t.numel()}, expected {descriptor_dim}. "
            "Use matching descriptor keys or omit --feature-stats-from."
        )
    return mean_t, std_t


class DescriptorExtractor:
    def __init__(self, cfg: DecoderPriorConfig, *, device: torch.device, compile_level: int = 1):
        self.cfg = cfg
        self.device = device
        self.feature_mean: torch.Tensor | None = None
        self.feature_std: torch.Tensor | None = None
        self.activation_fn = None

        if cfg.descriptor_mode == "mae":
            postprocess_fn = get_postprocess_fn(use_cache=True, has_clip=False)
            self.activation_fn, _ = build_activation_function(
                mae_path=cfg.mae_path,
                use_mae=True,
                use_convnext=False,
                postprocess_fn=postprocess_fn,
                compile_level=int(compile_level),
                device=device,
            )
            # The default keys are layer4 mean/std from the latent MAE:
            # 5120 + 5120 = 10240, matching the prior used in earlier runs.
            if cfg.feature_stats_from:
                self.feature_mean, self.feature_std = _load_feature_stats(cfg.feature_stats_from, descriptor_dim=None)
        elif cfg.descriptor_mode == "raw":
            self.activation_fn = None
        else:
            raise ValueError("--descriptor-mode must be 'mae' or 'raw'")

    def __call__(self, latents: torch.Tensor, batch_size: int = 128, *, requires_grad: bool = False) -> torch.Tensor:
        outs = []
        context = torch.enable_grad() if requires_grad else torch.inference_mode()
        with context:
            for chunk in latents.split(int(batch_size), dim=0):
                x = chunk.to(self.device, dtype=torch.float32, non_blocking=True)
                if self.cfg.descriptor_mode == "raw":
                    desc = x.reshape(x.shape[0], -1)
                else:
                    assert self.activation_fn is not None
                    feats = self.activation_fn(
                        x,
                        patch_mean_size=[],
                        patch_std_size=[],
                        use_mean=True,
                        use_std=True,
                        every_k_block=float("inf"),
                    )
                    parts = []
                    for key in self.cfg.descriptor_keys:
                        if key not in feats:
                            raise KeyError(f"Descriptor key {key!r} missing from MAE activations: {sorted(feats)}")
                        parts.append(feats[key].reshape(feats[key].shape[0], -1).float())
                    desc = torch.cat(parts, dim=1)
                if self.feature_mean is not None and self.feature_std is not None:
                    mean = self.feature_mean.to(device=desc.device, dtype=desc.dtype)
                    std = self.feature_std.to(device=desc.device, dtype=desc.dtype)
                    if mean.numel() != desc.shape[1]:
                        raise ValueError(f"Feature stats dim {mean.numel()} does not match descriptor dim {desc.shape[1]}")
                    desc = (desc - mean) / std
                outs.append(desc if requires_grad else desc.detach().cpu())
        return torch.cat(outs, dim=0)


def _sq_dists(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    x2 = (x.float() * x.float()).sum(dim=1, keepdim=True)
    c2 = (centers.float() * centers.float()).sum(dim=1).unsqueeze(0)
    return (x2 + c2 - 2.0 * x.float().matmul(centers.float().t())).clamp_min_(0.0)


def _kmeans_assign(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return _sq_dists(x, centers).argmin(dim=1)


def _fit_kmeans(x: torch.Tensor, k: int, *, iters: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(x.shape[0])
    if n == 0:
        raise ValueError("Cannot fit k-means with zero samples")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    init_idx = torch.randint(0, n, (k,), generator=gen)
    centers = x[init_idx].to(device=device, dtype=torch.float32).clone()
    x_dev = x.to(device=device, dtype=torch.float32)
    assign = torch.zeros((n,), device=device, dtype=torch.long)

    for _ in range(max(1, int(iters))):
        assign = _kmeans_assign(x_dev, centers)
        new_centers = torch.zeros_like(centers)
        counts = torch.bincount(assign, minlength=k).to(device=device, dtype=torch.float32)
        new_centers.index_add_(0, assign, x_dev)
        nonempty = counts > 0
        new_centers[nonempty] = new_centers[nonempty] / counts[nonempty, None]
        if (~nonempty).any():
            repl = torch.randint(0, n, (int((~nonempty).sum().item()),), device=device)
            new_centers[~nonempty] = x_dev[repl]
        centers = new_centers

    assign = _kmeans_assign(x_dev, centers).detach().cpu()
    return centers.detach().cpu(), assign


def _latent_moments_for_assignments(
    latents: torch.Tensor,
    assign: torch.Tensor,
    k: int,
    *,
    min_scale: float,
    max_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = latents.reshape(latents.shape[0], -1).to(device=device, dtype=torch.float32)
    assign_dev = assign.to(device=device, dtype=torch.long)
    dim = flat.shape[1]
    sums = torch.zeros((k, dim), device=device)
    sq_sums = torch.zeros((k, dim), device=device)
    counts = torch.bincount(assign_dev, minlength=k).to(device=device, dtype=torch.float32)
    sums.index_add_(0, assign_dev, flat)
    sq_sums.index_add_(0, assign_dev, flat * flat)
    means = torch.zeros_like(sums)
    nonempty = counts > 0
    means[nonempty] = sums[nonempty] / counts[nonempty, None]
    var = torch.zeros_like(sums)
    var[nonempty] = sq_sums[nonempty] / counts[nonempty, None] - means[nonempty] * means[nonempty]
    scales = var.clamp_min(0.0).sqrt()
    if max_scale > 0:
        scales = scales.clamp_max(float(max_scale))
    if min_scale > 0:
        scales = scales.clamp_min(float(min_scale))

    if (~nonempty).any():
        repl = torch.randint(0, flat.shape[0], (int((~nonempty).sum().item()),), device=device)
        means[~nonempty] = flat[repl]
        scales[~nonempty] = 0.0

    out_shape = (k, *latents.shape[1:])
    return means.reshape(out_shape).cpu(), scales.reshape(out_shape).cpu(), counts.cpu()


def _latent_components_for_assignments(
    latents: torch.Tensor,
    descriptors: torch.Tensor,
    descriptor_centers: torch.Tensor,
    assign: torch.Tensor,
    k: int,
    *,
    center_mode: str,
    min_scale: float,
    max_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if center_mode == "mean":
        return _latent_moments_for_assignments(
            latents,
            assign,
            k,
            min_scale=min_scale,
            max_scale=max_scale,
            device=device,
        )
    if center_mode != "medoid":
        raise ValueError("--center-mode must be 'mean' or 'medoid'")

    del device
    flat = latents.reshape(latents.shape[0], -1).to(dtype=torch.float32)
    desc = descriptors.to(dtype=torch.float32)
    desc_centers = descriptor_centers.to(dtype=torch.float32)
    assign_cpu = assign.to(dtype=torch.long)
    counts = torch.bincount(assign_cpu, minlength=k).to(dtype=torch.float32)

    center_flat = torch.empty((k, flat.shape[1]), dtype=torch.float32)
    for j in range(k):
        member_idx = torch.nonzero(assign_cpu == j, as_tuple=False).flatten()
        if member_idx.numel() == 0:
            repl = torch.randint(0, flat.shape[0], (1,)).item()
            center_flat[j] = flat[repl]
            continue
        member_desc = desc[member_idx]
        d2 = (member_desc - desc_centers[j]).square().mean(dim=1)
        center_flat[j] = flat[member_idx[int(d2.argmin().item())]]

    sq_dev_sums = torch.zeros_like(center_flat)
    dev = flat - center_flat[assign_cpu]
    sq_dev_sums.index_add_(0, assign_cpu, dev * dev)
    scales = torch.zeros_like(center_flat)
    nonempty = counts > 0
    scales[nonempty] = (sq_dev_sums[nonempty] / counts[nonempty, None]).clamp_min(0.0).sqrt()
    if max_scale > 0:
        scales = scales.clamp_max(float(max_scale))
    if min_scale > 0:
        scales = scales.clamp_min(float(min_scale))
    scales[~nonempty] = 0.0

    out_shape = (k, *latents.shape[1:])
    return center_flat.reshape(out_shape).cpu(), scales.reshape(out_shape).cpu(), counts.cpu()


def _read_class_latents(paths: list[str], *, use_flips: bool, max_per_class: int, seed: int) -> torch.Tensor:
    if max_per_class and len(paths) > max_per_class:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        idx = torch.randperm(len(paths), generator=gen)[: int(max_per_class)].tolist()
        paths = [paths[i] for i in idx]
    latents = []
    for path in paths:
        latents.append(_load_latent_file(path, use_flip=False))
        if use_flips:
            latents.append(_load_latent_file(path, use_flip=True))
    return torch.stack(latents, dim=0).float()


def fit_decoder_prior(args: argparse.Namespace) -> Path | None:
    maybe_init_distributed()
    rank = _rank()
    world = _world_size()
    device = _device()
    cfg = DecoderPriorConfig(
        component_count=int(args.component_count),
        num_classes=int(args.num_classes),
        descriptor_mode=str(args.descriptor_mode),
        descriptor_keys=tuple(str(x) for x in args.descriptor_keys.split(",") if x),
        mae_path=str(args.mae_path),
        feature_stats_from=str(args.feature_stats_from),
        kmeans_iters=int(args.kmeans_iters),
        center_mode=str(args.center_mode),
        use_flips=bool(args.use_flips),
        max_classes=int(args.max_classes),
        max_per_class=int(args.max_per_class),
        seed=int(args.seed),
        min_scale=float(args.min_scale),
        max_scale=float(args.max_scale),
    )
    out_dir = Path(args.out_dir).resolve()
    shard_dir = out_dir / "shards"
    if _is_rank_zero():
        shard_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    counts_by_class = _class_component_counts(cfg.component_count, cfg.num_classes)
    offsets = _class_offsets(counts_by_class)
    paths_by_class = _build_class_paths(args.cache_root, cfg.num_classes)
    class_ids = list(range(cfg.num_classes))
    if cfg.max_classes > 0:
        class_ids = class_ids[: cfg.max_classes]

    extractor = DescriptorExtractor(cfg, device=device, compile_level=int(args.compile_level))
    local_classes = [y for y in class_ids if y % world == rank]
    local_indices = []
    local_centers = []
    local_scales = []
    local_weights = []
    started = time.time()

    iterator: Iterable[int] = local_classes
    if _is_rank_zero():
        iterator = tqdm(local_classes, desc=f"fit-rank{rank}", dynamic_ncols=True)

    for y in iterator:
        k = int(counts_by_class[y].item())
        component_start = int(offsets[y].item())
        z = _read_class_latents(
            paths_by_class[y],
            use_flips=cfg.use_flips,
            max_per_class=cfg.max_per_class,
            seed=cfg.seed + y,
        )
        desc = extractor(z, batch_size=int(args.descriptor_batch_size))
        desc_centers, assign = _fit_kmeans(desc, k, iters=cfg.kmeans_iters, seed=cfg.seed + y * 1009, device=device)
        means, scales, weights = _latent_components_for_assignments(
            z,
            desc,
            desc_centers,
            assign,
            k,
            center_mode=cfg.center_mode,
            min_scale=cfg.min_scale,
            max_scale=cfg.max_scale,
            device=device,
        )
        local_indices.append(torch.arange(component_start, component_start + k, dtype=torch.long))
        local_centers.append(means)
        local_scales.append(scales)
        local_weights.append(weights)

    if local_indices:
        shard = {
            "component_indices": torch.cat(local_indices, dim=0),
            "centers": torch.cat(local_centers, dim=0),
            "scales": torch.cat(local_scales, dim=0),
            "weights": torch.cat(local_weights, dim=0),
            "rank": rank,
            "world_size": world,
            "elapsed_s": time.time() - started,
        }
    else:
        shard = {
            "component_indices": torch.empty((0,), dtype=torch.long),
            "centers": torch.empty((0, 32, 32, 4), dtype=torch.float32),
            "scales": torch.empty((0, 32, 32, 4), dtype=torch.float32),
            "weights": torch.empty((0,), dtype=torch.float32),
            "rank": rank,
            "world_size": world,
            "elapsed_s": time.time() - started,
        }
    shard_path = shard_dir / f"rank_{rank:05d}.pt"
    torch.save(shard, shard_path)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if not _is_rank_zero():
        return None

    prior_path = merge_decoder_prior(out_dir=out_dir, cfg=cfg, expected_world_size=world)
    return prior_path


def merge_decoder_prior(*, out_dir: Path, cfg: DecoderPriorConfig, expected_world_size: int) -> Path:
    counts_by_class = _class_component_counts(cfg.component_count, cfg.num_classes)
    offsets = _class_offsets(counts_by_class)
    centers = torch.empty((cfg.component_count, 32, 32, 4), dtype=torch.float32)
    scales = torch.empty_like(centers)
    weights = torch.zeros((cfg.component_count,), dtype=torch.float32)
    shard_paths = sorted((out_dir / "shards").glob("rank_*.pt"))
    if len(shard_paths) != expected_world_size:
        raise RuntimeError(f"Expected {expected_world_size} shards, found {len(shard_paths)} in {out_dir / 'shards'}")
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        idx = torch.as_tensor(shard["component_indices"], dtype=torch.long)
        if idx.numel() == 0:
            continue
        centers[idx] = shard["centers"].float()
        scales[idx] = shard["scales"].float()
        weights[idx] = torch.as_tensor(shard["weights"], dtype=torch.float32).clamp_min(0.0)

    artifact = {
        "format": "decoder_likelihood_prior.v1",
        "config": asdict(cfg),
        "centers": centers,
        "scales": scales,
        "component_weights": weights,
        "class_offsets": offsets,
        "class_counts": counts_by_class,
        "metadata": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "expected_world_size": expected_world_size,
            "descriptor": (
                "SDVAE latent components; responsibilities fitted by "
                f"{cfg.descriptor_mode} k-means over {','.join(cfg.descriptor_keys)}; "
                f"center_mode={cfg.center_mode}"
            ),
        },
    }
    prior_path = out_dir / "decoder_likelihood_prior.pt"
    torch.save(artifact, prior_path)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "format": artifact["format"],
                "config": artifact["config"],
                "metadata": artifact["metadata"],
                "component_count": int(cfg.component_count),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return prior_path


def _config_from_payload(payload: dict) -> DecoderPriorConfig:
    raw = dict(payload.get("config", {}) or {})
    if "descriptor_keys" in raw:
        raw["descriptor_keys"] = tuple(raw["descriptor_keys"])
    defaults = asdict(DecoderPriorConfig())
    defaults.update(raw)
    defaults["descriptor_keys"] = tuple(defaults["descriptor_keys"])
    return DecoderPriorConfig(**defaults)


def _class_nll(
    real_desc: torch.Tensor,
    gen_desc: torch.Tensor,
    logits: torch.Tensor,
    *,
    sigma: float,
    mc_samples: int,
) -> torch.Tensor:
    k = logits.shape[0]
    log_w = torch.log_softmax(logits.float(), dim=0)
    if mc_samples > 1:
        log_w = log_w.repeat(int(mc_samples)) - math.log(float(mc_samples))
    mse = (real_desc[:, None, :].float() - gen_desc[None, :, :].float()).square().mean(dim=-1)
    log_prob = log_w[None, :] - 0.5 * mse / max(float(sigma) ** 2, 1e-12)
    return -(torch.logsumexp(log_prob, dim=1)).mean()


def train_decoder_prior(args: argparse.Namespace) -> Path | None:
    maybe_init_distributed()
    rank = _rank()
    world = _world_size()
    device = _device()
    prior_in = Path(args.prior).resolve()
    payload = torch.load(prior_in, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)

    out_dir = Path(args.out_dir).resolve()
    shard_dir = out_dir / "train_shards"
    if _is_rank_zero():
        shard_dir.mkdir(parents=True, exist_ok=True)
        train_cfg = {
            "prior": str(prior_in),
            "steps_per_class": int(args.steps_per_class),
            "batch_size": int(args.batch_size),
            "mc_samples": int(args.mc_samples),
            "sigma": float(args.sigma),
            "lr": float(args.lr),
            "anchor_weight": float(args.anchor_weight),
            "scale_weight": float(args.scale_weight),
            "scale_mult": float(args.train_scale_mult),
            "max_per_class": int(args.max_per_class),
            "max_classes": int(args.max_classes),
        }
        (out_dir / "train_config.json").write_text(json.dumps(train_cfg, indent=2) + "\n", encoding="utf-8")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    centers_all = payload["centers"].float()
    scales_all = payload["scales"].float().clamp_min(1e-6)
    weights_all = payload["component_weights"].float().clamp_min(1.0)
    class_offsets = payload["class_offsets"].long()
    class_counts = payload["class_counts"].long()
    paths_by_class = _build_class_paths(args.cache_root, cfg.num_classes)
    extractor = DescriptorExtractor(cfg, device=device, compile_level=int(args.compile_level))

    class_ids = list(range(cfg.num_classes))
    if int(args.max_classes) > 0:
        class_ids = class_ids[: int(args.max_classes)]
    local_classes = [y for y in class_ids if y % world == rank]
    iterator: Iterable[int] = local_classes
    if _is_rank_zero():
        iterator = tqdm(local_classes, desc=f"train-rank{rank}", dynamic_ncols=True)

    local_indices = []
    local_centers = []
    local_scales = []
    local_weights = []
    metrics = []
    start_time = time.time()

    for y in iterator:
        start = int(class_offsets[y].item())
        count = int(class_counts[y].item())
        comp_slice = slice(start, start + count)
        z_real = _read_class_latents(
            paths_by_class[y],
            use_flips=cfg.use_flips,
            max_per_class=int(args.max_per_class),
            seed=int(args.seed) + y,
        )
        real_desc = extractor(z_real, batch_size=int(args.descriptor_batch_size)).to(device=device, dtype=torch.float32)

        center0 = centers_all[comp_slice].to(device=device, dtype=torch.float32)
        scale0 = scales_all[comp_slice].to(device=device, dtype=torch.float32).clamp_min(1e-5)
        center = torch.nn.Parameter(center0.clone())
        log_scale = torch.nn.Parameter(scale0.log())
        logits = torch.nn.Parameter(weights_all[comp_slice].to(device=device, dtype=torch.float32).log())
        opt = torch.optim.AdamW(
            [
                {"params": [center], "lr": float(args.lr)},
                {"params": [log_scale], "lr": float(args.lr) * float(args.scale_lr_mult)},
                {"params": [logits], "lr": float(args.lr) * float(args.logit_lr_mult)},
            ],
            weight_decay=0.0,
        )
        gen = torch.Generator(device=device).manual_seed(int(args.seed) + y * 9176 + rank)
        last = {}
        for step in range(int(args.steps_per_class)):
            idx = torch.randint(0, real_desc.shape[0], (int(args.batch_size),), generator=gen, device=device)
            batch_desc = real_desc[idx]
            scale = log_scale.exp().clamp(max=float(args.max_scale))
            if int(args.mc_samples) > 1:
                eps = torch.randn(
                    (int(args.mc_samples), count, *center.shape[1:]),
                    generator=gen,
                    device=device,
                    dtype=center.dtype,
                )
                u = center.unsqueeze(0) + eps * scale.unsqueeze(0) * float(args.train_scale_mult)
                u = u.reshape(int(args.mc_samples) * count, *center.shape[1:])
            else:
                if float(args.train_scale_mult) > 0:
                    eps = torch.randn(center.shape, generator=gen, device=device, dtype=center.dtype)
                    u = center + eps * scale * float(args.train_scale_mult)
                else:
                    u = center
            gen_desc = extractor(
                u,
                batch_size=int(args.generated_descriptor_batch_size),
                requires_grad=True,
            )
            nll = _class_nll(
                batch_desc,
                gen_desc,
                logits,
                sigma=float(args.sigma),
                mc_samples=int(args.mc_samples),
            )
            anchor = (center - center0).square().mean()
            scale_pen = (log_scale - scale0.log()).square().mean()
            loss = nll + float(args.anchor_weight) * anchor + float(args.scale_weight) * scale_pen
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([center, log_scale, logits], float(args.grad_clip))
            opt.step()
            last = {
                "class": y,
                "step": step + 1,
                "loss": float(loss.detach().cpu().item()),
                "nll": float(nll.detach().cpu().item()),
                "anchor": float(anchor.detach().cpu().item()),
                "scale_pen": float(scale_pen.detach().cpu().item()),
            }

        probs = torch.softmax(logits.detach().float(), dim=0)
        class_weight_total = weights_all[comp_slice].sum().clamp_min(1.0).to(device=device)
        local_indices.append(torch.arange(start, start + count, dtype=torch.long))
        local_centers.append(center.detach().cpu())
        local_scales.append(log_scale.detach().exp().clamp(max=float(args.max_scale)).cpu())
        local_weights.append((probs * class_weight_total).detach().cpu())
        metrics.append(last)

    shard = {
        "component_indices": torch.cat(local_indices, dim=0) if local_indices else torch.empty((0,), dtype=torch.long),
        "centers": torch.cat(local_centers, dim=0) if local_centers else torch.empty((0, 32, 32, 4), dtype=torch.float32),
        "scales": torch.cat(local_scales, dim=0) if local_scales else torch.empty((0, 32, 32, 4), dtype=torch.float32),
        "weights": torch.cat(local_weights, dim=0) if local_weights else torch.empty((0,), dtype=torch.float32),
        "rank": rank,
        "world_size": world,
        "elapsed_s": time.time() - start_time,
        "metrics": metrics,
    }
    torch.save(shard, shard_dir / f"rank_{rank:05d}.pt")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if not _is_rank_zero():
        return None

    # Reuse the original class layout and replace only trained shards.
    out_payload = dict(payload)
    out_payload["centers"] = centers_all.clone()
    out_payload["scales"] = scales_all.clone()
    out_payload["component_weights"] = weights_all.clone()
    for shard_path in sorted(shard_dir.glob("rank_*.pt")):
        shard_i = torch.load(shard_path, map_location="cpu", weights_only=False)
        idx = shard_i["component_indices"].long()
        if idx.numel() == 0:
            continue
        out_payload["centers"][idx] = shard_i["centers"].float()
        out_payload["scales"][idx] = shard_i["scales"].float()
        out_payload["component_weights"][idx] = shard_i["weights"].float().clamp_min(0.0)
    out_payload["format"] = "decoder_likelihood_prior.v1.trained"
    out_payload["training"] = {
        "prior": str(prior_in),
        "steps_per_class": int(args.steps_per_class),
        "batch_size": int(args.batch_size),
        "mc_samples": int(args.mc_samples),
        "sigma": float(args.sigma),
        "lr": float(args.lr),
        "anchor_weight": float(args.anchor_weight),
        "scale_weight": float(args.scale_weight),
        "elapsed_s": time.time() - start_time,
    }
    out_path = out_dir / "decoder_likelihood_prior_trained.pt"
    torch.save(out_payload, out_path)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "format": out_payload["format"],
                "config": out_payload.get("config", {}),
                "training": out_payload["training"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out_path


class DecoderPriorSampler:
    def __init__(
        self,
        prior_path: str | Path,
        *,
        scale_mult: float = 0.0,
        noise_floor: float = 0.0,
        weight_power: float = 1.0,
        device: torch.device | None = None,
    ):
        self.path = Path(prior_path)
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        self.config = payload.get("config", {})
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.centers = payload["centers"].to(self.device, dtype=torch.float32)
        self.scales = payload["scales"].to(self.device, dtype=torch.float32)
        self.weights = payload["component_weights"].to(self.device, dtype=torch.float32).clamp_min(0.0)
        self.class_offsets = payload["class_offsets"].to(self.device, dtype=torch.long)
        self.class_counts = payload["class_counts"].to(self.device, dtype=torch.long)
        self.scale_mult = float(scale_mult)
        self.noise_floor = float(noise_floor)
        self.weight_power = float(weight_power)
        self.postprocess_fn = get_postprocess_fn(use_cache=True)
        self.usage = torch.zeros((self.centers.shape[0],), dtype=torch.long)
        self.samples_tracked = 0

    def _sample_indices(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(self.device, dtype=torch.long)
        out = torch.empty_like(labels)
        for i, y_t in enumerate(labels):
            y = int(y_t.item())
            start = int(self.class_offsets[y].item())
            count = int(self.class_counts[y].item())
            comp = torch.arange(start, start + count, device=self.device)
            w = self.weights[comp].float()
            if self.weight_power != 1.0:
                w = w.clamp_min(0.0).pow(self.weight_power)
            if float(w.sum().item()) <= 0.0:
                local = torch.randint(0, count, (1,), device=self.device)
            else:
                local = torch.multinomial(w, num_samples=1, replacement=True)
            out[i] = comp[local]
        return out

    def gen(self, batch, params=None, apply_fn=None, postprocess_fn=None, cfg_scale: float = 1.0):
        del params, apply_fn, postprocess_fn, cfg_scale
        labels = batch[1]
        idx = self._sample_indices(labels)
        z = self.centers[idx]
        noise_scale = self.scales[idx] * self.scale_mult
        if self.noise_floor > 0:
            noise_scale = torch.sqrt(noise_scale * noise_scale + self.noise_floor**2)
        if self.scale_mult > 0 or self.noise_floor > 0:
            z = z + torch.randn_like(z) * noise_scale
        with torch.no_grad():
            bincount = torch.bincount(idx.detach().cpu(), minlength=self.usage.numel())
            self.usage += bincount.to(self.usage.dtype)
            self.samples_tracked += int(idx.numel())
        return self.postprocess_fn(z)

    def usage_metrics(self) -> dict[str, float]:
        counts = self.usage.float()
        total = float(counts.sum().item())
        if total <= 0:
            return {}
        p = counts[counts > 0] / total
        entropy = float(-(p * p.log()).sum().item())
        return {
            "decoder_component_usage_entropy": entropy,
            "decoder_component_usage_perplexity": float(math.exp(entropy)),
            "decoder_component_unique": float((counts > 0).sum().item()),
            "decoder_samples_tracked": float(total),
        }


def eval_decoder_prior(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = DecoderPriorSampler(
        args.prior,
        scale_mult=float(args.scale_mult),
        noise_floor=float(args.noise_floor),
        weight_power=float(args.weight_power),
        device=device,
    )
    eval_loader, _, _ = create_imagenet_split(
        resolution=256,
        split="val",
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        pin_memory=True,
    )
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    logger = WandbLogger()
    logger.set_logging(
        project=str(args.wandb_project),
        entity=str(args.wandb_entity) if args.wandb_entity else None,
        name=str(args.wandb_name or workdir.name),
        use_wandb=bool(args.use_wandb),
        workdir=str(workdir),
        log_every_k=1,
        mode=str(args.wandb_mode) if args.wandb_mode else None,
    )
    logger.set_step(0)
    metrics = evaluate_fid(
        dataset_name="imagenet256",
        gen_func=sampler.gen,
        gen_params={},
        eval_loader=eval_loader,
        logger=logger,
        num_samples=int(args.num_samples),
        log_folder="eval",
        log_prefix=f"decoder_prior_scale{float(args.scale_mult):g}",
        eval_prc_recall=int(args.num_samples) >= 50000,
        eval_isc=True,
        eval_fid=True,
    )
    metrics.update(sampler.usage_metrics())
    result = {
        "prior": str(Path(args.prior).resolve()),
        "cfg_scale": float(args.cfg_scale),
        "num_samples": int(args.num_samples),
        "eval_batch_size": int(args.eval_batch_size),
        "scale_mult": float(args.scale_mult),
        "noise_floor": float(args.noise_floor),
        "weight_power": float(args.weight_power),
        "config": sampler.config,
        **metrics,
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    logger.log_dict({f"decoder/{k}": v for k, v in metrics.items()})
    logger.finish()
    print(json.dumps(result, indent=2, allow_nan=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit/evaluate a decoder-style ImageNet latent likelihood prior.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit")
    fit.add_argument("--out-dir", required=True)
    fit.add_argument("--cache-root", default=IMAGENET_CACHE_PATH)
    fit.add_argument("--component-count", type=int, default=8192)
    fit.add_argument("--num-classes", type=int, default=1000)
    fit.add_argument("--descriptor-mode", choices=["mae", "raw"], default="mae")
    fit.add_argument("--descriptor-keys", default=",".join(DEFAULT_DESCRIPTOR_KEYS))
    fit.add_argument("--mae-path", default="hf://mae_latent_640")
    fit.add_argument("--feature-stats-from", default="")
    fit.add_argument("--kmeans-iters", type=int, default=8)
    fit.add_argument("--center-mode", choices=["mean", "medoid"], default="mean")
    fit.add_argument("--descriptor-batch-size", type=int, default=128)
    fit.add_argument("--compile-level", type=int, default=1)
    fit.add_argument("--use-flips", action="store_true")
    fit.add_argument("--max-classes", type=int, default=0)
    fit.add_argument("--max-per-class", type=int, default=0)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--min-scale", type=float, default=0.0)
    fit.add_argument("--max-scale", type=float, default=2.0)

    train = sub.add_parser("train")
    train.add_argument("--prior", required=True)
    train.add_argument("--out-dir", required=True)
    train.add_argument("--cache-root", default=IMAGENET_CACHE_PATH)
    train.add_argument("--steps-per-class", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--mc-samples", type=int, default=1)
    train.add_argument("--sigma", type=float, default=0.875)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--scale-lr-mult", type=float, default=0.25)
    train.add_argument("--logit-lr-mult", type=float, default=0.25)
    train.add_argument("--anchor-weight", type=float, default=0.05)
    train.add_argument("--scale-weight", type=float, default=0.001)
    train.add_argument("--train-scale-mult", type=float, default=0.25)
    train.add_argument("--max-scale", type=float, default=2.0)
    train.add_argument("--grad-clip", type=float, default=5.0)
    train.add_argument("--descriptor-batch-size", type=int, default=128)
    train.add_argument("--generated-descriptor-batch-size", type=int, default=64)
    train.add_argument("--compile-level", type=int, default=0)
    train.add_argument("--max-classes", type=int, default=0)
    train.add_argument("--max-per-class", type=int, default=0)
    train.add_argument("--seed", type=int, default=42)

    ev = sub.add_parser("eval")
    ev.add_argument("--prior", required=True)
    ev.add_argument("--workdir", required=True)
    ev.add_argument("--json-out", default="")
    ev.add_argument("--num-samples", type=int, default=50000)
    ev.add_argument("--eval-batch-size", type=int, default=512)
    ev.add_argument("--num-workers", type=int, default=4)
    ev.add_argument("--cfg-scale", type=float, default=1.0)
    ev.add_argument("--scale-mult", type=float, default=0.0)
    ev.add_argument("--noise-floor", type=float, default=0.0)
    ev.add_argument("--weight-power", type=float, default=1.0)
    ev.add_argument("--use-wandb", action="store_true")
    ev.add_argument("--wandb-project", default="drift")
    ev.add_argument("--wandb-entity", default="ucph-dk")
    ev.add_argument("--wandb-name", default="")
    ev.add_argument("--wandb-mode", default="")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "fit":
        prior = fit_decoder_prior(args)
        if prior is not None:
            print(f"saved {prior}")
    elif args.cmd == "train":
        prior = train_decoder_prior(args)
        if prior is not None:
            print(f"saved {prior}")
    elif args.cmd == "eval":
        eval_decoder_prior(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
