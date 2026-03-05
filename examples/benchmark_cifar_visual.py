"""Run a visual CIFAR-10 benchmark across rectified flow and drifting variants."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from torchvision.utils import make_grid, save_image

from drifting_models_pytorch import compute_drifting_field
from drifting_models_pytorch.data import build_cifar10_loader
from drifting_models_pytorch.utils import ensure_dir, save_json, set_seed

try:
    from rectified_flow_pytorch import RectifiedFlow, Unet
except Exception:  # pragma: no cover - optional import path
    RectifiedFlow = None  # type: ignore[assignment]
    Unet = None  # type: ignore[assignment]


@dataclass(slots=True)
class VariantSpec:
    """Specifies one benchmark variant."""

    name: str
    label: str
    method: str
    drifting_mode: str


@dataclass(slots=True)
class BenchmarkConfig:
    """Shared benchmark hyperparameters used by all variants."""

    steps: int
    batch_size: int
    lr: float
    eval_every: int
    eval_n_samples: int
    eval_batch_size: int
    snapshot_batch_size: int
    rf_sample_steps: int
    unet_dim: int
    unet_dim_mults: tuple[int, ...]
    taus: tuple[float, ...]
    feature_stages: tuple[int, ...]
    feature_backbone: str
    include_raw_loss_in_feature_mode: bool
    seed: int
    data_root: str
    use_ema: bool


@dataclass(slots=True)
class EvalRecord:
    """Per-checkpoint metrics used for plotting and machine-readable artifacts."""

    global_step: int
    images_seen: int
    elapsed_seconds: float
    train_loss: float
    fid: float
    inception_score: float
    inception_score_std: float


@dataclass(slots=True)
class RunOutputs:
    """Paths and summaries produced by one variant run."""

    spec: VariantSpec
    outdir: Path
    records: list[EvalRecord]


@dataclass(slots=True)
class FeatureGroup:
    """A group of feature vectors sharing normalization statistics."""

    name: str
    generated: Tensor  # [batch, num_vectors, channels]
    real: Tensor  # [batch, num_vectors, channels]


class DriftingUNetGenerator(nn.Module):
    """U-Net generator that maps image-shaped noise to images."""

    def __init__(self, dim: int, dim_mults: tuple[int, ...]) -> None:
        """Initialize the generator.

        Args:
            dim: Base model width.
            dim_mults: Width multipliers at each stage.
        """
        super().__init__()
        if Unet is None:
            raise RuntimeError("rectified_flow_pytorch.Unet is required for CIFAR training")
        self.unet = Unet(dim=dim, channels=3, dim_mults=dim_mults, accept_time=False)

    def forward(self, noise_img: Tensor) -> Tensor:
        """Generate images in [-1, 1] from image-shaped noise."""
        return torch.tanh(self.unet(noise_img))


class FrozenFeatureEncoder(nn.Module):
    """Frozen ResNet feature encoder for feature-space drifting loss."""

    def __init__(self, stage_ids: tuple[int, ...], backbone_name: str) -> None:
        """Initialize encoder and selected stage outputs."""
        super().__init__()
        self.stage_ids = stage_ids
        self.backbone_name = backbone_name
        if backbone_name == "resnet18":
            try:
                backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                backbone = resnet18(weights=None)
        elif backbone_name == "resnet50":
            try:
                backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            except Exception:
                backbone = resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported feature backbone: {backbone_name}")
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def _normalize_images(images: Tensor) -> Tensor:
        """Normalize images from [-1, 1] into ImageNet feature space."""
        x = (images + 1.0) * 0.5
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    @staticmethod
    def _collect_stage_outputs(stage: nn.Sequential, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Collect outputs from every 2 blocks and stage end."""
        outputs: list[Tensor] = []
        for block_idx, block in enumerate(stage):
            x = block(x)
            if (block_idx + 1) % 2 == 0 or block_idx == len(stage) - 1:
                outputs.append(x)
        return x, outputs

    def forward(self, images: Tensor) -> tuple[list[Tensor], Tensor]:
        """Extract selected multi-scale maps and encoder-input squared-channel stats."""
        x = self._normalize_images(images)
        input_sq = (x.pow(2)).mean(dim=(2, 3))
        x = self.stem(x)
        feature_maps: list[Tensor] = []
        stages = (self.layer1, self.layer2, self.layer3, self.layer4)
        for stage_id, stage in enumerate(stages, start=1):
            x, outputs = self._collect_stage_outputs(stage, x)
            if stage_id in self.stage_ids:
                feature_maps.extend(outputs)
        return feature_maps, input_sq


class CifarMetricEvaluator:
    """Computes periodic FID and Inception Score with cached real-image stats."""

    def __init__(
        self,
        *,
        loader_cycle: itertools.cycle,
        eval_n_samples: int,
        eval_batch_size: int,
        device: torch.device,
    ) -> None:
        """Initialize evaluator and precompute real features for FID."""
        self.loader_cycle = loader_cycle
        self.eval_n_samples = eval_n_samples
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.fid_metric: Any = FrechetInceptionDistance(normalize=True, reset_real_features=False).to(device)
        self.is_metric: Any = InceptionScore(normalize=True).to(device)
        self._init_real_features()

    def _init_real_features(self) -> None:
        """Populate persistent real features from the CIFAR train split."""
        remaining = self.eval_n_samples
        while remaining > 0:
            real, _ = next(self.loader_cycle)
            real = real.to(self.device)
            current = min(remaining, real.shape[0])
            self.fid_metric.update((real[:current].clamp(-1.0, 1.0) + 1.0) * 0.5, real=True)
            remaining -= current

    def evaluate(self, sampler: Callable[[int, int], Tensor], seed: int) -> tuple[float, float, float]:
        """Evaluate generated samples with FID and IS."""
        self.fid_metric.reset()
        self.is_metric.reset()
        remaining = self.eval_n_samples
        batch_idx = 0
        while remaining > 0:
            current = min(remaining, self.eval_batch_size)
            generated = sampler(current, seed + batch_idx * 10_000).clamp(-1.0, 1.0)
            generated_01 = (generated + 1.0) * 0.5
            self.fid_metric.update(generated_01, real=False)
            self.is_metric.update(generated_01)
            remaining -= current
            batch_idx += 1
        fid = float(self.fid_metric.compute().item())
        inception_mean, inception_std = self.is_metric.compute()
        return fid, float(inception_mean.item()), float(inception_std.item())


def _parse_tuple_int(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("tuple cannot be empty")
    return values


def _parse_tuple_float(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("tuple cannot be empty")
    return values


def _as_feature_vectors(feature_map: Tensor) -> Tensor:
    """Convert [B, C, H, W] feature maps to [B, H*W, C] vectors."""
    return feature_map.flatten(2).transpose(1, 2).contiguous()


def _patch_mean_std_vectors(feature_map: Tensor, patch_size: int) -> tuple[Tensor, Tensor] | None:
    """Return per-patch mean/std vectors as [B, num_patches, C]."""
    height, width = feature_map.shape[-2:]
    if height < patch_size or width < patch_size:
        return None
    mean_map = F.avg_pool2d(feature_map, kernel_size=patch_size, stride=patch_size)
    sq_mean_map = F.avg_pool2d(feature_map.pow(2), kernel_size=patch_size, stride=patch_size)
    std_map = (sq_mean_map - mean_map.pow(2)).clamp_min(1e-12).sqrt()
    return _as_feature_vectors(mean_map), _as_feature_vectors(std_map)


def _feature_map_groups(map_name: str, generated_map: Tensor, real_map: Tensor) -> list[FeatureGroup]:
    """Build feature groups from one map following Appendix A descriptors."""
    groups: list[FeatureGroup] = [
        FeatureGroup(
            name=f"{map_name}/loc", generated=_as_feature_vectors(generated_map), real=_as_feature_vectors(real_map)
        ),
        FeatureGroup(
            name=f"{map_name}/global_mean",
            generated=generated_map.mean(dim=(2, 3), keepdim=False).unsqueeze(1),
            real=real_map.mean(dim=(2, 3), keepdim=False).unsqueeze(1),
        ),
        FeatureGroup(
            name=f"{map_name}/global_std",
            generated=generated_map.flatten(2).std(dim=2, unbiased=False).clamp_min(1e-6).unsqueeze(1),
            real=real_map.flatten(2).std(dim=2, unbiased=False).clamp_min(1e-6).unsqueeze(1),
        ),
    ]
    for patch in (2, 4):
        generated_patch = _patch_mean_std_vectors(generated_map, patch_size=patch)
        real_patch = _patch_mean_std_vectors(real_map, patch_size=patch)
        if generated_patch is None or real_patch is None:
            continue
        generated_mean, generated_std = generated_patch
        real_mean, real_std = real_patch
        groups.append(FeatureGroup(name=f"{map_name}/patch{patch}_mean", generated=generated_mean, real=real_mean))
        groups.append(FeatureGroup(name=f"{map_name}/patch{patch}_std", generated=generated_std, real=real_std))
    return groups


def _build_feature_groups(
    generated_maps: list[Tensor],
    real_maps: list[Tensor],
    generated_input_sq: Tensor,
    real_input_sq: Tensor,
) -> list[FeatureGroup]:
    """Build all feature groups used for feature-space drifting loss."""
    groups: list[FeatureGroup] = []
    for map_idx, (generated_map, real_map) in enumerate(zip(generated_maps, real_maps, strict=True)):
        groups.extend(_feature_map_groups(f"map_{map_idx:02d}", generated_map, real_map))
    groups.append(
        FeatureGroup(
            name="encoder_input_sq",
            generated=generated_input_sq.unsqueeze(1),
            real=real_input_sq.unsqueeze(1),
        )
    )
    return groups


def _single_descriptor_drifting_loss(generated: Tensor, real: Tensor, taus: tuple[float, ...]) -> Tensor:
    """Compute normalized multi-temperature drifting loss for one descriptor."""
    dim = generated.shape[1]
    sqrt_dim = dim**0.5
    all_samples = torch.cat((real, generated.detach()), dim=0)
    scale_s = (torch.cdist(generated, all_samples).mean() / sqrt_dim).detach().clamp_min(1e-6)
    generated_norm = generated / scale_s
    real_norm = real / scale_s
    combined_drift = torch.zeros_like(generated_norm)
    for tau in taus:
        drift = compute_drifting_field(
            x=generated_norm,
            y_pos=real_norm,
            y_neg=generated_norm,
            tau=tau * sqrt_dim,
            dual_softmax=True,
            exclude_self_from_negatives=True,
        )
        lambda_scale = ((drift.pow(2).sum(dim=1) / dim).mean().sqrt()).detach().clamp_min(1e-6)
        combined_drift = combined_drift + drift / lambda_scale
    target = (generated_norm + combined_drift).detach()
    return F.mse_loss(generated_norm, target)


def _groupwise_drifting_loss(feature_group: FeatureGroup, taus: tuple[float, ...]) -> Tensor:
    """Compute paper-style drifting loss for one feature group.

    This follows Appendix A with shared feature/drift normalization across
    locations in the same feature map group.
    """
    generated = feature_group.generated
    real = feature_group.real
    if generated.ndim != 3 or real.ndim != 3:
        raise ValueError("Feature groups must have shape [batch, num_vectors, channels]")
    batch_size, num_vectors, channels = generated.shape
    if batch_size < 2:
        raise ValueError("Batch size must be >= 2 for drifting field computation")
    sqrt_channels = channels**0.5

    generated_flat = generated.reshape(batch_size * num_vectors, channels)
    real_flat = real.reshape(batch_size * num_vectors, channels)
    all_samples = torch.cat((real_flat, generated_flat.detach()), dim=0)
    scale_s = (torch.cdist(generated_flat, all_samples).mean() / sqrt_channels).detach().clamp_min(1e-6)
    generated_norm = generated / scale_s
    real_norm = real / scale_s

    generated_loc = generated_norm.permute(1, 0, 2).contiguous()  # [L, B, C]
    real_loc = real_norm.permute(1, 0, 2).contiguous()  # [L, B, C]
    combined_drift = torch.zeros_like(generated_loc)
    eye = torch.eye(batch_size, device=generated.device, dtype=torch.bool).unsqueeze(0)
    for tau in taus:
        tau_scaled = tau * sqrt_channels
        dist_pos = torch.cdist(generated_loc, real_loc)
        dist_neg = torch.cdist(generated_loc, generated_loc)
        dist_neg = dist_neg.masked_fill(eye, 1e6)
        logits_pos = -dist_pos / tau_scaled
        logits_neg = -dist_neg / tau_scaled
        logits = torch.cat((logits_pos, logits_neg), dim=2)
        affinity_row = logits.softmax(dim=2)
        affinity_col = logits.softmax(dim=1)
        affinity = torch.sqrt(affinity_row * affinity_col + 1e-12)
        affinity_pos = affinity[:, :, :batch_size]
        affinity_neg = affinity[:, :, batch_size:]
        weight_pos = affinity_pos * affinity_neg.sum(dim=2, keepdim=True)
        weight_neg = affinity_neg * affinity_pos.sum(dim=2, keepdim=True)
        drift = torch.bmm(weight_pos, real_loc) - torch.bmm(weight_neg, generated_loc)
        lambda_scale = ((drift.pow(2).sum(dim=2) / channels).mean().sqrt()).detach().clamp_min(1e-6)
        combined_drift = combined_drift + drift / lambda_scale

    target = (generated_loc + combined_drift).detach()
    per_vector_loss = (generated_loc - target).pow(2).mean(dim=(1, 2))
    return per_vector_loss.sum()


def _pixel_drifting_loss(generated: Tensor, real: Tensor, taus: tuple[float, ...]) -> Tensor:
    """Compute drifting loss directly in pixel space."""
    return _single_descriptor_drifting_loss(generated.flatten(1), real.flatten(1), taus=taus)


def _paper_feature_drifting_loss(
    generated_images: Tensor,
    real_images: Tensor,
    feature_encoder: FrozenFeatureEncoder,
    taus: tuple[float, ...],
    include_raw_loss: bool,
) -> Tensor:
    """Compute feature drifting loss with Appendix-A style normalization."""
    generated_maps, generated_input_sq = feature_encoder(generated_images)
    with torch.no_grad():
        real_maps, real_input_sq = feature_encoder(real_images)
    feature_groups = _build_feature_groups(generated_maps, real_maps, generated_input_sq, real_input_sq)
    loss = torch.zeros((), device=generated_images.device)
    for feature_group in feature_groups:
        loss = loss + _groupwise_drifting_loss(feature_group, taus=taus)
    if include_raw_loss:
        loss = loss + _pixel_drifting_loss(generated_images, real_images, taus=taus)
    return loss


def _noise_like(batch_size: int, device: torch.device, seed: int, dtype: torch.dtype) -> Tensor:
    """Build deterministic Gaussian image noise for repeatable sampling."""
    generator_device = "cpu" if device.type == "cpu" else device.type
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return torch.randn((batch_size, 3, 32, 32), generator=generator, device=device, dtype=dtype)


def _write_records_csv(records: list[EvalRecord], path: Path) -> None:
    """Write per-checkpoint metrics to CSV."""
    fieldnames = [
        "global_step",
        "images_seen",
        "elapsed_seconds",
        "train_loss",
        "fid",
        "inception_score",
        "inception_score_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _write_records_json(records: list[EvalRecord], path: Path) -> None:
    """Write per-checkpoint metrics to JSON."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(record) for record in records], handle, indent=2, sort_keys=True)


def _read_records_csv(path: Path) -> list[EvalRecord]:
    """Read EvalRecord rows from CSV."""
    rows: list[EvalRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            rows.append(
                EvalRecord(
                    global_step=int(item["global_step"]),
                    images_seen=int(item["images_seen"]),
                    elapsed_seconds=float(item["elapsed_seconds"]),
                    train_loss=float(item["train_loss"]),
                    fid=float(item["fid"]),
                    inception_score=float(item["inception_score"]),
                    inception_score_std=float(item["inception_score_std"]),
                )
            )
    return rows


def _build_common_steps(outputs: list[RunOutputs]) -> list[int]:
    """Find matched checkpoints available for all methods."""
    sets = [{record.global_step for record in output.records} for output in outputs]
    common = sorted(set.intersection(*sets))
    if not common:
        raise RuntimeError("No common checkpoints found across methods")
    if len(common) <= 3:
        return common
    return [common[0], common[len(common) // 2], common[-1]]


def _available_sample_steps(samples_dir: Path) -> list[int]:
    """Return sorted step ids for saved sample grids."""
    steps: list[int] = []
    for path in samples_dir.glob("step_*.png"):
        stem = path.stem
        steps.append(int(stem.split("_")[1]))
    return sorted(steps)


def _resolve_sample_path(samples_dir: Path, target_step: int) -> Path:
    """Find the closest available sample image at or before target step."""
    candidates = _available_sample_steps(samples_dir)
    if not candidates:
        raise RuntimeError(f"No sample grids found in {samples_dir}")
    chosen = candidates[0]
    for step in candidates:
        if step <= target_step:
            chosen = step
        else:
            break
    return samples_dir / f"step_{chosen:06d}.png"


def _plot_comparison(
    *,
    outputs: list[RunOutputs],
    checkpoint_steps: list[int],
    figure_path: Path,
    add_svg_pdf: bool,
) -> None:
    """Build the README-ready visual comparison figure."""
    colors = {
        "rectified_flow": "#1f77b4",
        "drifting_feature": "#2ca02c",
        "drifting_pixel": "#d62728",
    }
    figure = plt.figure(figsize=(18, 10 + 2 * len(outputs)), constrained_layout=True)
    grid = figure.add_gridspec(
        nrows=3 + len(outputs),
        ncols=len(checkpoint_steps),
        height_ratios=[1.2, 1.2, 1.2] + [1.0] * len(outputs),
    )

    ax_fid = figure.add_subplot(grid[0, :])
    ax_is = figure.add_subplot(grid[1, :])
    ax_loss = figure.add_subplot(grid[2, :])

    for output in outputs:
        x = [record.images_seen for record in output.records]
        ax_fid.plot(
            x, [record.fid for record in output.records], label=output.spec.label, color=colors[output.spec.name]
        )
        ax_is.plot(
            x,
            [record.inception_score for record in output.records],
            label=output.spec.label,
            color=colors[output.spec.name],
        )
        ax_loss.plot(
            x,
            [record.train_loss for record in output.records],
            label=output.spec.label,
            color=colors[output.spec.name],
        )

    for axis, ylabel in ((ax_fid, "FID"), (ax_is, "Inception Score"), (ax_loss, "Training Loss")):
        axis.set_xlabel("images_seen")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")

    for row, output in enumerate(outputs):
        for col, checkpoint in enumerate(checkpoint_steps):
            axis = figure.add_subplot(grid[3 + row, col])
            image_path = _resolve_sample_path(output.outdir / "samples", checkpoint)
            image = mpimg.imread(image_path)
            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            if col == 0:
                axis.set_ylabel(output.spec.label, fontsize=10)
            if row == 0:
                axis.set_title(f"step {checkpoint}")

    figure.suptitle("CIFAR-10 Training Benchmark: Rectified Flow vs Drifting Variants", fontsize=18)
    ensure_dir(figure_path.parent)
    figure.savefig(figure_path, dpi=240)
    if add_svg_pdf:
        figure.savefig(figure_path.with_suffix(".svg"), dpi=240)
        figure.savefig(figure_path.with_suffix(".pdf"), dpi=240)
    plt.close(figure)


def _build_variant_specs() -> list[VariantSpec]:
    """Return all benchmark variants in plotting order."""
    return [
        VariantSpec(
            name="rectified_flow",
            label="Rectified Flow",
            method="rectified_flow",
            drifting_mode="feature",
        ),
        VariantSpec(
            name="drifting_feature",
            label="Drifting Model (feature)",
            method="drifting",
            drifting_mode="feature",
        ),
        VariantSpec(
            name="drifting_pixel",
            label="Drifting Model (pixel)",
            method="drifting",
            drifting_mode="pixel",
        ),
    ]


def _select_variant_specs(raw: str, all_specs: list[VariantSpec]) -> list[VariantSpec]:
    """Filter variants from a comma-separated list of variant names."""
    requested = [name.strip() for name in raw.split(",") if name.strip()]
    if not requested:
        raise ValueError("At least one variant must be specified")
    by_name = {spec.name: spec for spec in all_specs}
    selected: list[VariantSpec] = []
    seen: set[str] = set()
    for name in requested:
        if name not in by_name:
            valid = ",".join(sorted(by_name))
            raise ValueError(f"Unknown variant '{name}'. Valid variants: {valid}")
        if name in seen:
            continue
        selected.append(by_name[name])
        seen.add(name)
    return selected


def _run_variant(spec: VariantSpec, config: BenchmarkConfig, outdir: Path, device: torch.device) -> RunOutputs:
    """Train one method variant and persist metrics/sample artifacts."""
    set_seed(config.seed)
    run_outdir = ensure_dir(outdir / spec.name)
    samples_dir = ensure_dir(run_outdir / "samples")

    train_loader = build_cifar10_loader(batch_size=config.batch_size, root=config.data_root, train=True)
    eval_loader = build_cifar10_loader(batch_size=config.eval_batch_size, root=config.data_root, train=True)
    train_iterator = itertools.cycle(train_loader)
    eval_iterator = itertools.cycle(eval_loader)

    drifting_generator = DriftingUNetGenerator(dim=config.unet_dim, dim_mults=config.unet_dim_mults).to(device)
    feature_encoder: FrozenFeatureEncoder | None = None
    if spec.method == "drifting" and spec.drifting_mode == "feature":
        feature_encoder = FrozenFeatureEncoder(
            stage_ids=config.feature_stages,
            backbone_name=config.feature_backbone,
        ).to(device)
        feature_encoder.eval()
    rf_unet = Unet(dim=config.unet_dim, channels=3, dim_mults=config.unet_dim_mults).to(device)
    rf_runner = RectifiedFlow(model=rf_unet, data_shape=(3, 32, 32)).to(device)

    model = drifting_generator if spec.method == "drifting" else rf_runner
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    evaluator = CifarMetricEvaluator(
        loader_cycle=eval_iterator,
        eval_n_samples=config.eval_n_samples,
        eval_batch_size=config.eval_batch_size,
        device=device,
    )

    def sample_images(batch_size: int, seed: int) -> Tensor:
        """Generate deterministic samples used by metrics and snapshots."""
        with torch.no_grad():
            noise = _noise_like(batch_size, device, seed, next(model.parameters()).dtype)
            if spec.method == "drifting":
                return drifting_generator(noise)
            return rf_runner.sample(
                batch_size=batch_size,
                noise=noise,
                data_shape=(3, 32, 32),
                steps=config.rf_sample_steps,
                use_ema=config.use_ema,
            )

    eval_steps = set(range(config.eval_every, config.steps + 1, config.eval_every))
    eval_steps.add(1)
    eval_steps.add(config.steps)
    fixed_snapshot_noise = _noise_like(
        config.snapshot_batch_size,
        device,
        seed=config.seed + 40,
        dtype=next(model.parameters()).dtype,
    )

    losses: list[float] = []
    records: list[EvalRecord] = []
    start = time.perf_counter()

    for step in range(1, config.steps + 1):
        real, _ = next(train_iterator)
        real = real.to(device)
        optimizer.zero_grad(set_to_none=True)
        if spec.method == "drifting":
            generated = drifting_generator(torch.randn_like(real))
            if spec.drifting_mode == "feature":
                if feature_encoder is None:
                    raise RuntimeError("feature encoder must be initialized for drifting_feature mode")
                loss = _paper_feature_drifting_loss(
                    generated_images=generated,
                    real_images=real,
                    feature_encoder=feature_encoder,
                    taus=config.taus,
                    include_raw_loss=config.include_raw_loss_in_feature_mode,
                )
            else:
                loss = _pixel_drifting_loss(generated, real, taus=config.taus)
        else:
            loss = rf_runner(real)

        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if step in eval_steps:
            with torch.no_grad():
                if spec.method == "drifting":
                    snapshots = drifting_generator(fixed_snapshot_noise)
                else:
                    snapshots = rf_runner.sample(
                        batch_size=config.snapshot_batch_size,
                        noise=fixed_snapshot_noise,
                        data_shape=(3, 32, 32),
                        steps=config.rf_sample_steps,
                        use_ema=config.use_ema,
                    )
                grid = make_grid((snapshots.clamp(-1.0, 1.0) + 1.0) * 0.5, nrow=8)
                save_image(grid, samples_dir / f"step_{step:06d}.png")

            fid, inception_mean, inception_std = evaluator.evaluate(sample_images, seed=config.seed + step * 97)
            elapsed = time.perf_counter() - start
            record = EvalRecord(
                global_step=step,
                images_seen=step * config.batch_size,
                elapsed_seconds=elapsed,
                train_loss=losses[-1],
                fid=fid,
                inception_score=inception_mean,
                inception_score_std=inception_std,
            )
            records.append(record)
            print(
                f"[{spec.name}] step={step:06d} images_seen={record.images_seen} "
                f"loss={record.train_loss:.6f} fid={fid:.4f} is={inception_mean:.4f} "
                f"elapsed={elapsed:.1f}s"
            )

    _write_records_csv(records, run_outdir / "metrics_history.csv")
    _write_records_json(records, run_outdir / "metrics_history.json")
    save_json(
        {
            "variant": asdict(spec),
            "config": asdict(config),
            "final_metrics": asdict(records[-1]),
        },
        run_outdir / "metrics_summary.json",
    )
    return RunOutputs(spec=spec, outdir=run_outdir, records=records)


def _build_config(args: argparse.Namespace) -> BenchmarkConfig:
    """Construct benchmark config from mode presets and CLI overrides."""
    presets: dict[str, dict[str, Any]] = {
        "quick": {
            "steps": 120,
            "batch_size": 64,
            "eval_every": 40,
            "eval_n_samples": 512,
            "eval_batch_size": 128,
            "snapshot_batch_size": 64,
            "rf_sample_steps": 8,
        },
        "full": {
            "steps": 6000,
            "batch_size": 256,
            "eval_every": 1000,
            "eval_n_samples": 5000,
            "eval_batch_size": 256,
            "snapshot_batch_size": 64,
            "rf_sample_steps": 16,
        },
    }
    preset = presets[args.mode]
    taus = _parse_tuple_float(args.taus)
    canonical_taus = (0.02, 0.05, 0.2)
    if tuple(round(value, 6) for value in taus) != canonical_taus:
        raise ValueError(f"This benchmark uses paper triple taus only: {canonical_taus}")
    return BenchmarkConfig(
        steps=args.steps if args.steps is not None else int(preset["steps"]),
        batch_size=args.batch_size if args.batch_size is not None else int(preset["batch_size"]),
        lr=args.lr,
        eval_every=args.eval_every if args.eval_every is not None else int(preset["eval_every"]),
        eval_n_samples=args.eval_n_samples if args.eval_n_samples is not None else int(preset["eval_n_samples"]),
        eval_batch_size=args.eval_batch_size if args.eval_batch_size is not None else int(preset["eval_batch_size"]),
        snapshot_batch_size=(
            args.snapshot_batch_size if args.snapshot_batch_size is not None else int(preset["snapshot_batch_size"])
        ),
        rf_sample_steps=args.rf_sample_steps if args.rf_sample_steps is not None else int(preset["rf_sample_steps"]),
        unet_dim=args.unet_dim,
        unet_dim_mults=_parse_tuple_int(args.unet_dim_mults),
        taus=taus,
        feature_stages=_parse_tuple_int(args.feature_stages),
        feature_backbone=args.feature_backbone,
        include_raw_loss_in_feature_mode=not args.disable_raw_feature_loss,
        seed=args.seed,
        data_root=args.data_root,
        use_ema=args.use_ema,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="quick", choices=("quick", "full"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-n-samples", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--snapshot-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rf-sample-steps", type=int, default=None)
    parser.add_argument("--unet-dim", type=int, default=64)
    parser.add_argument("--unet-dim-mults", type=str, default="1,2,4")
    parser.add_argument("--taus", type=str, default="0.02,0.05,0.2")
    parser.add_argument("--feature-stages", type=str, default="1,2,3,4")
    parser.add_argument("--feature-backbone", type=str, default="resnet18", choices=("resnet18", "resnet50"))
    parser.add_argument("--disable-raw-feature-loss", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--export-vector", action="store_true")
    parser.add_argument(
        "--variants",
        type=str,
        default="rectified_flow,drifting_feature,drifting_pixel",
        help="Comma-separated subset from {rectified_flow,drifting_feature,drifting_pixel}.",
    )
    return parser.parse_args()


def _copy_metric_artifacts(outputs: list[RunOutputs], metrics_dir: Path) -> dict[str, Path]:
    """Copy canonical CSV/JSON artifacts to results/metrics."""
    ensure_dir(metrics_dir)
    mapping: dict[str, Path] = {}
    for output in outputs:
        csv_source = output.outdir / "metrics_history.csv"
        json_source = output.outdir / "metrics_history.json"
        if output.spec.name == "rectified_flow":
            csv_target = metrics_dir / "cifar_rectified_flow.csv"
            json_target = metrics_dir / "cifar_rectified_flow.json"
        elif output.spec.name == "drifting_feature":
            csv_target = metrics_dir / "cifar_drifting_feature.csv"
            json_target = metrics_dir / "cifar_drifting_feature.json"
        else:
            csv_target = metrics_dir / "cifar_drifting_pixel.csv"
            json_target = metrics_dir / "cifar_drifting_pixel.json"
        shutil.copy2(csv_source, csv_target)
        shutil.copy2(json_source, json_target)
        mapping[output.spec.name] = csv_target
    return mapping


def main() -> None:
    """Run the full 3-way CIFAR visual benchmark and build output artifacts."""
    if RectifiedFlow is None or Unet is None:
        raise RuntimeError("Please install rectified-flow-pytorch to run CIFAR benchmarks")

    args = parse_args()
    config = _build_config(args)
    selected_specs = _select_variant_specs(args.variants, _build_variant_specs())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_root = ensure_dir(args.results_root)
    run_root = ensure_dir(results_root / "benchmarks_visual" / "cifar10")
    metrics_dir = ensure_dir(results_root / "metrics")
    figures_dir = ensure_dir(results_root / "figures")

    save_json(
        {
            "mode": args.mode,
            "device": str(device),
            "config": asdict(config),
            "variants": [asdict(item) for item in selected_specs],
        },
        metrics_dir / "run_config.json",
    )

    outputs: list[RunOutputs] = []
    for spec in selected_specs:
        print(f"\n=== Running {spec.name} ===")
        output = _run_variant(spec=spec, config=config, outdir=run_root, device=device)
        outputs.append(output)

    _copy_metric_artifacts(outputs, metrics_dir)
    checkpoint_steps = _build_common_steps(outputs)
    figure_path = figures_dir / "cifar_training_comparison.png"
    _plot_comparison(
        outputs=outputs, checkpoint_steps=checkpoint_steps, figure_path=figure_path, add_svg_pdf=args.export_vector
    )

    summary = {
        "checkpoints_used": checkpoint_steps,
        "figure": str(figure_path),
        "variants": {
            output.spec.name: {
                "label": output.spec.label,
                "final": asdict(output.records[-1]),
                "history_csv": str(output.outdir / "metrics_history.csv"),
                "history_json": str(output.outdir / "metrics_history.json"),
            }
            for output in outputs
        },
    }
    save_json(summary, metrics_dir / "summary.json")

    print("\n=== Benchmark Complete ===")
    print(f"Figure: {figure_path}")
    print(f"Metrics: {metrics_dir}")
    for output in outputs:
        final = output.records[-1]
        print(f"- {output.spec.name}: fid={final.fid:.4f} is={final.inception_score:.4f} step={final.global_step}")


if __name__ == "__main__":
    main()
