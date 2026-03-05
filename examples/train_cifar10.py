"""Train Drifting Models or Rectified Flow on CIFAR-10."""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18
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
class ImageStats:
    """Basic tensor statistics used for sanity checks."""

    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    is_finite: bool


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
        """Generate images from image-shaped Gaussian noise.

        Args:
            noise_img: Input noise with shape [batch, 3, 32, 32].

        Returns:
            Generated images in [-1, 1].
        """
        return torch.tanh(self.unet(noise_img))


def _image_stats(images: Tensor) -> ImageStats:
    """Compute simple sanity statistics for an image batch."""
    return ImageStats(
        min_val=float(images.min().item()),
        max_val=float(images.max().item()),
        mean_val=float(images.mean().item()),
        std_val=float(images.std().item()),
        is_finite=bool(torch.isfinite(images).all().item()),
    )


def _parse_dim_mults(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("dim-mults cannot be empty")
    return values


def _parse_taus(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("taus cannot be empty")
    return values


def _drifting_features(images: Tensor) -> Tensor:
    """Extract lower-dimensional normalized features for drifting loss.

    Args:
        images: Input image tensor with shape [batch, 3, 32, 32].

    Returns:
        Normalized feature vectors with shape [batch, 192].
    """
    pooled = F.avg_pool2d(images, kernel_size=4)  # [batch, 3, 8, 8]
    return F.normalize(pooled.flatten(1), dim=1)


class FrozenFeatureEncoder(nn.Module):
    """Frozen ResNet feature encoder for drifting loss."""

    def __init__(self, stage_ids: tuple[int, ...]) -> None:
        """Initialize encoder and selected stage outputs.

        Args:
            stage_ids: Stage ids from {1,2,3,4}.
        """
        super().__init__()
        self.stage_ids = stage_ids
        try:
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = resnet18(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, images: Tensor) -> list[Tensor]:
        """Extract multi-scale feature maps.

        Args:
            images: Input images in [-1, 1], shape [batch, 3, 32, 32].

        Returns:
            Selected feature maps.
        """
        x = (images + 1.0) * 0.5
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        feats: dict[int, Tensor] = {}
        x = self.stem(x)
        x = self.layer1(x)
        feats[1] = x
        x = self.layer2(x)
        feats[2] = x
        x = self.layer3(x)
        feats[3] = x
        x = self.layer4(x)
        feats[4] = x
        return [feats[idx] for idx in self.stage_ids]


def _parse_stage_ids(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("feature-stages cannot be empty")
    for value in values:
        if value not in {1, 2, 3, 4}:
            raise ValueError("feature-stages must only contain values in {1,2,3,4}")
    return values


def _feature_descriptors(images: Tensor, feature_encoder: FrozenFeatureEncoder) -> list[Tensor]:
    """Build compact per-stage descriptors inspired by appendix features."""
    descriptors: list[Tensor] = []
    for feat in feature_encoder(images):
        global_mean = feat.mean(dim=(2, 3))
        global_std = feat.flatten(2).std(dim=2, unbiased=False).clamp_min(1e-6)
        pooled_2x2 = F.adaptive_avg_pool2d(feat, output_size=(2, 2)).flatten(1)
        descriptor = torch.cat((global_mean, global_std, pooled_2x2), dim=1)
        descriptors.append(descriptor)
    input_sq = (images**2).mean(dim=(2, 3))
    descriptors.append(input_sq)
    return descriptors


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


def _pixel_drifting_loss(generated: Tensor, real: Tensor, taus: tuple[float, ...]) -> Tensor:
    """Compute drifting loss directly in raw pixel space."""
    generated_flat = generated.flatten(1)
    real_flat = real.flatten(1)
    return _single_descriptor_drifting_loss(generated_flat, real_flat, taus=taus)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="drifting", choices=("drifting", "rectified_flow"))
    parser.add_argument("--drifting-mode", type=str, default="feature", choices=("feature", "pixel"))
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--taus", type=str, default="0.02,0.05,0.2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--rf-sample-steps", type=int, default=16)
    parser.add_argument("--unet-dim", type=int, default=64)
    parser.add_argument("--unet-dim-mults", type=str, default="1,2,4")
    parser.add_argument("--feature-stages", type=str, default="2,4")
    parser.add_argument("--use-early-stop", action="store_true")
    parser.add_argument("--min-steps", type=int, default=2_000)
    parser.add_argument("--early-stop-window", type=int, default=300)
    parser.add_argument("--early-stop-loss-delta", type=float, default=5e-4)
    parser.add_argument("--outdir", type=str, default="results/cifar10")
    parser.add_argument("--data-root", type=str, default="./data")
    return parser.parse_args()


def main() -> None:
    """Run CIFAR-10 training."""
    args = parse_args()
    set_seed(args.seed)
    if RectifiedFlow is None or Unet is None:
        raise RuntimeError("Please install rectified-flow-pytorch to run CIFAR experiments")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = ensure_dir(args.outdir)
    samples_dir = ensure_dir(outdir / "samples")

    loader = build_cifar10_loader(batch_size=args.batch_size, root=args.data_root, train=True)
    iterator = itertools.cycle(loader)
    dim_mults = _parse_dim_mults(args.unet_dim_mults)
    taus = _parse_taus(args.taus)
    stage_ids = _parse_stage_ids(args.feature_stages)

    drifting_generator = DriftingUNetGenerator(dim=args.unet_dim, dim_mults=dim_mults).to(device)
    feature_encoder = FrozenFeatureEncoder(stage_ids=stage_ids).to(device).eval()
    rf_unet = Unet(dim=args.unet_dim, channels=3, dim_mults=dim_mults).to(device)
    rf_runner = RectifiedFlow(model=rf_unet, data_shape=(3, 32, 32)).to(device)
    model = drifting_generator if args.method == "drifting" else rf_runner
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    losses: list[float] = []
    start = time.perf_counter()
    actual_steps = args.steps

    for step in range(1, args.steps + 1):
        real, _ = next(iterator)
        real = real.to(device)
        optimizer.zero_grad(set_to_none=True)
        if args.method == "drifting":
            generated = drifting_generator(torch.randn_like(real))
            if args.drifting_mode == "feature":
                loss = torch.zeros((), device=device)
                generated_descriptors = _feature_descriptors(generated, feature_encoder=feature_encoder)
                with torch.no_grad():
                    real_descriptors = _feature_descriptors(real, feature_encoder=feature_encoder)
                for generated_desc, real_desc in zip(generated_descriptors, real_descriptors, strict=True):
                    loss = loss + _single_descriptor_drifting_loss(generated_desc, real_desc, taus=taus)
            else:
                loss = _pixel_drifting_loss(generated, real, taus=taus)
        else:
            loss = rf_runner(real)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % args.save_every == 0 or step == 1 or step == args.steps:
            with torch.no_grad():
                if args.method == "drifting":
                    eval_images = drifting_generator(torch.randn(64, 3, 32, 32, device=device))
                else:
                    eval_images = rf_runner.sample(batch_size=64, data_shape=(3, 32, 32), steps=args.rf_sample_steps)
                stats = _image_stats(eval_images)
                if not stats.is_finite:
                    raise RuntimeError("Non-finite values found in generated samples")
                print(
                    "sample_stats "
                    f"min={stats.min_val:.3f} max={stats.max_val:.3f} "
                    f"mean={stats.mean_val:.3f} std={stats.std_val:.3f}"
                )
                grid = make_grid((eval_images.clamp(-1.0, 1.0) + 1.0) / 2.0, nrow=8)
                save_image(grid, samples_dir / f"step_{step:06d}.png")

        if (
            args.use_early_stop
            and step >= args.min_steps
            and len(losses) >= args.early_stop_window * 2
            and (
                float(torch.tensor(losses[-args.early_stop_window * 2 : -args.early_stop_window]).mean())
                - float(torch.tensor(losses[-args.early_stop_window :]).mean())
            )
            < args.early_stop_loss_delta
        ):
            actual_steps = step
            print(f"early-stop step={step:06d} due to converged loss trend")
            break

        if step % 200 == 0 or step == args.steps:
            print(f"step={step:06d} loss={loss.item():.6f}")

    elapsed = time.perf_counter() - start
    summary = {
        "method": args.method,
        "drifting_mode": args.drifting_mode,
        "steps": actual_steps,
        "batch_size": args.batch_size,
        "taus": taus,
        "feature_stages": stage_ids,
        "unet_dim": args.unet_dim,
        "unet_dim_mults": dim_mults,
        "elapsed_sec": elapsed,
        "final_loss": losses[-1],
        "mean_loss_last_100": float(torch.tensor(losses[-100:]).mean()),
    }
    save_json(summary, outdir / "metrics.json")
    print(f"Saved metrics to {outdir / 'metrics.json'}")


if __name__ == "__main__":
    main()
