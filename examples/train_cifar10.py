"""Train Drifting Models or Rectified Flow on CIFAR-10."""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchvision.utils import make_grid, save_image

from drifting_models_pytorch import drifting_loss
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="drifting", choices=("drifting", "rectified_flow"))
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--tau", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--rf-sample-steps", type=int, default=16)
    parser.add_argument("--unet-dim", type=int, default=64)
    parser.add_argument("--unet-dim-mults", type=str, default="1,2,4")
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

    drifting_generator = DriftingUNetGenerator(dim=args.unet_dim, dim_mults=dim_mults).to(device)
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
            loss, _ = drifting_loss(generated, real, tau=args.tau)
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
            step >= args.min_steps
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
        "steps": actual_steps,
        "batch_size": args.batch_size,
        "tau": args.tau,
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
