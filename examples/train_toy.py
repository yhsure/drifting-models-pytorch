"""Train Drifting Models or Rectified Flow on toy 2D datasets."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

from drifting_models_pytorch import (
    DriftingModelConfig,
    TimeConditionedMLP,
    build_model,
    drifting_loss,
    rectified_flow_loss,
)
from drifting_models_pytorch.data import get_toy_batch
from drifting_models_pytorch.utils import ensure_dir, mmd_rbf, save_json, set_seed

try:
    from rectified_flow_pytorch import RectifiedFlow
except Exception:  # pragma: no cover - optional import path
    RectifiedFlow = None  # type: ignore[assignment]


class LucidRainsToyModel(nn.Module):
    """Time-conditioned MLP adapter for lucidrains RectifiedFlow."""

    def __init__(self, dim: int, hidden_dim: int, depth: int) -> None:
        """Initialize the adapter model.

        Args:
            dim: Data dimension.
            hidden_dim: Hidden width.
            depth: Number of hidden layers.
        """
        super().__init__()
        self.model = TimeConditionedMLP(dim=dim, hidden_dim=hidden_dim, depth=depth)

    def forward(self, x: Tensor, times: Tensor) -> Tensor:
        """Predict flow vectors.

        Args:
            x: Noised inputs with shape [batch, dim].
            times: Time values with shape [batch].

        Returns:
            Predicted flow with shape [batch, dim].
        """
        return self.model(x, times.unsqueeze(-1))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="swissroll", choices=("swissroll", "checkerboard"))
    parser.add_argument("--method", type=str, default="drifting", choices=("drifting", "rectified_flow"))
    parser.add_argument("--steps", type=int, default=8_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.08)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--rf-impl", type=str, default="internal", choices=("lucidrains", "internal"))
    parser.add_argument("--rf-sample-steps", type=int, default=32)
    parser.add_argument("--min-steps", type=int, default=2_000)
    parser.add_argument("--early-stop-window", type=int, default=200)
    parser.add_argument("--early-stop-mmd", type=float, default=0.0020)
    parser.add_argument("--outdir", type=str, default="results/toy")
    return parser.parse_args()


def _save_scatter(path: Path, generated: Tensor, real: Tensor, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(real[:, 0].cpu().numpy(), real[:, 1].cpu().numpy(), s=6, alpha=0.35, label="data")
    ax.scatter(generated[:, 0].cpu().numpy(), generated[:, 1].cpu().numpy(), s=6, alpha=0.35, label="generated")
    ax.legend(loc="upper right")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run a toy 2D experiment."""
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = ensure_dir(args.outdir)
    samples_dir = ensure_dir(outdir / "samples")
    cfg = DriftingModelConfig(dim=2, hidden_dim=args.hidden_dim, depth=args.depth, latent_dim=args.latent_dim)
    generator = build_model(cfg).to(device)
    rf_model = TimeConditionedMLP(dim=2, hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    lucid_model = LucidRainsToyModel(dim=2, hidden_dim=args.hidden_dim, depth=args.depth).to(device)

    rf_runner = None
    if args.method == "rectified_flow" and args.rf_impl == "lucidrains" and RectifiedFlow is not None:
        try:
            rf_runner = RectifiedFlow(model=lucid_model, data_shape=(2,))
        except Exception as error:  # pragma: no cover - runtime dependency behavior
            print(f"RectifiedFlow wrapper unavailable ({error}), falling back to internal objective.")

    model = generator if args.method == "drifting" else (lucid_model if rf_runner is not None else rf_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    losses: list[float] = []
    mmds: list[float] = []
    start = time.perf_counter()
    actual_steps = args.steps

    for step in range(1, args.steps + 1):
        real = get_toy_batch(dataset=args.dataset, batch_size=args.batch_size, device=device)
        noise = torch.randn(args.batch_size, args.latent_dim, device=device)
        optimizer.zero_grad(set_to_none=True)
        if args.method == "drifting":
            loss, _ = drifting_loss(x=generator(noise), y_pos=real, tau=args.tau)
        else:
            if rf_runner is not None:
                loss = rf_runner(real)
            else:
                noise_data = torch.randn_like(real)
                loss = rectified_flow_loss(rf_model, data=real, noise=noise_data)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if args.method == "rectified_flow":
                if rf_runner is not None:
                    generated_eval = rf_runner.sample(
                        batch_size=args.batch_size, data_shape=(2,), steps=args.rf_sample_steps
                    )
                    if isinstance(generated_eval, tuple):
                        generated_eval = generated_eval[0]
                    generated_eval = generated_eval.view(args.batch_size, 2)
                else:
                    z = torch.randn(args.batch_size, 2, device=device)
                    generated_eval = z
                    dt = 1.0 / args.rf_sample_steps
                    for i in range(args.rf_sample_steps):
                        t = torch.full((z.shape[0], 1), fill_value=i * dt, device=device)
                        generated_eval = generated_eval + dt * rf_model(generated_eval, t)
            else:
                generated_eval = generator(torch.randn(args.batch_size, args.latent_dim, device=device))

            mmd = mmd_rbf(generated_eval, real).item()
            losses.append(loss.item())
            mmds.append(mmd)

            if step % args.save_every == 0 or step == 1 or step == args.steps:
                title = f"{args.method} | {args.dataset} | step={step} | loss={loss.item():.4f} | mmd={mmd:.4f}"
                _save_scatter(samples_dir / f"step_{step:06d}.png", generated_eval, real, title=title)

            if (
                step >= args.min_steps
                and len(mmds) >= args.early_stop_window
                and float(torch.tensor(mmds[-args.early_stop_window :]).mean()) <= args.early_stop_mmd
            ):
                actual_steps = step
                print(f"early-stop step={step:06d} mean_mmd_last_window={mmds[-1]:.6f}")
                break

        if step % 200 == 0 or step == args.steps:
            print(f"step={step:06d} loss={loss.item():.6f} mmd={mmds[-1]:.6f}")

    elapsed = time.perf_counter() - start
    summary = {
        "dataset": args.dataset,
        "method": args.method,
        "steps": actual_steps,
        "batch_size": args.batch_size,
        "tau": args.tau,
        "elapsed_sec": elapsed,
        "final_loss": losses[-1],
        "final_mmd": mmds[-1],
        "mean_loss_last_100": float(torch.tensor(losses[-100:]).mean()),
        "mean_mmd_last_100": float(torch.tensor(mmds[-100:]).mean()),
    }
    save_json(summary, outdir / "metrics.json")
    print(f"Saved metrics to {outdir / 'metrics.json'}")


if __name__ == "__main__":
    main()
