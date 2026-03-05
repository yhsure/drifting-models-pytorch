"""Build a merged toy-data figure from multi-tau drifting runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import ScalarFormatter

from drifting_models_pytorch.utils import ensure_dir


@dataclass(slots=True)
class HistoryRecord:
    """Per-step training row loaded from toy history CSV."""

    global_step: int
    elapsed_seconds: float
    train_loss: float
    mmd: float


@dataclass(slots=True)
class RunSpec:
    """One toy run used in the merged figure."""

    label: str
    run_dir: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--swiss-drifting-dir", type=str, required=True)
    parser.add_argument("--checker-drifting-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/figures/toy_training_comparison_swiss_checker.png")
    parser.add_argument("--export-vector", action="store_true")
    return parser.parse_args()


def _read_records(path: Path) -> list[HistoryRecord]:
    records: list[HistoryRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                HistoryRecord(
                    global_step=int(row["global_step"]),
                    elapsed_seconds=float(row["elapsed_seconds"]),
                    train_loss=float(row["train_loss"]),
                    mmd=float(row["mmd"]),
                )
            )
    if not records:
        raise ValueError(f"No history rows found in {path}")
    return records


def _sample_path(run_dir: Path, step: int) -> Path:
    sample = run_dir / "samples" / f"step_{step:06d}.png"
    if not sample.exists():
        raise ValueError(f"Missing sample image: {sample}")
    return sample


def _available_sample_steps(run_dir: Path) -> list[int]:
    steps: list[int] = []
    for path in sorted((run_dir / "samples").glob("step_*.png")):
        try:
            steps.append(int(path.stem.split("_", maxsplit=1)[1]))
        except (IndexError, ValueError):
            continue
    if not steps:
        raise ValueError(f"No sample checkpoints found in {run_dir / 'samples'}")
    return steps


def _pick_checkpoints(steps: list[int]) -> list[int]:
    if not steps:
        raise ValueError("No checkpoints found for toy run")
    if len(steps) <= 3:
        return steps
    return [steps[0], steps[len(steps) // 2], steps[-1]]


def _plot_curve(axis, records: list[HistoryRecord], key: str, ylabel: str) -> None:
    x = [row.global_step for row in records]
    y = [getattr(row, key) for row in records]
    axis.plot(x, y, linewidth=2.7, color="#6A1B9A")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.xaxis.set_major_formatter(ScalarFormatter())


def _plasma_pair() -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Pick two distinct plasma-inspired colors for swiss/checker plots."""
    cmap = plt.cm.plasma
    return cmap(0.25), cmap(0.82)


def main() -> None:
    args = parse_args()
    swiss = RunSpec(label="Swissroll (drifting, multi-tau)", run_dir=Path(args.swiss_drifting_dir))
    checker = RunSpec(label="Checkerboard (drifting, multi-tau)", run_dir=Path(args.checker_drifting_dir))
    for run in [swiss, checker]:
        if not run.run_dir.exists():
            raise ValueError(f"Run path does not exist: {run.run_dir}")

    swiss_records = _read_records(swiss.run_dir / "metrics_history.csv")
    checker_records = _read_records(checker.run_dir / "metrics_history.csv")
    swiss_ckpts = _pick_checkpoints(_available_sample_steps(swiss.run_dir))
    checker_ckpts = _pick_checkpoints(_available_sample_steps(checker.run_dir))

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )

    fig = plt.figure(figsize=(20, 10), constrained_layout=False)
    grid = fig.add_gridspec(nrows=3, ncols=6, height_ratios=[1.2, 1.2, 1.6])

    swiss_mmd_ax = fig.add_subplot(grid[0, :3])
    checker_mmd_ax = fig.add_subplot(grid[0, 3:])
    swiss_loss_ax = fig.add_subplot(grid[1, :3], sharex=swiss_mmd_ax)
    checker_loss_ax = fig.add_subplot(grid[1, 3:], sharex=checker_mmd_ax)

    swiss_color, checker_color = _plasma_pair()
    _plot_curve(swiss_mmd_ax, swiss_records, key="mmd", ylabel="MMD")
    _plot_curve(checker_mmd_ax, checker_records, key="mmd", ylabel="MMD")
    _plot_curve(swiss_loss_ax, swiss_records, key="train_loss", ylabel="Train Loss")
    _plot_curve(checker_loss_ax, checker_records, key="train_loss", ylabel="Train Loss")
    for axis in (swiss_mmd_ax, swiss_loss_ax):
        axis.lines[-1].set_color(swiss_color)
    for axis in (checker_mmd_ax, checker_loss_ax):
        axis.lines[-1].set_color(checker_color)

    swiss_mmd_ax.set_title("Swissroll")
    checker_mmd_ax.set_title("Checkerboard")
    swiss_mmd_ax.set_title("Swissroll", color=to_rgba(swiss_color))
    checker_mmd_ax.set_title("Checkerboard", color=to_rgba(checker_color))
    swiss_loss_ax.set_xlabel("global_step")
    checker_loss_ax.set_xlabel("global_step")

    for col, checkpoint in enumerate(swiss_ckpts):
        axis = fig.add_subplot(grid[2, col])
        axis.imshow(plt.imread(_sample_path(swiss.run_dir, checkpoint)))
        axis.set_xticks([])
        axis.set_yticks([])
        if col == 0:
            axis.set_ylabel("Swissroll samples")
        axis.set_title(f"step {checkpoint}")

    for col, checkpoint in enumerate(checker_ckpts):
        axis = fig.add_subplot(grid[2, 3 + col])
        axis.imshow(plt.imread(_sample_path(checker.run_dir, checkpoint)))
        axis.set_xticks([])
        axis.set_yticks([])
        if col == 0:
            axis.set_ylabel("Checkerboard samples")
        axis.set_title(f"step {checkpoint}")

    fig.suptitle("Toy Training (Multi-tau Drifting): Swissroll and Checkerboard", fontsize=20, y=0.98)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.05, wspace=0.22, hspace=0.30)

    output = Path(args.output)
    ensure_dir(output.parent)
    fig.savefig(output, dpi=220)
    if args.export_vector:
        fig.savefig(output.with_suffix(".svg"), dpi=220)
        fig.savefig(output.with_suffix(".pdf"), dpi=220)
    plt.close(fig)
    print(f"Merged figure saved to {output}")


if __name__ == "__main__":
    main()
