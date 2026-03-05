"""Build a merged CIFAR comparison figure from existing run artifacts."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, ScalarFormatter

from drifting_models_pytorch.utils import ensure_dir


@dataclass(slots=True)
class EvalRecord:
    """Per-checkpoint training/eval row loaded from CSV."""

    global_step: int
    images_seen: int
    elapsed_seconds: float
    train_loss: float
    fid: float
    inception_score: float
    inception_score_std: float


@dataclass(slots=True)
class RunSpec:
    """One run included in the merged figure."""

    label: str
    run_dir: Path


def _parse_runs(raw: str) -> list[RunSpec]:
    """Parse comma-separated run specs in label|path format."""
    runs: list[RunSpec] = []
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        label, path = item.split("|", maxsplit=1)
        run_dir = Path(path)
        if not run_dir.exists():
            raise ValueError(f"Run path does not exist: {run_dir}")
        runs.append(RunSpec(label=label, run_dir=run_dir))
    if not runs:
        raise ValueError("At least one run must be provided")
    return runs


def _read_records(path: Path) -> list[EvalRecord]:
    """Load metrics CSV rows."""
    records: list[EvalRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                EvalRecord(
                    global_step=int(row["global_step"]),
                    images_seen=int(row["images_seen"]),
                    elapsed_seconds=float(row["elapsed_seconds"]),
                    train_loss=float(row["train_loss"]),
                    fid=float(row["fid"]),
                    inception_score=float(row["inception_score"]),
                    inception_score_std=float(row["inception_score_std"]),
                )
            )
    if not records:
        raise ValueError(f"No metric rows found in {path}")
    return records


def _build_checkpoints(records_per_run: list[list[EvalRecord]]) -> list[int]:
    """Find common steps across all runs and pick early/mid/late checkpoints."""
    common = sorted(set.intersection(*[{record.global_step for record in records} for records in records_per_run]))
    if not common:
        raise ValueError("No common checkpoints across runs")
    if len(common) <= 3:
        return common
    return [common[0], common[len(common) // 2], common[-1]]


def _sample_path(run_dir: Path, step: int) -> Path:
    """Resolve sample grid path for a given step."""
    sample_path = run_dir / "samples" / f"step_{step:06d}.png"
    if not sample_path.exists():
        raise ValueError(f"Missing sample image: {sample_path}")
    return sample_path


def _crop_sample_grid_rows(image: np.ndarray, keep_rows: int = 5, total_rows: int = 8) -> np.ndarray:
    """Keep only top grid rows using cell-aligned boundaries."""
    if image.ndim < 2:
        return image
    if keep_rows >= total_rows:
        return image
    height = image.shape[0]
    padding = 2
    numerator = height - padding * (total_rows + 1)
    if numerator > 0 and numerator % total_rows == 0:
        cell = numerator // total_rows
        keep_height = padding + keep_rows * (cell + padding)
    else:
        keep_height = int(round(height * (keep_rows / total_rows)))
    keep_height = max(1, min(keep_height, height))
    return image[:keep_height, ...]


def _display_label(label: str) -> str:
    """Format labels for plot readability and requested two-line ResNet titles."""
    if "resnet50" in label.lower():
        return "Drifting Feat.\n(ResNet50)"
    if "resnet18" in label.lower():
        return "Drifting Feat.\n(ResNet18)"
    return label.replace("Drifting Feature", "Drifting Feat.")


def _plasma_colors(n: int) -> list[tuple[float, float, float, float]]:
    """Get n visually separated colors from the plasma colormap."""
    if n <= 1:
        return [plt.cm.plasma(0.7)]
    points = np.linspace(0.12, 0.92, n)
    return [plt.cm.plasma(float(point)) for point in points]


def _lighten_color(color: tuple[float, float, float, float], amount: float = 0.82) -> tuple[float, float, float]:
    """Blend a color with white to create a muted background color."""
    r, g, b = color[:3]
    return (
        r * (1.0 - amount) + amount,
        g * (1.0 - amount) + amount,
        b * (1.0 - amount) + amount,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        type=str,
        required=True,
        help=(
            "Comma-separated entries in label|run_dir format. "
            "Each run_dir must contain metrics_history.csv and samples/step_*.png."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/figures/cifar_training_comparison_4way.png",
    )
    parser.add_argument("--export-vector", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Generate the merged plot from existing run outputs."""
    args = parse_args()
    runs = _parse_runs(args.runs)
    records_per_run = [_read_records(run.run_dir / "metrics_history.csv") for run in runs]
    checkpoints = _build_checkpoints(records_per_run)
    colors = _plasma_colors(len(runs))

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 30,
            "axes.labelsize": 27,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 24,
        }
    )

    figure = plt.figure(figsize=(23, 13 + 4.2 * len(runs)), constrained_layout=False)
    grid = figure.add_gridspec(
        nrows=4 + len(runs),
        ncols=len(checkpoints),
        height_ratios=[3.4, 3.4, 3.4, 1.6] + [3.8] * len(runs),
    )
    ax_fid = figure.add_subplot(grid[0, :])
    ax_is = figure.add_subplot(grid[1, :], sharex=ax_fid)
    ax_loss = figure.add_subplot(grid[2, :], sharex=ax_fid)

    for run, records, color in zip(runs, records_per_run, colors, strict=True):
        x = [record.images_seen for record in records]
        label = _display_label(run.label)
        ax_fid.plot(x, [record.fid for record in records], label=label, color=color, linewidth=4.0)
        ax_is.plot(x, [record.inception_score for record in records], label=label, color=color, linewidth=4.0)
        ax_loss.plot(x, [record.train_loss for record in records], label=label, color=color, linewidth=4.0)

    x_formatter = FuncFormatter(lambda x, pos: f"{x / 1_000_000:.1f}M")
    for axis, ylabel in ((ax_fid, "FID (log)"), (ax_is, "Inception Score"), (ax_loss, "Train Loss (log)")):
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.xaxis.set_major_formatter(x_formatter)
        axis.xaxis.offsetText.set_visible(False)
    ax_fid.set_yscale("log")
    all_fids = [r.fid for records in records_per_run for r in records if r.fid > 0]
    if all_fids:
        fid_lo = max(1.0, min(all_fids) * 0.9)
        fid_hi = max(all_fids) * 1.1
        exp_lo = math.floor(math.log10(fid_lo))
        exp_hi = math.ceil(math.log10(fid_hi))
        subs = (1.0, 2.0, 5.0)
        fid_ticks = []
        for exp in range(exp_lo, exp_hi + 1):
            for s in subs:
                t = s * (10**exp)
                if fid_lo <= t <= fid_hi:
                    fid_ticks.append(t)
        if fid_ticks:
            ax_fid.set_yticks(sorted(set(fid_ticks)))
    ax_fid.yaxis.set_major_formatter(ScalarFormatter())
    plt.setp(ax_fid.get_xticklabels(), visible=False)
    plt.setp(ax_is.get_xticklabels(), visible=False)
    ax_fid.set_xlabel("")
    ax_is.set_xlabel("")
    ax_loss.set_xlabel("images_seen")
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("Train Loss (log)")
    legend_handles, legend_labels = ax_fid.get_legend_handles_labels()
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(runs),
        framealpha=0.95,
    )
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.04, top=0.92, wspace=0.10, hspace=0.08)

    sample_rows: list[tuple[list[Axes], tuple[float, float, float]]] = []
    for row, (run, _, color) in enumerate(zip(runs, records_per_run, colors, strict=True)):
        row_background = _lighten_color(color, amount=0.86)
        row_axes: list[Axes] = []
        for col, checkpoint in enumerate(checkpoints):
            axis = figure.add_subplot(grid[4 + row, col])
            image = _crop_sample_grid_rows(mpimg.imread(_sample_path(run.run_dir, checkpoint)))
            axis.imshow(image, extent=(0.005, 0.995, 0.005, 0.995), interpolation="nearest")
            axis.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xticks([])
            axis.set_yticks([])
            if col == 0:
                axis.set_ylabel(_display_label(run.label), fontsize=26)
            if row == 0:
                axis.set_title(f"step {checkpoint}", fontsize=24)
            row_axes.append(axis)
        sample_rows.append((row_axes, row_background))

    # Draw one continuous muted background per sample row.
    figure.canvas.draw()
    bounds = []
    for row_axes, row_background in sample_rows:
        x0 = min(axis.get_position().x0 for axis in row_axes)
        x1 = max(axis.get_position().x1 for axis in row_axes)
        y0 = min(axis.get_position().y0 for axis in row_axes)
        y1 = max(axis.get_position().y1 for axis in row_axes)
        bounds.append((x0, x1, y0, y1, row_background))
    gaps = []
    for row_idx in range(len(bounds) - 1):
        top_row_bottom = bounds[row_idx][2]
        next_row_top = bounds[row_idx + 1][3]
        gaps.append(max(0.0, top_row_bottom - next_row_top))
    if not gaps:
        gaps = [0.0]
    keep_fraction = 0.0  # remove retained gap between rows
    for row_idx, (x0, x1, y0, y1, row_background) in enumerate(bounds):
        gap_above = gaps[row_idx - 1] if row_idx > 0 else gaps[0]
        gap_below = gaps[row_idx] if row_idx < len(bounds) - 1 else gaps[-1]
        top_edge = y1 + 0.5 * (1.0 - keep_fraction) * gap_above
        bottom_edge = y0 - 0.5 * (1.0 - keep_fraction) * gap_below
        figure.patches.append(
            Rectangle(
                (x0, bottom_edge),
                x1 - x0,
                top_edge - bottom_edge,
                transform=figure.transFigure,
                facecolor=row_background,
                edgecolor="none",
                zorder=-10,
            )
        )

    figure.suptitle("CIFAR-10 Training Comparison", fontsize=38, y=0.985)
    output = Path(args.output)
    ensure_dir(output.parent)
    figure.savefig(output, dpi=240)
    if args.export_vector:
        figure.savefig(output.with_suffix(".svg"), dpi=240)
        figure.savefig(output.with_suffix(".pdf"), dpi=240)
    plt.close(figure)
    print(f"Merged figure saved to {output}")


if __name__ == "__main__":
    main()
