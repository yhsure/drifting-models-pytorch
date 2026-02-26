"""Run fair-compute comparisons between Drifting Models and Rectified Flow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from drifting_models_pytorch.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="toy", choices=("toy", "cifar10"))
    parser.add_argument("--dataset", type=str, default="swissroll", choices=("swissroll", "checkerboard"))
    parser.add_argument("--steps", type=int, default=8_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="results/comparisons")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    """Run both methods under an identical step budget."""
    args = parse_args()
    outdir = ensure_dir(Path(args.outdir) / f"{args.domain}_{args.dataset}_steps_{args.steps}")
    if args.domain == "toy":
        for method in ("drifting", "rectified_flow"):
            method_out = outdir / method
            _run(
                [
                    sys.executable,
                    "examples/train_toy.py",
                    "--dataset",
                    args.dataset,
                    "--method",
                    method,
                    "--steps",
                    str(args.steps),
                    "--batch-size",
                    str(args.batch_size),
                    "--seed",
                    str(args.seed),
                    "--outdir",
                    str(method_out),
                ]
            )
    else:
        for method in ("drifting", "rectified_flow"):
            method_out = outdir / method
            _run(
                [
                    sys.executable,
                    "examples/train_cifar10.py",
                    "--method",
                    method,
                    "--steps",
                    str(args.steps),
                    "--batch-size",
                    str(args.batch_size),
                    "--seed",
                    str(args.seed),
                    "--outdir",
                    str(method_out),
                ]
            )
    save_json(
        {
            "domain": args.domain,
            "dataset": args.dataset,
            "fixed_compute": {"steps": args.steps, "batch_size": args.batch_size},
            "methods": ["drifting", "rectified_flow"],
            "output_dir": str(outdir),
        },
        outdir / "comparison_config.json",
    )
    print(f"Comparison outputs saved to {outdir}")


if __name__ == "__main__":
    main()
