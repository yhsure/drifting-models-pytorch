# drifting-models-pytorch

PyTorch implementation of **Drifting Models** for one-step generation, following
`paper/paper.pdf` and `paper/main.tex`.

Implemented components:

- Drifting loss with stop-gradient target and mean-shift drifting field.
- Corrected canonical toy samplers for `swissroll` and `checkerboard`.
- Convergence-aware early stopping for toy and CIFAR experiments.
- CIFAR-10 training with lucidrains-style U-Net backbones.
- Fixed-compute comparison runner for Drifting Models vs Rectified Flow.
- CI workflow for push / pull request quality checks.

## Setup

```bash
uv sync --group dev
uv run pre-commit install
```

## Commands

```bash
uv run invoke --list
uv run invoke format
uv run invoke lint
uv run invoke typecheck
uv run invoke test
```

## Run Experiments

### Toy Data (GPU recommended)

Swissroll:

```bash
uv run examples/train_toy.py --dataset swissroll --method drifting --tau 0.08 --outdir results/toy/swiss_drifting
uv run examples/train_toy.py --dataset swissroll --method rectified_flow --rf-impl internal --outdir results/toy/swiss_rf
```

Checkerboard:

```bash
uv run examples/train_toy.py --dataset checkerboard --method drifting --tau 0.20 --outdir results/toy/checker_drifting
uv run examples/train_toy.py --dataset checkerboard --method rectified_flow --rf-impl internal --outdir results/toy/checker_rf
```

Early-stop controls (optional):

```bash
--min-steps 2000 --early-stop-window 200 --early-stop-mmd 0.0020
```

### CIFAR-10 (lucidrains U-Net)

Drifting:

```bash
uv run examples/train_cifar10.py --method drifting --unet-dim 64 --unet-dim-mults 1,2,4 --outdir results/cifar10/drifting
```

Rectified Flow (lucidrains):

```bash
uv run examples/train_cifar10.py --method rectified_flow --unet-dim 64 --unet-dim-mults 1,2,4 --outdir results/cifar10/rf
```

### Fixed Compute Comparison

```bash
uv run examples/compare_methods.py --domain toy --dataset swissroll --steps 8000 --batch-size 256
uv run examples/compare_methods.py --domain cifar10 --steps 20000 --batch-size 256
```

## Quality Automation

GitHub Actions workflow: `.github/workflows/ci.yml`

- `uv sync --group dev`
- `ruff format --check`
- `ruff check`
- `ty check`
- `pytest tests/`
- `pre-commit run --all-files`

## Structure

- `drifting_models_pytorch/`: package source.
- `examples/train_toy.py`: toy experiments.
- `examples/train_cifar10.py`: CIFAR-10 training.
- `examples/compare_methods.py`: fair-compute comparison launcher.
- `.github/workflows/ci.yml`: push / PR quality gates.
- `tests/`: test suite.
- `paper/`: copied paper assets.
