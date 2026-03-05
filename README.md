# Generative Modeling via Drifting (PyTorch)

PyTorch implementation of **Drifting Models** for one-step generation, following `paper/paper.pdf` and `paper/main.tex`.

## Status (to-do)

The available codebases on GitHub are still quite simple and lack most of the details from the paper. This repo has a status like this:

### Implemented now

- [x] Core drifting objective with stop-gradient target.
- [x] Mean-shift attraction/repulsion drifting field from positive/negative samples.
- [x] Softmax-based kernel normalization and multi-temperature support (e.g., `tau in {0.02, 0.05, 0.2}`).
- [x] Toy experiments on swissroll and checkerboard with sample snapshots and MMD tracking.
- [x] Pixel-space training path (CIFAR-10) with feature-space drifting loss.
- [x] Rectified Flow baseline runs with fixed-compute comparison scripts.
- [x] Systematic evaluation pipeline (FID/IS and visual benchmark report).
- [x] Paper-style multi-scale feature extraction for drifting loss (per-scale/per-location descriptors).
- [x] Full feature and drift normalization from Appendix A (shared scale estimation across locations).
- [x] Basic quality automation (`ruff`, `ty`, `pytest`, `pre-commit`, CI workflow).

### Future paper details to implement
- [ ] Strong custom feature encoders used in paper (ResNet-MAE / ConvNeXt-V2) and related ablations.
- [ ] CFG training/inference path for drifting models (paper Appendix CFG details).
- [ ] ImageNet-scale experiments and paper-like reporting metrics/ablation tables.

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
uv run examples/train_toy.py --dataset swissroll --method drifting --taus 0.02,0.05,0.2 --batch-size 1024 --dataset-noise 0.0 --hidden-dim 384 --depth 8 --steps 5000 --save-every 1000 --use-output-bound --output-scale 1.0 --outdir results/publication_toy_fixstyle_v3/swiss_drifting
```

Checkerboard:

```bash
uv run examples/train_toy.py --dataset checkerboard --method drifting --taus 0.02,0.05,0.2 --batch-size 1024 --dataset-noise 0.0 --hidden-dim 384 --depth 8 --steps 5000 --save-every 1000 --use-output-bound --output-scale 1.0 --outdir results/publication_toy_fixstyle_v3/checker_drifting
```

Early-stop controls (optional):

```bash
--min-steps 2000 --early-stop-window 200 --early-stop-mmd 0.0020
```

### CIFAR-10 (lucidrains U-Net)

Single run (feature drifting):

```bash
uv run examples/train_cifar10.py --method drifting --drifting-mode feature --unet-dim 64 --unet-dim-mults 1,2,4 --outdir results/cifar10/drifting_feature
```

Single run (pixel drifting):

```bash
uv run examples/train_cifar10.py --method drifting --drifting-mode pixel --unet-dim 64 --unet-dim-mults 1,2,4 --outdir results/cifar10/drifting_pixel
```

Single run (rectified flow):

```bash
uv run examples/train_cifar10.py --method rectified_flow --unet-dim 64 --unet-dim-mults 1,2,4 --outdir results/cifar10/rf
```

### CIFAR Visual Benchmark (3-way)

Quick smoke run:

```bash
uv run examples/benchmark_cifar_visual.py --mode quick
```

Full benchmark run:

```bash
uv run examples/benchmark_cifar_visual.py --mode full --export-vector
```

Main output figure:

- `results/figures/cifar_training_comparison.png`

Publication comparison figure (existing runs):

- `results/figures/cifar_training_comparison_flow_r50_r18_pixel.png`
- generated via `results/plot_cifar_existing_comparison.py`
- styling note: run curves use a plasma colormap for visual consistency

Toy-data comparison figure (swissroll + checkerboard):

```bash
uv run results/plot_toy_existing_comparison.py \
  --swiss-drifting-dir results/publication_toy_fixstyle_v3/swiss_drifting \
  --checker-drifting-dir results/publication_toy_fixstyle_v3/checker_drifting \
  --output results/figures/toy_training_comparison_swiss_checker.png \
  --export-vector
```

Toy figure styling note:

- plasma-inspired palette is used for swissroll/checkerboard training curves.

## Figures

### CIFAR-10 (Drifting Feature/Pixel + RF)

![CIFAR comparison](results/figures/cifar_training_comparison_flow_r50_r18_pixel.png)

### Toy final samples (matched architecture)

Swissroll (`hidden_dim=384`, `depth=8`, `steps=5000`, multi-tau):

![Swissroll final sample](results/publication_toy_fixstyle_v3/swiss_drifting/samples/step_005000.png)


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
- `examples/benchmark_cifar_visual.py`: CIFAR visual benchmark orchestrator.
- `results/plot_*.py`: scripts for publication-ready comparison figures.
- `.github/workflows/ci.yml`: push / PR quality gates.
- `tests/`: test suite.
- `paper/`: copied paper assets.
