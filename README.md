# Generative Modeling via Drifting - PyTorch Port

PyTorch port of the JAX release in `lambertae-drifting`, preserving the same file layout:

- `configs/`
- `dataset/`
- `models/`
- `utils/`
- `main.py`, `train.py`, `train_mae.py`, `inference.py`

## Environment (uv + Jupiter)

Use `uv` for dependency and venv management.

On Jupiter:

```bash
module load Stages/2026
module load GCCcore/14.3.0 CUDA/13 Python/3.13.5 uv/0.8.17
```

Use the `uv` venv for `torch` / `torchvision` (do not load `PyTorch/*`; CUDA 12, conflicts with this stack).

Set cache paths to project storage (avoids home/scratch permission and quota issues):

```bash
export UV_CACHE_DIR=/e/project1/e-dev-2026d02-064/_abj/.uv-cache
export TORCHINDUCTOR_CACHE_DIR=$PWD/.torch-cache/torchinductor
export TORCH_HOME=$PWD/.torch-cache/torch
mkdir -p "$UV_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_HOME"
```

CLI `--workdir` is rewritten to a stamped folder: `runs/foo` → `runs/MMDD_HHMM_foo`; default `runs` → `runs/MMDD_HHMM`. Point `--init-from` / resumes at that path.

After `uv sync`, prefer `.venv/bin/python …` for repeated local runs; `uv run` re-checks the env each time and can be slow on shared storage.

Install and run:

```bash
uv sync --group dev
uv run python -V
uv run python main.py --gen --config configs/gen/latent_ablation.yaml --workdir runs/gen_latent_ablation
```

Quick login-node smoke runs on cached latents:

```bash
uv run python main.py --config configs/dev/login_smoke_mae.yaml --workdir runs/login_smoke_mae
uv run python main.py --gen --config configs/dev/login_smoke_gen.yaml --workdir runs/login_smoke_gen
```

## Paths

Runtime paths are configured in `utils/env.py` and can be overridden by environment variables:

- `IMAGENET_PATH`
- `IMAGENET_CACHE_PATH`
- `IMAGENET_FID_NPZ`
- `IMAGENET_PR_NPZ`
- `HF_ROOT`
- `HF_REPO_ID`

Defaults are set to the shared workspace layout used on Jupiter (`imagenet/`, `imagenet_latent_cache/`, `imagenet_stats/`).

## Build latent cache

```bash
uv run python -m dataset.latent \
  --data-path "$IMAGENET_PATH" \
  --target-path "$IMAGENET_CACHE_PATH" \
  --local-batch-size 128 \
  --num-workers 8 \
  --pin-memory
```

## Train

Generator training:

```bash
uv run python main.py --gen --config configs/gen/latent_sota_B.yaml --workdir runs/gen_latent_sota_B
```

MAE training:

```bash
uv run python main.py --config configs/mae/latent_640.yaml --workdir runs/mae_latent_640
```

## FID inference

```bash
uv run python inference.py \
  --init-from runs/gen_latent_sota_B \
  --cfg-scale 1.0 \
  --num-samples 50000 \
  --eval-batch-size 512 \
  --json-out runs/fid/result.json
```

## CI

Basic GitHub Actions CI is in `.github/workflows/ci.yml`:

- `uv sync --group dev`
- `pytest -q`

## Slurm

Shared setup: `scripts/slurm/common.sh`. Submit from the repository root; see `scripts/slurm/README.md` for allocation flags (`--exclusive`, 288 CPUs per node).

```bash
sbatch scripts/slurm/mae_login_smoke.sbatch
sbatch scripts/slurm/gen_login_smoke.sbatch
sbatch scripts/slurm/gen_2node_smoke.sbatch
```
