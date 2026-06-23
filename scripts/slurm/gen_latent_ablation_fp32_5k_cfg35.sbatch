#!/usr/bin/env bash
#SBATCH --job-name=torch-latent-fp32-5k
#SBATCH --account=e-dev-2026d02-064
#SBATCH --partition=booster
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}" || exit 1
else
  _d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$(cd "${_d}/../.." && pwd)" || exit 1
fi
source "${PWD}/scripts/slurm/common.sh"
export DRIFT_HF_HOME="${DRIFT_HF_HOME:-/e/project1/e-dev-2026d02-064/hf_cache}"
export DRIFT_CACHE_ROOT="${DRIFT_CACHE_ROOT:-/e/project1/e-dev-2026d02-064/.cache}"
export IMAGENET_CACHE_PATH="${IMAGENET_CACHE_PATH:-/e/project1/e-dev-2026d02-064/imagenet_latent_cache}"
export IMAGENET_FID_NPZ="${IMAGENET_FID_NPZ:-/e/project1/e-dev-2026d02-064/imagenet_stats/imagenet_256_fid_stats.npz}"
export IMAGENET_PR_NPZ="${IMAGENET_PR_NPZ:-/e/project1/e-dev-2026d02-064/imagenet_stats/imagenet_val_prc_arr0.npz}"
export UV_OFFLINE="${UV_OFFLINE:-1}"
drift_setup

CONFIG="${CONFIG:-configs/gen/latent_ablation_fp32_5k_cfg35.yaml}"
WORKDIR="${WORKDIR:-runs/fp32_latent_ablation_5k_cfg35_8nodes}"
MAE_NAME="$("${REPO_ROOT}/.venv/bin/python" - "${CONFIG}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
mae_path = str(cfg.get("feature", {}).get("mae_path", ""))
print(mae_path.removeprefix("hf://"))
PY
)"
MAE_CACHE="${HF_ROOT}/models/mae/pytorch/${MAE_NAME}"
if [[ ! -f "${MAE_CACHE}/metadata.json" || ! -f "${MAE_CACHE}/ema_params.pt" ]]; then
  echo "Missing offline MAE cache at ${MAE_CACHE}" >&2
  exit 1
fi

NPROC="$(drift_nproc_per_node)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Torch run: config=${CONFIG} workdir=${WORKDIR}"
echo "Topology: nodes=${SLURM_NNODES} torch processes per node=${NPROC}"

exec srun env -u CUDA_VISIBLE_DEVICES "${REPO_ROOT}/.venv/bin/python" -u -m torchrun_jsc \
  --nnodes="${SLURM_NNODES}" \
  --node-rank="${SLURM_NODEID}" \
  --nproc-per-node="${NPROC}" \
  --rdzv-id="${SLURM_JOB_ID}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  main.py --gen \
    --config "${CONFIG}" \
    --workdir "${WORKDIR}"
