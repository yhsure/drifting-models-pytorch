#!/usr/bin/env bash
#SBATCH --job-name=drift-dec-like-fit
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --time=03:00:00
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
drift_setup

NPROC="$(drift_nproc_per_node)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

K="${K:-8192}"
DESC="${DESC:-mae}"
ITERS="${ITERS:-8}"
CENTER_MODE="${CENTER_MODE:-mean}"
DESC_BATCH="${DESC_BATCH:-128}"
COMPILE_LEVEL="${COMPILE_LEVEL:-1}"
RUN_NAME="${RUN_NAME:-${SLURM_JOB_ID}_k${K}_${DESC}_iters${ITERS}}"
OUT_DIR="${OUT_DIR:-runs/imagenet_decoder_likelihood_prior/${RUN_NAME}}"
FEATURE_STATS_FROM="${FEATURE_STATS_FROM:-runs/imagenet_likelihood_sparse_soft_5k/427705_k8192_top8_tau0.45/likelihood_prior.pt}"
MAX_CLASSES="${MAX_CLASSES:-0}"
MAX_PER_CLASS="${MAX_PER_CLASS:-0}"
USE_FLIPS="${USE_FLIPS:-0}"

echo "Fitting decoder-style latent mixture prior:"
echo "  out=${OUT_DIR}"
echo "  K=${K}, descriptor=${DESC}, iters=${ITERS}, center_mode=${CENTER_MODE}, world=${NPROC}"
echo "  feature_stats_from=${FEATURE_STATS_FROM}"

ARGS=(
  scripts/run_imagenet_decoder_likelihood.py fit
  --out-dir "${OUT_DIR}"
  --component-count "${K}"
  --descriptor-mode "${DESC}"
  --kmeans-iters "${ITERS}"
  --center-mode "${CENTER_MODE}"
  --descriptor-batch-size "${DESC_BATCH}"
  --compile-level "${COMPILE_LEVEL}"
  --feature-stats-from "${FEATURE_STATS_FROM}"
  --max-classes "${MAX_CLASSES}"
  --max-per-class "${MAX_PER_CLASS}"
)

if [[ "${USE_FLIPS}" == "1" ]]; then
  ARGS+=(--use-flips)
fi

exec srun env -u CUDA_VISIBLE_DEVICES "${REPO_ROOT}/.venv/bin/python" -u -m torchrun_jsc \
  --nnodes="${SLURM_NNODES}" \
  --node-rank="${SLURM_NODEID}" \
  --nproc-per-node="${NPROC}" \
  --rdzv-id="${SLURM_JOB_ID}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  "${ARGS[@]}"
