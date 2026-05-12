#!/usr/bin/env bash
#SBATCH --job-name=drift-dec-like-train
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

PRIOR="${PRIOR:?Set PRIOR=/path/to/decoder_likelihood_prior.pt}"
STEPS_PER_CLASS="${STEPS_PER_CLASS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MC_SAMPLES="${MC_SAMPLES:-1}"
SIGMA="${SIGMA:-0.875}"
LR="${LR:-0.001}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.05}"
SCALE_WEIGHT="${SCALE_WEIGHT:-0.001}"
TRAIN_SCALE_MULT="${TRAIN_SCALE_MULT:-0.25}"
RUN_NAME="${RUN_NAME:-${SLURM_JOB_ID}_trained_s${STEPS_PER_CLASS}}"
OUT_DIR="${OUT_DIR:-runs/imagenet_decoder_likelihood_prior/${RUN_NAME}}"
MAX_CLASSES="${MAX_CLASSES:-0}"
MAX_PER_CLASS="${MAX_PER_CLASS:-0}"
COMPILE_LEVEL="${COMPILE_LEVEL:-0}"

echo "Training decoder-style latent mixture prior:"
echo "  prior=${PRIOR}"
echo "  out=${OUT_DIR}"
echo "  steps_per_class=${STEPS_PER_CLASS}, batch=${BATCH_SIZE}, mc=${MC_SAMPLES}, world=${NPROC}"

exec srun env -u CUDA_VISIBLE_DEVICES "${REPO_ROOT}/.venv/bin/python" -u -m torchrun_jsc \
  --nnodes="${SLURM_NNODES}" \
  --node-rank="${SLURM_NODEID}" \
  --nproc-per-node="${NPROC}" \
  --rdzv-id="${SLURM_JOB_ID}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  scripts/run_imagenet_decoder_likelihood.py train \
    --prior "${PRIOR}" \
    --out-dir "${OUT_DIR}" \
    --steps-per-class "${STEPS_PER_CLASS}" \
    --batch-size "${BATCH_SIZE}" \
    --mc-samples "${MC_SAMPLES}" \
    --sigma "${SIGMA}" \
    --lr "${LR}" \
    --anchor-weight "${ANCHOR_WEIGHT}" \
    --scale-weight "${SCALE_WEIGHT}" \
    --train-scale-mult "${TRAIN_SCALE_MULT}" \
    --max-classes "${MAX_CLASSES}" \
    --max-per-class "${MAX_PER_CLASS}" \
    --compile-level "${COMPILE_LEVEL}"
