#!/usr/bin/env bash
#SBATCH --job-name=gen-mae640-mog-loss
#SBATCH --partition=booster
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --array=0-3
#SBATCH --output=logs/slurm/%x-%A_%a.log
#SBATCH --error=logs/slurm/%x-%A_%a.log

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}" || exit 1
else
  _d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$(cd "${_d}/../.." && pwd)" || exit 1
fi
source "${PWD}/scripts/slurm/common.sh"
drift_setup

VARIANTS=(
  drift_mog_jsd
  drift_mog_nll
  mog_only_nll
  mog_only_jsd
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[${IDX}]}"
CONFIG="configs/gen/latent_ablation_mae640_short_${VARIANT}.yaml"
WORKDIR="runs/gen_latent_ablation_mae640_short_${VARIANT}"

NPROC="$(drift_nproc_per_node)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Running variant: ${VARIANT}"
echo "Config: ${CONFIG}"
echo "Workdir: ${WORKDIR}"

exec srun env -u CUDA_VISIBLE_DEVICES "${REPO_ROOT}/.venv/bin/python" -u -m torchrun_jsc \
  --nnodes="${SLURM_NNODES}" \
  --node-rank="${SLURM_NODEID}" \
  --nproc-per-node="${NPROC}" \
  --rdzv-id="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  main.py --gen \
    --config "${CONFIG}" \
    --workdir "${WORKDIR}"
