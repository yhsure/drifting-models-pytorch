#!/usr/bin/env bash
#SBATCH --job-name=drift-gen-mae640
#SBATCH --partition=booster
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}" || exit 1
else
  _d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$(cd "${_d}/../.." && pwd)" || exit 1
fi
source "${PWD}/scripts/slurm/common.sh"
drift_setup

NPROC="$(drift_nproc_per_node)"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT=$((10000 + SLURM_JOB_ID % 55000))
export MASTER_ADDR MASTER_PORT
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec srun "${REPO_ROOT}/.venv/bin/torchrun" \
  --nnodes="${SLURM_NNODES}" \
  --node-rank="${SLURM_NODEID}" \
  --nproc-per-node="${NPROC}" \
  --rdzv-id="${SLURM_JOB_ID}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  main.py --gen \
    --config configs/gen/latent_ablation_mae640_short.yaml \
    --workdir runs/gen_latent_ablation_mae640_short_fastwarmup
