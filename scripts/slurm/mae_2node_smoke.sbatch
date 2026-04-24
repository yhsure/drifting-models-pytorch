#!/usr/bin/env bash
#SBATCH --job-name=drift-mae-2node
#SBATCH --nodes=2
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

exec srun "${REPO_ROOT}/.venv/bin/torchrun" \
  --nnodes="${SLURM_NNODES}" \
  --node-rank="${SLURM_NODEID}" \
  --nproc-per-node="${NPROC}" \
  --rdzv-id="${SLURM_JOB_ID}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  main.py --config configs/dev/smoke_mae.yaml --workdir runs/slurm_2node_mae_smoke
