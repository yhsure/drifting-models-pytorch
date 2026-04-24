#!/usr/bin/env bash
#SBATCH --job-name=drift-mae-1node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --time=00:20:00
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
exec srun --ntasks=1 "${REPO_ROOT}/.venv/bin/torchrun" \
  --standalone \
  --nproc-per-node="${NPROC}" \
  main.py --config configs/dev/smoke_mae.yaml --workdir runs/slurm_1node_mae_smoke
