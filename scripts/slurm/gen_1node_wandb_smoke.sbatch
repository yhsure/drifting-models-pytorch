#!/usr/bin/env bash
#SBATCH --job-name=drift-gen-wandb-smoke
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
exec srun env -u CUDA_VISIBLE_DEVICES "${REPO_ROOT}/.venv/bin/python" -u -m torchrun_jsc \
  --standalone \
  --nproc-per-node="${NPROC}" \
  main.py --gen --config configs/dev/smoke_gen_wandb.yaml --workdir runs/slurm_1node_gen_wandb_smoke
