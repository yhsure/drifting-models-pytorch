#!/usr/bin/env bash
#SBATCH --job-name=drift-gen-2node
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

exec srun env -u CUDA_VISIBLE_DEVICES "${REPO_ROOT}/.venv/bin/python" -u -m torchrun_jsc \
  --nnodes="${SLURM_NNODES}" \
  --node-rank="${SLURM_NODEID}" \
  --nproc-per-node="${NPROC}" \
  --rdzv-id="${SLURM_JOB_ID}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  main.py --gen --config configs/dev/smoke_gen.yaml --workdir runs/slurm_2node_gen_smoke
