#!/usr/bin/env bash
# Run the explicit-likelihood manifest evals, summary, and completion audit on
# one GPU after a training checkpoint has been written.
#SBATCH --job-name=drift-like-eval
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=250G
#SBATCH --time=12:00:00
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

RUN_DIR="${RUN_DIR:-}"
RUN_ROOT="${RUN_ROOT:-runs/imagenet_likelihood_sparse_soft_20k_1node}"
INCLUDE="${INCLUDE:-primary,secondary,variants,panel}"
FORCE_EVAL="${FORCE_EVAL:-0}"

args=()
if [[ -n "${RUN_DIR}" ]]; then
  args+=("${RUN_DIR}")
else
  args+=("--run-root" "${RUN_ROOT}")
fi
args+=("--include" "${INCLUDE}")
if [[ "${FORCE_EVAL}" = "1" ]]; then
  args+=("--force-eval")
fi

echo "Running explicit-likelihood postrun: ${args[*]}"
"${REPO_ROOT}/.venv/bin/python" scripts/run_imagenet_likelihood_postrun.py "${args[@]}"
