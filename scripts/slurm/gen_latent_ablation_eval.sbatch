#!/usr/bin/env bash
#SBATCH --job-name=drift-eval-latent-ablation
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --time=02:00:00
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

if [[ -z "${TRAIN_RUN:-}" ]]; then
  TRAIN_RUN=""
  while IFS= read -r candidate; do
    if [[ -f "${candidate}/checkpoints/step_000010000.pt" || -f "${candidate}/params_ema/ema_params.pt" ]]; then
      TRAIN_RUN="${candidate}"
      break
    fi
  done < <(ls -dt runs/*_gen_latent_ablation 2>/dev/null || true)
fi
if [[ -z "${TRAIN_RUN}" ]]; then
  echo "No completed gen_latent_ablation run found. Set TRAIN_RUN=/path/to/run." >&2
  exit 1
fi
if [[ ! -f "${TRAIN_RUN}/checkpoints/step_000010000.pt" && ! -f "${TRAIN_RUN}/params_ema/ema_params.pt" ]]; then
  echo "Step-10000 checkpoint or EMA artifact missing in ${TRAIN_RUN}" >&2
  exit 1
fi

echo "Evaluating run: ${TRAIN_RUN}"
exec srun env -u CUDA_VISIBLE_DEVICES "${REPO_ROOT}/.venv/bin/python" -u inference.py \
  --init-from "${TRAIN_RUN}" \
  --workdir "${TRAIN_RUN}" \
  --cfg-scale "${CFG_SCALE:-2.5}" \
  --num-samples "${NUM_SAMPLES:-50000}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-512}" \
  --json-out "${TRAIN_RUN}/eval_cfg${CFG_SCALE:-2.5}.json"
