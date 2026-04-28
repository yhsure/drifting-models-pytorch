#!/usr/bin/env bash
#SBATCH --job-name=drift-eval-mae640
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --time=00:30:00
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

TRAIN_RUN="${TRAIN_RUN:-$(ls -dt runs/*_gen_latent_ablation_mae640_short* 2>/dev/null | head -n1 || true)}"
if [[ -z "${TRAIN_RUN}" ]]; then
  echo "No gen_latent_ablation_mae640_short run found. Set TRAIN_RUN=/path/to/run." >&2
  exit 1
fi
if [[ ! -f "${TRAIN_RUN}/checkpoints/step_000000800.pt" && ! -f "${TRAIN_RUN}/params_ema/ema_params.pt" ]]; then
  echo "Step-800 checkpoint or EMA artifact missing in ${TRAIN_RUN}" >&2
  exit 1
fi

echo "Evaluating run: ${TRAIN_RUN}"
exec srun env -u CUDA_VISIBLE_DEVICES "${REPO_ROOT}/.venv/bin/python" -u inference.py \
  --init-from "${TRAIN_RUN}" \
  --workdir "${TRAIN_RUN}" \
  --cfg-scale "${CFG_SCALE:-2.0}" \
  --num-samples "${NUM_SAMPLES:-1024}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-256}" \
  --json-out "${TRAIN_RUN}/eval_cfg${CFG_SCALE:-2.0}.json"
