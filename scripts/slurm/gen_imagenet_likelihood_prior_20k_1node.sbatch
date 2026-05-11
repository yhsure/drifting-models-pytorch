#!/usr/bin/env bash
# ImageNet explicit-likelihood pilot matching the 1-node / 4-GPU 20k profile
# documented in _abj/drifty-samples/EXPERIMENTS.md.
#SBATCH --job-name=drift-gen-like-20k-1n
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
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

BASE_CONFIG="${BASE_CONFIG:-configs/gen/latent_sota_B.yaml}"
RUN_ROOT="${RUN_ROOT:-runs/imagenet_likelihood_sparse_soft_20k_1node}"
COMPONENT_COUNT="${COMPONENT_COUNT:-8192}"
CONDITIONING="${CONDITIONING:-sparse_soft}"
TOPK_START="${TOPK_START:-16}"
TOPK_FINAL="${TOPK_FINAL:-8}"
TEMP_START="${TEMP_START:-1.5}"
TEMP_FINAL="${TEMP_FINAL:-0.45}"
TEMP_ANNEAL_STEPS="${TEMP_ANNEAL_STEPS:-3500}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
DATASET_BATCH_SIZE="${DATASET_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
GEN_PER_LABEL="${GEN_PER_LABEL:-8}"
TOTAL_STEPS="${TOTAL_STEPS:-20000}"
FIT_SAMPLES="${FIT_SAMPLES:-0}"
FIT_BATCH_SIZE="${FIT_BATCH_SIZE:-256}"
KMEANS_ITERS="${KMEANS_ITERS:-40}"
KMEANS_BATCH_SIZE="${KMEANS_BATCH_SIZE:-2048}"
RESPONSIBILITY_BATCH_SIZE="${RESPONSIBILITY_BATCH_SIZE:-512}"
EXPECTED_WORLD_SIZE="${EXPECTED_WORLD_SIZE:-4}"
COMPILE_LEVEL="${COMPILE_LEVEL:-1}"
PRIOR_CACHE_PATH="${PRIOR_CACHE_PATH:-}"

MAE_NAME="$("${REPO_ROOT}/.venv/bin/python" - "${BASE_CONFIG}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
mae_path = str(cfg.get("feature", {}).get("mae_path", ""))
print(mae_path.removeprefix("hf://"))
PY
)"
if [[ -n "${MAE_NAME}" ]]; then
  MAE_CACHE="${HF_ROOT}/models/mae/pytorch/${MAE_NAME}"
  if [[ ! -f "${MAE_CACHE}/metadata.json" || ! -f "${MAE_CACHE}/ema_params.pt" ]]; then
    echo "Missing offline MAE cache at ${MAE_CACHE}" >&2
    echo "Expected metadata.json and ema_params.pt; compute nodes do not have internet access." >&2
    exit 1
  fi
fi

NPROC="$(drift_nproc_per_node)"
export DRIFT_NPROC_PER_NODE="${NPROC}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Training explicit likelihood prior (1-node small profile): K=${COMPONENT_COUNT}, topk=${TOPK_START}->${TOPK_FINAL}, tau=${TEMP_START}->${TEMP_FINAL}, expected_world_size=${EXPECTED_WORLD_SIZE}, compile=${COMPILE_LEVEL}"
if [[ -n "${PRIOR_CACHE_PATH}" ]]; then
  echo "Reusing explicit likelihood prior cache: ${PRIOR_CACHE_PATH}"
fi

exec srun env -u CUDA_VISIBLE_DEVICES "${REPO_ROOT}/.venv/bin/python" -u -m torchrun_jsc \
  --standalone \
  --nproc-per-node="${NPROC}" \
  scripts/run_imagenet_likelihood_prior.py \
    --base-config "${BASE_CONFIG}" \
    --run-root "${RUN_ROOT}" \
    --total-steps "${TOTAL_STEPS}" \
    --component-count "${COMPONENT_COUNT}" \
    --conditioning "${CONDITIONING}" \
    --topk-start "${TOPK_START}" --topk-final "${TOPK_FINAL}" \
    --temp-start "${TEMP_START}" --temp-final "${TEMP_FINAL}" \
    --temp-anneal-steps "${TEMP_ANNEAL_STEPS}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --dataset-batch-size "${DATASET_BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --gen-per-label "${GEN_PER_LABEL}" \
    --save-per-step 2000 --eval-per-step 0 --eval-samples 0 \
    --fixed-cfg 1.0 \
    --fit-samples "${FIT_SAMPLES}" \
    --fit-batch-size "${FIT_BATCH_SIZE}" \
    --kmeans-iters "${KMEANS_ITERS}" \
    --kmeans-batch-size "${KMEANS_BATCH_SIZE}" \
    --responsibility-batch-size "${RESPONSIBILITY_BATCH_SIZE}" \
    --expected-world-size "${EXPECTED_WORLD_SIZE}" \
    --compile "${COMPILE_LEVEL}" \
    --comparison-family imagenet_20k_1node \
    --prior-cache-path "${PRIOR_CACHE_PATH}"
