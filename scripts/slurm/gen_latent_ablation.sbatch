#!/usr/bin/env bash
#SBATCH --job-name=drift-gen-latent-ablation
#SBATCH --partition=booster
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --time=08:00:00
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

CONFIG="${CONFIG:-configs/gen/latent_ablation.yaml}"
MAE_NAME="$("${REPO_ROOT}/.venv/bin/python" - "${CONFIG}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
mae_path = str(cfg.get("feature", {}).get("mae_path", ""))
print(mae_path.removeprefix("hf://"))
PY
)"
MAE_CACHE="${HF_ROOT}/models/mae/pytorch/${MAE_NAME}"
if [[ ! -f "${MAE_CACHE}/metadata.json" || ! -f "${MAE_CACHE}/ema_params.pt" ]]; then
  echo "Missing offline MAE cache at ${MAE_CACHE}" >&2
  echo "Expected metadata.json and ema_params.pt; compute nodes do not have internet access." >&2
  exit 1
fi

NPROC="$(drift_nproc_per_node)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec srun env -u CUDA_VISIBLE_DEVICES "${REPO_ROOT}/.venv/bin/python" -u -m torchrun_jsc \
  --nnodes="${SLURM_NNODES}" \
  --node-rank="${SLURM_NODEID}" \
  --nproc-per-node="${NPROC}" \
  --rdzv-id="${SLURM_JOB_ID}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  main.py --gen \
    --config "${CONFIG}" \
    --workdir runs/gen_latent_ablation
