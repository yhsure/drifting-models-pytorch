#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${REPO_ROOT}/../../.." && pwd)}"

function drift_load_modules() {
  module load Stages/2026
  module load GCCcore/14.3.0 CUDA/13 Python/3.13.5 uv/0.8.17
}

function drift_export_env() {
  export PYTHONUNBUFFERED=1
  export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE_ROOT}/.cache/uv}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${WORKSPACE_ROOT}/.cache/torchinductor}"
  export TORCH_HOME="${TORCH_HOME:-${WORKSPACE_ROOT}/.cache/torch}"
  export HF_ROOT="${HF_ROOT:-${WORKSPACE_ROOT}/hf_cache}"

  mkdir -p "${UV_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_HOME}" "${HF_ROOT}" "${REPO_ROOT}/logs/slurm"
}

function drift_setup() {
  drift_load_modules
  drift_export_env
  cd "${REPO_ROOT}"
  uv sync --group dev
}

function drift_nproc_per_node() {
  cd "${REPO_ROOT}"
  local n
  n="$("${REPO_ROOT}/.venv/bin/python" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || true)"
  if [[ -z "${n}" || "${n}" == "0" ]]; then
    n=4
  fi
  echo "${n}"
}
