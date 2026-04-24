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

  local cache_root="${DRIFT_CACHE_ROOT:-${WORKSPACE_ROOT}/.cache}"
  local hf_home="${DRIFT_HF_HOME:-${WORKSPACE_ROOT}/hf_cache}"

  export UV_CACHE_DIR="${cache_root}/uv"
  export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
  export TORCH_HOME="${cache_root}/torch"
  export HF_HOME="${hf_home}"
  export HF_ROOT="${HF_HOME}"
  export HF_HUB_CACHE="${HF_HOME}/hub"
  export DRIFT_SDVAE_PATH="${DRIFT_SDVAE_PATH:-${HF_HUB_CACHE}/models--stabilityai--sd-vae-ft-mse/snapshots/31f26fdeee1355a5c34592e401dd41e45d25a493}"
  # Keep transformers on HF_HOME/HF_HUB_CACHE path
  unset TRANSFORMERS_CACHE

  mkdir -p "${UV_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_HOME}" "${HF_HOME}" "${HF_HUB_CACHE}" "${REPO_ROOT}/logs/slurm"
}

function drift_export_dist_env() {
  if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
  fi

  if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    local master_addr
    master_addr="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
    case "${SYSTEMNAME:-}" in
      juwelsbooster|juwels|jurecadc|jusuf)
        master_addr="${master_addr}i"
        ;;
    esac
    export MASTER_ADDR="${MASTER_ADDR:-${master_addr}}"
  fi

  export MASTER_PORT="${MASTER_PORT:-54123}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ib0}"
}

function drift_setup() {
  drift_load_modules
  drift_export_env
  drift_export_dist_env
  cd "${REPO_ROOT}"
  uv sync --group dev
}

# Copies DATA_PATH into /dev/shm on every node (once per node) and updates
# DRIFT_DATA_PATH to point at the shm location.
# Usage: drift_copy_to_shm <data_path>  or set DRIFT_DATA_PATH before calling.
# The tar archive is created on first call and reused on subsequent calls.
# Call this after drift_setup and before the training srun.
function drift_copy_to_shm() {
  local data_path="${1:-${DRIFT_DATA_PATH:?DRIFT_DATA_PATH must be set or passed as argument}}"
  local scratch="${SCRATCH:?SCRATCH must be set}"
  local tar_path="${scratch}/$(basename "${data_path}").tar"
  local shm_path="/dev/shm/$(whoami)/$(basename "${data_path}")"

  # Create tar archive on the launch node (skipped if already exists).
  if [[ ! -e "${tar_path}" ]]; then
    echo "[drift] Creating tar archive: ${tar_path}"
    tar cf "${tar_path}" -C "$(dirname "${data_path}")" "$(basename "${data_path}")"
  fi

  export _DRIFT_TAR_PATH="${tar_path}"
  export _DRIFT_SHM_PATH="${shm_path}"

  _drift_extract_to_shm() {
    rm -rf "${_DRIFT_SHM_PATH}"
    mkdir -p "${_DRIFT_SHM_PATH}"
    tar xf "${_DRIFT_TAR_PATH}" -C "$(dirname "${_DRIFT_SHM_PATH}")"
    chmod 700 "${_DRIFT_SHM_PATH}"
  }
  export -f _drift_extract_to_shm

  # Run exactly one extraction task per node.
  echo "[drift] Copying data to /dev/shm on all nodes..."
  srun --ntasks-per-node=1 bash -c _drift_extract_to_shm
  echo "[drift] Data available at: ${shm_path}"

  export DRIFT_DATA_PATH="${shm_path}"
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
