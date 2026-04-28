#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${REPO_ROOT}/../../.." && pwd)}"

RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs}"
WANDB_WATCH_DIR="${WANDB_WATCH_DIR:-${RUNS_ROOT}}"
SLEEP_SECONDS="${WANDB_SYNC_INTERVAL:-60}"
PROJECT_ARG=()
ENTITY_ARG=()

if [[ -f "${REPO_ROOT}/scripts/slurm/common.sh" ]]; then
  # The uv venv on Jupiter depends on the Python module loaded by common.sh.
  source "${REPO_ROOT}/scripts/slurm/common.sh"
  drift_load_modules >/dev/null 2>&1 || true
fi

if [[ -n "${WANDB_PROJECT:-}" ]]; then
  PROJECT_ARG=(--project "${WANDB_PROJECT}")
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  ENTITY_ARG=(--entity "${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}"
if [[ -x "${REPO_ROOT}/.venv/bin/wandb" ]]; then
  WANDB_BIN="${REPO_ROOT}/.venv/bin/wandb"
elif command -v wandb >/dev/null 2>&1; then
  WANDB_BIN="$(command -v wandb)"
else
  echo "wandb command not found. Run 'uv sync --group dev' first." >&2
  exit 1
fi

mkdir -p "${WANDB_WATCH_DIR}"

echo "Watching ${WANDB_WATCH_DIR}"
echo "Press Ctrl-C to stop."
echo "If WandB asks for a login, stop here and run: ${WANDB_BIN} login"

while true; do
  while IFS= read -r -d "" run_dir; do
    if [[ -d "${run_dir}" ]]; then
      SYNC_ARGS=()
      shopt -s nullglob
      synced=("${run_dir}"/run-*.wandb.synced)
      shopt -u nullglob
      if (( ${#synced[@]} > 0 )); then
        newest_synced="$(find "${run_dir}" -maxdepth 1 -name 'run-*.wandb.synced' -printf '%T@\n' | sort -nr | head -n1)"
        newest_payload="$(find "${run_dir}" -type f ! -name '*.synced' -printf '%T@\n' | sort -nr | head -n1)"
        if [[ -n "${newest_synced}" && -n "${newest_payload}" ]]; then
          if awk "BEGIN { exit !(${newest_payload} <= ${newest_synced}) }"; then
            continue
          fi
          SYNC_ARGS=(--include-synced)
        fi
      fi
      "${WANDB_BIN}" sync "${PROJECT_ARG[@]}" "${ENTITY_ARG[@]}" "${SYNC_ARGS[@]}" "${run_dir}" || true
    fi
  done < <(find "${WANDB_WATCH_DIR}" -type d -name 'offline-run-*' -path '*/wandb/offline-run-*' -prune -print0)

  sleep "${SLEEP_SECONDS}"
done
