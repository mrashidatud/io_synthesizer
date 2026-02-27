#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_FILE="${SCRIPT_DIR}/machine_ssh_map.csv"
LOCAL_ROOT="/Users/user/dirlab/io_stack_tuning/synth_logs"
DRY_RUN=0
DELETE_MODE=0

REMOTE_OUT_SYNTH="/mnt/hasanfs/out_synth"
REMOTE_SAMPLES="/mnt/hasanfs/samples"

usage() {
  cat <<'EOF'
Usage:
  bash remote_orchestration/sync_outputs_from_remote.sh [--dry-run] [--delete] [--map <path>] [--local-dir <path>]

Syncs remote output/log directories from node0 (machine_type=client) to local storage.

Defaults:
  local root: /Users/user/dirlab/io_stack_tuning/synth_logs
  remote dirs:
    /mnt/hasanfs/out_synth
    /mnt/hasanfs/samples

Options:
  --dry-run            Show what would be synced without writing files.
  --delete             Mirror remote exactly (delete local files missing remotely).
  --map <path>         Override machine map CSV path.
  --local-dir <path>   Override local destination root.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --delete)
      DELETE_MODE=1
      shift
      ;;
    --map)
      MAP_FILE="${2:-}"
      shift 2
      ;;
    --local-dir)
      LOCAL_ROOT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${MAP_FILE}" ]]; then
  echo "ERROR: machine map CSV not found: ${MAP_FILE}" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "ERROR: rsync is required but not found on local machine." >&2
  exit 1
fi

node0_ssh_command=""
while IFS=, read -r id ssh_command machine_type; do
  if [[ "${id}" == "id" ]]; then
    continue
  fi
  if [[ "${id}" == "node0" && "${machine_type}" == "client" ]]; then
    node0_ssh_command="${ssh_command}"
    break
  fi
done < "${MAP_FILE}"

if [[ -z "${node0_ssh_command}" ]]; then
  echo "ERROR: node0 with machine_type=client not found in ${MAP_FILE}" >&2
  exit 1
fi

read -r -a ssh_parts <<< "${node0_ssh_command}"
if [[ "${#ssh_parts[@]}" -lt 2 || "${ssh_parts[0]}" != "ssh" ]]; then
  echo "ERROR: invalid ssh_command for node0: ${node0_ssh_command}" >&2
  exit 1
fi

last_index=$(( ${#ssh_parts[@]} - 1 ))
ssh_target="${ssh_parts[$last_index]}"
ssh_opts=()
if [[ "${#ssh_parts[@]}" -gt 2 ]]; then
  ssh_opts=("${ssh_parts[@]:1:${#ssh_parts[@]}-2}")
fi

local_out_synth="${LOCAL_ROOT}/out_synth"
local_samples="${LOCAL_ROOT}/samples"
mkdir -p "${local_out_synth}" "${local_samples}"

rsync_args=(
  -az
  --progress
  --rsync-path="sudo rsync"
)

if [[ "${DELETE_MODE}" -eq 1 ]]; then
  rsync_args+=( --delete )
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
  rsync_args+=( --dry-run --itemize-changes )
fi

echo "Machine map: ${MAP_FILE}"
echo "Remote target: ${node0_ssh_command}"
echo "Local root: ${LOCAL_ROOT}"
echo "Remote source 1: ${REMOTE_OUT_SYNTH}/"
echo "Local destination 1: ${local_out_synth}/"
echo "Remote source 2: ${REMOTE_SAMPLES}/"
echo "Local destination 2: ${local_samples}/"
if [[ "${DELETE_MODE}" -eq 1 ]]; then
  echo "Mode: mirror (--delete enabled)"
else
  echo "Mode: incremental (no delete)"
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry-run: enabled"
fi
echo

if [[ "${#ssh_opts[@]}" -gt 0 ]]; then
  echo "[node0] syncing ${REMOTE_OUT_SYNTH} -> ${local_out_synth}"
  rsync "${rsync_args[@]}" -e "ssh ${ssh_opts[*]}" "${ssh_target}:${REMOTE_OUT_SYNTH}/" "${local_out_synth}/"

  echo "[node0] syncing ${REMOTE_SAMPLES} -> ${local_samples}"
  rsync "${rsync_args[@]}" -e "ssh ${ssh_opts[*]}" "${ssh_target}:${REMOTE_SAMPLES}/" "${local_samples}/"
else
  echo "[node0] syncing ${REMOTE_OUT_SYNTH} -> ${local_out_synth}"
  rsync "${rsync_args[@]}" "${ssh_target}:${REMOTE_OUT_SYNTH}/" "${local_out_synth}/"

  echo "[node0] syncing ${REMOTE_SAMPLES} -> ${local_samples}"
  rsync "${rsync_args[@]}" "${ssh_target}:${REMOTE_SAMPLES}/" "${local_samples}/"
fi

echo "[node0] output/log sync completed."
