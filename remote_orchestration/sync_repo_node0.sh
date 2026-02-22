#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAP_FILE="${SCRIPT_DIR}/machine_ssh_map.csv"
REMOTE_DIR="/mnt/hasanfs/io_synthesizer"
DELETE_MODE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash remote_orchestration/sync_repo_node0.sh [--delete] [--dry-run] [--map <path/to/machine_ssh_map.csv>]

Options:
  --delete    Mirror local repo exactly (deletes remote files missing locally).
  --dry-run   Show what would be synced without making changes.
  --map       Override machine map CSV path.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --delete)
      DELETE_MODE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --map)
      MAP_FILE="${2:-}"
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

if [[ "$(basename "${REPO_DIR}")" != "io_synthesizer" ]]; then
  echo "ERROR: expected repo directory name 'io_synthesizer', got: ${REPO_DIR}" >&2
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
remote_user="${ssh_target%@*}"
if [[ "${remote_user}" == "${ssh_target}" ]]; then
  remote_user=""
fi
ssh_opts=()
if [[ "${#ssh_parts[@]}" -gt 2 ]]; then
  ssh_opts=("${ssh_parts[@]:1:${#ssh_parts[@]}-2}")
fi

echo "Machine map: ${MAP_FILE}"
echo "Remote target: ${node0_ssh_command}"
echo "Source: ${REPO_DIR}/"
echo "Destination: ${REMOTE_DIR}/"
if [[ "${DELETE_MODE}" -eq 1 ]]; then
  echo "Mode: mirror (--delete enabled)"
else
  echo "Mode: incremental (no delete)"
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry-run: enabled"
fi
echo

echo "[node0] ensuring remote target directory exists..."
if [[ "${#ssh_opts[@]}" -gt 0 ]]; then
  ssh "${ssh_opts[@]}" "${ssh_target}" -n "sudo -i mkdir -p ${REMOTE_DIR}"
else
  ssh "${ssh_target}" -n "sudo -i mkdir -p ${REMOTE_DIR}"
fi

rsync_args=(
  -az
  --progress
  --rsync-path="sudo rsync"
  --exclude ".git/"
  --exclude "__pycache__/"
  --exclude "*.pyc"
  --exclude ".DS_Store"
)

if [[ "${DELETE_MODE}" -eq 1 ]]; then
  rsync_args+=( --delete )
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
  rsync_args+=( --dry-run --itemize-changes )
fi

echo "[node0] syncing local repository to remote..."
if [[ "${#ssh_opts[@]}" -gt 0 ]]; then
  rsync "${rsync_args[@]}" -e "ssh ${ssh_opts[*]}" "${REPO_DIR}/" "${ssh_target}:${REMOTE_DIR}/"
else
  rsync "${rsync_args[@]}" "${REPO_DIR}/" "${ssh_target}:${REMOTE_DIR}/"
fi

if [[ "${DRY_RUN}" -eq 0 && -n "${remote_user}" ]]; then
  echo "[node0] fixing ownership on remote repository path..."
  if [[ "${#ssh_opts[@]}" -gt 0 ]]; then
    ssh "${ssh_opts[@]}" "${ssh_target}" -n "sudo -i bash -lc 'set -euo pipefail; grp=\$(id -gn ${remote_user} 2>/dev/null || true); if [[ -n \"\${grp}\" ]]; then chown -R ${remote_user}:\${grp} ${REMOTE_DIR}; else chown -R ${remote_user} ${REMOTE_DIR}; fi'"
  else
    ssh "${ssh_target}" -n "sudo -i bash -lc 'set -euo pipefail; grp=\$(id -gn ${remote_user} 2>/dev/null || true); if [[ -n \"\${grp}\" ]]; then chown -R ${remote_user}:\${grp} ${REMOTE_DIR}; else chown -R ${remote_user} ${REMOTE_DIR}; fi'"
  fi
fi

echo "[node0] sync completed."

