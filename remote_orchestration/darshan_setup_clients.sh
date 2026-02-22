#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_FILE="${1:-${SCRIPT_DIR}/machine_ssh_map.csv}"

if [[ ! -f "${MAP_FILE}" ]]; then
  echo "ERROR: CSV file not found: ${MAP_FILE}" >&2
  exit 1
fi

NODE0_SETUP_CMD="$(cat <<'EOF'
sudo -i bash -lc 'set -euo pipefail; DARSHAN_LIB="/mnt/hasanfs/darshan-3.4.7/darshan-runtime/install/lib/libdarshan.so"; LOG_DIR="/mnt/hasanfs/darshan-logs"; MK_LOG_DIRS="/mnt/hasanfs/darshan-3.4.7/darshan-runtime/install/bin/darshan-mk-log-dirs.pl"; if [[ -f "${DARSHAN_LIB}" ]]; then echo "Darshan library exists at ${DARSHAN_LIB}; skipping runtime/util setup."; else cd /custom-install/scripts/io_profiler_setup_scripts; bash darshan_runtime_setup.sh; bash darshan_util_setup.sh; fi; if [[ ! -d "${LOG_DIR}" ]]; then mkdir -p "${LOG_DIR}"; "${MK_LOG_DIRS}"; else echo "Log directory exists at ${LOG_DIR}; skipping darshan-mk-log-dirs.pl."; fi'
EOF
)"
BASHRC_UPDATE_CMD="sudo -i bash -lc 'set -euo pipefail; if [[ -f /root/.bashrc ]]; then sed -Ei \"s/3\\.4\\.[0-9]+/3.4.7/g\" /root/.bashrc; else echo \"/root/.bashrc not found; skipping.\"; fi'"

prepare_ssh_parts() {
  local ssh_command="$1"
  read -r -a SSH_PARTS <<< "${ssh_command}"
  if [[ "${#SSH_PARTS[@]}" -eq 0 || "${SSH_PARTS[0]}" != "ssh" ]]; then
    return 1
  fi

  local has_tty=0
  local token
  for token in "${SSH_PARTS[@]}"; do
    if [[ "${token}" == "-t" || "${token}" == "-tt" ]]; then
      has_tty=1
      break
    fi
  done
  if [[ "${has_tty}" -eq 0 ]]; then
    SSH_PARTS=(ssh -tt "${SSH_PARTS[@]:1}")
  fi
  return 0
}

run_remote_cmd() {
  local id="$1"
  local ssh_command="$2"
  local remote_cmd="$3"

  if ! prepare_ssh_parts "${ssh_command}"; then
    echo "[${id}] ERROR: invalid ssh_command in CSV: ${ssh_command}" >&2
    return 1
  fi

  "${SSH_PARTS[@]}" -n "${remote_cmd}"
}

node0_ssh_command=""
total_clients=0
processed_clients=0
failed=0
failed_nodes=()

while IFS=, read -r id ssh_command machine_type; do
  if [[ "${id}" == "id" ]]; then
    continue
  fi
  if [[ "${machine_type}" != "client" ]]; then
    continue
  fi

  total_clients=$((total_clients + 1))
  if [[ "${id}" == "node0" ]]; then
    node0_ssh_command="${ssh_command}"
  fi
done < "${MAP_FILE}"

if [[ -z "${node0_ssh_command}" ]]; then
  echo "ERROR: node0 with machine_type=client not found in ${MAP_FILE}" >&2
  exit 1
fi

echo "[node0] running one-time Darshan setup/checks..."
if run_remote_cmd "node0" "${node0_ssh_command}" "${NODE0_SETUP_CMD}"; then
  echo "[node0] setup/checks completed successfully."
else
  echo "[node0] setup/checks FAILED." >&2
  failed=$((failed + 1))
  failed_nodes+=("node0")
fi

while IFS=, read -r id ssh_command machine_type; do
  if [[ "${id}" == "id" ]]; then
    continue
  fi
  if [[ "${machine_type}" != "client" ]]; then
    continue
  fi

  echo "[${id}] updating /root/.bashrc Darshan version to 3.4.7..."
  if run_remote_cmd "${id}" "${ssh_command}" "${BASHRC_UPDATE_CMD}"; then
    echo "[${id}] /root/.bashrc update completed."
    processed_clients=$((processed_clients + 1))
  else
    echo "[${id}] /root/.bashrc update FAILED." >&2
    failed=$((failed + 1))
    failed_nodes+=("${id}")
  fi
done < "${MAP_FILE}"

echo
echo "Summary: total_clients=${total_clients}, bashrc_updates_success=${processed_clients}, failed=${failed}"
if [[ "${failed}" -gt 0 ]]; then
  echo "Failed nodes: ${failed_nodes[*]}" >&2
  exit 1
fi
