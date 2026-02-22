#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAP_FILE="${1:-${SCRIPT_DIR}/machine_ssh_map.csv}"

if [[ ! -f "${MAP_FILE}" ]]; then
  echo "ERROR: CSV file not found: ${MAP_FILE}" >&2
  exit 1
fi

if [[ "$(basename "${REPO_DIR}")" != "io_synthesizer" ]]; then
  echo "ERROR: Expected repo directory name 'io_synthesizer', got: ${REPO_DIR}" >&2
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

echo "[node0] ensuring /mnt/hasanfs exists..."
if [[ "${#ssh_opts[@]}" -gt 0 ]]; then
  ssh "${ssh_opts[@]}" "${ssh_target}" -n "mkdir -p /mnt/hasanfs"
else
  ssh "${ssh_target}" -n "mkdir -p /mnt/hasanfs"
fi

echo "[node0] transferring local repo to /mnt/hasanfs/io_synthesizer ..."
if [[ "${#ssh_opts[@]}" -gt 0 ]]; then
  rsync -az --progress --rsync-path="sudo rsync" -e "ssh ${ssh_opts[*]}" "${REPO_DIR}" "${ssh_target}:/mnt/hasanfs/"
else
  rsync -az --progress --rsync-path="sudo rsync" "${REPO_DIR}" "${ssh_target}:/mnt/hasanfs/"
fi

echo "[node0] creating initial synth directories..."
if [[ "${#ssh_opts[@]}" -gt 0 ]]; then
  ssh "${ssh_opts[@]}" "${ssh_target}" -n "sudo -i bash -lc 'set -euo pipefail; mkdir -p /mnt/hasanfs/bin /mnt/hasanfs/out_synth'"
else
  ssh "${ssh_target}" -n "sudo -i bash -lc 'set -euo pipefail; mkdir -p /mnt/hasanfs/bin /mnt/hasanfs/out_synth'"
fi

if [[ -n "${remote_user}" ]]; then
  echo "[node0] setting ownership for writable synth paths..."
  if [[ "${#ssh_opts[@]}" -gt 0 ]]; then
    ssh "${ssh_opts[@]}" "${ssh_target}" -n "sudo -i bash -lc 'set -euo pipefail; grp=\$(id -gn ${remote_user} 2>/dev/null || true); if [[ -n \"\${grp}\" ]]; then chown -R ${remote_user}:\${grp} /mnt/hasanfs/io_synthesizer /mnt/hasanfs/bin /mnt/hasanfs/out_synth; else chown -R ${remote_user} /mnt/hasanfs/io_synthesizer /mnt/hasanfs/bin /mnt/hasanfs/out_synth; fi'"
  else
    ssh "${ssh_target}" -n "sudo -i bash -lc 'set -euo pipefail; grp=\$(id -gn ${remote_user} 2>/dev/null || true); if [[ -n \"\${grp}\" ]]; then chown -R ${remote_user}:\${grp} /mnt/hasanfs/io_synthesizer /mnt/hasanfs/bin /mnt/hasanfs/out_synth; else chown -R ${remote_user} /mnt/hasanfs/io_synthesizer /mnt/hasanfs/bin /mnt/hasanfs/out_synth; fi'"
  fi
else
  echo "[node0] WARNING: could not infer remote user from ssh target (${ssh_target}); skipping chown step."
fi

echo "[node0] initial synthesizer setup completed."
