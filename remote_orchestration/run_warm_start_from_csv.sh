#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAP_FILE="${1:-${SCRIPT_DIR}/machine_ssh_map.csv}"
OPTIONS_FILE="${2:-${SCRIPT_DIR}/warm_start_options.csv}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-/Users/user/dirlab/io_stack_tuning/synth_logs}"
REMOTE_ROOT="/mnt/hasanfs/io_synthesizer"
REMOTE_WARM_SCRIPT="${REMOTE_ROOT}/warm_start_pipeline.py"

if [[ ! -f "${MAP_FILE}" ]]; then
  echo "ERROR: machine map CSV not found: ${MAP_FILE}" >&2
  exit 1
fi
if [[ ! -f "${OPTIONS_FILE}" ]]; then
  echo "ERROR: warm-start options CSV not found: ${OPTIONS_FILE}" >&2
  exit 1
fi

# Map local path to remote repo path when possible.
REMOTE_OPTIONS_FILE="${OPTIONS_FILE}"
if [[ "${OPTIONS_FILE}" == "${LOCAL_REPO_ROOT}/"* ]]; then
  rel="${OPTIONS_FILE#${LOCAL_REPO_ROOT}/}"
  REMOTE_OPTIONS_FILE="${REMOTE_ROOT}/${rel}"
elif [[ "${OPTIONS_FILE}" != /* ]]; then
  REMOTE_OPTIONS_FILE="${REMOTE_ROOT}/${OPTIONS_FILE}"
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

mkdir -p "${LOCAL_LOG_DIR}"
stamp="$(date +%Y%m%d_%H%M%S)"
local_log="${LOCAL_LOG_DIR}/warm_start_remote_${stamp}.log"

remote_cmd=( python3 "${REMOTE_WARM_SCRIPT}" --options-csv "${REMOTE_OPTIONS_FILE}" )
printf -v remote_cmd_quoted '%q ' "${remote_cmd[@]}"
remote_cmd_quoted="${remote_cmd_quoted% }"
inner_root_cmd="cd ${REMOTE_ROOT} && ${remote_cmd_quoted}"
printf -v inner_root_cmd_quoted '%q' "${inner_root_cmd}"
remote_shell_cmd="sudo -i bash -lc ${inner_root_cmd_quoted}"

echo "Machine map: ${MAP_FILE}"
echo "Options file: ${OPTIONS_FILE}"
echo "Remote options file: ${REMOTE_OPTIONS_FILE}"
echo "Remote target: ${node0_ssh_command}"
echo "Remote command: ${remote_shell_cmd}"
echo "Local log: ${local_log}"
echo

set +e
"${ssh_parts[@]}" -n "${remote_shell_cmd}" 2>&1 | tee -a "${local_log}"
ssh_rc=${PIPESTATUS[0]}
set -e

if [[ "${ssh_rc}" -ne 0 ]]; then
  echo "Remote warm-start pipeline failed with exit code ${ssh_rc}" >&2
  exit "${ssh_rc}"
fi

echo "Remote warm-start pipeline completed successfully."
