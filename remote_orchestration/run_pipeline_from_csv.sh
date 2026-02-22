#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_FILE="${1:-${SCRIPT_DIR}/machine_ssh_map.csv}"
OPTIONS_FILE="${2:-${SCRIPT_DIR}/pipeline_options.csv}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-/Users/user/dirlab/io_stack_tuning/synth_logs}"
REMOTE_PIPELINE="/mnt/hasanfs/io_synthesizer/pipeline.sh"
REMOTE_ROOT="/mnt/hasanfs/io_synthesizer"

if [[ ! -f "${MAP_FILE}" ]]; then
  echo "ERROR: machine map CSV not found: ${MAP_FILE}" >&2
  exit 1
fi
if [[ ! -f "${OPTIONS_FILE}" ]]; then
  echo "ERROR: pipeline options CSV not found: ${OPTIONS_FILE}" >&2
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

# Do not force TTY allocation here; this script is intended to support
# detached/nohup background runs from local machine.

cap=""
nprocs=""
nprocs_cap=""
inputs=""
force_build="0"
delete_darshan="0"
filters=""

while IFS=, read -r option value; do
  if [[ "${option}" == "option" ]]; then
    continue
  fi
  case "${option}" in
    cap) cap="${value}" ;;
    nprocs) nprocs="${value}" ;;
    nprocs_cap) nprocs_cap="${value}" ;;
    inputs) inputs="${value}" ;;
    force_build) force_build="${value}" ;;
    delete_darshan) delete_darshan="${value}" ;;
    filters) filters="${value}" ;;
    "") ;;
    *)
      echo "WARN: Unknown option in ${OPTIONS_FILE}: ${option}" >&2
      ;;
  esac
done < "${OPTIONS_FILE}"

args=()
if [[ -n "${cap}" ]]; then args+=( --cap "${cap}" ); fi
if [[ -n "${nprocs}" ]]; then args+=( --nprocs "${nprocs}" ); fi
if [[ -n "${nprocs_cap}" ]]; then args+=( --nprocs-cap "${nprocs_cap}" ); fi
if [[ -n "${inputs}" ]]; then args+=( --inputs "${inputs}" ); fi
if [[ "${force_build}" == "1" ]]; then args+=( --force-build ); fi
if [[ "${delete_darshan}" == "1" ]]; then args+=( --delete-darshan ); fi

if [[ -n "${filters}" ]]; then
  read -r -a filter_parts <<< "${filters}"
  args+=( -- "${filter_parts[@]}" )
fi

mkdir -p "${LOCAL_LOG_DIR}"
stamp="$(date +%Y%m%d_%H%M%S)"
local_log="${LOCAL_LOG_DIR}/pipeline_remote_${stamp}.log"

remote_cmd=( bash "${REMOTE_PIPELINE}" "${args[@]}" )
printf -v remote_cmd_quoted '%q ' "${remote_cmd[@]}"
remote_cmd_quoted="${remote_cmd_quoted% }"
inner_root_cmd="cd ${REMOTE_ROOT} && ${remote_cmd_quoted}"
printf -v inner_root_cmd_quoted '%q' "${inner_root_cmd}"
remote_shell_cmd="sudo -i bash -lc ${inner_root_cmd_quoted}"

echo "Machine map: ${MAP_FILE}"
echo "Options file: ${OPTIONS_FILE}"
echo "Remote target: ${node0_ssh_command}"
echo "Remote command: ${remote_shell_cmd}"
echo "Local log: ${local_log}"
echo

set +e
"${ssh_parts[@]}" -n "${remote_shell_cmd}" 2>&1 | tee -a "${local_log}"
ssh_rc=${PIPESTATUS[0]}
set -e

if [[ "${ssh_rc}" -ne 0 ]]; then
  echo "Remote pipeline failed with exit code ${ssh_rc}" >&2
  exit "${ssh_rc}"
fi

echo "Remote pipeline completed successfully."
