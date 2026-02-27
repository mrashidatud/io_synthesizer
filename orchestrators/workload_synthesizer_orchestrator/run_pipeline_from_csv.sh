#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OPTIONS_FILE="${1:-${SCRIPT_DIR}/pipeline_options.csv}"
PIPELINE_SCRIPT="${SCRIPT_DIR}/workload_synthesizer_pipeline.py"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-/mnt/hasanfs/out_synth}"

if [[ ! -f "${OPTIONS_FILE}" ]]; then
  echo "ERROR: pipeline options CSV not found: ${OPTIONS_FILE}" >&2
  exit 1
fi
if [[ ! -f "${PIPELINE_SCRIPT}" ]]; then
  echo "ERROR: workload synthesizer pipeline not found: ${PIPELINE_SCRIPT}" >&2
  exit 1
fi
if [[ "$(basename "${REPO_ROOT}")" != "io_synthesizer" ]]; then
  echo "ERROR: expected repo root name 'io_synthesizer', got: ${REPO_ROOT}" >&2
  exit 1
fi

pipeline_cmd=( python3 "${PIPELINE_SCRIPT}" --options-csv "${OPTIONS_FILE}" )
printf -v pipeline_cmd_quoted '%q ' "${pipeline_cmd[@]}"
pipeline_cmd_quoted="${pipeline_cmd_quoted% }"

mkdir -p "${LOCAL_LOG_DIR}"
stamp="$(date +%Y%m%d_%H%M%S)"
local_log="${LOCAL_LOG_DIR}/workload_synthesizer_${stamp}.log"

echo "Repo root: ${REPO_ROOT}"
echo "Options file: ${OPTIONS_FILE}"
echo "Command: ${pipeline_cmd_quoted}"
echo "Local log dir: ${LOCAL_LOG_DIR}"
echo "Local log: ${local_log}"
echo

set +e
"${pipeline_cmd[@]}" 2>&1 | tee -a "${local_log}"
pipeline_rc=${PIPESTATUS[0]}
set -e

if [[ "${pipeline_rc}" -ne 0 ]]; then
  echo "Workload synthesizer pipeline failed with exit code ${pipeline_rc}" >&2
  exit "${pipeline_rc}"
fi

echo "Workload synthesizer pipeline completed successfully."
