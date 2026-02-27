#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OPTIONS_FILE="${1:-${SCRIPT_DIR}/warm_start_options.csv}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${REPO_ROOT}/logs}"
WARM_SCRIPT="${SCRIPT_DIR}/warm_start_pipeline.py"

if [[ ! -f "${OPTIONS_FILE}" ]]; then
  echo "ERROR: warm-start options CSV not found: ${OPTIONS_FILE}" >&2
  exit 1
fi
if [[ ! -f "${WARM_SCRIPT}" ]]; then
  echo "ERROR: warm-start pipeline not found: ${WARM_SCRIPT}" >&2
  exit 1
fi

if [[ "$(basename "${REPO_ROOT}")" != "io_synthesizer" ]]; then
  echo "ERROR: expected repo root name 'io_synthesizer', got: ${REPO_ROOT}" >&2
  exit 1
fi

mkdir -p "${LOCAL_LOG_DIR}"
stamp="$(date +%Y%m%d_%H%M%S)"
local_log="${LOCAL_LOG_DIR}/warm_start_remote_${stamp}.log"

warm_cmd=( python3 "${WARM_SCRIPT}" --options-csv "${OPTIONS_FILE}" )
printf -v warm_cmd_quoted '%q ' "${warm_cmd[@]}"
warm_cmd_quoted="${warm_cmd_quoted% }"

echo "Repo root: ${REPO_ROOT}"
echo "Options file: ${OPTIONS_FILE}"
echo "Command: ${warm_cmd_quoted}"
echo "Local log: ${local_log}"
echo

set +e
"${warm_cmd[@]}" 2>&1 | tee -a "${local_log}"
warm_rc=${PIPESTATUS[0]}
set -e

if [[ "${warm_rc}" -ne 0 ]]; then
  echo "Warm-start pipeline failed with exit code ${warm_rc}" >&2
  exit "${warm_rc}"
fi

echo "Warm-start pipeline completed successfully."
