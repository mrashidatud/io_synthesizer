#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OPTIONS_FILE="${1:-${SCRIPT_DIR}/active_sampling_options.csv}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${REPO_ROOT}/logs}"
ACTIVE_SCRIPT="${SCRIPT_DIR}/active_sampling_pipeline.py"

if [[ ! -f "${OPTIONS_FILE}" ]]; then
  echo "ERROR: active-sampling options CSV not found: ${OPTIONS_FILE}" >&2
  exit 1
fi
if [[ ! -f "${ACTIVE_SCRIPT}" ]]; then
  echo "ERROR: active-sampling pipeline not found: ${ACTIVE_SCRIPT}" >&2
  exit 1
fi
if [[ "$(basename "${REPO_ROOT}")" != "io_synthesizer" ]]; then
  echo "ERROR: expected repo root name 'io_synthesizer', got: ${REPO_ROOT}" >&2
  exit 1
fi

mkdir -p "${LOCAL_LOG_DIR}"
stamp="$(date +%Y%m%d_%H%M%S)"
local_log="${LOCAL_LOG_DIR}/active_sampling_${stamp}.log"

active_cmd=( python3 "${ACTIVE_SCRIPT}" --options-csv "${OPTIONS_FILE}" )
printf -v active_cmd_quoted '%q ' "${active_cmd[@]}"
active_cmd_quoted="${active_cmd_quoted% }"

echo "Repo root: ${REPO_ROOT}"
echo "Options file: ${OPTIONS_FILE}"
echo "Command: ${active_cmd_quoted}"
echo "Local log: ${local_log}"
echo

set +e
"${active_cmd[@]}" 2>&1 | tee -a "${local_log}"
active_rc=${PIPESTATUS[0]}
set -e

if [[ "${active_rc}" -ne 0 ]]; then
  echo "Active-sampling pipeline failed with exit code ${active_rc}" >&2
  exit "${active_rc}"
fi

echo "Active-sampling pipeline completed successfully."
