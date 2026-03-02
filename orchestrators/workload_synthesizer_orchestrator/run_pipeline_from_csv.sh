#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OPTIONS_FILE="${1:-${SCRIPT_DIR}/pipeline_options.csv}"
PIPELINE_SCRIPT="${SCRIPT_DIR}/workload_synthesizer_pipeline.py"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-/mnt/hasanfs/out_synth}"

# Baseline Lustre settings (intentionally hardcoded here instead of reading
# io_recommender/config.yaml).
BASE_STRIPE_COUNT=1
BASE_STRIPE_SIZE="1M"
BASE_OSC_MAX_PAGES_PER_RPC=1024
BASE_MDC_MAX_PAGES_PER_RPC=256
BASE_OSC_MAX_RPCS_IN_FLIGHT=8
BASE_MDC_MAX_RPCS_IN_FLIGHT=8

# Set to 1 to run Lustre commands with sudo.
USE_SUDO_FOR_LUSTRE="${USE_SUDO_FOR_LUSTRE:-0}"

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

run_lustre_cmd() {
  if [[ "${USE_SUDO_FOR_LUSTRE}" == "1" ]]; then
    sudo "$@"
  else
    "$@"
  fi
}

apply_baseline_settings() {
  local output_root
  output_root="$(awk -F',' 'NR > 1 && $1 == "output_root" { print $2; exit }' "${OPTIONS_FILE}" | tr -d '\r')"
  if [[ -z "${output_root}" ]]; then
    output_root="${LOCAL_LOG_DIR}"
  fi
  mkdir -p "${output_root}"

  if ! command -v lctl >/dev/null 2>&1; then
    echo "ERROR: lctl not found in PATH; cannot apply baseline Lustre settings." >&2
    exit 1
  fi
  if ! command -v lfs >/dev/null 2>&1; then
    echo "ERROR: lfs not found in PATH; cannot apply baseline Lustre settings." >&2
    exit 1
  fi

  echo "Applying baseline Lustre settings before orchestration..."
  echo "  stripe_count=${BASE_STRIPE_COUNT} stripe_size=${BASE_STRIPE_SIZE}"
  echo "  osc.max_pages_per_rpc=${BASE_OSC_MAX_PAGES_PER_RPC}"
  echo "  mdc.max_pages_per_rpc=${BASE_MDC_MAX_PAGES_PER_RPC}"
  echo "  osc.max_rpcs_in_flight=${BASE_OSC_MAX_RPCS_IN_FLIGHT}"
  echo "  mdc.max_rpcs_in_flight=${BASE_MDC_MAX_RPCS_IN_FLIGHT}"
  echo "  output_root=${output_root}"

  # Global client RPC settings.
  run_lustre_cmd lctl set_param "osc.*.max_pages_per_rpc=${BASE_OSC_MAX_PAGES_PER_RPC}"
  run_lustre_cmd lctl set_param "osc.*.max_rpcs_in_flight=${BASE_OSC_MAX_RPCS_IN_FLIGHT}"
  run_lustre_cmd lctl set_param "mdc.*.max_pages_per_rpc=${BASE_MDC_MAX_PAGES_PER_RPC}"
  run_lustre_cmd lctl set_param "mdc.*.max_rpcs_in_flight=${BASE_MDC_MAX_RPCS_IN_FLIGHT}"

  # Default stripe policy for newly created workload files under output root.
  run_lustre_cmd lfs setstripe -c "${BASE_STRIPE_COUNT}" -S "${BASE_STRIPE_SIZE}" "${output_root}"
}

apply_baseline_settings

pipeline_cmd=( python3 -u "${PIPELINE_SCRIPT}" --options-csv "${OPTIONS_FILE}" )
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
