#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/hasanfs/io_synthesizer"
BIN_DIR="/mnt/hasanfs/bin"
OUT_ROOT="/mnt/hasanfs/out_synth"
INPUT_DIR="${ROOT}/inputs/exemplar_jsons"   # per your request
FEATS_SCRIPT="${ROOT}/scripts/features2synth_opsaware.py"

# Defaults (override on CLI: --cap 512 --nprocs 4 --io posix --meta posix --coll none)
CAP_TOTAL_GIB=512
NPROCS=""
IO_API=""
META_API=""
MPI_COLL_MODE=""
FORCE_BUILD=0

# Parse optional overrides
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cap) CAP_TOTAL_GIB="$2"; shift 2;;
    --nprocs) NPROCS="$2"; shift 2;;
    --io) IO_API="$2"; shift 2;;
    --meta) META_API="$2"; shift 2;;
    --coll) MPI_COLL_MODE="$2"; shift 2;;
    --inputs) INPUT_DIR="$2"; shift 2;;
    --force-build) FORCE_BUILD=1; shift 1;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

timestamp() { date +"%Y%m%d_%H%M%S"; }
LOG="${OUT_ROOT}/pipeline_$(timestamp).log"
mkdir -p "${OUT_ROOT}"

# tee everything to a logfile
exec > >(tee -a "$LOG") 2>&1

echo "=== STEP 0: Build mpi_synthio ==="
if [[ $FORCE_BUILD -eq 1 || ! -x "${BIN_DIR}/mpi_synthio" ]]; then
  pushd "${ROOT}/scripts" >/dev/null
  make clean && make
  mv -f mpi_synthio "${BIN_DIR}/"
  popd >/dev/null
  echo "✅ Build complete; moved mpi_synthio to ${BIN_DIR}"
else
  echo "⏭️  mpi_synthio already exists at ${BIN_DIR}/mpi_synthio — skipping build (use --force-build to rebuild)"
fi

# Gather JSONs
mapfile -t JSONS < <(find "${INPUT_DIR}" -maxdepth 1 -type f -name "*.json" | sort)
if [[ ${#JSONS[@]} -eq 0 ]]; then
  echo "No JSON inputs found in ${INPUT_DIR}"
  exit 0
fi

echo "=== STEP 1..N: Plan, run, validate, analyze for each input ==="
for J in "${JSONS[@]}"; do
  JSON_ABS="$(realpath "$J")"
  JSON_NAME="$(basename "$J")"
  JSON_BASE="${JSON_NAME%.json}"
  echo ""
  echo "---- Processing: ${JSON_NAME} ----"

  # Plan (generate prep/run scripts)
  echo "[Plan] features2synth_opsaware.py for ${JSON_NAME}"
  cd "${ROOT}"
  CMD=( python3 "${FEATS_SCRIPT}" --features "${JSON_ABS}" --cap-total-gib "${CAP_TOTAL_GIB}" )
  [[ -n "${NPROCS}" ]] && CMD+=( --nprocs "${NPROCS}" )
  [[ -n "${IO_API}" ]] && CMD+=( --io-api "${IO_API}" )
  [[ -n "${META_API}" ]] && CMD+=( --meta-api "${META_API}" )
  [[ -n "${MPI_COLL_MODE}" ]] && CMD+=( --mpi-collective-mode "${MPI_COLL_MODE}" )

  printf 'Running: '; printf '%q ' "${CMD[@]}"; echo
  "${CMD[@]}"

  # Locate generated run script (prefer new per-JSON layout; fall back to legacy)
  CAND1="${OUT_ROOT}/${JSON_BASE}/run_from_features.sh"
  CAND2="${OUT_ROOT}/run_from_features.sh"
  if [[ -f "${CAND1}" ]]; then
    RUN_SH="${CAND1}"
    RUN_ROOT="$(dirname "${RUN_SH}")"
  elif [[ -f "${CAND2}" ]]; then
    RUN_SH="${CAND2}"
    RUN_ROOT="$(dirname "${RUN_SH}")"
    echo "⚠️  Using legacy run script location: ${RUN_SH}"
  else
    echo "❌ Could not find run_from_features.sh in ${CAND1} or ${CAND2}"
    continue
  fi

  # Extract DARSHAN_LOGFILE='…' from the run script (quoted with single quotes)
  EXPECTED="$(awk -F"'" '/^export[[:space:]]+DARSHAN_LOGFILE=/{print $2}' "${RUN_SH}" || true)"

  # If the exact darshan file already exists, skip the run
  if [[ -n "${EXPECTED}" && -f "${EXPECTED}" ]]; then
    echo "⏭️  Found existing Darshan artifact:"
    echo "    ${EXPECTED}"
    echo "    Skipping execution and proceeding to analysis."
  else
    # Run it
    echo "[Run] ${RUN_SH}"
    bash "${RUN_SH}"

    echo "[Validate] Sleep 10s to allow Darshan to flush…"
    sleep 10

    if [[ -n "${EXPECTED}" ]]; then
      if [[ -f "${EXPECTED}" ]]; then
        echo "✅ Found Darshan: ${EXPECTED}"
      else
        echo "⚠️  Expected Darshan not found: ${EXPECTED}"
        echo "    Present .darshan files in ${RUN_ROOT}:"
        ls -l "${RUN_ROOT}"/*.darshan 2>/dev/null || true
        continue
      fi
    else
      # Fallback: try to find a single .darshan file if ENV var wasn't set
      shopt -s nullglob
      found=( "${RUN_ROOT}"/*.darshan )
      shopt -u nullglob
      if [[ ${#found[@]} -eq 1 && -f "${found[0]}" ]]; then
        EXPECTED="${found[0]}"
        echo "ℹ️  Using discovered darshan file: ${EXPECTED}"
      else
        echo "❌ Could not determine Darshan artifact to analyze."
        continue
      fi
    fi
  fi

  # Analyze (merged script)
  echo "[Analyze] merged analysis for ${JSON_NAME}"
  ANALYZE="${ROOT}/analysis/scripts_analysis/analyze_darshan_merged.py"
  if [[ ! -f "${ANALYZE}" ]]; then
    echo "❌ Missing ${ANALYZE}. Please place the merged script there."
    continue
  fi
  python3 "${ANALYZE}" \
      --darshan "${EXPECTED}" \
      --input-json "${JSON_ABS}" \
      --outdir "${RUN_ROOT}"

  echo "---- Done: ${JSON_NAME} ----"
done

echo ""
echo "🎉 Pipeline complete. Log saved to ${LOG}"
