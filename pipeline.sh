#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/hasanfs/io_synthesizer"
BIN_DIR="/mnt/hasanfs/bin"
OUT_ROOT="/mnt/hasanfs/out_synth"
INPUT_DIR="${ROOT}/inputs/exemplar_jsons"
FEATS_SCRIPT="${ROOT}/scripts/features2synth_opsaware.py"

# Defaults
CAP_TOTAL_GIB=512
NPROCS_OVERRIDE=""
NPROCS_CAP=64
FORCE_BUILD=0
DELETE_DARSHAN=0

# -------- Parse options (flags first, then filters/ranges) --------
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cap) CAP_TOTAL_GIB="$2"; shift 2;;
    --nprocs) NPROCS_OVERRIDE="$2"; shift 2;;
    --nprocs-cap) NPROCS_CAP="$2"; shift 2;;
    --inputs) INPUT_DIR="$2"; shift 2;;
    --force-build) FORCE_BUILD=1; shift 1;;
    --delete-darshan) DELETE_DARSHAN=1; shift 1;;
    --) shift; break;;
    -*)
      echo "Unknown option: $1" >&2; exit 1;;
    *)
      ARGS+=("$1"); shift 1;;
  esac
done
# Anything after -- is also treated as filters
if [[ $# -gt 0 ]]; then ARGS+=("$@"); fi
FILTERS=( "${ARGS[@]}" )

# -------- Basic validation --------
if [[ -n "${NPROCS_OVERRIDE}" && ! "${NPROCS_OVERRIDE}" =~ ^[0-9]+$ ]]; then
  echo "Error: --nprocs must be an integer" >&2; exit 1
fi
if [[ -n "${NPROCS_CAP}" && ! "${NPROCS_CAP}" =~ ^[0-9]+$ ]]; then
  echo "Error: --nprocs-cap must be an integer" >&2; exit 1
fi

# Timestamps
timestamp() { date +"%Y%m%d_%H%M%S"; }      # for filenames
ts() { date +"%F %T"; }                     # for log lines

STAMP="$(timestamp)"
LOG="${OUT_ROOT}/pipeline_${STAMP}.log"
mkdir -p "${OUT_ROOT}"

# tee everything to a logfile
exec > >(tee -a "$LOG") 2>&1

# ---- init master comparison rollups (new) ----
MASTER_TXT="${OUT_ROOT}/pct_compare_master_${STAMP}.txt"
MASTER_CSV="${OUT_ROOT}/pct_compare_master_${STAMP}.csv"
echo "# pct_* comparison rollup (all cases)  @ ${STAMP}" > "${MASTER_TXT}"
if [[ ! -f "${MASTER_CSV}" ]]; then
  echo "json,status,key,input,produced,abs_diff" > "${MASTER_CSV}"
fi

echo "$(ts)  === STEP 0: Build mpi_synthio ==="
if [[ $FORCE_BUILD -eq 1 || ! -x "${BIN_DIR}/mpi_synthio" ]]; then
  pushd "${ROOT}/scripts" >/dev/null
  make clean && make
  mv -f mpi_synthio "${BIN_DIR}/"
  popd >/dev/null
  echo "$(ts)  ✅ Build complete; moved mpi_synthio to ${BIN_DIR}"
else
  echo "$(ts)  ⏭️  mpi_synthio already exists at ${BIN_DIR}/mpi_synthio — skipping build (use --force-build to rebuild)"
fi

# -------- Collect JSONs to run (supports ranges & explicit names) --------
declare -a JSONS=()
declare -A SEEN=()  # to dedupe
add_json() {
  local path="$1"
  if [[ -f "$path" ]]; then
    local rp; rp="$(realpath "$path")"
    if [[ -z "${SEEN[$rp]:-}" ]]; then
      JSONS+=( "$rp" ); SEEN[$rp]=1
    fi
  fi
}

if [[ ${#FILTERS[@]} -gt 0 ]]; then
  echo "$(ts)  ℹ️  Limiting to filters: ${FILTERS[*]}"
  for tok in "${FILTERS[@]}"; do
    if [[ "$tok" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      # Range like 1-10  → include all topN_*.json for N in [start..end]
      start="${BASH_REMATCH[1]}"; end="${BASH_REMATCH[2]}"
      if (( start > end )); then tmp="$start"; start="$end"; end="$tmp"; fi
      for ((n=start; n<=end; n++)); do
        shopt -s nullglob
        for f in "${INPUT_DIR}/top${n}_"*.json; do add_json "$f"; done
        shopt -u nullglob
      done
    elif [[ "$tok" =~ ^[0-9]+$ ]]; then
      # Single number like 7 → top7_*.json
      n="$tok"
      shopt -s nullglob
      for f in "${INPUT_DIR}/top${n}_"*.json; do add_json "$f"; done
      shopt -u nullglob
    else
      # Base name like top8_48 (with or without .json)
      base="${tok%.json}.json"
      add_json "${INPUT_DIR}/${base}"
    fi
  done
else
  # Default: all *.json in INPUT_DIR
  while IFS= read -r f; do add_json "$f"; done \
    < <(find "${INPUT_DIR}" -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 -I{} realpath "{}" | sort)
fi

if [[ ${#JSONS[@]} -eq 0 ]]; then
  echo "No JSON inputs selected/found (INPUT_DIR=${INPUT_DIR})."
  exit 0
fi

echo "$(ts)  === STEP 1..N: Plan, run, validate, analyze for each input ==="
for JSON_ABS in "${JSONS[@]}"; do
  JSON_NAME="$(basename "$JSON_ABS")"
  JSON_BASE="${JSON_NAME%.json}"
  echo ""
  echo "$(ts)  ---- Processing: ${JSON_NAME} ----"

  # Determine nprocs to pass (cap at NPROCS_CAP if provided or present in input JSON)
  DESIRED_NPROCS=""
  if [[ -n "${NPROCS_OVERRIDE}" ]]; then
    if (( NPROCS_OVERRIDE > NPROCS_CAP )); then DESIRED_NPROCS="${NPROCS_CAP}"; else DESIRED_NPROCS="${NPROCS_OVERRIDE}"; fi
  else
    JSON_NPROCS="$(python3 - "$JSON_ABS" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    n=d.get("nprocs","")
    if isinstance(n,int): print(n)
    elif isinstance(n,str) and n.isdigit(): print(int(n))
    else: print("")
except Exception:
    print("")
PY
)"
    if [[ -n "${JSON_NPROCS}" ]]; then
      if (( JSON_NPROCS > NPROCS_CAP )); then DESIRED_NPROCS="${NPROCS_CAP}"; else DESIRED_NPROCS="${JSON_NPROCS}"; fi
    fi
  fi

  # Plan (generate prep/run scripts) — always force posix/posix/none
  echo "$(ts)  [Plan] features2synth_opsaware.py for ${JSON_NAME}"
  cd "${ROOT}"
  CMD=( python3 "${FEATS_SCRIPT}" --features "${JSON_ABS}" --cap-total-gib "${CAP_TOTAL_GIB}" )
  CMD+=( --io-api posix --meta-api posix --mpi-collective-mode none )
  [[ -n "${DESIRED_NPROCS}" ]] && CMD+=( --nprocs "${DESIRED_NPROCS}" )

  printf '%s  Running: ' "$(ts)"; printf '%q ' "${CMD[@]}"; echo
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
    echo "$(ts)  ⚠️  Using legacy run script location: ${RUN_SH}"
  else
    echo "$(ts)  ❌ Could not find run_from_features.sh in ${CAND1} or ${CAND2}"
    continue
  fi

  # Optional: delete existing Darshan files for this case before running
  if [[ ${DELETE_DARSHAN} -eq 1 ]]; then
    echo "$(ts)  [Pre-run] --delete-darshan enabled → removing any existing Darshan files in ${RUN_ROOT}"
    rm -f "${RUN_ROOT}"/*.darshan 2>/dev/null || true
  fi

  # Extract DARSHAN_LOGFILE from the run script (used for validation)
  EXPECTED="$(awk -F"'" '/^export[[:space:]]+DARSHAN_LOGFILE=/{print $2}' "${RUN_SH}" || true)"

  # If the exact .darshan already exists AND we didn't delete it, skip execution
  if [[ ${DELETE_DARSHAN} -eq 0 && -n "${EXPECTED}" && -f "${EXPECTED}" ]]; then
    echo "$(ts)  ⏭️  Found existing Darshan artifact:"
    echo "    ${EXPECTED}"
    echo "    Skipping execution and proceeding to analysis."
  else
    echo "$(ts)  [Run] ${RUN_SH}"
    bash "${RUN_SH}"

    echo "$(ts)  [Validate] Sleep 10s to allow Darshan to flush…"
    sleep 10

    if [[ -n "${EXPECTED}" ]]; then
      if [[ -f "${EXPECTED}" ]]; then
        echo "$(ts)  ✅ Found Darshan: ${EXPECTED}"
      else
        echo "$(ts)  ⚠️  Expected Darshan not found: ${EXPECTED}"
        echo "    Present .darshan files in ${RUN_ROOT}:"
        ls -l "${RUN_ROOT}"/*.darshan 2>/dev/null || true
        # Cleanup subdirs in payload before moving on
        PAYLOAD_DIR="${RUN_ROOT}/payload"
        if [[ -d "${PAYLOAD_DIR}" ]]; then
          echo "$(ts)  [Cleanup] Removing subdirectories inside ${PAYLOAD_DIR}"
          find "${PAYLOAD_DIR}" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
        fi
        continue
      fi
    else
      # Fallback: try to find a single .darshan if ENV var wasn't set
      shopt -s nullglob
      found=( "${RUN_ROOT}"/*.darshan )
      shopt -u nullglob
      if [[ ${#found[@]} -eq 1 && -f "${found[0]}" ]]; then
        EXPECTED="${found[0]}"
        echo "$(ts)  ℹ️  Using discovered darshan file: ${EXPECTED}"
      else
        echo "$(ts)  ❌ Could not determine Darshan artifact to analyze."
        # Cleanup subdirs in payload before moving on
        PAYLOAD_DIR="${RUN_ROOT}/payload"
        if [[ -d "${PAYLOAD_DIR}" ]]; then
          echo "$(ts)  [Cleanup] Removing subdirectories inside ${PAYLOAD_DIR}"
          find "${PAYLOAD_DIR}" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
        fi
        continue
      fi
    fi
  fi

  # Analyze (merged script) — tolerant of non-zero to keep loop going
  echo "$(ts)  [Analyze] merged analysis for ${JSON_NAME}"
  ANALYZE="${ROOT}/analysis/scripts_analysis/analyze_darshan_merged.py"
  if [[ ! -f "${ANALYZE}" ]]; then
    echo "$(ts)  ❌ Missing ${ANALYZE}. Please place the merged script there."
    ANALYZE_RC=127
  else
    set +e
    python3 "${ANALYZE}" \
        --darshan "${EXPECTED}" \
        --input-json "${JSON_ABS}" \
        --outdir "${RUN_ROOT}"
    ANALYZE_RC=$?
    set -e
  fi

  if [[ ${ANALYZE_RC} -ne 0 ]]; then
    echo "$(ts)  ⚠️  Analysis returned non-zero (${ANALYZE_RC}) for ${JSON_NAME}. Continuing with next input."
  fi

  # -------- NEW: Append per-case report to master rollups --------
  REPORT="${RUN_ROOT}/pct_compare_report.txt"
  if [[ -f "${REPORT}" ]]; then
    {
      echo ""
      echo "=== ${JSON_BASE} ==="
      cat "${REPORT}"
    } >> "${MASTER_TXT}"

    # Parse details into CSV: status ∈ {within,outside}
    # Lines look like: "  - pct_key: <iv> vs <pv> (Δ=<d>)"
    awk -v json="${JSON_BASE}" '
      BEGIN{status=""}
      /^Within tolerance/ {status="within"; next}
      /^Outside tolerance/ {status="outside"; next}
      # skip headers/summary lines
      /^Pct-field comparison report/ || /^Time:/ || /^Input JSON:/ || /^Produced JSON:/ || /^Total pct_/ || /^Exact matches:/ || /^Within / || /^Differences/ {next}
      # detail rows
      /^  - / {
        match($0, /^  - ([^:]+):[[:space:]]*([^[:space:]]+)[[:space:]]+vs[[:space:]]+([^[:space:]]+)[[:space:]]+\(Δ=([^)]+)\)/, m)
        if (m[1] != "" && status != "") {
          # json,status,key,input,produced,abs_diff
          printf("%s,%s,%s,%s,%s,%s\n", json, status, m[1], m[2], m[3], m[4]);
        }
      }
    ' "${REPORT}" >> "${MASTER_CSV}"
  fi

  # -------- Post-analysis cleanup: remove only subdirectories inside payload --------
  PAYLOAD_DIR="${RUN_ROOT}/payload"
  if [[ -d "${PAYLOAD_DIR}" ]]; then
    echo "$(ts)  [Cleanup] Removing subdirectories inside ${PAYLOAD_DIR}"
    find "${PAYLOAD_DIR}" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
  fi

  echo "$(ts)  ---- Done: ${JSON_NAME} ----"
done

echo ""
echo "🧾 Master comparison (txt): ${MASTER_TXT}"
echo "🧾 Master comparison (csv): ${MASTER_CSV}"
echo "🎉 Pipeline complete. Log saved to ${LOG}"
