#!/usr/bin/env bash
set -euo pipefail
mkdir -p /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/rw /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/meta
truncate_with_fallback() {
  local path="$1"
  local requested="$2"
  local applied=""
  local attempts=("$requested" 17592186044415 8796093022207 4398046511103 2199023255551 1099511627775 549755813887)
  for s in "${attempts[@]}"; do
    [[ "$s" -le 0 ]] && continue
    if truncate -s "$s" "$path" 2>/dev/null; then
      applied="$s"
      break
    fi
  done
  if [[ -z "$applied" ]]; then
    echo "ERROR: truncate failed for $path (requested=$requested)" >&2
    return 1
  fi
  if [[ "$applied" != "$requested" ]]; then
    echo "WARN: truncate fallback for $path (requested=$requested applied=$applied)" >&2
  fi
}
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_0.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_1.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_10.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_11.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_12.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_13.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_14.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_15.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_16.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_17.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_18.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_19.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_2.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_20.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_21.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_22.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_23.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_24.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_25.dat' 74993328
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_26.dat' 74993228
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_27.dat' 74993228
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_28.dat' 74993228
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_29.dat' 74993228
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_3.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_4.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_5.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_6.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_7.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_8.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/ro/ro_9.dat' 74994352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/rw/rw_0.dat' 6174015488
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/rw/rw_1.dat' 6308233216
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/rw/rw_2.dat' 6308233216
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/rw/rw_3.dat' 6308233216
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/rw/rw_4.dat' 6308233216
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_0.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_1.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_10.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_11.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_12.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_13.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_14.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_15.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_16.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_17.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_18.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_19.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_2.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_20.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_21.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_22.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_23.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_24.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_25.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_26.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_27.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_28.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_3.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_4.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_5.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_6.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_7.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_8.dat' 1073741824
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/data/wo/wo_9.dat' 1073741824
truncate -s 0 '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top18_7/payload/meta/meta_only.dat'
