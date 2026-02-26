#!/usr/bin/env bash
set -euo pipefail
mkdir -p /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/ro /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/rw /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/meta
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
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/ro/ro_0.dat' 11676942536
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_0.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_1.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_10.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_11.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_12.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_13.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_14.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_15.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_16.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_17.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_18.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_19.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_2.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_20.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_21.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_22.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_23.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_24.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_25.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_26.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_27.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_28.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_29.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_3.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_30.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_31.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_32.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_33.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_34.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_35.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_36.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_37.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_38.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_39.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_4.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_40.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_41.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_42.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_43.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_44.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_45.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_46.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_47.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_48.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_49.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_5.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_50.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_51.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_52.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_53.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_54.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_55.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_56.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_57.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_58.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_59.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_6.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_60.dat' 8589934592
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_61.dat' 8589934592
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_62.dat' 8589934592
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_7.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_8.dat' 8724152320
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/data/wo/wo_9.dat' 8724152320
truncate -s 0 '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top7_28/payload/meta/meta_only.dat'
