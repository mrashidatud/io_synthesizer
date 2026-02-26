#!/usr/bin/env bash
set -euo pipefail
mkdir -p /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/rw /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/wo /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/meta
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
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_0.dat' 133680857088
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_1.dat' 133680857088
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_10.dat' 133680857088
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_11.dat' 133680857088
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_12.dat' 133546639360
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_13.dat' 133546639360
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_14.dat' 133546639360
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_15.dat' 133546639360
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_16.dat' 133546639360
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_17.dat' 133546639360
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_18.dat' 133546639360
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_19.dat' 133546639360
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_2.dat' 133680857088
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_20.dat' 133546639360
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_21.dat' 133546639360
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_22.dat' 133546639360
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_23.dat' 133546639360
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_24.dat' 133546639360
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_25.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_26.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_27.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_28.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_29.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_3.dat' 133680857088
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_30.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_31.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_32.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_33.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_34.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_35.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_36.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_37.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_38.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_39.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_4.dat' 133680857088
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_40.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_41.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_42.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_43.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_44.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_45.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_46.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_47.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_48.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_49.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_5.dat' 133680857088
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_50.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_51.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_52.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_53.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_54.dat' 133412421632
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_6.dat' 133680857088
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_7.dat' 133680857088
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_8.dat' 133680857088
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/ro/ro_9.dat' 133680857088
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/rw/rw_0.dat' 298366009344
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/wo/wo_0.dat' 6710886400
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/wo/wo_1.dat' 6710886400
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/wo/wo_2.dat' 6576671772
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/wo/wo_3.dat' 6576671772
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/wo/wo_4.dat' 6576671772
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/wo/wo_5.dat' 6576671772
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/wo/wo_6.dat' 6576671772
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/data/wo/wo_7.dat' 6576671772
truncate -s 0 '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top13_237/payload/meta/meta_only.dat'
