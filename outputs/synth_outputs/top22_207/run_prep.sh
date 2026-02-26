#!/usr/bin/env bash
set -euo pipefail
mkdir -p /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/ro /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/wo /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/meta
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
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_0.dat' 81738596352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_1.dat' 81738596352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_10.dat' 81335943168
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_11.dat' 81335943168
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_12.dat' 81201725440
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_13.dat' 81201725440
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_14.dat' 81201725440
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_15.dat' 81201725440
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_16.dat' 81201725440
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_17.dat' 81201725440
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_18.dat' 80933289984
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_19.dat' 80933289984
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_2.dat' 81738596352
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_20.dat' 80933289984
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_21.dat' 80933289984
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_22.dat' 80933289984
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_23.dat' 80933289984
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_24.dat' 80933289984
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_25.dat' 80933289984
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_26.dat' 80799072256
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_27.dat' 80799072256
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_28.dat' 80799072256
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_29.dat' 80799072256
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_3.dat' 81470160896
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_30.dat' 80799072256
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_31.dat' 80799072256
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_4.dat' 81470160896
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_5.dat' 81470160896
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_6.dat' 81470160896
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_7.dat' 81470160896
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_8.dat' 81470160896
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/rw/rw_9.dat' 81335943168
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/data/wo/wo_0.dat' 534857646080
truncate -s 0 '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top22_207/payload/meta/meta_only.dat'
