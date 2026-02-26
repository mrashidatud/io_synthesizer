#!/usr/bin/env bash
set -euo pipefail
mkdir -p /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/ro /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/wo /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/meta
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
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_0.dat' 43620761700
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_1.dat' 43620761700
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_10.dat' 43352326144
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_11.dat' 43352326144
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_12.dat' 43352326144
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_13.dat' 43352326144
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_14.dat' 43352326144
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_15.dat' 43352326144
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_16.dat' 43352326144
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_17.dat' 43352326144
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_18.dat' 43218108416
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_19.dat' 43218108416
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_2.dat' 43620761700
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_20.dat' 43218108416
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_21.dat' 43083890688
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_22.dat' 43083890688
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_23.dat' 43083890688
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_24.dat' 43083890688
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_3.dat' 43620761700
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_4.dat' 43620761700
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_5.dat' 43486544072
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_6.dat' 43486544072
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_7.dat' 43486544072
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_8.dat' 43352326344
truncate_with_fallback '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/data/rw/rw_9.dat' 43352326144
truncate -s 0 '/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top19_83/payload/meta/meta_only.dat'
