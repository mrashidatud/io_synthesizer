#!/usr/bin/env bash
set -euo pipefail
bash /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top15_50/run_prep.sh
export DARSHAN_LOGFILE="${DARSHAN_LOGFILE:-/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top15_50/top15_50_cap_1_nproc_4_io_mpiio_meta_posix_coll_False.darshan}"
mpiexec -n 4 -genv LD_PRELOAD /mnt/hasanfs/darshan-3.4.7/darshan-runtime/install/lib/libdarshan.so -genv DARSHAN_LOGFILE "$DARSHAN_LOGFILE" /mnt/hasanfs/bin/mpi_synthio --plan /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top15_50/payload/plan.csv --io-api mpiio --meta-api posix --collective False
