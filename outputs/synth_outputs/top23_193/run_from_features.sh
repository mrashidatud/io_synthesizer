#!/usr/bin/env bash
set -euo pipefail
bash /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top23_193/run_prep.sh
export DARSHAN_LOGFILE="${DARSHAN_LOGFILE:-/Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top23_193/top23_193_cap_1_nproc_512_io_mpiio_meta_posix_coll_True.darshan}"
mpiexec -n 512 -genv LD_PRELOAD /mnt/hasanfs/darshan-3.4.7/darshan-runtime/install/lib/libdarshan.so -genv DARSHAN_LOGFILE "$DARSHAN_LOGFILE" /mnt/hasanfs/bin/mpi_synthio --plan /Users/user/dirlab/repos/io_synthesizer/outputs/audit_top25_after_fix_v8_round_rank2_verif/top23_193/payload/plan.csv --io-api mpiio --meta-api posix --collective True
