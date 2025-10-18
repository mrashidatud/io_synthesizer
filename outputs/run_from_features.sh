#!/usr/bin/env bash
set -euo pipefail
bash /mnt/hasanfs/out_synth/run_prep.sh
mpiexec -n 1 -genv LD_PRELOAD /mnt/hasanfs/darshan-3.4.7/darshan-runtime/install/lib/libdarshan.so /mnt/hasanfs/bin/mpi_synthio --plan /mnt/hasanfs/out_synth/payload/plan.csv --io-api posix --meta-api posix --collective none
