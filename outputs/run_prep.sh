#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$(dirname /mnt/hasanfs/repos/io_synthesizer/outputs/ior_shared_sequential_r_small.dat)"
mpiexec -hostfile ~/hfile -n 4 /mnt/hasanfs/bin/ior -a MPIIO -k -w -c -b 4294967296 -t 64k -o /mnt/hasanfs/repos/io_synthesizer/outputs/ior_shared_sequential_r_small.dat
mkdir -p "$(dirname /mnt/hasanfs/repos/io_synthesizer/outputs/ior_shared_sequential_r_large.dat)"
mpiexec -hostfile ~/hfile -n 4 /mnt/hasanfs/bin/ior -a MPIIO -k -w -c -b 4294967296 -t 128m -o /mnt/hasanfs/repos/io_synthesizer/outputs/ior_shared_sequential_r_large.dat
mkdir -p "$(dirname /mnt/hasanfs/repos/io_synthesizer/outputs/ior_shared_random_r_small.dat)"
mpiexec -hostfile ~/hfile -n 4 /mnt/hasanfs/bin/ior -a MPIIO -k -w -c -b 4294967296 -t 64k -o /mnt/hasanfs/repos/io_synthesizer/outputs/ior_shared_random_r_small.dat
mkdir -p "$(dirname /mnt/hasanfs/repos/io_synthesizer/outputs/ior_shared_random_r_large.dat)"
mpiexec -hostfile ~/hfile -n 4 /mnt/hasanfs/bin/ior -a MPIIO -k -w -c -b 4294967296 -t 128m -o /mnt/hasanfs/repos/io_synthesizer/outputs/ior_shared_random_r_large.dat
