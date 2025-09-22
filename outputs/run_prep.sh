#!/usr/bin/env bash
set -euo pipefail

mkdir -p "/mnt/hasanfs/synth_from_features/mdtree"
mkdir -p "$(dirname /mnt/hasanfs/synth_from_features/ior_shared_sequential_r_small.dat)"
mpiexec -hostfile ~/hfile -n 10 /mnt/hasanfs/bin/ior -a MPIIO -k -w -c -b 309264384 -t 64k -o /mnt/hasanfs/synth_from_features/ior_shared_sequential_r_small.dat
mkdir -p "$(dirname /mnt/hasanfs/synth_from_features/ior_shared_sequential_r_large.dat)"
mpiexec -hostfile ~/hfile -n 10 /mnt/hasanfs/bin/ior -a MPIIO -k -w -c -b 402653184 -t 128m -o /mnt/hasanfs/synth_from_features/ior_shared_sequential_r_large.dat
mkdir -p "$(dirname /mnt/hasanfs/synth_from_features/ior_shared_random_r_small.dat)"
mpiexec -hostfile ~/hfile -n 10 /mnt/hasanfs/bin/ior -a MPIIO -k -w -c -b 309264384 -t 64k -o /mnt/hasanfs/synth_from_features/ior_shared_random_r_small.dat
mkdir -p "$(dirname /mnt/hasanfs/synth_from_features/ior_shared_random_r_large.dat)"
mpiexec -hostfile ~/hfile -n 10 /mnt/hasanfs/bin/ior -a MPIIO -k -w -c -b 402653184 -t 128m -o /mnt/hasanfs/synth_from_features/ior_shared_random_r_large.dat
