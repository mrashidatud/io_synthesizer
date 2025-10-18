#!/usr/bin/env bash
set -euo pipefail
mkdir -p /mnt/hasanfs/out_synth/payload/data/ro /mnt/hasanfs/out_synth/payload/data/rw /mnt/hasanfs/out_synth/payload/data/wo /mnt/hasanfs/out_synth/payload/meta
truncate -s 50570657792 /mnt/hasanfs/out_synth/payload/data/ro/ro_shared_0.dat
truncate -s 50570657792 /mnt/hasanfs/out_synth/payload/data/ro/ro_shared_1.dat
truncate -s 4096 /mnt/hasanfs/out_synth/payload/data/wo/wo_shared_0.dat
truncate -s 4096 /mnt/hasanfs/out_synth/payload/meta/meta_only.dat
