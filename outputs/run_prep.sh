#!/usr/bin/env bash
set -euo pipefail
mkdir -p /mnt/hasanfs/out_synth/payload/data/ro /mnt/hasanfs/out_synth/payload/data/rw /mnt/hasanfs/out_synth/payload/data/wo /mnt/hasanfs/out_synth/payload/meta
truncate -s 137663725568 '/mnt/hasanfs/out_synth/payload/data/ro/ro_0.dat'
truncate -s 137221668288 '/mnt/hasanfs/out_synth/payload/data/ro/ro_1.dat'
truncate -s 3414160 '/mnt/hasanfs/out_synth/payload/data/wo/wo_0.dat'
truncate -s 0 '/mnt/hasanfs/out_synth/payload/meta/meta_only.dat'
