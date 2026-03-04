# MPI-IO Collective/Independent Validation Report (2026-03-03)

## Scope
Validated on real workload patterns using distributed execution across all client machines from:
- `/mnt/hasanfs/io_synthesizer/remote_orchestration/machine_ssh_map.csv`

Requested validations:
1. Collective synthesis and produced features for `top23` pattern
2. Independent synthesis and produced features for `top11` pattern
3. One collective run with tuned MPI knobs for `top23`

## Environment and orchestration
- Execution host: `node0.mrashid2-291681.dirr-pg0.utah.cloudlab.us`
- MPI: MPICH 4.3.0 (`mpiexec` from `/custom-install/hpc-tools/mpich-4.3.0/install/bin/mpiexec`)
- Synthesizer binary rebuilt and installed:
  - `/mnt/hasanfs/io_synthesizer/scripts/mpi_synthio`
  - `/mnt/hasanfs/bin/mpi_synthio`
- Hostfile generated from client entries in machine map:
  - `/mnt/hasanfs/io_synthesizer/outputs/mpiio_validation_2026-03-03_actual/client_hosts.txt`
- Distributed launch mechanism: `HYDRA_HOST_FILE=<hostfile>`

## Common run settings
- `cap_total_gib=16`
- `nprocs=64`
- `io_api=mpiio`
- `meta_api=posix`
- `meta_scope=data_files`
- planner policies: `optimizer=lexicographic`, `seq_policy=nonconsec_strict`, `alignment_policy=structure_preserving`

## Test matrix
| Case | Pattern | mpi_collective_mode | MPI knobs |
|---|---|---|---|
| case1_top23_collective_default | top23_193 | yes | none |
| case2_top11_independent_default | top11_249 | no | none |
| case3_top23_collective_tuned | top23_193 | yes | `num_aggregators=8`, `collective_buffer_size=67108864`, `aggregators_per_client=2` |

Artifacts root:
- `/mnt/hasanfs/io_synthesizer/outputs/mpiio_validation_2026-03-03_actual`

## Distributed execution proof
Each case includes `host_distribution.txt`, showing balanced rank placement across all four clients:
- `16 node0...`
- `16 node1...`
- `16 node2...`
- `16 node3...`

## Results summary
| Case | Elapsed (s) | MPIIO bytes written | MPIIO indep writes | MPIIO coll writes | MPIIO agg_perf_by_slowest |
|---|---:|---:|---:|---:|---:|
| case1_top23_collective_default | 8.462 | 23471587328 | 0 | 9536 | 3557.605296 |
| case2_top11_independent_default | 11.417 | 4027056128 | 9362 | 0 | 424.411637 |
| case3_top23_collective_tuned | 7.263 | 23471587328 | 0 | 9536 | 4973.507813 |

Tuned collective performance impact (case3 vs case1):
- `+39.8%` in `MPIIO_agg_perf_by_slowest`

## Validation by requested test

### 1) Collective pattern synthesis + feature check (top23)
Case: `case1_top23_collective_default`

Behavior evidence:
- `run.log` shows: `INFO: mode io_api=mpiio collective=yes`
- Backend lines are collective only: `backend=MPIIO_COLL`
- Darshan counters: `MPIIO_COLL_WRITES=9536`, `MPIIO_INDEP_WRITES=0`

Synthesis shape (plan):
- Data rows: 5 (all `|shared|`, all writes)
- Mix of `seq|shared|` and `consec|shared|` rows with 256KiB/1MiB/4MiB transfers

Feature fidelity:
- `pct_compare_report.txt`: compared=32, exact=24, within tol=4, outside tol=4
- Module basis used: `MPIIO`
- Main outside-tolerance diffs:
  - `pct_seq_writes: 1.0 -> 0.0`
  - `pct_io_access: 0.78 -> 1.0`
  - `pct_meta_open_access: 0.11 -> 0.0`
  - `pct_meta_sync_access: 0.11 -> 0.0`

### 2) Independent pattern synthesis + feature check (top11)
Case: `case2_top11_independent_default`

Behavior evidence:
- `run.log` shows: `INFO: mode io_api=mpiio collective=no`
- Backend lines are independent only: `backend=MPIIO_INDEP`
- Darshan counters: `MPIIO_INDEP_WRITES=9362`, `MPIIO_COLL_WRITES=0`

Synthesis shape (plan):
- Data rows: 2 (both `seq|shared|`, writes)
- Transfers: 256KiB and 1MiB

Feature fidelity:
- `pct_compare_report.txt`: compared=32, exact=27, within tol=3, outside tol=2
- Module basis used: `MPIIO`
- Outside-tolerance diffs:
  - `pct_file_not_aligned: 1.0 -> 0.0`
  - `pct_seq_writes: 1.0 -> 0.0`

### 3) Collective with tuned knobs (top23)
Case: `case3_top23_collective_tuned`

Behavior evidence:
- `run.log` shows knob activation:
  - `INFO: MPI hint inputs ... active=yes`
  - requested hints:
    - `cb_nodes=8`
    - `cb_buffer_size=67108864`
    - `cb_config_list=*:2`
  - effective hints reported with same values (accepted by ROMIO in this run)
- Backend and counters remain collective-only:
  - `backend=MPIIO_COLL`
  - `MPIIO_COLL_WRITES=9536`, `MPIIO_INDEP_WRITES=0`

Feature fidelity:
- Same profile as case1 (collective top23), including same outside-tolerance fields.

## Assessment

### What is working properly
- Collective vs independent mode selection works correctly in synthesized execution.
- MPI-IO path is used for both modes (not POSIX fallback), and Darshan MPIIO counters match expected mode.
- Runs are correctly orchestrated across all client machines listed in the map.
- MPI tuning knobs are applied dynamically via env passthrough and visibly affect effective ROMIO hints/performance.

### Gaps observed
- Feature reproduction is not fully faithful for MPIIO-derived sequential/alignment and metadata-access fractions in these patterns.
- Repeated mismatch on `pct_seq_writes` (input 1.0, produced 0.0) indicates the current MPIIO feature extraction basis does not currently capture/derive this as intended.

## Key artifact paths
- Root: `/mnt/hasanfs/io_synthesizer/outputs/mpiio_validation_2026-03-03_actual`
- Case folders:
  - `/mnt/hasanfs/io_synthesizer/outputs/mpiio_validation_2026-03-03_actual/case1_top23_collective_default`
  - `/mnt/hasanfs/io_synthesizer/outputs/mpiio_validation_2026-03-03_actual/case2_top11_independent_default`
  - `/mnt/hasanfs/io_synthesizer/outputs/mpiio_validation_2026-03-03_actual/case3_top23_collective_tuned`
- Per-case key files:
  - `run.log`
  - `host_distribution.txt`
  - `darshan_summary.csv`
  - `darshan_features_updated.json`
  - `pct_compare_report.txt`
  - `case_summary.json`
- Rollup table:
  - `/mnt/hasanfs/io_synthesizer/outputs/mpiio_validation_2026-03-03_actual/summary_table.json`
