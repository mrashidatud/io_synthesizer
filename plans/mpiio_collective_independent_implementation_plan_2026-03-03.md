# MPI-IO Collective + Independent Implementation Plan

Date: 2026-03-03  
Scope: core runtime path in `/mnt/hasanfs/io_synthesizer/scripts/features2synth_opsaware.py` and `/mnt/hasanfs/io_synthesizer/scripts/mpi_synthio.c`, plus config/CSV/parser surfaces that feed runtime options.

## 1. Objective

Implement MPI-IO execution support in the synthesizer with a binary collective control, and add ROMIO-ready tuning hooks for near-term collective performance experiments:

- `--collective no`: MPI-IO independent behavior (each rank performs rank-local MPI-IO operations via `MPI_File_read_at` / `MPI_File_write_at`)
- `--collective yes`: MPI-IO collective behavior for shared/legacy rows (using `MPI_File_read_at_all` / `MPI_File_write_at_all`) with unique-row owner fallback to independent MPI-IO

Keep legacy POSIX mode unchanged for backward compatibility.

## 2. User-Constrained Behavioral Rules

From the design direction in this thread:

1. We expose only two `--collective` options: `yes` and `no`.
2. `--collective no` means rank-local MPI-IO (not direct POSIX calls in the MPI-IO harness path).
3. `--collective yes` enables collective MPI-IO for shared/legacy rows.
4. Legacy POSIX behavior must remain as-is.
5. `--collective` only materializes for `--io-api mpiio`; with `--io-api posix`, it is ignored (effective mode is POSIX path, `collective=no`).
6. We should be ROMIO-aware now: support passing MPI-IO hints so collective tuning can begin without redesign.
7. Near-term MPI tuning knobs to expose are:
   - `num_aggregators`
   - `collective_buffer_size`
   - `aggregators_per_client`

## 3. Current-State Findings (Grounded)

### 3.1 Planner (`features2synth_opsaware.py`)

- Already emits runner args `--io-api` and `--collective`.
- Uses raw `mpi_collective_mode` value from features JSON.
- Existing exemplar JSONs include boolean forms (`true` / `false`) for `mpi_collective_mode`, which are currently stringified to `True` / `False` in runner command and not parsed by C harness.

### 3.2 Synthesizer (`mpi_synthio.c`)

- Parses `--io-api posix|mpiio` and `--collective yes|no`.
- Data path currently always executes POSIX `pread/pwrite` logic.
- Parsed `ioapi` and `cm` are currently not used in execution logic.
- Metadata path is POSIX-only and should remain unchanged in this phase.

### 3.3 ROMIO hint keys available in current MPICH/ROMIO stack

For the planned MPI knobs, use explicit ROMIO mappings:

- `num_aggregators` -> `cb_nodes`
- `collective_buffer_size` -> `cb_buffer_size`
- `aggregators_per_client` -> `cb_config_list` (using wildcard layout form `*:<K>`)

These keys are available in the current MPICH ROMIO build and should be treated as best-effort hints.

### 3.4 Current pre-execution knob application model (reference behavior)

Current Lustre knobs (`stripe_*`, `osc_*`, `mdc_*`) are applied before each run in the execution pipelines:

- `io_recommender/runner_real.py` via `_apply_knobs_before_run(...)`
- warm/active/validation orchestrators via `apply_lustre_knobs(...)`

This means config-space knobs are selected per trial and applied immediately before `run_from_features.sh` execution.  
MPI tuning knobs should follow this same lifecycle.

### 3.5 Current validation gap: MPI mode checks and metrics are under-specified

Observed gaps relative to campaign goals:

- mode correctness is not currently tied to explicit Darshan MPIIO counter checks.
- feature-fidelity comparison exists (`analyze_darshan_merged.py`) but is effectively POSIX-derived today.
- recommender scoring defaults are POSIX-centric (`POSIX_agg_perf_by_slowest` and POSIX-byte/time fallback), which is not ideal for MPI-IO tuning impact studies.

## 4. Target Runtime Semantics Matrix

| io_api | collective arg | Effective data backend | Notes |
|---|---|---|---|
| `posix` | `yes` or `no` | POSIX (existing) | collective option is ignored in POSIX mode |
| `mpiio` | `no` | MPI-IO independent | rank-local MPI-IO calls |
| `mpiio` | `yes` | MPI-IO collective for shared/legacy rows; independent fallback for unique-owner execution | avoids collective mismatch/deadlock for unique rows |

## 5. Planner + Pre-Execution Control Surface Changes

## 5.1 Normalize I/O mode fields before runner generation

Add a small normalization utility near runner generation:

- Normalize `io_api` to lowercase and validate: `posix|mpiio`.
- Normalize `mpi_collective_mode` from:
  - booleans: `true -> yes`, `false -> no`
  - strings: `yes|no` (case-insensitive)
  - compatibility aliases accepted as input: `collective -> yes`, `independent|none -> no`
- Compatibility rules:
  - if `io_api=posix`, force `collective=no` (ignore requested collective setting)
  - if `io_api=mpiio`, interpret `collective=no` as independent MPI-IO behavior

## 5.2 Use normalized values everywhere runner identity is emitted

Apply normalized values to:

- generated command line in `run_from_features.sh`
- output naming suffix (`..._io_<...>_coll_<...>.darshan`)
- notes text for clarity/debugability

## 5.3 Put MPI knobs in recommender config space (same as Lustre knobs)

These three knobs should be added to `io_recommender/config.yaml` under `parameters` (+ `baseline`) so sampling selects them per configuration:

- `num_aggregators`
- `collective_buffer_size`
- `aggregators_per_client`

Important correction:

- do **not** add these knobs to workload feature JSONs (`inputs/features.json`, exemplar feature summary CSVs), because they are execution-time system knobs, not workload descriptors.

## 5.4 Add per-run pass-through from pre-execution env to `mpi_synthio`

Because one `run_from_features.sh` is reused across many sampled configs, MPI knob values must be injected at run time (per trial), not baked into plan generation.

Required approach:

- update generated `run_from_features.sh` to append optional extra args for `mpi_synthio` from environment-derived values.
- recommended env keys set by runner/orchestrator before each run:
  - `SYNTH_MPI_NUM_AGGREGATORS`
  - `SYNTH_MPI_COLLECTIVE_BUFFER_SIZE`
  - `SYNTH_MPI_AGGREGATORS_PER_CLIENT`
- script converts these env vars into `mpi_synthio` args:
  - `--mpi-num-aggregators`
  - `--mpi-collective-buffer-size`
  - `--mpi-aggregators-per-client`

Compatibility rules:

- `io_api=posix`: ignore MPI knob env vars for execution.
- `io_api=mpiio` + `collective=no`: accept values but mark inactive.

## 5.5 Pre-execution knob builder in runners/orchestrators

Add a helper that mirrors `apply_lustre_knobs(...)` lifecycle and prepares MPI env overrides per config before invoking `run_from_features.sh`.

Target files:

- `io_recommender/runner_real.py`
- `orchestrators/warm_start_sampling_orchestrator/warm_start_pipeline.py`
- `orchestrators/active_train_sampling_orchestrator/active_sampling_pipeline.py`
- `io_recommender/validation/run_closest_workload_validation.py`

Behavior:

- derive env from sampled config for each run
- clear/unset MPI env keys when not applicable to avoid leakage across trials
- keep recorded metadata of requested knob values in observation rows/logs

## 5.6 Module-aware analysis and metric selection for MPI validation

To make validation meaningful for MPI-IO modes:

- add module-aware metric preference in recommender/validation paths:
  - for `io_api=mpiio`, prefer `MPIIO_agg_perf_by_slowest` when present
  - fallback chain: `agg_perf_by_slowest` -> `POSIX_agg_perf_by_slowest` -> bytes/time fallback
- extend analysis output artifacts to include MPIIO counters (if present), at minimum:
  - `MPIIO_INDEP_READS`, `MPIIO_INDEP_WRITES`
  - `MPIIO_COLL_READS`, `MPIIO_COLL_WRITES`
  - `MPIIO_BYTES_READ`, `MPIIO_BYTES_WRITTEN`
  - `MPIIO_F_READ_TIME`, `MPIIO_F_WRITE_TIME`, `MPIIO_F_META_TIME`
- for feature-fidelity checks in MPI mode:
  - keep existing POSIX-derived comparison for backward compatibility
  - add MPI-aware comparison mode that can derive `pct_*` using MPIIO counters when POSIX counters are absent or when `io_api=mpiio` is explicitly requested for validation

## 6. Synthesizer Changes (`mpi_synthio.c`)

## 6.1 Wire runtime mode into execution dispatch

Refactor `run_plan(...)` signature to accept parsed runtime mode:

- `io_api_t ioapi`
- `col_mode_t cm`

Call `run_plan(fp, ioapi, cm)` from `main()`.

## 6.2 Keep POSIX path exactly intact

Do not alter existing POSIX data and metadata behavior when `io_api=posix`.

## 6.3 Add MPI-IO file-handle cache

Introduce MPI-IO cache analogous to POSIX fd cache:

- key: `path`, access intent (read/write), communicator kind
  - world communicator for shared/legacy rows
  - self communicator for owner-only unique fallback
- value: `MPI_File fh`, `MPI_Offset file_size`, validity flag
- open flags:
  - write: `MPI_MODE_RDWR | MPI_MODE_CREATE`
  - read: `MPI_MODE_RDONLY`
- retrieve size via `MPI_File_get_size`
- finalize with `MPI_File_close` on all valid entries

## 6.4 Abstract single-operation write/read primitive

Create backend operation wrappers:

- POSIX wrapper (existing `do_prw`)
- MPI independent wrapper:
  - `MPI_File_write_at` / `MPI_File_read_at`
- MPI collective wrapper:
  - `MPI_File_write_at_all` / `MPI_File_read_at_all`

Operation arguments:

- `(offset, buffer, xfer, is_write)`

Keep `xfer` unit as bytes, using `MPI_BYTE` datatype.

## 6.5 Reuse existing offset-generation semantics

Retain current generation and quotas for:

- phase partitioning (`consec`, `seq`, `random`)
- file alignment and mem alignment quotas
- global-to-local quota mapping via `n_ops` + `start_idx`
- shared/unique sharding policy

Implementation strategy:

- factor the existing core loop body in `exec_phase_data_local` into backend-selectable operation issue calls
- avoid changing math that defines offsets/ordering

## 6.6 Collective safety and deadlock prevention

For collective MPI-IO mode:

- Apply collectives only on rows where all ranks participate (shared/legacy execution branch).
- Ensure every participating rank issues the same number/order of collective calls:
  - compute per-rank local op count
  - global `max_ops = MPI_Allreduce(max(local_ops))`
  - loop for `i in [0, max_ops)`:
    - if rank has real op at `i`, use real offset/xfer
    - otherwise issue zero-byte collective call (`count=0`, any valid offset, e.g. 0)

This prevents collective call-count mismatch.

## 6.7 Unique-row behavior under `--collective=yes`

Keep existing unique owner-wave scheduler.

When runtime asks for `--collective=yes`:

- unique-owner row executes via MPI-IO independent on owner rank only
- non-owner ranks do not open or touch file

Rationale:

- preserves unique semantics
- avoids forcing artificial collectives on non-participating ranks
- matches existing ownership model

## 6.8 Keep metadata POSIX-only in this milestone

No change to `do_posix_meta()` path.  
`--meta-api` remains effectively POSIX-only.

## 6.9 Logging updates

Add explicit backend label in per-phase logs:

- `POSIX`
- `MPIIO_INDEP`
- `MPIIO_COLL`
- `MPIIO_INDEP_UNIQUE_FALLBACK`

Also log normalized mode on rank 0 at startup.

## 6.10 ROMIO tuning hooks for three MPI knobs

Add dedicated runtime args consumed by `mpi_synthio` (delivered via per-run env expansion in run script):

- `--mpi-num-aggregators <N>`
- `--mpi-collective-buffer-size <BYTES>`
- `--mpi-aggregators-per-client <K>`

Map them to ROMIO `MPI_Info` keys:

- `num_aggregators` -> `cb_nodes=N`
- `collective_buffer_size` -> `cb_buffer_size=BYTES`
- `aggregators_per_client` -> `cb_config_list=*:<K>`

Applicability policy:

- apply only when `io_api=mpiio` and `collective=yes` (collective tuning path)
- if `io_api=posix`, ignore all three
- if `io_api=mpiio` and `collective=no`, accept inputs but log they are inactive for independent path

Runtime implementation details:

- create `MPI_Info info` before each `MPI_File_open`
- set only provided keys via `MPI_Info_set`
- pass `info` to `MPI_File_open`; free with `MPI_Info_free`
- keep malformed/invalid numeric values non-fatal (warn + skip)
- keep CLI parsing as the final source of truth; env variables are only a transport mechanism from orchestration layer to CLI args

Observability requirements:

- log MPI library string via `MPI_Get_library_version`
- after open, call `MPI_File_get_info` and log effective values for:
  - `cb_nodes`
  - `cb_buffer_size`
  - `cb_config_list`
- log requested vs effective values to make ROMIO precedence/normalization explicit

## 7. CLI and Validation Rules

## 7.1 Planner-side normalization (preferred enforcement)

Planner should emit only normalized combos to runner.

## 7.2 Synthesizer-side defensive parsing

In C parser:

- keep strict accepted values for `--collective` (`yes|no`)
- if `io_api=posix`, ignore `--collective` and force effective `no`
- map legacy aliases (`collective`, `independent`, `none`, booleans) to `yes|no` before runtime dispatch when encountered in upstream config
- if unknown value appears, print warning and fall back to safe default:
  - `io_api`: `posix`
  - `collective`: `no`
- for MPI tuning knobs:
  - parse `mpi_num_aggregators` as positive integer
  - parse `mpi_collective_buffer_size` as positive byte count (accept plain integer bytes only in first implementation)
  - parse `mpi_aggregators_per_client` as positive integer
  - invalid values: warn and treat as unset
  - apply to `MPI_Info` only for `io_api=mpiio` and `collective=yes`
- planner/workload feature schema remains unchanged for MPI tuning knobs; they come from per-run config space

## 8. Test Plan

## 8.1 Planner unit tests (`tests/test_features2synth_opsaware.py`)

Add tests for normalization:

1. boolean `mpi_collective_mode=true` -> runner contains `--collective yes`
2. boolean `mpi_collective_mode=false` -> runner contains `--collective no`
3. `io_api=posix` forces effective `--collective no` (ignore requested `yes`)
4. `io_api=mpiio` + legacy `mpi_collective_mode=none|independent` emits `--collective no`
5. run-script contains optional env-to-CLI pass-through for MPI knobs (per-run injection path)
6. run-script generation remains workload-static; per-config MPI values are not baked into workload features

## 8.2 Build/compile checks

- `mpicc -O3 -Wall -Wextra -o mpi_synthio mpi_synthio.c -lm`

## 8.3 Runtime smoke tests (2+ ranks)

1. POSIX regression: existing POSIX workload still runs unchanged.
2. MPIIO independent shared workload: no deadlock, successful completion.
3. MPIIO collective shared workload: no deadlock, successful completion.
4. Mixed shared+unique workload in collective mode:
   - shared rows use collective path
   - unique rows use owner independent fallback
   - no deadlock
5. MPIIO collective with hints:
   - pass the three knobs together
   - verify run completes and logs requested/effective `cb_nodes`, `cb_buffer_size`, `cb_config_list` + MPI library version
6. MPIIO collective knob isolation:
   - vary one knob at a time (`num_aggregators`, `collective_buffer_size`, `aggregators_per_client`)
   - verify each one appears independently in requested/effective hint logs
7. Pre-execution lifecycle parity with Lustre knobs:
   - run two configs back-to-back with different MPI knob values
   - verify second run does not inherit stale MPI knob values from first run

## 8.4 Functional sanity checks

- Verify total executed ops still equals planned `sum(n_ops)` semantics.
- Verify phase logs show expected backend labels by row type/flag.
- Verify MPI knob hints are ignored in POSIX mode and do not alter POSIX execution path.
- Verify `collective=no` runs remain deadlock-free and do not claim active collective tuning.

## 8.5 Goal-Oriented Validation Gates (Explicit Pass/Fail)

This section defines sufficiency criteria for the three campaign questions.

### Goal A: Can we produce both independent and collective I/O from feature-based patterns?

Required evidence (same workload family, `io_api=mpiio`):

1. `collective=no` run:
   - no deadlock
   - MPIIO independent counters are active (`MPIIO_INDEP_READS + MPIIO_INDEP_WRITES > 0`) when those counters are available
2. `collective=yes` run (shared-row workload):
   - no deadlock
   - MPIIO collective counters are active (`MPIIO_COLL_READS + MPIIO_COLL_WRITES > 0`) when available
3. mixed shared+unique run with `collective=yes`:
   - shared path uses collective branch
   - unique path remains owner-only independent fallback

Pass rule:

- all three scenarios complete successfully, and counter/log evidence is consistent with expected mode behavior.

### Goal B: Can we synthesize the same features as input?

Required evidence:

1. Run `analyze_darshan_merged.py` comparison for each validation workload.
2. Produce and archive `pct_compare_report.txt`.
3. In MPI mode, run module-aware comparison path (POSIX-compatible + MPIIO-aware) and document which module basis was used.

Pass rule:

- no core `pct_*` features outside tolerance and at least 90% of compared `pct_*` keys within tolerance.
- core features include:
  - `pct_reads`, `pct_writes`, `pct_byte_reads`, `pct_byte_writes`
  - `pct_read_0_100K`, `pct_read_100K_10M`, `pct_read_10M_1G_PLUS`
  - `pct_write_0_100K`, `pct_write_100K_10M`, `pct_write_10M_1G_PLUS`
  - `pct_seq_reads`, `pct_seq_writes`, `pct_consec_reads`, `pct_consec_writes`
  - `pct_rw_switches`, `pct_file_not_aligned`, `pct_mem_not_aligned`
  - `pct_shared_files`, `pct_unique_files`, `pct_read_only_files`, `pct_read_write_files`, `pct_write_only_files`

### Goal C: Can MPI tuning knobs be changed dynamically, and do they impact outcomes?

Required evidence:

1. Dynamic changeability:
   - run at least two back-to-back configs with different MPI knob values
   - logs show requested and effective (`MPI_File_get_info`) values changed accordingly
2. Impact test:
   - for each knob, perform one-factor A/B against baseline (`collective=yes`, same workload, same scale)
   - run at least 5 repeats per point
   - compare median selected metric using MPI-aware metric preference

Pass rule:

- dynamic application: requested/effective values differ as expected without leakage across runs.
- measurable impact: at least one knob setting change yields >=5% median metric shift from baseline (or a justified threshold if workload noise dictates otherwise), with repeat-level consistency documented.

## 9. Risk Register and Mitigations

1. **Collective mismatch deadlock**
   - mitigation: equalized collective call loop with zero-count calls.
2. **Behavior drift from current offset semantics**
   - mitigation: preserve existing offset/quota logic; change only backend issue call.
3. **Legacy JSON boolean incompatibility**
   - mitigation: planner normalization of booleans and compatibility mapping.
4. **Unique rows in collective mode ambiguity**
   - mitigation: explicit independent-owner fallback contract.
5. **ROMIO/environment variability**
   - mitigation: log MPI library version + applied hints and treat unsupported hints as best-effort.
6. **MPI knob interaction ambiguity (`cb_nodes` vs `cb_config_list`)**
   - mitigation: always log requested vs effective values from `MPI_File_get_info`; keep parser deterministic and non-fatal.
7. **Per-run env leakage across sampled configs**
   - mitigation: explicitly set/unset MPI env keys on every run invocation.
8. **False-negative impact conclusions due to POSIX-centric metrics in MPI mode**
   - mitigation: enforce MPI-aware metric preference and record metric source used per run.
9. **Feature-fidelity ambiguity in MPI mode**
   - mitigation: require module-basis annotation (POSIX vs MPIIO) in feature-comparison artifacts.

## 10. Implementation Sequence (Decision-Complete)

1. Add planner normalization helper and wire it into runner generation + naming.
2. Add planner unit tests for normalization mappings.
3. Refactor synthesizer `run_plan` signature to receive `ioapi/cm`.
4. Implement MPI-IO handle cache and close lifecycle.
5. Add backend op wrappers (POSIX/indep/collective).
6. Refactor data execution to dispatch op wrappers while preserving offset logic.
7. Implement collective safe-loop equalization for shared rows.
8. Implement unique-row independent fallback when collective requested.
9. Add backend-mode logging.
10. Add three MPI knobs to recommender config space (`io_recommender/config.yaml` parameters + baseline).
11. Add per-run MPI env builder in runner/orchestrators (same lifecycle as Lustre knob application).
12. Update `run_from_features.sh` template to convert MPI env vars into optional `mpi_synthio` CLI args.
13. Add runtime parser support in `mpi_synthio.c` for the three knobs.
14. Map knobs to ROMIO hints via `MPI_Info_set` on `MPI_File_open`.
15. Add requested-vs-effective hint logging using `MPI_File_get_info` + MPI library version.
16. Add MPI-aware metric preference and metric-source logging in recommender analysis path.
17. Add explicit goal-gate validation scripts/checks for mode correctness + feature fidelity + tuning impact.
18. Compile + run smoke tests + verify no POSIX regression.

## 11. Out of Scope (for this milestone)

- Meta API conversion to MPI-IO metadata equivalents.
- Plan CSV schema redesign.
- Re-derivation of features from MPIIO module counters in analysis pipeline.
- Automatic ROMIO hint search/optimization policy.
- Modeling MPI tuning knobs as workload-feature inputs (not needed for this design).

## 12. Acceptance Criteria

Implementation is complete when:

1. `--io-api mpiio --collective no` executes full workload via MPI-IO independent data calls.
2. `--io-api mpiio --collective yes` executes shared rows with collective MPI-IO and mixed unique rows without deadlock.
3. `--io-api posix` behavior and outputs are unchanged from current baseline.
4. Planner handles boolean/string collective inputs and emits normalized `yes|no` runtime args deterministically.
5. `num_aggregators`, `collective_buffer_size`, and `aggregators_per_client` are sampled from recommender config space and applied per run (pre-execution) via env -> CLI pass-through.
6. Runtime logs include requested and effective ROMIO hint values (`MPI_File_get_info`) plus MPI library version.
7. POSIX mode remains unchanged; MPI knob fields are ignored for execution in POSIX path.
8. Independent vs collective correctness is demonstrated by explicit mode-evidence gates (counter/log based) on shared and mixed workloads.
9. Feature-fidelity gate passes with documented tolerance outcomes and module basis.
10. Dynamic MPI knob changes are shown to be effective per run and at least one knob demonstrates measurable impact under repeat-based A/B validation.

## 13. File-Level Modification Map for MPI Knobs

Core runtime:

- `scripts/features2synth_opsaware.py`: keep collective normalization and emit run script with MPI env-to-CLI pass-through hooks.
- `scripts/mpi_synthio.c`: parse new CLI args, apply `MPI_Info` hints, log requested/effective values.

Recommender config surfaces:

- `io_recommender/config.yaml`
- optional override paths that already feed recommender config loading (`cfg.*` options in orchestrators)

Parser/plumbing surfaces:

- `orchestrators/warm_start_sampling_orchestrator/warm_start_pipeline.py`
- `orchestrators/active_train_sampling_orchestrator/active_sampling_pipeline.py`
- `io_recommender/runner_real.py`
- `io_recommender/validation/run_closest_workload_validation.py`
- `analysis/scripts_analysis/analyze_darshan_recommender.py` (MPI-aware metric preference + metric-source logging)
- `analysis/scripts_analysis/analyze_darshan_merged.py` and/or `analysis/scripts_analysis/generate_features.py` (module-aware feature-fidelity path for MPI mode)

Tests:

- `tests/test_features2synth_opsaware.py` (collective normalization + run script pass-through presence)
- new or extended runtime tests for `mpi_synthio.c` hint parsing and logging behavior
- runner/orchestrator tests for per-run MPI env synthesis and leakage prevention
- validation tests/checklists for the 3 goal gates (mode correctness, feature fidelity, tuning impact)
