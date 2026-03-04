# Outstanding Work Discussion Plan (Sequenced)

## Summary
Purpose: prepare a co-author discussion on outstanding items and campaign sizing before any implementation.

Sequence to discuss:
1. Re-clustering
2. Workload Pattern Scales
3. MPI Configs
4. Sampling Estimation (single iteration per sample)
5. Benchmark Validation

---

## 1) Re-clustering (First)
Previous clustering used Polaris Darshan data from **04/2024 to 04/2025**.  
Discussion focus is to refresh clustering with data extended through **03/2026**.

What needs to be discussed:
- Exact dataset window and inclusion rules for the refreshed corpus.
- Whether feature definitions stay unchanged or need updates before re-clustering.
- Clustering method/settings to keep comparability with prior results.
- How many representative workloads should be selected after refresh.
- How to quantify drift from old clusters to new clusters.

Output to agree in meeting:
- Final re-clustered representative workload set and drift summary.

---

## 2) Workload Pattern Scales (Processes + Data Volume Only)
For now, workload scaling will use:
- **Number of processes** with three levels: **low, medium, high**
- **Data volume** with three levels: **low, medium, high**

What needs to be discussed:
- Concrete low/medium/high values for process counts.
- Concrete low/medium/high values for data volume.
- Whether one global scale definition applies to all representative workloads.
- Whether any workload-specific exclusions are needed for infeasible scale points.
- Final scenario expansion rule: **3 x 3 = 9 scale points per workload**.

Output to agree in meeting:
- Final Low/Medium/High table for process counts and data volume.

---

## 3) MPI Configs
MPI parameters to include:
- `num_aggregators`
- `collective_buffer_size`
- `aggregators_per_client`

Assumption for estimation:
- Each MPI parameter has **6 discrete values**.

What needs to be discussed:
- The exact 6 values for each MPI parameter.
- Baseline/default MPI config for comparisons.
- Whether tuning is MPI-collective-only in the first study phase.
- Any constraints/compatibility rules across MPI parameters and rank counts.

Config-space implications to review:
- MPI-only space: `6 x 6 x 6 = 216`
- Existing Lustre space: `52,920`
- Joint Lustre+MPI space per scenario: `52,920 x 216 = 11,430,720`

Output to agree in meeting:
- Frozen MPI parameter value sets and baseline.

### 3.1) Platform-Scale Transfer Validation (Config Scale Generalization)
Current concern:
- Current exploration scale is limited on some knobs (for example, observed ceiling like `stripe_count = 16`).
- We need to validate if recommendations still hold when platform scale changes.

Two discussion options:
1. Explore in down-scale regime (for example, max `stripe_count = 8`), then execute at higher scale (for example, `stripe_count = 32`) and test if recommendations transfer.
2. Repeat the same transfer validation on the Perlmutter supercomputer.

What needs to be discussed:
- Exact train-scale vs test-scale boundaries per knob.
- Transfer success criteria for “recommendation still holds”.
- Whether Option 1 is mandatory before Option 2, or both run in parallel.

Output to agree in meeting:
- Final scale-transfer validation protocol across Polaris and Perlmutter.

### 3.2) Recommendation Matrix Strategy (MPI-IO Primary, Single Matrix)
Current direction:
- Build **one recommendation matrix**, focused on **MPI-IO**.
- Incorporate POSIX-only recommendations inside this same matrix space.

What needs to be discussed:
- How POSIX-only applications are represented in the MPI-IO matrix space.
- Whether POSIX-only mapping uses a constrained MPI profile (for example, fixed MPI control values) or a dedicated mode tag inside the same matrix.
- How to validate that POSIX-only recommendations are not degraded under a single MPI-IO-centric matrix.

Output to agree in meeting:
- Final POSIX-to-MPI matrix embedding strategy and validation criteria.

---

## 4) Sampling Estimation (Assuming Single Iteration Per Sample + Heuristic Pruning)
Assumptions:
- Each sampled config is executed once for this estimate.
- Search uses heuristic-based pruning to reduce collected samples.

Measured runtime basis from `/mnt/hasanfs/out_synth`:
- `25` `darshan_summary.csv` files analyzed.
- Mean `run_time` per config: **217.68 sec** (3.63 min).

Formulas:
- Pre-pruning runs: `Total runs = W x 9 x (N_warm + N_active)`
- Post-pruning runs: `Total runs_pruned = W x 9 x (N_warm + N_active) x keep_ratio`
- Runtime hours: `Total hours ~= runs x 217.68 / 3600`

Where:
- `W` = number of representative workloads after re-clustering.
- `N_warm + N_active` = sampled configs per scenario before pruning.
- `keep_ratio` = fraction of candidates kept by pruning heuristics.

Example pre-pruning projections for `W = 25`:
- `96` samples/scenario (`60 warm + 36 active`): `21,600` runs, `1,306` hours.
- `180` samples/scenario: `40,500` runs, `2,449` hours.
- `210` samples/scenario: `47,250` runs, `2,857` hours.

What needs to be discussed:
- Heuristic pruning design and guardrails so quality is not compromised.
- Reasonable warm-start + active budget before and after pruning.
- Allowed parallel concurrency for wallclock planning.

Output to agree in meeting:
- Final pruning policy and final sampling budget policy.

---

## 5) Benchmark Validation (Mandatory)
Required benchmark families:
- IO-500
- h5Bench
- MLPerf

What needs to be discussed:
- Which benchmark workloads/subtests from each suite are in scope.
- How benchmark runs map onto process/data-volume scale levels (low/medium/high).
- How transfer checks (down-scale to high-scale and Perlmutter) are represented in benchmark validation.
- Which tuned configs are validated (baseline + top-K policy).
- Validation metrics and pass criteria for publication claims.
- Required repeat policy for benchmark validation runs.

Output to agree in meeting:
- Final benchmark validation matrix and acceptance criteria.

---

## Final Decisions to Lock at End of Discussion
1. Re-clustering dataset and representative workload list (extended through 03/2026).
2. Low/Medium/High scale definition for process counts and data volume.
3. MPI parameter value sets for `num_aggregators`, `collective_buffer_size`, `aggregators_per_client`.
4. Platform scale-transfer validation protocol (down-scale to high-scale and Perlmutter path).
5. Single recommendation matrix policy (MPI-IO primary) and POSIX-only embedding strategy.
6. Heuristic pruning policy and warm-start + active sample budget under single-iteration assumptions.
7. IO-500/h5Bench/MLPerf validation matrix and evidence threshold for the paper.
