# Improvement Plan (Remaining Work) - Pipeline Style

Date: 2026-03-02  
Scope: only pending work, written in the same flow as the recommender pipeline.

## What Is Already Done
- Leakage fix in active history metrics (`oracle_mode`, `oracle_data_source`).
- Held-out validation framework (`evaluation_report.json`).
- Adaptive candidate mode (`enumerated` vs `sampled`) with logging.
- Replicate support + uncertainty stats + robust ranking fields.
- Contribution reporting (`contribution_report.json`) + markdown summary.

## End-to-End Flow (Remaining)

`Objective -> Prior Build -> Warm-Start Bootstrap -> Active Sampling -> Stop Policy -> Validation/Report`

---

## Step 1: Set the Right Objective (TODO)

### Goal
Switch optimization target from throughput-only to end-to-end behavior (I/O + metadata time).

### Why
Throughput can look good while metadata cost is high. We want the objective to match user-facing runtime.

### Plan
- Add objective config:
  - `objective.metric: throughput | e2e_time | speedup`
  - `objective.direction: maximize | minimize`
- Compute `e2e_time` from Darshan summary:
  - `POSIX_F_READ_TIME + POSIX_F_WRITE_TIME + POSIX_F_META_TIME`
- Convert to model-friendly gain:
  - time-minimization form: `gain = baseline_e2e - current_e2e`
  - optional relative form: `(baseline_e2e / current_e2e) - 1`
- Record objective metadata in summary/evaluation/contribution artifacts.

### Note on your intuition
Your intuition is correct: for ranking configs **within the same workload**, scale differences across workloads are less problematic. Still, for regression mode we should prefer normalized targets.

### Done when
- Pipeline runs with `objective.metric=e2e_time`.
- Held-out report can compare throughput objective vs e2e objective.

---

## Step 2: Build Prior From Observations (TODO)

### Goal
Use prior observations to guide search, instead of hand-written empirical rules.

### Why
This gives sample efficiency without brittle manual heuristics.

### Plan
Build two priors:
- **Global prior**: learned from all sampled workloads.
- **Local prior**: learned from top-K similar workloads to current workload.

Combine them:
- `prior_score = alpha * global_prior + (1 - alpha) * local_prior`
- make `alpha` adaptive:
  - early (low local evidence): higher global weight
  - later (more local evidence): higher local weight

Use prior in two modes:
- soft bias (default): re-rank/down-weight low-prior candidates
- hard prune (optional): filter candidates below `P(gain>0)` threshold

### Done when
- Candidate count drops with prior enabled.
- Held-out quality does not regress beyond tolerance.
- Logs show global/local contribution and before/after candidate counts.

---

## Step 3: Prior-Guided Warm-Start Workload Ordering (TODO)

### Goal
Choose which workloads to sample first so early observations are maximally reusable.

### Why
If we explore "influential" workloads first, later workloads get stronger local prior sooner.

### Plan
- Build workload similarity graph from workload features.
- Compute centrality over **yet-unexplored** workloads.
- Use adaptive ordering (recomputed as exploration proceeds), not one-time static ordering.
- Prioritize workloads with highest coverage influence.

Config examples:
- `workload_ordering: static | centrality_first | adaptive_centrality`
- `transfer.k_neighbors`, `transfer.weight`

### Done when
- Ordering artifact is generated (auditable per round).
- Adaptive ordering reaches same/better quality with fewer runs than static ordering.

---

## Step 4: Active Sampling Loop (TODO Refinement)

### Goal
Keep existing explore/exploit/diverse selection, but make it prior-guided and transfer-aware.

### Clear flow per active iteration
1. Choose next workload(s) using adaptive centrality/priority.  
2. Build candidate pool.  
3. Apply prior bias (global + local).  
4. Run explore/exploit/diversity selection on biased pool.  
5. Execute selected configs.  
6. Optional replicates for top candidates.  
7. Update posterior/model and priors.

### Why this is not duplicate work
- Prior-guided constraints (Step 2) act at **config candidate** level.
- Adaptive centrality ordering (Step 3) acts at **workload scheduling** level.
- Both are needed, but they should be presented as one coherent pipeline.

### Done when
- Active loop logs include workload priority, prior usage, candidate filtering stats, and replicate stats.

---

## Step 5: Better Stop Policy (TODO)

### Goal
Stop based on evidence, not only fixed iteration count.

### Plan
Add composite stopping:
- `min_iters` guard,
- robust gain plateau patience,
- uncertainty collapse threshold,
- optional budget cap (`max_runs`/wall-clock).

Stop reason must be logged explicitly.

### Done when
- Runs end with deterministic reason codes.
- Lower average run count at comparable held-out quality.

---

## Step 6: Validate and Decide (TODO)

### Goal
Prove that changes improve generalization and operational value.

### Plan
For each milestone run:
- held-out validation (`evaluation_report.json`): top1/topk regret, hit@k, ndcg@k, CI
- contribution report (`contribution_report.json`): active value-add trajectory
- markdown summary for decision-making

### Done when
- One run folder clearly answers:
  - Did quality improve?
  - Did sample cost drop?
  - Did active/prior guidance actually add value?

---

## Suggested Implementation Order
1. Step 1 (objective = e2e)
2. Step 2 (observation-based priors)
3. Step 3 (adaptive workload ordering)
4. Step 4 (wire into active loop)
5. Step 5 (stop policy)
6. Step 6 (validation + decision checks)

This order keeps the objective correct first, then improves search efficiency, then controls runtime budget.

---

## Quick Validation Protocol (No Full Execution)
For each step, run only short stub/small pilot checks:
- schema checks: `summary.json`, `evaluation_report.json`, `contribution_report.json`
- deterministic seed check
- expected new fields/logs present
- no full-scale workload execution
