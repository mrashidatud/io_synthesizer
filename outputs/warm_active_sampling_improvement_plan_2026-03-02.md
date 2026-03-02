# Improvement Plan: Warm-Start + Active-Sampling Recommender

Date: 2026-03-02  
Goal: Convert current high in-sample quality into reliable, measurable, and operationally robust recommendation quality.

## Plan Overview

This plan addresses 5 identified shortcomings:
1. Evaluation leakage in active history metrics
2. Missing held-out validation protocol
3. Candidate search strategy near full-space boundary
4. Single-shot active measurements (noise sensitivity)
5. Missing active-vs-warm contribution reporting

## Phase 1: Fix Evaluation Metrics (Leakage Removal)

### Objective
Make `regret_at_3`, `hit_at_3`, and `ndcg_at_3` meaningful and non-trivial.

### Tasks
- Change oracle construction in active pipeline to use a fixed reference set (not current observed best).
- Support two oracle modes:
  - `oracle_mode=heldout` (preferred)
  - `oracle_mode=warm_only` (fallback)
- Add explicit metadata in `summary.json`:
  - `oracle_mode`
  - `oracle_data_source`

### Deliverables
- Code updates in active pipeline.
- Backward-compatible output schema update.

### Acceptance Criteria
- History metrics are no longer constant/perfect by construction.
- A/B rerun shows non-trivial metric curves over iterations.

## Phase 2: Add True Validation Protocol

### Objective
Measure generalization quality instead of in-sample ranking only.

### Tasks
- Add evaluation split options:
  - `split_mode=heldout_configs_per_pattern`
  - `split_mode=heldout_patterns`
  - optional `temporal_split` for resumed runs
- Generate `evaluation_report.json` with:
  - top-1/top-3 regret vs held-out oracle
  - hit@k and NDCG@k on held-out set
  - confidence intervals (bootstrap)

### Deliverables
- Evaluation module and CLI flags.
- Saved validation report artifact.

### Acceptance Criteria
- Reproducible held-out results from fixed seed.
- Report includes per-workload and aggregate metrics.

## Phase 3: Candidate Search Strategy Upgrade

### Objective
Reduce miss-risk from sampled candidate pools when full enumeration is feasible.

### Tasks
- Add adaptive candidate mode:
  - if total space <= `enum_threshold_hard`, use full enumeration.
- Set `enum_threshold_hard` >= 52,920 for this environment (or auto by memory/time guard).
- Log candidate mode per iteration/pattern (`enumerated` vs `sampled`).

### Deliverables
- Candidate pool selector update.
- Runtime diagnostics in logs.

### Acceptance Criteria
- For current 52,920-space config, mode resolves to enumerated without failure.
- Runtime overhead remains within agreed budget.

## Phase 4: Add Replicate Policy for Active Top Candidates

### Objective
Improve ranking reliability under run-to-run variability.

### Tasks
- Add replicate policy after each active iteration:
  - re-run top-1 and top-2 candidates for each pattern (configurable).
- Store replicate index and aggregate stats per config:
  - mean, std, CV, 95% CI
- Rank by robust score (e.g., lower confidence bound or mean-penalized-std).

### Deliverables
- Replicate execution path.
- Aggregated scoring logic.

### Acceptance Criteria
- Active top-K becomes stable across reruns.
- Ranking report includes uncertainty columns.

## Phase 5: Active Contribution and Operational Reporting

### Objective
Make improvement attribution explicit and decision-ready.

### Tasks
- Add `contribution_report.json` with:
  - `% top-K from active`
  - `best_active_vs_best_warm` delta per pattern
  - `new_best_found_iter`
  - cumulative improvement trajectory
- Add markdown summary generation from artifacts.

### Deliverables
- Contribution report artifact.
- Auto-generated concise markdown summary for experiment runs.

### Acceptance Criteria
- One command produces summary showing active value-add clearly.
- Decision-makers can tell whether active sampling paid off per workload.

## Suggested Execution Order

1. Phase 1 (metric correctness)  
2. Phase 2 (validation framework)  
3. Phase 3 (candidate strategy)  
4. Phase 4 (replicates + robust ranking)  
5. Phase 5 (reporting and attribution)

## Estimated Effort (Engineering)

- Phase 1: 0.5-1 day
- Phase 2: 1-2 days
- Phase 3: 0.5-1 day
- Phase 4: 1-2 days
- Phase 5: 0.5-1 day

Total: **3.5 to 7 days** depending on test/runtime cycles.

## Risks and Mitigations

- Risk: Increased runtime from enumeration/replicates  
  Mitigation: make both configurable and budget-guarded.

- Risk: Held-out oracle quality depends on split design  
  Mitigation: support multiple split modes and report both.

- Risk: Schema changes break downstream tooling  
  Mitigation: keep backward-compatible fields and version artifacts.

## Immediate Next Step

Implement Phase 1 first, then rerun one short active experiment (e.g., 4 iterations) to verify metrics are no longer degenerate before moving to broader changes.
