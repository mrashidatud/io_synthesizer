# Warm-Start and Active-Sampling Report

Date: 2026-03-02  
Scope: Review of collected samples under `/mnt/hasanfs/samples`, active-sampling method, and recommendation quality for 4 trained workloads.

## 1) Current State of Collection and Sample Observations

### 1.1 Collection Summary

- Warm-start observations: **522**
- Active-sampling observations: **144**
- Total observations used in active pipeline summary: **666**
- Trained workload patterns: **4** (`top1_101`, `top2_20`, `top3_18`, `top4_192`)

Collection windows from observation timestamps:
- Warm-start: **2026-02-24T14:41:52** to **2026-02-26T18:33:07** (~51.85 hours)
- Active-sampling: **2026-02-26T20:02:36** to **2026-02-27T06:07:15** (~10.08 hours)

### 1.2 Per-Workload Collection Coverage

| Workload | Warm runs | Active runs | Total runs | Unique configs (warm) | Unique configs (active) | Unique configs (total) |
|---|---:|---:|---:|---:|---:|---:|
| top1_101 | 174 | 36 | 210 | 58 | 36 | 94 |
| top2_20  | 116 | 36 | 152 | 58 | 36 | 94 |
| top3_18  | 116 | 36 | 152 | 58 | 36 | 94 |
| top4_192 | 116 | 36 | 152 | 58 | 36 | 94 |

Notes:
- Warm-start config set size is **58**, generated from `warm_start_target=60` (actual produced count 58).
- Warm-start pairwise coverage is **100%**: **569/569** parameter-value pairs covered.
- Active added **36 new configs per workload** (no config-id overlap with warm in this run).

### 1.3 Active Iteration Structure

- Active iterations configured: **12**
- Batch per iteration: **3**
- Patterns: **4**
- Expected active rows: `12 x 3 x 4 = 144` (matches observed)
- Every iteration has exactly 12 runs (3 per workload)

### 1.4 Metric Distribution Snapshot (`agg_perf_by_slowest`, MiB/s)

Warm-start min/median/max:
- top1_101: **65.88 / 968.66 / 1246.86**
- top2_20: **55.96 / 859.99 / 1131.09**
- top3_18: **61.79 / 922.25 / 1167.53**
- top4_192: **44.77 / 921.54 / 1230.68**

Active min/median/max:
- top1_101: **66.98 / 1124.72 / 1159.72**
- top2_20: **171.89 / 1076.74 / 1124.31**
- top3_18: **221.68 / 1135.13 / 1169.51**
- top4_192: **133.12 / 1068.20 / 1117.23**

Key observation:
- Active found a new best for **top3_18** (+1.98 MiB/s over warm best), but did not exceed warm best for top1_101, top2_20, or top4_192.

### 1.5 Run-to-Run Stability (Warm repeated runs)

All warm configs have repeated runs (2x or 3x depending on workload), enabling variance checks.

Median coefficient of variation (CV) across repeated warm runs:
- top1_101: **0.45%**
- top2_20: **0.55%**
- top3_18: **0.77%**
- top4_192: **0.35%**

Worst-case range/mean outliers do exist (up to ~17.5% for one `top2_20` config), so confidence intervals are still needed for top-rank decisions.

---

## 2) How Active Sampling Is Done

### 2.1 Initialization

1. Read recommender config and option overrides.
2. Load warm observations per selected workload.
3. Compute baseline performance per workload from baseline config.
4. Build initial training observations with gain = `perf - baseline_perf`.

### 2.2 Model

- Ensemble of **6** models.
- Mode: **ranking** (`LightGBM Ranker`, LambdaRank objective).
- Feature vector = workload features + encoded config vector.

### 2.3 Candidate Generation

Per workload per iteration:
- Build candidate pool from untested configs.
- Since total space is **52,920** (> enum threshold 50,000), it uses a sampled/mutated pool (`max_pool=12,000`) instead of full enumeration.
- Uses top observed configs as mutation anchors plus random fill.

### 2.4 Hybrid Batch Selection (size 3)

For each workload and iteration:
1. **Exploit pick**: `argmax(mu)`
2. **Explore pick**: `argmax(mu + beta*sigma - lambda*redundancy)` in UCB mode
3. **Diversity pick**: max distance from tested + selected set

Active params used:
- `iterations=12`
- `batch_per_iter=3`
- `beta_start=1.2`, `beta_end=0.4` (linear decay)
- `lambda_redundancy=0.2`
- `explore_mode=ucb`

### 2.5 Execution and Logging

For each selected config:
- Apply filesystem/client knobs (`lfs setstripe`, `lctl set_param`)
- Run workload script (`run_from_features.sh`) with Darshan output
- Parse metric via `analyze_darshan_recommender.py`
- Append to:
  - iteration observations CSV
  - workload observations CSV
  - global `observations_all.csv`

### 2.6 Final Recommendation Matrix

- After active loop, recommender matrix is built per workload by sorting configs by **best observed gain**.
- Top-k per workload saved to `recommendation_matrix.json` (`topk_per_pattern=20`).

---

## 3) Recommendation Quality for the 4 Trained Workloads

Evaluation basis used here: observed data in warm + active samples.

| Workload | Top-1 recommended config | Source of top-1 | Matches best observed config? | Top-1 best observed metric (MiB/s) | Mean gain vs baseline (MiB/s) | Mean gain vs baseline (%) |
|---|---|---|---|---:|---:|---:|
| top1_101 | `19ec6a6b28b318b9` | warm | Yes | 1246.86 | 124.32 | 11.1% |
| top2_20  | `7bc28a2b9e93b3a6` | warm | Yes | 1131.09 | 422.74 | 59.7% |
| top3_18  | `5cb9bb31d6a2a2ac` | active | Yes | 1169.51 | 314.53 | 36.8% |
| top4_192 | `96781dd229a515b8` | warm | Yes | 1230.68 | 776.68 | 172.7% |

Top-3 tightness relative to top-1 (mean metric):
- top1_101: #2 = 94.7%, #3 = 93.2%
- top2_20: #2 = 99.4%, #3 = 99.3%
- top3_18: #2 = 98.5%, #3 = 99.8%
- top4_192: #2 = 99.7%, #3 = 97.7%

Interpretation:
- In this observed dataset, top-1 recommendations are excellent and align with the best observed configs.
- For top2_20/top3_18/top4_192, top-2 is very close to top-1; recommendation is robust among a small set of near-optimal choices.

Important caveat:
- Current logged active history metrics (`regret_at_3`, `hit_at_3`, `ndcg_at_3`) are not valid as external quality measures in this run, because oracle is set to observed best in-loop (self-referential), which yields trivial perfect values.

---

## 4) Shortcomings and Scope of Improvement (Summary)

1. **Evaluation leakage in active history metrics**
   - `oracle = best3` in loop makes regret/hit/NDCG trivially perfect and non-diagnostic.

2. **No held-out validation protocol**
   - Recommendation quality currently reflects in-sample ranking quality, not generalization.

3. **Candidate enumeration threshold is just below full space size**
   - Search space = 52,920 and threshold = 50,000; full enumeration may be feasible and safer.

4. **Active configs are single-shot measurements**
   - No repeated measurements for active-selected configs; ranking can be sensitive to run noise.

5. **Insufficient attribution tracking of active contribution**
   - Need explicit reporting of how much active changed top-K vs warm-only baseline.

A detailed action plan is provided in:
- `/mnt/hasanfs/io_synthesizer/outputs/warm_active_sampling_improvement_plan_2026-03-02.md`

---

## References (Primary Artifacts)

- `/mnt/hasanfs/samples/warm-start/observations_all.csv`
- `/mnt/hasanfs/samples/active-sampling/observations_all.csv`
- `/mnt/hasanfs/samples/active-sampling/summary.json`
- `/mnt/hasanfs/samples/active-sampling/recommendation_matrix.json`
- `/mnt/hasanfs/samples/active-sampling/effective_config.yaml`
- `/mnt/hasanfs/samples/warm-start/warm_start_configs.json`
- `/mnt/hasanfs/samples/warm-start/logs/warm_start_remote_20260225_031612.log`
- `/mnt/hasanfs/samples/warm-start/logs/warm_start_remote_20260225_121215.log`
- `/mnt/hasanfs/samples/active-sampling/logs/active_sampling_20260226_195907.log`

Code references:
- `/mnt/hasanfs/io_synthesizer/orchestrators/active_train_sampling_orchestrator/active_sampling_pipeline.py`
- `/mnt/hasanfs/io_synthesizer/io_recommender/active/acquisition.py`
- `/mnt/hasanfs/io_synthesizer/io_recommender/active/candidates.py`
- `/mnt/hasanfs/io_synthesizer/io_recommender/deploy/recommender.py`
- `/mnt/hasanfs/io_synthesizer/io_recommender/sampling/pairwise.py`
- `/mnt/hasanfs/io_synthesizer/orchestrators/warm_start_sampling_orchestrator/warm_start_pipeline.py`
