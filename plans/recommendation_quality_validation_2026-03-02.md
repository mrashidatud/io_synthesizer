# Recommendation Quality Validation Report (2026-03-02)

## Run Artifacts
- Validation script: `/mnt/hasanfs/io_synthesizer/io_recommender/validation/run_closest_workload_validation.py`
- Validation output root: `/mnt/hasanfs/samples/validation/recommendation_quality_20260302_run2`
- Summary JSON: `/mnt/hasanfs/samples/validation/recommendation_quality_20260302_run2/summary.json`
- Generated at (UTC): `2026-03-02T13:03:01.179698`

## 1) Best-Match Workload Per Top-4 Trained Pattern

| Reference Pattern | Best-Match Candidate | Candidate Rank | Weighted Distance |
|---|---:|---:|---:|
| top1_101 | top10_252 | 10 | 2.0128 |
| top2_20 | top12_236 | 12 | 2.5176 |
| top3_18 | top21_258 | 21 | 1.4224 |
| top4_192 | top23_193 | 23 | 2.0256 |

## 2) Execution Design

- For each selected candidate workload, executed 4 runs: `baseline + rec1 + rec2 + rec3` (total `16` runs).
- Baseline config id: `75984fca0ee010f8` (`stripe_count=1`, `stripe_size=1M`, `osc_pages=1024`, `mdc_pages=256`, `osc_rpcs=8`, `mdc_rpcs=8`).
- Metric used: `agg_perf_by_slowest` (MiB/s), parsed from Darshan after each run.

## 3) Per-Workload Results (MiB/s)

| Reference -> Candidate | Baseline | Rec1 | Rec2 | Rec3 | Best Rec | Best Gain vs Baseline |
|---|---:|---:|---:|---:|---:|---:|
| top1_101 -> top10_252 | 737.857 | 850.777 | 850.140 | 849.943 | rec1 (850.777) | 15.30% |
| top2_20 -> top12_236 | 402.800 | 884.866 | 897.214 | 896.546 | rec2 (897.214) | 122.74% |
| top3_18 -> top21_258 | 1398.112 | 1709.442 | 1721.025 | 1716.722 | rec2 (1721.025) | 23.10% |
| top4_192 -> top23_193 | 410.332 | 864.564 | 897.447 | 876.272 | rec2 (897.447) | 118.71% |

## 4) Overall Recommendation Quality

- Avg baseline throughput: **737.275 MiB/s**
- Avg best-of-top3 throughput: **1091.616 MiB/s**
- Avg best-of-top3 gain vs baseline: **69.96%**
- Workloads where all top-3 recs beat baseline: **4/4 (100.0%)**
- Top-1 ranking hit rate (predicted best == actual best among rec1..3): **1/4 (25.0%)**
- Avg regret from picking predicted top-1 vs actual best in top-3: **14.203 MiB/s** (max **32.883 MiB/s**)

## 5) Key Observations

- Absolute recommendation quality is strong: every candidate workload improved over baseline with every top-3 recommendation.
- The most consistently strong config in this validation set is `210b61d8361cb3c2` (best on 3/4 workloads).
- Ranking quality is weaker than absolute quality: the model-predicted top recommendation matched actual best only 1 out of 4 times.
- Largest wins were on write-dominant patterns (`top12_236`, `top23_193`) where throughput more than doubled vs baseline.

## 6) Shortcomings Seen in This Validation

- **Top-1 misranking**: predicted order among the top-3 is often not the actual order, even though all are better than baseline.
- **Small score separation**: predicted scores for rec1..3 are close on several workloads, but observed throughput gaps can be meaningful.
- **Single-run noise sensitivity**: this validation uses one run per config; close results (for example within ~1-2%) may switch with reruns.

## 7) Practical Next Steps (Validation-Focused)

- Add a lightweight tie-break stage when top-3 predicted scores are close (for example, run a short pilot for top-2 and choose by measured throughput).
- Calibrate ranking with pairwise/listwise fine-tuning on recent active-sampling observations to improve top-1 ordering.
- Re-run each `(workload, config)` at least 3 times and report mean/std + confidence intervals before finalizing ranking judgments.

---
Report path: `/mnt/hasanfs/io_synthesizer/outputs/recommendation_quality_validation_2026-03-02.md`
