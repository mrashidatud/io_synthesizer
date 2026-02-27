[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mrashidatud/io_synthesizer)


## Active Learning Recommender Pipeline

This repository now includes a production-style Python package for:
1. Warm-start configuration sampling (anchors + pairwise coverage)
2. Active-learning recommender training with uncertainty-aware hybrid acquisition
3. Deployment-time top-k recommendation using nearest-workload matching

### Architecture

Package: `io_recommender/`

- `io_recommender/sampling/`
  - `anchors.py`: baseline/all-min/all-max/checkerboards
  - `pairwise.py`: mixed-level strength-2 greedy pairwise coverage
  - `distance.py`: normalized L1 distances in parameter-index space
- `io_recommender/model/`
  - `encoder.py`: stable workload/config encoders
  - `ensemble.py`: LightGBM ensemble (ranking/regression) with mean/std prediction
  - `labels.py`: gain -> graded relevance for ranking
- `io_recommender/active/`
  - `candidates.py`: scalable candidate generation
  - `acquisition.py`: UCB/Thompson + redundancy penalty + hybrid selector
  - `loop.py`: iterative active-learning controller
- `io_recommender/deploy/`
  - `recommender.py`: kNN workload retrieval + candidate merge + rerank + top-k
- `io_recommender/eval/`
  - `metrics.py`: regret@3, hit@3, ndcg@3 utilities
  - `curves.py`: learning-curve plots
- `io_recommender/runner.py`
  - deterministic stub `run_testbed` oracle for development/testing
- `io_recommender/pipeline.py`
  - end-to-end orchestration helpers

CLI entrypoint: `io_recommender/run_pipeline.py`

### Install

```bash
pip install -r io_recommender/requirements.txt
```

### Configure

Edit `io_recommender/config.yaml`.

The template includes:
- parameter specs (`name`, `values`, `is_ordered`)
- warm-start target size
- active loop settings (`ensemble_size`, `iterations`, `batch_per_iter`, beta schedule)
- model mode (`ranking` default, or `regression`)
- deployment settings (`topk_per_pattern`, `knn_neighbors`, return top-k)

### Run End-to-End

```bash
python -m io_recommender.run_pipeline --config io_recommender/config.yaml --output-dir artifacts
```

### Warm-Start Collection Workflow (Separated)

Use the dedicated script for warm-start data collection:

```bash
python3 orchestrators/warm_start_sampling_orchestrator/warm_start_pipeline.py \
  --options-csv orchestrators/warm_start_sampling_orchestrator/warm_start_options.csv
```

This workflow is separated from `pipeline.sh` and is designed for recommender warm-start sampling:
- plan generation once per workload
- no re-generation of `plan.csv` / `run_prep.sh` / `run_from_features.sh` across iterations
- run each workload for `iterations` (default `3`)
- apply warm-start configs each iteration and extract recommender-ready metrics from Darshan

Remote launcher (same style as existing CSV orchestration):

```bash
orchestrators/warm_start_sampling_orchestrator/run_warm_start_from_csv.sh \
  orchestrators/warm_start_sampling_orchestrator/warm_start_options.csv
```

Output layout:
- root: `/mnt/hasanfs/samples/warm-start`
- per workload: `/mnt/hasanfs/samples/warm-start/<workload_key>`
- static artifacts in workload dir: `payload/`, `run_prep.sh`, `run_from_features.sh`
- per iteration: `iter_01/`, `iter_02/`, `iter_03/`
- per config-run (inside iteration): `cfg_<idx>_<config_id>/` with darshan + parsed metrics
- observation tables:
  - per-iteration: `iter_xx/observations.csv`
  - per-workload: `<workload>/observations.csv`
  - global: `/mnt/hasanfs/samples/warm-start/observations_all.csv`

Outputs:
- `artifacts/summary.json`
- `artifacts/recommendation_matrix.json`
- `artifacts/plots/regret_at_3.png`
- `artifacts/plots/hit_at_3.png`
- `artifacts/plots/ndcg_at_3.png`

### Plug In Real Testbed Execution

The bridge is implemented in `io_recommender/runner_real.py` and selected with:

- `runner.mode: real` in `io_recommender/config.yaml`

Runner flow per active-learning trial:
1. Use `scripts/features2synth_opsaware.py` with selected pattern JSON.
2. Apply Lustre knobs from config to both OSC and MDC explicitly:
   - `osc_max_pages_per_rpc` -> `osc.*.max_pages_per_rpc`
   - `mdc_max_pages_per_rpc` -> `mdc.*.max_pages_per_rpc`
   - `osc_max_rpcs_in_flight` -> `osc.*.max_rpcs_in_flight`
   - `mdc_max_rpcs_in_flight` -> `mdc.*.max_rpcs_in_flight`
   - `mdc.*.max_rpcs_in_flight` is range-limited to `[2,512]` if needed.
3. Run generated `run_from_features.sh`.
4. Run `analysis/scripts_analysis/analyze_darshan_merged.py`.
5. Read objective metric from `darshan_summary.csv` (default `POSIX_agg_perf_by_slowest`).

Where striping is applied:
- Data files are created by generated `run_prep.sh` under:
  - `/mnt/hasanfs/out_synth/<pattern>/payload/data/ro`
  - `/mnt/hasanfs/out_synth/<pattern>/payload/data/rw`
  - `/mnt/hasanfs/out_synth/<pattern>/payload/data/wo`
  - `/mnt/hasanfs/out_synth/<pattern>/payload/meta`
- The bridge applies `lfs setstripe` on these directories before `run_prep.sh` creates files, so files inherit that striping policy.

### Warm-Start + Active Sampling Behavior

- Warm-start:
  - Anchors: baseline, all-min, all-max, checkerboard A/B
  - Greedy pairwise fill: adds configs that cover maximum currently-uncovered value pairs
  - Tie-breaks by diversity (normalized L1 distance), then seeded randomness
  - Enumerates full space when size <= 50k, else uses large sampled pools
- Active loop (per workload, per iteration):
  1. Exploit: best predicted mean gain
  2. Explore: best UCB (`mu + beta*sigma`) or Thompson (optional)
  3. Diversify: farthest from tested set in knob index space
  4. Redundancy penalty applied in acquisition scoring

### Tests

```bash
pytest io_recommender/tests
```

Included tests:
- pairwise coverage reaches 100% in provided 4-knob setting
- warm-start determinism for fixed seed
- active loop adds exactly `B` new configs per workload/iteration with no duplicates
