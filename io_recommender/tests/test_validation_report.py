import numpy as np

from io_recommender.active import ActiveLoopConfig, run_active_loop
from io_recommender.eval.validation import evaluate_on_heldout
from io_recommender.model import ConfigEncoder, EnsembleConfig, WorkloadEncoder
from io_recommender.pipeline import collect_observations
from io_recommender.runner import StubTestbedRunner
from io_recommender.sampling import build_warm_start_set
from io_recommender.types import ParameterSpec, WorkloadPattern


def _build_dataset():
    specs = [
        ParameterSpec("a", [0, 1, 2]),
        ParameterSpec("b", [0, 1, 2]),
        ParameterSpec("c", [0, 1, 2]),
    ]
    patterns = [
        WorkloadPattern(f"p{i}", np.array([0.1 * (i + 1), -0.2, 0.3, 0.4]))
        for i in range(6)
    ]
    wenc = WorkloadEncoder().fit(patterns)
    cenc = ConfigEncoder(specs).fit()
    baseline = {s.name: s.values[0] for s in specs}
    warm = build_warm_start_set(specs, baseline=baseline, n_target=8, seed=5)

    runner = StubTestbedRunner(specs, seed=5)
    warm_obs, baseline_perf = collect_observations(patterns, warm.configs, wenc, cenc, runner, baseline)
    active_cfg = ActiveLoopConfig(
        iterations=2,
        batch_per_iter=1,
        ensemble_size=4,
        model_mode="regression",
        use_lightgbm=False,
        seed=5,
    )
    obs, _, _ = run_active_loop(
        patterns=patterns,
        specs=specs,
        workload_encoder=wenc,
        config_encoder=cenc,
        runner=runner,
        baseline_perf_by_pattern=baseline_perf,
        initial_observations=warm_obs,
        cfg=active_cfg,
    )
    return obs, patterns, wenc, cenc


def test_evaluate_on_heldout_patterns_report() -> None:
    observations, patterns, wenc, cenc = _build_dataset()
    report = evaluate_on_heldout(
        observations=observations,
        patterns=patterns,
        workload_encoder=wenc,
        config_encoder=cenc,
        ensemble_cfg=EnsembleConfig(mode="regression", ensemble_size=4, seed=5, use_lightgbm=False),
        split_mode="heldout_patterns",
        top_k=3,
        bootstrap_iters=30,
        seed=5,
    )
    assert report["enabled"] is True
    assert report["n_eval_patterns"] > 0
    assert "aggregate" in report
    assert set(report["aggregate"].keys()) == {"top1_regret", "topk_regret", "hit_at_k", "ndcg_at_k"}


def test_temporal_split_supported() -> None:
    observations, patterns, wenc, cenc = _build_dataset()
    report = evaluate_on_heldout(
        observations=observations,
        patterns=patterns,
        workload_encoder=wenc,
        config_encoder=cenc,
        ensemble_cfg=EnsembleConfig(mode="regression", ensemble_size=4, seed=5, use_lightgbm=False),
        split_mode="heldout_configs_per_pattern",
        temporal_split=True,
        temporal_holdout_fraction=0.4,
        top_k=3,
        bootstrap_iters=20,
        seed=5,
    )
    assert report["enabled"] is True
    assert report["split_meta"]["split_mode_effective"] == "temporal"
