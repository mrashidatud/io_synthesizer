import numpy as np

from io_recommender.active import ActiveLoopConfig, run_active_loop
from io_recommender.model import ConfigEncoder, WorkloadEncoder
from io_recommender.pipeline import collect_observations
from io_recommender.runner import StubTestbedRunner
from io_recommender.sampling import build_warm_start_set
from io_recommender.types import ParameterSpec, WorkloadPattern


def test_active_loop_batching_and_no_duplicates() -> None:
    specs = [
        ParameterSpec("a", [0, 1, 2, 3]),
        ParameterSpec("b", [0, 1, 2, 3]),
        ParameterSpec("c", [0, 1, 2]),
        ParameterSpec("d", [0, 1, 2]),
    ]
    patterns = [
        WorkloadPattern("p0", np.array([0.1, -0.4, 0.2, 1.0])),
        WorkloadPattern("p1", np.array([0.5, 0.4, -0.1, 0.0])),
        WorkloadPattern("p2", np.array([-0.2, 0.3, 0.7, -0.4])),
    ]

    wenc = WorkloadEncoder().fit(patterns)
    cenc = ConfigEncoder(specs).fit()
    baseline = {s.name: s.values[0] for s in specs}
    warm = build_warm_start_set(specs, baseline=baseline, n_target=12, seed=9)

    runner = StubTestbedRunner(specs, seed=9)
    warm_obs, baseline_perf = collect_observations(patterns, warm.configs, wenc, cenc, runner, baseline)

    cfg = ActiveLoopConfig(
        iterations=3,
        batch_per_iter=2,
        ensemble_size=4,
        model_mode="regression",
        use_lightgbm=False,
        seed=9,
    )
    final_obs, _, history = run_active_loop(
        patterns=patterns,
        specs=specs,
        workload_encoder=wenc,
        config_encoder=cenc,
        runner=runner,
        baseline_perf_by_pattern=baseline_perf,
        initial_observations=warm_obs,
        cfg=cfg,
    )

    assert len(history) == 3
    expected_new = len(patterns) * cfg.batch_per_iter * cfg.iterations
    assert len(final_obs) == len(warm_obs) + expected_new

    for p in patterns:
        seen = set()
        for obs in [o for o in final_obs if o.pattern_id == p.pattern_id]:
            key = tuple((k, obs.config_params[k]) for k in sorted(obs.config_params))
            assert key not in seen
            seen.add(key)
