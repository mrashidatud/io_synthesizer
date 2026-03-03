import numpy as np

from io_recommender.eval.contribution import build_contribution_report
from io_recommender.types import Observation


def _obs(
    pattern_id: str,
    config_id: str,
    gain: float,
    source: str,
    iteration: int,
) -> Observation:
    return Observation(
        pattern_id=pattern_id,
        config_id=config_id,
        config_params={"x": config_id},
        workload_vec=np.array([0.0, 1.0]),
        config_vec=np.array([0.0]),
        perf=100.0 + gain,
        baseline_perf=100.0,
        gain=gain,
        source=source,
        iteration=iteration,
        replicate_index=0,
    )


def test_contribution_report_has_required_fields() -> None:
    observations = [
        _obs("p0", "warm_a", gain=1.0, source="warm", iteration=0),
        _obs("p0", "act_b", gain=2.5, source="active", iteration=1),
        _obs("p0", "act_b", gain=2.2, source="replicate", iteration=1),
        _obs("p1", "warm_c", gain=1.8, source="warm", iteration=0),
        _obs("p1", "act_d", gain=1.2, source="active", iteration=1),
    ]

    report = build_contribution_report(observations, top_k=1)
    assert report["n_patterns"] == 2
    assert report["pct_top_k_from_active"] == 50.0
    assert len(report["per_pattern"]) == 2
    assert len(report["cumulative_improvement_trajectory"]) == 2

    p0 = [p for p in report["per_pattern"] if p["pattern_id"] == "p0"][0]
    assert p0["best_active_vs_best_warm_delta"] > 0
    assert p0["new_best_found_iter"] == 1
