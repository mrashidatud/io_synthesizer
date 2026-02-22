from io_recommender.sampling import build_warm_start_set
from io_recommender.types import ParameterSpec


def test_pairwise_coverage_full_or_report_when_truncated() -> None:
    specs = [
        ParameterSpec("stripe_count", [1, 2, 4, 8, 16, 32]),
        ParameterSpec("stripe_size", [64, 128, 256, 512, 1024, 2048, 4096]),
        ParameterSpec("max_pages_per_rpc", [32, 64, 128, 256, 512, 1024]),
        ParameterSpec("max_rpcs_in_flight", [1, 2, 4, 8, 16, 32]),
    ]
    baseline = {s.name: s.values[0] for s in specs}

    warm_full = build_warm_start_set(specs, baseline=baseline, n_target=47, seed=7)
    assert warm_full.pairwise_coverage_percent == 100.0
    assert warm_full.covered_pairs == warm_full.total_pairs

    warm_truncated = build_warm_start_set(specs, baseline=baseline, n_target=45, seed=7)
    assert 0.0 < warm_truncated.pairwise_coverage_percent < 100.0
    assert warm_truncated.covered_pairs < warm_truncated.total_pairs
