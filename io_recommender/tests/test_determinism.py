from io_recommender.sampling import build_warm_start_set
from io_recommender.types import ParameterSpec


def test_warm_start_deterministic_given_seed() -> None:
    specs = [
        ParameterSpec("a", [0, 1, 2]),
        ParameterSpec("b", [0, 1, 2, 3]),
        ParameterSpec("c", [0, 1, 2]),
        ParameterSpec("d", [0, 1, 2]),
    ]
    baseline = {s.name: s.values[0] for s in specs}

    one = build_warm_start_set(specs, baseline=baseline, n_target=20, seed=123)
    two = build_warm_start_set(specs, baseline=baseline, n_target=20, seed=123)

    assert one.configs == two.configs
    assert one.pairwise_coverage_percent == two.pairwise_coverage_percent
