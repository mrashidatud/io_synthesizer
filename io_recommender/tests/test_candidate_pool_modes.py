from io_recommender.active.candidates import generate_candidate_pool_details
from io_recommender.types import ParameterSpec


def test_candidate_pool_mode_enumerated_when_space_under_hard_threshold() -> None:
    specs = [
        ParameterSpec("a", [0, 1]),
        ParameterSpec("b", [0, 1]),
    ]
    res = generate_candidate_pool_details(
        specs=specs,
        observations=[],
        pattern_id="p0",
        top_configs=[{"a": 0, "b": 0}],
        seed=7,
        enum_threshold_hard=52_920,
        max_pool=100,
    )
    assert res.mode == "enumerated"
    assert res.total_space == 4
    assert len(res.configs) == 4


def test_candidate_pool_mode_sampled_when_space_over_hard_threshold() -> None:
    specs = [ParameterSpec(f"p{i}", [0, 1, 2, 3, 4]) for i in range(7)]  # 5^7 = 78,125
    baseline = {s.name: s.values[0] for s in specs}
    res = generate_candidate_pool_details(
        specs=specs,
        observations=[],
        pattern_id="p0",
        top_configs=[baseline],
        seed=11,
        enum_threshold_hard=52_920,
        max_pool=150,
    )
    assert res.mode == "sampled"
    assert res.total_space == 78_125
    assert len(res.configs) == 150
