from __future__ import annotations

from typing import Dict, List, Sequence

from io_recommender.types import Config, ParameterSpec


ANCHOR_NAMES = ["baseline", "all_min", "all_max", "checkerboard_a", "checkerboard_b"]


def make_anchor_configs(specs: Sequence[ParameterSpec], baseline: Config) -> List[Config]:
    all_min = {s.name: s.values[0] for s in specs}
    all_max = {s.name: s.values[-1] for s in specs}
    checker_a: Dict[str, object] = {}
    checker_b: Dict[str, object] = {}
    for i, spec in enumerate(specs):
        low = spec.values[0]
        high = spec.values[-1]
        checker_a[spec.name] = low if i % 2 == 0 else high
        checker_b[spec.name] = high if i % 2 == 0 else low

    anchors = [baseline, all_min, all_max, checker_a, checker_b]
    dedup = []
    seen = set()
    for conf in anchors:
        key = tuple((k, conf[k]) for k in sorted(conf))
        if key not in seen:
            dedup.append(dict(conf))
            seen.add(key)
    return dedup
