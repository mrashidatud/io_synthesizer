from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from io_recommender.types import ParameterSpec


def config_index_vector(config: Mapping[str, object], specs: Sequence[ParameterSpec]) -> np.ndarray:
    idx = []
    for spec in specs:
        idx.append(spec.values.index(config[spec.name]))
    return np.array(idx, dtype=float)


def normalized_l1_distance(
    a: Mapping[str, object],
    b: Mapping[str, object],
    specs: Sequence[ParameterSpec],
) -> float:
    va = config_index_vector(a, specs)
    vb = config_index_vector(b, specs)
    num = np.abs(va - vb)
    den = np.array([max(len(s.values) - 1, 1) for s in specs], dtype=float)
    return float(np.mean(num / den))


def min_distance_to_set(config: Mapping[str, object], configs: Sequence[Mapping[str, object]], specs: Sequence[ParameterSpec]) -> float:
    if not configs:
        return 1.0
    return min(normalized_l1_distance(config, c, specs) for c in configs)
