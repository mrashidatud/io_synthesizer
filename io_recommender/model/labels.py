from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from io_recommender.types import Observation


def gains_to_relevance(gains: np.ndarray, levels: int = 5) -> np.ndarray:
    if gains.size == 0:
        return np.array([], dtype=int)
    if np.allclose(gains, gains[0]):
        return np.zeros_like(gains, dtype=int)

    quantiles = np.linspace(0, 1, levels + 1)
    bins = np.quantile(gains, quantiles[1:-1])
    rel = np.digitize(gains, bins, right=True)
    return rel.astype(int)


def build_ranking_labels(observations: Iterable[Observation], levels: int = 5) -> List[int]:
    by_pattern: Dict[str, List[Observation]] = {}
    for obs in observations:
        by_pattern.setdefault(obs.pattern_id, []).append(obs)

    labels: List[int] = []
    for pattern in sorted(by_pattern):
        gains = np.array([o.gain for o in by_pattern[pattern]], dtype=float)
        rel = gains_to_relevance(gains, levels=levels)
        labels.extend(rel.tolist())
    return labels
