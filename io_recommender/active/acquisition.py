from __future__ import annotations

from typing import List, Mapping, Sequence

import numpy as np

from io_recommender.sampling.distance import min_distance_to_set, normalized_l1_distance
from io_recommender.types import Config, ParameterSpec


def redundancy_penalty(config: Mapping[str, object], tested: Sequence[Mapping[str, object]], specs: Sequence[ParameterSpec]) -> float:
    if not tested:
        return 0.0
    min_dist = min_distance_to_set(config, tested, specs)
    return 1.0 - min_dist


def acquisition_scores(
    candidates: Sequence[Config],
    mu: np.ndarray,
    sigma: np.ndarray,
    tested: Sequence[Config],
    specs: Sequence[ParameterSpec],
    beta: float,
    lam: float,
) -> np.ndarray:
    penalties = np.array([redundancy_penalty(c, tested, specs) for c in candidates], dtype=float)
    return mu + beta * sigma - lam * penalties


def select_hybrid(
    candidates: Sequence[Config],
    mu: np.ndarray,
    sigma: np.ndarray,
    tested: Sequence[Config],
    specs: Sequence[ParameterSpec],
    b: int,
    beta: float,
    lam: float,
    explore_mode: str = "ucb",
    seed: int = 7,
) -> List[Config]:
    if len(candidates) == 0:
        return []
    rng = np.random.default_rng(seed)
    selected: List[Config] = []
    used = set()

    def add_idx(idx: int) -> None:
        key = tuple(candidates[idx][s.name] for s in specs)
        if key in used:
            return
        used.add(key)
        selected.append(candidates[idx])

    exploit_idx = int(np.argmax(mu))
    add_idx(exploit_idx)

    if len(selected) < b:
        if explore_mode == "thompson":
            theta = rng.normal(mu, np.maximum(sigma, 1e-9))
            explore_idx = int(np.argmax(theta))
        else:
            acq = acquisition_scores(candidates, mu, sigma, tested, specs, beta, lam)
            explore_idx = int(np.argmax(acq))
        add_idx(explore_idx)

    if len(selected) < b:
        best_idx = None
        best_div = -1.0
        reference = list(tested) + selected
        for i, cand in enumerate(candidates):
            key = tuple(cand[s.name] for s in specs)
            if key in used:
                continue
            if not reference:
                div = 1.0
            else:
                div = min(normalized_l1_distance(cand, r, specs) for r in reference)
            if div > best_div:
                best_idx = i
                best_div = div
        if best_idx is not None:
            add_idx(best_idx)

    if len(selected) < b:
        acq = acquisition_scores(candidates, mu, sigma, tested, specs, beta, lam)
        for idx in np.argsort(acq)[::-1]:
            add_idx(int(idx))
            if len(selected) >= b:
                break

    return selected[:b]
