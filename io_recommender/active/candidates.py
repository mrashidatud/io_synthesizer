from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from io_recommender.sampling.pairwise import enumerate_all_configs, total_space_size
from io_recommender.types import Config, Observation, ParameterSpec


def _config_key(config: Mapping[str, object], specs: Sequence[ParameterSpec]) -> tuple:
    return tuple(config[s.name] for s in specs)


def _mutate_one(config: Config, specs: Sequence[ParameterSpec], rng: np.random.Generator) -> Config:
    c = dict(config)
    spec = specs[rng.integers(0, len(specs))]
    options = [v for v in spec.values if v != c[spec.name]]
    c[spec.name] = options[rng.integers(0, len(options))]
    return c


def _mutate_two(config: Config, specs: Sequence[ParameterSpec], rng: np.random.Generator) -> Config:
    c = dict(config)
    idxs = rng.choice(np.arange(len(specs)), size=min(2, len(specs)), replace=False)
    for idx in idxs:
        spec = specs[idx]
        options = [v for v in spec.values if v != c[spec.name]]
        c[spec.name] = options[rng.integers(0, len(options))]
    return c


def generate_candidate_pool(
    specs: Sequence[ParameterSpec],
    observations: Iterable[Observation],
    pattern_id: str,
    top_configs: List[Config],
    seed: int,
    enum_threshold: int = 50_000,
    max_pool: int = 12_000,
) -> List[Config]:
    rng = np.random.default_rng(seed)
    tested = {
        _config_key(obs.config_params, specs)
        for obs in observations
        if obs.pattern_id == pattern_id
    }

    if total_space_size(specs) <= enum_threshold:
        return [c for c in enumerate_all_configs(specs) if _config_key(c, specs) not in tested]

    pool: Dict[tuple, Config] = {}
    for cfg in top_configs:
        for _ in range(300):
            c1 = _mutate_one(cfg, specs, rng)
            k1 = _config_key(c1, specs)
            if k1 not in tested:
                pool[k1] = c1
            if rng.random() < 0.25:
                c2 = _mutate_two(cfg, specs, rng)
                k2 = _config_key(c2, specs)
                if k2 not in tested:
                    pool[k2] = c2
            if len(pool) >= max_pool:
                break
        if len(pool) >= max_pool:
            break

    while len(pool) < max_pool:
        c = {s.name: s.values[rng.integers(0, len(s.values))] for s in specs}
        k = _config_key(c, specs)
        if k in tested:
            continue
        pool[k] = c
    return list(pool.values())
