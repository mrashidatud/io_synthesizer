from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

from io_recommender.model.encoder import ConfigEncoder
from io_recommender.types import Config, ParameterSpec


def _stable_hash_int(text: str) -> int:
    return abs(hash(text)) % (2**31 - 1)


@dataclass
class StubTestbedRunner:
    specs: Sequence[ParameterSpec]
    seed: int = 7
    noise_std: float = 0.0

    def __post_init__(self) -> None:
        self.cfg_encoder = ConfigEncoder(self.specs).fit()
        self.max_idx = np.array([max(len(s.values) - 1, 1) for s in self.specs], dtype=float)

    def run_testbed(self, pattern_id: str, config: Config, workload_vec: np.ndarray | None = None) -> float:
        if workload_vec is None:
            workload_vec = np.zeros(35, dtype=float)
        x = self.cfg_encoder.index_vector(config) / self.max_idx

        pid_seed = self.seed + _stable_hash_int(pattern_id)
        rng = np.random.default_rng(pid_seed)
        w1 = rng.normal(loc=0.0, scale=1.0, size=x.size)
        pair_mat = rng.normal(loc=0.0, scale=0.5, size=(x.size, x.size))
        pair_mat = np.triu(pair_mat, k=1)

        wl = workload_vec
        wl_score = float(np.sin(wl[: min(10, wl.size)].mean()) + 0.2 * wl.std())
        lin = float(np.dot(w1, x))
        pair = float((x.reshape(1, -1) @ pair_mat @ x.reshape(-1, 1)).squeeze())
        perf = 100.0 + 10.0 * wl_score + 8.0 * lin + 6.0 * pair

        if self.noise_std > 0:
            noise_rng = np.random.default_rng(self.seed + _stable_hash_int(pattern_id + str(config)))
            perf += float(noise_rng.normal(0, self.noise_std))
        return float(perf)
