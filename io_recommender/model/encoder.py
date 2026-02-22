from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from io_recommender.types import ParameterSpec, WorkloadPattern


@dataclass
class WorkloadEncoder:
    feature_names: List[str] | None = None

    def fit(self, patterns: Sequence[WorkloadPattern]) -> "WorkloadEncoder":
        if not patterns:
            raise ValueError("patterns cannot be empty")
        first = patterns[0].features
        if isinstance(first, np.ndarray):
            self.feature_names = [f"f_{i}" for i in range(first.size)]
        else:
            self.feature_names = sorted(first.keys())
        return self

    def encode_workload(self, pattern: WorkloadPattern | Mapping[str, float] | np.ndarray) -> np.ndarray:
        if self.feature_names is None:
            raise ValueError("WorkloadEncoder must be fitted before encoding")
        if isinstance(pattern, WorkloadPattern):
            feat = pattern.features
        else:
            feat = pattern

        if isinstance(feat, np.ndarray):
            return feat.astype(float)
        return np.array([float(feat[name]) for name in self.feature_names], dtype=float)

    def encode_many(self, patterns: Iterable[WorkloadPattern]) -> np.ndarray:
        return np.vstack([self.encode_workload(p) for p in patterns])


@dataclass
class ConfigEncoder:
    specs: Sequence[ParameterSpec]
    columns: List[str] | None = None
    value_to_index: Dict[str, Dict[Any, int]] | None = None

    def fit(self) -> "ConfigEncoder":
        self.columns = []
        self.value_to_index = {}
        for spec in self.specs:
            self.value_to_index[spec.name] = {value: idx for idx, value in enumerate(spec.values)}
            for value in spec.values:
                self.columns.append(f"{spec.name}={value}")
        return self

    def encode_config(self, config: Mapping[str, Any]) -> np.ndarray:
        if self.columns is None or self.value_to_index is None:
            raise ValueError("ConfigEncoder must be fitted before encoding")
        vec = np.zeros(len(self.columns), dtype=float)
        offset = 0
        for spec in self.specs:
            value = config[spec.name]
            idx = self.value_to_index[spec.name][value]
            vec[offset + idx] = 1.0
            offset += len(spec.values)
        return vec

    def index_vector(self, config: Mapping[str, Any]) -> np.ndarray:
        if self.value_to_index is None:
            raise ValueError("ConfigEncoder must be fitted before encoding")
        return np.array([self.value_to_index[s.name][config[s.name]] for s in self.specs], dtype=float)
