from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    values: List[Any]
    is_ordered: bool = True


Config = Dict[str, Any]


@dataclass(frozen=True)
class WorkloadPattern:
    pattern_id: str
    features: Mapping[str, float] | np.ndarray


@dataclass
class Observation:
    pattern_id: str
    config_id: str
    config_params: Config
    workload_vec: np.ndarray
    config_vec: np.ndarray
    perf: float
    baseline_perf: float
    gain: float


def config_id_from_params(config: Mapping[str, Any]) -> str:
    items = sorted((k, str(v)) for k, v in config.items())
    payload = "|".join(f"{k}={v}" for k, v in items)
    return sha1(payload.encode("utf-8")).hexdigest()[:16]


def normalize_specs(specs: Sequence[ParameterSpec]) -> List[ParameterSpec]:
    return [ParameterSpec(name=s.name, values=list(s.values), is_ordered=s.is_ordered) for s in specs]
