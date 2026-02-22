from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


def dcg_at_k(rels: Sequence[float], k: int = 3) -> float:
    vals = np.asarray(rels[:k], dtype=float)
    if vals.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, vals.size + 2))
    return float(np.sum((2**vals - 1) * discounts))


def ndcg_at_k(actual_rels_sorted_by_model: Sequence[float], ideal_rels_sorted: Sequence[float], k: int = 3) -> float:
    best = dcg_at_k(ideal_rels_sorted, k=k)
    if best <= 0:
        return 0.0
    return dcg_at_k(actual_rels_sorted_by_model, k=k) / best


def regret_at_3(observed_best3: Mapping[str, float], oracle_best: Mapping[str, float]) -> float:
    vals = []
    for pid, opt in oracle_best.items():
        got = observed_best3.get(pid, float("-inf"))
        denom = max(abs(opt), 1e-9)
        vals.append((opt - got) / denom)
    return float(np.mean(vals)) if vals else 0.0


def hit_at_3_within(observed_best3: Mapping[str, float], oracle_best: Mapping[str, float], tol: float = 0.05) -> float:
    hits = []
    for pid, opt in oracle_best.items():
        got = observed_best3.get(pid, float("-inf"))
        hits.append(1.0 if got >= (1.0 - tol) * opt else 0.0)
    return float(np.mean(hits)) if hits else 0.0
