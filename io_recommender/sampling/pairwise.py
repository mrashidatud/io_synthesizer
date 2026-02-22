from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from io_recommender.sampling.anchors import make_anchor_configs
from io_recommender.sampling.distance import min_distance_to_set
from io_recommender.types import Config, ParameterSpec


@dataclass
class WarmStartResult:
    configs: List[Config]
    pairwise_coverage_percent: float
    covered_pairs: int
    total_pairs: int


def enumerate_all_configs(specs: Sequence[ParameterSpec]) -> List[Config]:
    names = [s.name for s in specs]
    values = [s.values for s in specs]
    return [dict(zip(names, combo)) for combo in itertools.product(*values)]


def total_space_size(specs: Sequence[ParameterSpec]) -> int:
    size = 1
    for s in specs:
        size *= len(s.values)
    return size


def _pair_universe(specs: Sequence[ParameterSpec]) -> Dict[Tuple[int, int], set[Tuple[object, object]]]:
    out: Dict[Tuple[int, int], set[Tuple[object, object]]] = {}
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            out[(i, j)] = set(itertools.product(specs[i].values, specs[j].values))
    return out


def _covered_pairs_by_config(config: Mapping[str, object], specs: Sequence[ParameterSpec]) -> Dict[Tuple[int, int], Tuple[object, object]]:
    covered = {}
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            covered[(i, j)] = (config[specs[i].name], config[specs[j].name])
    return covered


def _coverage_score(config: Mapping[str, object], uncovered: Dict[Tuple[int, int], set[Tuple[object, object]]], specs: Sequence[ParameterSpec]) -> int:
    c = _covered_pairs_by_config(config, specs)
    return sum(1 for pair_key, pair_val in c.items() if pair_val in uncovered[pair_key])


def _remove_covered(config: Mapping[str, object], uncovered: Dict[Tuple[int, int], set[Tuple[object, object]]], specs: Sequence[ParameterSpec]) -> None:
    c = _covered_pairs_by_config(config, specs)
    for pair_key, pair_val in c.items():
        uncovered[pair_key].discard(pair_val)


def _covered_count(configs: Sequence[Mapping[str, object]], specs: Sequence[ParameterSpec]) -> int:
    universe = _pair_universe(specs)
    covered = {k: set() for k in universe}
    for cfg in configs:
        for i in range(len(specs)):
            for j in range(i + 1, len(specs)):
                covered[(i, j)].add((cfg[specs[i].name], cfg[specs[j].name]))
    return sum(len(v) for v in covered.values())


def _candidate_pool(
    specs: Sequence[ParameterSpec],
    rng: np.random.Generator,
    enum_threshold: int,
    pool_size: int,
) -> List[Config]:
    if total_space_size(specs) <= enum_threshold:
        return enumerate_all_configs(specs)

    names = [s.name for s in specs]
    pool = []
    seen = set()
    while len(pool) < pool_size:
        c = {s.name: s.values[rng.integers(0, len(s.values))] for s in specs}
        key = tuple((k, c[k]) for k in names)
        if key in seen:
            continue
        seen.add(key)
        pool.append(c)
    return pool


def build_warm_start_set(
    specs: Sequence[ParameterSpec],
    baseline: Config,
    n_target: int = 45,
    seed: int = 7,
    enum_threshold: int = 50_000,
    sampled_pool_size: int = 10_000,
    restarts: int = 24,
) -> WarmStartResult:
    if n_target <= 0:
        raise ValueError("n_target must be positive")

    total_pairs = sum(len(v) for v in _pair_universe(specs).values())

    def _build_once(local_seed: int) -> WarmStartResult:
        rng = np.random.default_rng(local_seed)
        anchors = make_anchor_configs(specs, baseline)
        selected = list(anchors)[:n_target]
        locked_count = len(selected)

        uncovered = _pair_universe(specs)
        for c in selected:
            _remove_covered(c, uncovered, specs)

        while len(selected) < n_target and any(uncovered[k] for k in uncovered):
            candidates = _candidate_pool(specs, rng, enum_threshold=enum_threshold, pool_size=sampled_pool_size)
            used = {tuple((s.name, cfg[s.name]) for s in specs) for cfg in selected}
            candidates = [c for c in candidates if tuple((s.name, c[s.name]) for s in specs) not in used]
            if not candidates:
                break

            best = None
            best_score = -1
            best_div = -1.0
            for cand in candidates:
                score = _coverage_score(cand, uncovered, specs)
                if score < best_score:
                    continue
                div = min_distance_to_set(cand, selected, specs)
                if score > best_score:
                    best, best_score, best_div = cand, score, div
                    continue
                if div > best_div:
                    best, best_score, best_div = cand, score, div
                    continue
                if np.isclose(div, best_div) and rng.random() < 0.5:
                    best, best_score, best_div = cand, score, div

            if best is None or best_score <= 0:
                break
            selected.append(best)
            _remove_covered(best, uncovered, specs)

        if any(uncovered[k] for k in uncovered) and total_space_size(specs) <= enum_threshold:
            all_configs = _candidate_pool(specs, rng, enum_threshold=enum_threshold, pool_size=sampled_pool_size)
            used = {tuple((s.name, cfg[s.name]) for s in specs) for cfg in selected}

            for _ in range(200):
                remaining_pairs = [(k, list(v)) for k, v in uncovered.items() if v]
                if not remaining_pairs:
                    break

                pair_key, vals = max(remaining_pairs, key=lambda x: len(x[1]))
                target_val = vals[rng.integers(0, len(vals))]

                feasible = [
                    c
                    for c in all_configs
                    if (c[specs[pair_key[0]].name], c[specs[pair_key[1]].name]) == target_val
                ]
                if not feasible:
                    break

                current_cover = _covered_count(selected, specs)
                best_delta = 0
                best_action = None

                for cand in feasible:
                    cand_key = tuple((s.name, cand[s.name]) for s in specs)
                    if cand_key in used:
                        continue

                    if len(selected) < n_target:
                        trial = selected + [cand]
                        delta = _covered_count(trial, specs) - current_cover
                        if delta > best_delta:
                            best_delta = delta
                            best_action = ("add", cand)
                    else:
                        for ridx in range(locked_count, len(selected)):
                            trial = list(selected)
                            trial[ridx] = cand
                            delta = _covered_count(trial, specs) - current_cover
                            if delta > best_delta:
                                best_delta = delta
                                best_action = ("swap", ridx, cand)

                if best_action is None:
                    break

                if best_action[0] == "add":
                    cand = best_action[1]
                    selected.append(cand)
                    used.add(tuple((s.name, cand[s.name]) for s in specs))
                else:
                    _, ridx, cand = best_action
                    old = selected[ridx]
                    used.discard(tuple((s.name, old[s.name]) for s in specs))
                    selected[ridx] = cand
                    used.add(tuple((s.name, cand[s.name]) for s in specs))

                uncovered = _pair_universe(specs)
                for c in selected:
                    _remove_covered(c, uncovered, specs)

        remaining = sum(len(v) for v in uncovered.values())
        covered = total_pairs - remaining
        pct = 100.0 * covered / total_pairs if total_pairs else 100.0
        return WarmStartResult(
            configs=selected,
            pairwise_coverage_percent=pct,
            covered_pairs=covered,
            total_pairs=total_pairs,
        )

    best_result = _build_once(seed)
    for r in range(1, max(1, restarts)):
        trial = _build_once(seed + r)
        if trial.covered_pairs > best_result.covered_pairs:
            best_result = trial
        if trial.covered_pairs == total_pairs:
            best_result = trial
            break
    return best_result


def pairwise_coverage_percent(configs: Iterable[Mapping[str, object]], specs: Sequence[ParameterSpec]) -> float:
    universe = _pair_universe(specs)
    total = sum(len(v) for v in universe.values())
    covered_sets = {k: set() for k in universe}
    for cfg in configs:
        for i in range(len(specs)):
            for j in range(i + 1, len(specs)):
                covered_sets[(i, j)].add((cfg[specs[i].name], cfg[specs[j].name]))
    covered = sum(len(v) for v in covered_sets.values())
    return 100.0 * covered / total if total else 100.0
