from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import numpy as np

from io_recommender.active.acquisition import select_hybrid
from io_recommender.active.candidates import generate_candidate_pool
from io_recommender.eval.metrics import hit_at_3_within, regret_at_3
from io_recommender.model import ConfigEncoder, EnsembleConfig, EnsembleModel, WorkloadEncoder
from io_recommender.types import Observation, ParameterSpec, WorkloadPattern, config_id_from_params


@dataclass
class ActiveLoopConfig:
    iterations: int = 8
    batch_per_iter: int = 3
    beta_start: float = 1.2
    beta_end: float = 0.4
    lambda_redundancy: float = 0.2
    ensemble_size: int = 6
    explore_mode: str = "ucb"
    model_mode: str = "ranking"
    use_lightgbm: bool = True
    early_stop_rounds: int = 0
    low_sigma_threshold: float = 0.0
    seed: int = 7


def _beta_for_iter(cfg: ActiveLoopConfig, t: int) -> float:
    if cfg.iterations <= 1:
        return cfg.beta_end
    alpha = (t - 1) / (cfg.iterations - 1)
    return cfg.beta_start + alpha * (cfg.beta_end - cfg.beta_start)


def _best_by_pattern(obs: Sequence[Observation]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for o in obs:
        out[o.pattern_id] = max(out.get(o.pattern_id, float("-inf")), o.gain)
    return out


def _ndcg3_from_observed(obs: Sequence[Observation], oracle_best: Mapping[str, float]) -> float:
    # proxy NDCG@3 over observed gains using each workload's own ideal ordering
    vals = []
    by_pattern: Dict[str, List[Observation]] = {}
    for o in obs:
        by_pattern.setdefault(o.pattern_id, []).append(o)
    for pid in oracle_best:
        ranked = sorted(by_pattern.get(pid, []), key=lambda x: x.gain, reverse=True)
        rel = [max(0.0, x.gain) for x in ranked]
        ideal = sorted(rel, reverse=True)
        if not ideal:
            vals.append(0.0)
            continue
        dcg = 0.0
        for i, r in enumerate(rel[:3]):
            dcg += (2**r - 1) / np.log2(i + 2)
        idcg = 0.0
        for i, r in enumerate(ideal[:3]):
            idcg += (2**r - 1) / np.log2(i + 2)
        vals.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def run_active_loop(
    patterns: Sequence[WorkloadPattern],
    specs: Sequence[ParameterSpec],
    workload_encoder: WorkloadEncoder,
    config_encoder: ConfigEncoder,
    runner,
    baseline_perf_by_pattern: Mapping[str, float],
    initial_observations: Sequence[Observation],
    cfg: ActiveLoopConfig,
    oracle_best_gain_by_pattern: Mapping[str, float] | None = None,
) -> tuple[list[Observation], EnsembleModel, list[dict]]:
    observations = list(initial_observations)
    ensemble = EnsembleModel(
        EnsembleConfig(
            mode=cfg.model_mode,
            ensemble_size=cfg.ensemble_size,
            seed=cfg.seed,
            use_lightgbm=cfg.use_lightgbm,
        )
    )

    history: List[dict] = []
    rounds_without_improve = 0
    prev_global_best = max((o.gain for o in observations), default=float("-inf"))

    for t in range(1, cfg.iterations + 1):
        ensemble.fit(observations)
        beta = _beta_for_iter(cfg, t)

        new_batch: List[Observation] = []

        for p_idx, pattern in enumerate(patterns):
            wvec = workload_encoder.encode_workload(pattern)
            pid_obs = [o for o in observations if o.pattern_id == pattern.pattern_id]
            tested_configs = [o.config_params for o in pid_obs]
            top_cfgs = [o.config_params for o in sorted(pid_obs, key=lambda x: x.gain, reverse=True)[:5]]

            candidates = generate_candidate_pool(
                specs,
                observations,
                pattern.pattern_id,
                top_configs=top_cfgs,
                seed=cfg.seed + 1000 * t + p_idx,
            )
            if not candidates:
                continue

            Xcand = np.vstack([
                np.concatenate([wvec, config_encoder.encode_config(c)])
                for c in candidates
            ])
            mu, sigma = ensemble.predict_mean_std(Xcand)

            picked = select_hybrid(
                candidates,
                mu,
                sigma,
                tested_configs,
                specs,
                b=cfg.batch_per_iter,
                beta=beta,
                lam=cfg.lambda_redundancy,
                explore_mode=cfg.explore_mode,
                seed=cfg.seed + t + p_idx,
            )

            for conf in picked:
                perf = runner.run_testbed(pattern.pattern_id, conf, wvec)
                base = baseline_perf_by_pattern[pattern.pattern_id]
                obs = Observation(
                    pattern_id=pattern.pattern_id,
                    config_id=config_id_from_params(conf),
                    config_params=dict(conf),
                    workload_vec=wvec,
                    config_vec=config_encoder.encode_config(conf),
                    perf=perf,
                    baseline_perf=base,
                    gain=perf - base,
                )
                new_batch.append(obs)

        observations.extend(new_batch)

        best_now = max((o.gain for o in observations), default=float("-inf"))
        rounds_without_improve = rounds_without_improve + 1 if best_now <= prev_global_best else 0
        prev_global_best = max(prev_global_best, best_now)

        best3 = _best_by_pattern(observations)
        oracle = dict(oracle_best_gain_by_pattern or best3)
        history.append(
            {
                "iter": t,
                "runs": len(observations),
                "best_gain_so_far": float(best_now),
                "regret_at_3": regret_at_3(best3, oracle),
                "hit_at_3": hit_at_3_within(best3, oracle, tol=0.05),
                "ndcg_at_3": _ndcg3_from_observed(observations, oracle),
            }
        )

        if cfg.early_stop_rounds > 0 and rounds_without_improve >= cfg.early_stop_rounds:
            break
        if cfg.low_sigma_threshold > 0:
            if new_batch:
                sigma_proxy = np.std([o.gain for o in new_batch])
                if sigma_proxy < cfg.low_sigma_threshold:
                    break

    ensemble.fit(observations)
    return observations, ensemble, history
