from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import numpy as np

from io_recommender.active.acquisition import select_hybrid
from io_recommender.active.candidates import generate_candidate_pool_details
from io_recommender.eval.metrics import hit_at_3_within, ndcg_at_k, regret_at_3
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
    oracle_mode: str = "warm_only"
    enum_threshold_hard: int = 52_920
    candidate_max_pool: int = 12_000
    replicate_top_n: int = 0
    replicate_repeats: int = 1
    robust_score_z: float = 1.0


def _beta_for_iter(cfg: ActiveLoopConfig, t: int) -> float:
    if cfg.iterations <= 1:
        return cfg.beta_end
    alpha = (t - 1) / (cfg.iterations - 1)
    return cfg.beta_start + alpha * (cfg.beta_end - cfg.beta_start)


def _config_stats(obs: Sequence[Observation], robust_z: float) -> List[dict]:
    by_cfg: Dict[str, List[Observation]] = {}
    for o in obs:
        by_cfg.setdefault(o.config_id, []).append(o)

    out: List[dict] = []
    for cid, rows in by_cfg.items():
        gains = np.asarray([r.gain for r in rows], dtype=float)
        mean_gain = float(gains.mean()) if gains.size else float("-inf")
        std_gain = float(gains.std(ddof=1)) if gains.size > 1 else 0.0
        cv = std_gain / (abs(mean_gain) + 1e-9) if gains.size > 1 else 0.0
        ci95 = 1.96 * std_gain / np.sqrt(max(int(gains.size), 1))
        robust = mean_gain - robust_z * std_gain
        out.append(
            {
                "config_id": cid,
                "config_params": dict(rows[0].config_params),
                "n": int(gains.size),
                "mean_gain": mean_gain,
                "std_gain": std_gain,
                "cv_gain": cv,
                "ci95_gain": ci95,
                "robust_score": robust,
            }
        )
    out.sort(key=lambda x: x["robust_score"], reverse=True)
    return out


def _top_configs_by_robust(obs: Sequence[Observation], top_n: int, robust_z: float) -> List[dict]:
    return [x["config_params"] for x in _config_stats(obs, robust_z=robust_z)[:top_n]]


def _best_by_pattern(obs: Sequence[Observation], robust_z: float) -> Dict[str, float]:
    by_pattern: Dict[str, List[Observation]] = {}
    for o in obs:
        by_pattern.setdefault(o.pattern_id, []).append(o)
    out: Dict[str, float] = {}
    for pid, rows in by_pattern.items():
        stats = _config_stats(rows, robust_z=robust_z)
        if not stats:
            continue
        out[pid] = float(stats[0]["mean_gain"])
    return out


def _ndcg3_vs_oracle(obs: Sequence[Observation], oracle_best: Mapping[str, float], robust_z: float) -> float:
    vals: List[float] = []
    by_pattern: Dict[str, List[Observation]] = {}
    for o in obs:
        by_pattern.setdefault(o.pattern_id, []).append(o)

    for pid, oracle_gain in oracle_best.items():
        stats = _config_stats(by_pattern.get(pid, []), robust_z=robust_z)
        rel = [max(0.0, float(s["mean_gain"])) for s in stats]
        ideal = [max(0.0, float(oracle_gain)), 0.0, 0.0]
        vals.append(float(ndcg_at_k(rel, ideal, k=3)))
    return float(np.mean(vals)) if vals else 0.0


def _uncertainty_snapshot(
    patterns: Sequence[WorkloadPattern],
    observations: Sequence[Observation],
    robust_z: float,
    top_n: int = 3,
) -> Dict[str, List[dict]]:
    by_pattern: Dict[str, List[Observation]] = {}
    for o in observations:
        by_pattern.setdefault(o.pattern_id, []).append(o)
    out: Dict[str, List[dict]] = {}
    for p in patterns:
        stats = _config_stats(by_pattern.get(p.pattern_id, []), robust_z=robust_z)[:top_n]
        out[p.pattern_id] = [
            {
                "config_id": s["config_id"],
                "mean_gain": float(s["mean_gain"]),
                "std_gain": float(s["std_gain"]),
                "cv_gain": float(s["cv_gain"]),
                "ci95_gain": float(s["ci95_gain"]),
                "n": int(s["n"]),
                "robust_score": float(s["robust_score"]),
            }
            for s in stats
        ]
    return out


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

    oracle = (
        dict(oracle_best_gain_by_pattern)
        if oracle_best_gain_by_pattern is not None
        else _best_by_pattern(initial_observations, robust_z=cfg.robust_score_z)
    )

    history: List[dict] = []
    rounds_without_improve = 0
    prev_global_best = max((o.gain for o in observations), default=float("-inf"))

    for t in range(1, cfg.iterations + 1):
        ensemble.fit(observations)
        beta = _beta_for_iter(cfg, t)

        new_batch: List[Observation] = []
        candidate_mode_counts: Dict[str, int] = {"enumerated": 0, "sampled": 0}
        candidate_mode_by_pattern: Dict[str, str] = {}

        for p_idx, pattern in enumerate(patterns):
            wvec = workload_encoder.encode_workload(pattern)
            pid_obs = [o for o in observations if o.pattern_id == pattern.pattern_id]
            tested_configs = [o.config_params for o in pid_obs]
            top_cfgs = _top_configs_by_robust(pid_obs, top_n=5, robust_z=cfg.robust_score_z)

            pool_info = generate_candidate_pool_details(
                specs=specs,
                observations=observations,
                pattern_id=pattern.pattern_id,
                top_configs=top_cfgs,
                seed=cfg.seed + 1000 * t + p_idx,
                enum_threshold_hard=cfg.enum_threshold_hard,
                max_pool=cfg.candidate_max_pool,
            )
            candidates = pool_info.configs
            candidate_mode_counts[pool_info.mode] = candidate_mode_counts.get(pool_info.mode, 0) + 1
            candidate_mode_by_pattern[pattern.pattern_id] = pool_info.mode
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
                    source="active",
                    iteration=t,
                    replicate_index=0,
                )
                new_batch.append(obs)

        replicate_runs = 0
        if cfg.replicate_top_n > 0 and cfg.replicate_repeats > 0:
            active_new_by_pattern: Dict[str, List[Observation]] = {}
            for o in new_batch:
                if o.source == "active":
                    active_new_by_pattern.setdefault(o.pattern_id, []).append(o)

            replicate_batch: List[Observation] = []
            for p in patterns:
                rows = sorted(
                    active_new_by_pattern.get(p.pattern_id, []),
                    key=lambda x: x.gain,
                    reverse=True,
                )
                uniq: List[Observation] = []
                seen_cfg: set[str] = set()
                for row in rows:
                    if row.config_id in seen_cfg:
                        continue
                    seen_cfg.add(row.config_id)
                    uniq.append(row)
                    if len(uniq) >= cfg.replicate_top_n:
                        break

                for row in uniq:
                    for rep_idx in range(1, cfg.replicate_repeats + 1):
                        perf = runner.run_testbed(p.pattern_id, row.config_params, row.workload_vec)
                        rep_obs = Observation(
                            pattern_id=p.pattern_id,
                            config_id=row.config_id,
                            config_params=dict(row.config_params),
                            workload_vec=row.workload_vec,
                            config_vec=row.config_vec,
                            perf=perf,
                            baseline_perf=row.baseline_perf,
                            gain=perf - row.baseline_perf,
                            source="replicate",
                            iteration=t,
                            replicate_index=rep_idx,
                        )
                        replicate_batch.append(rep_obs)
                        replicate_runs += 1
            new_batch.extend(replicate_batch)

        observations.extend(new_batch)

        best_now = max((o.gain for o in observations), default=float("-inf"))
        rounds_without_improve = rounds_without_improve + 1 if best_now <= prev_global_best else 0
        prev_global_best = max(prev_global_best, best_now)

        best3 = _best_by_pattern(observations, robust_z=cfg.robust_score_z)
        history.append(
            {
                "iter": t,
                "runs": len(observations),
                "best_gain_so_far": float(best_now),
                "regret_at_3": regret_at_3(best3, oracle),
                "hit_at_3": hit_at_3_within(best3, oracle, tol=0.05),
                "ndcg_at_3": _ndcg3_vs_oracle(observations, oracle, robust_z=cfg.robust_score_z),
                "candidate_mode_counts": candidate_mode_counts,
                "candidate_mode_by_pattern": candidate_mode_by_pattern,
                "replicate_runs": int(replicate_runs),
                "uncertainty_top": _uncertainty_snapshot(
                    patterns=patterns,
                    observations=observations,
                    robust_z=cfg.robust_score_z,
                    top_n=3,
                ),
            }
        )

        if cfg.early_stop_rounds > 0 and rounds_without_improve >= cfg.early_stop_rounds:
            break
        if cfg.low_sigma_threshold > 0 and new_batch:
            sigma_proxy = np.std([o.gain for o in new_batch])
            if sigma_proxy < cfg.low_sigma_threshold:
                break

    ensemble.fit(observations)
    return observations, ensemble, history
