from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from io_recommender.eval.metrics import ndcg_at_k
from io_recommender.model import ConfigEncoder, EnsembleConfig, EnsembleModel, WorkloadEncoder
from io_recommender.types import Observation, WorkloadPattern


def _split_heldout_patterns(
    observations: Sequence[Observation],
    holdout_fraction: float,
    seed: int,
) -> tuple[list[Observation], list[Observation], dict]:
    pids = sorted({o.pattern_id for o in observations})
    if len(pids) <= 1:
        return list(observations), [], {"split_mode_effective": "heldout_patterns", "heldout_patterns": []}

    n_holdout = max(1, int(round(len(pids) * holdout_fraction)))
    n_holdout = min(n_holdout, len(pids) - 1)
    rng = np.random.default_rng(seed)
    heldout = set(rng.choice(np.array(pids, dtype=object), size=n_holdout, replace=False).tolist())

    train = [o for o in observations if o.pattern_id not in heldout]
    test = [o for o in observations if o.pattern_id in heldout]
    return train, test, {"split_mode_effective": "heldout_patterns", "heldout_patterns": sorted(heldout)}


def _split_heldout_configs_per_pattern(
    observations: Sequence[Observation],
    holdout_fraction: float,
    seed: int,
) -> tuple[list[Observation], list[Observation], dict]:
    rng = np.random.default_rng(seed)
    by_pattern_cfg: Dict[str, set[str]] = {}
    for o in observations:
        by_pattern_cfg.setdefault(o.pattern_id, set()).add(o.config_id)

    heldout_cfgs: Dict[str, set[str]] = {}
    for pid, cfg_ids in by_pattern_cfg.items():
        cfgs = sorted(cfg_ids)
        if len(cfgs) <= 1:
            continue
        n_holdout = max(1, int(round(len(cfgs) * holdout_fraction)))
        n_holdout = min(n_holdout, len(cfgs) - 1)
        picked = rng.choice(np.array(cfgs, dtype=object), size=n_holdout, replace=False).tolist()
        heldout_cfgs[pid] = set(picked)

    train: List[Observation] = []
    test: List[Observation] = []
    for o in observations:
        if o.config_id in heldout_cfgs.get(o.pattern_id, set()):
            test.append(o)
        else:
            train.append(o)

    md = {
        "split_mode_effective": "heldout_configs_per_pattern",
        "heldout_config_counts_by_pattern": {k: len(v) for k, v in heldout_cfgs.items()},
    }
    return train, test, md


def _split_temporal(
    observations: Sequence[Observation],
    holdout_fraction: float,
) -> tuple[list[Observation], list[Observation], dict]:
    max_iter = max((int(o.iteration) for o in observations), default=0)
    if max_iter <= 0:
        return list(observations), [], {"split_mode_effective": "temporal", "cutoff_iter": 0, "applied": False}

    cutoff = int(np.floor(max_iter * (1.0 - holdout_fraction)))
    cutoff = min(max(cutoff, 0), max_iter - 1)
    train = [o for o in observations if int(o.iteration) <= cutoff]
    test = [o for o in observations if int(o.iteration) > cutoff]
    return train, test, {"split_mode_effective": "temporal", "cutoff_iter": cutoff, "applied": True}


def _bootstrap_ci(values: Sequence[float], n_bootstrap: int, ci_level: float, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        v = float(arr[0])
        return v, v

    rng = np.random.default_rng(seed)
    boots = []
    n = arr.size
    for _ in range(max(1, n_bootstrap)):
        sample = rng.choice(arr, size=n, replace=True)
        boots.append(float(sample.mean()))
    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    return lo, hi


def _mean_gain_by_pattern_config(
    observations: Sequence[Observation],
) -> Dict[str, List[dict]]:
    by_pattern_cfg: Dict[Tuple[str, str], List[Observation]] = {}
    for o in observations:
        by_pattern_cfg.setdefault((o.pattern_id, o.config_id), []).append(o)

    out: Dict[str, List[dict]] = {}
    for (pid, cid), rows in by_pattern_cfg.items():
        gains = np.asarray([r.gain for r in rows], dtype=float)
        out.setdefault(pid, []).append(
            {
                "pattern_id": pid,
                "config_id": cid,
                "config_params": dict(rows[0].config_params),
                "actual_gain": float(gains.mean()) if gains.size else float("-inf"),
                "n_obs": int(gains.size),
            }
        )
    return out


def evaluate_on_heldout(
    observations: Sequence[Observation],
    patterns: Sequence[WorkloadPattern],
    workload_encoder: WorkloadEncoder,
    config_encoder: ConfigEncoder,
    ensemble_cfg: EnsembleConfig,
    split_mode: str = "heldout_configs_per_pattern",
    holdout_fraction_configs: float = 0.2,
    holdout_fraction_patterns: float = 0.2,
    temporal_split: bool = False,
    temporal_holdout_fraction: float = 0.2,
    top_k: int = 3,
    hit_tolerance: float = 0.05,
    bootstrap_iters: int = 200,
    ci_level: float = 0.95,
    seed: int = 7,
) -> dict:
    if not observations:
        return {"enabled": False, "reason": "no_observations"}

    if temporal_split:
        train_obs, test_obs, split_meta = _split_temporal(observations, temporal_holdout_fraction)
    elif split_mode == "heldout_patterns":
        train_obs, test_obs, split_meta = _split_heldout_patterns(observations, holdout_fraction_patterns, seed=seed)
    elif split_mode == "heldout_configs_per_pattern":
        train_obs, test_obs, split_meta = _split_heldout_configs_per_pattern(observations, holdout_fraction_configs, seed=seed)
    else:
        raise ValueError(f"Unsupported split_mode={split_mode}")

    if not train_obs or not test_obs:
        return {
            "enabled": False,
            "reason": "empty_train_or_test_split",
            "split_mode_requested": split_mode,
            "temporal_split": temporal_split,
            "split_meta": split_meta,
            "n_train_observations": len(train_obs),
            "n_test_observations": len(test_obs),
        }

    pattern_by_id = {p.pattern_id: p for p in patterns}
    grouped_test = _mean_gain_by_pattern_config(test_obs)
    eval_patterns = sorted(grouped_test.keys())

    ensemble = EnsembleModel(ensemble_cfg).fit(train_obs)
    per_pattern: List[dict] = []

    for pid in eval_patterns:
        rows = grouped_test[pid]
        if not rows:
            continue

        if pid in pattern_by_id:
            wvec = workload_encoder.encode_workload(pattern_by_id[pid])
        else:
            # Fallback when workload id is missing from provided pattern list.
            wvec = np.asarray([o.workload_vec for o in test_obs if o.pattern_id == pid][0], dtype=float)

        X = np.vstack([
            np.concatenate([wvec, config_encoder.encode_config(r["config_params"])])
            for r in rows
        ])
        mu, _ = ensemble.predict_mean_std(X)
        order = np.argsort(mu)[::-1]
        ranked = [rows[i] for i in order]

        actual_ranked = np.asarray([float(r["actual_gain"]) for r in ranked], dtype=float)
        actual_all = np.asarray([float(r["actual_gain"]) for r in rows], dtype=float)
        oracle_best = float(actual_all.max()) if actual_all.size else 0.0
        top1_gain = float(actual_ranked[0]) if actual_ranked.size else float("-inf")
        topk_gain = float(actual_ranked[:top_k].max()) if actual_ranked.size else float("-inf")

        denom = max(abs(oracle_best), 1e-9)
        top1_regret = float((oracle_best - top1_gain) / denom)
        topk_regret = float((oracle_best - topk_gain) / denom)
        hit_at_k = 1.0 if topk_gain >= (1.0 - hit_tolerance) * oracle_best else 0.0
        rel_ranked = [max(0.0, x) for x in actual_ranked.tolist()]
        rel_ideal = sorted([max(0.0, x) for x in actual_all.tolist()], reverse=True)
        ndcg_k = float(ndcg_at_k(rel_ranked, rel_ideal, k=top_k))

        per_pattern.append(
            {
                "pattern_id": pid,
                "n_candidates": int(len(rows)),
                "oracle_best_gain": oracle_best,
                "pred_top1_gain": top1_gain,
                "pred_topk_best_gain": topk_gain,
                "top1_regret": top1_regret,
                "topk_regret": topk_regret,
                "hit_at_k": hit_at_k,
                "ndcg_at_k": ndcg_k,
                "pred_top1_config_id": ranked[0]["config_id"] if ranked else "",
            }
        )

    metrics = {
        "top1_regret": [p["top1_regret"] for p in per_pattern],
        "topk_regret": [p["topk_regret"] for p in per_pattern],
        "hit_at_k": [p["hit_at_k"] for p in per_pattern],
        "ndcg_at_k": [p["ndcg_at_k"] for p in per_pattern],
    }
    aggregate: Dict[str, dict] = {}
    for name, values in metrics.items():
        vals = np.asarray(values, dtype=float)
        mean = float(vals.mean()) if vals.size else 0.0
        lo, hi = _bootstrap_ci(values, n_bootstrap=bootstrap_iters, ci_level=ci_level, seed=seed + 13 * (len(name) + 1))
        aggregate[name] = {"mean": mean, "ci_low": lo, "ci_high": hi}

    return {
        "enabled": True,
        "split_mode_requested": split_mode,
        "temporal_split": temporal_split,
        "split_meta": split_meta,
        "oracle_data_source": "heldout_test_observations",
        "top_k": int(top_k),
        "n_train_observations": len(train_obs),
        "n_test_observations": len(test_obs),
        "n_eval_patterns": len(per_pattern),
        "aggregate": aggregate,
        "per_pattern": per_pattern,
    }
