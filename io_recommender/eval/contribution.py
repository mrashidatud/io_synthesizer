from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from io_recommender.types import Observation


ACTIVE_SOURCES = {"active", "replicate"}


def _group_pattern_config(observations: Sequence[Observation]) -> Dict[str, Dict[str, List[Observation]]]:
    by_pattern: Dict[str, Dict[str, List[Observation]]] = {}
    for o in observations:
        by_pattern.setdefault(o.pattern_id, {}).setdefault(o.config_id, []).append(o)
    return by_pattern


def build_contribution_report(observations: Sequence[Observation], top_k: int = 3) -> dict:
    by_pattern_cfg = _group_pattern_config(observations)
    warm_best_by_pattern: Dict[str, float] = {}
    per_pattern: List[dict] = []

    total_topk = 0
    active_topk = 0

    for pid, cfg_map in sorted(by_pattern_cfg.items()):
        warm_best = max((o.gain for rows in cfg_map.values() for o in rows if o.source == "warm"), default=float("-inf"))
        active_best = max((o.gain for rows in cfg_map.values() for o in rows if o.source in ACTIVE_SOURCES), default=float("-inf"))
        warm_best_by_pattern[pid] = warm_best

        best_rows = []
        for cid, rows in cfg_map.items():
            best_obs = max(rows, key=lambda x: x.gain)
            best_rows.append(
                {
                    "config_id": cid,
                    "best_gain": float(best_obs.gain),
                    "best_source": str(best_obs.source),
                    "best_iter": int(best_obs.iteration),
                }
            )
        best_rows.sort(key=lambda x: x["best_gain"], reverse=True)
        top_rows = best_rows[: max(1, top_k)]
        total_topk += len(top_rows)
        active_count = sum(1 for r in top_rows if r["best_source"] in ACTIVE_SOURCES)
        active_topk += active_count

        new_best_found_iter = None
        for o in sorted(
            [x for rows in cfg_map.values() for x in rows if x.source in ACTIVE_SOURCES],
            key=lambda x: (x.iteration, x.gain),
        ):
            if o.gain > warm_best:
                new_best_found_iter = int(o.iteration)
                break

        per_pattern.append(
            {
                "pattern_id": pid,
                "best_warm_gain": float(warm_best) if np.isfinite(warm_best) else None,
                "best_active_gain": float(active_best) if np.isfinite(active_best) else None,
                "best_active_vs_best_warm_delta": float(active_best - warm_best)
                if np.isfinite(warm_best) and np.isfinite(active_best)
                else None,
                "new_best_found_iter": new_best_found_iter,
                "topk_from_active_percent": (100.0 * active_count / len(top_rows)) if top_rows else 0.0,
                "topk": top_rows,
            }
        )

    max_iter = max((int(o.iteration) for o in observations), default=0)
    trajectory = []
    pattern_ids = sorted(by_pattern_cfg.keys())
    for t in range(0, max_iter + 1):
        improved = 0
        deltas = []
        for pid in pattern_ids:
            warm_best = warm_best_by_pattern.get(pid, float("-inf"))
            best_so_far = max(
                (
                    o.gain
                    for rows in by_pattern_cfg.get(pid, {}).values()
                    for o in rows
                    if int(o.iteration) <= t
                ),
                default=warm_best,
            )
            delta = best_so_far - warm_best if np.isfinite(warm_best) else 0.0
            deltas.append(delta)
            if delta > 0:
                improved += 1
        trajectory.append(
            {
                "iter": int(t),
                "mean_improvement_over_warm": float(np.mean(deltas)) if deltas else 0.0,
                "patterns_improved": int(improved),
            }
        )

    return {
        "top_k": int(top_k),
        "pct_top_k_from_active": (100.0 * active_topk / total_topk) if total_topk else 0.0,
        "n_patterns": len(per_pattern),
        "per_pattern": per_pattern,
        "cumulative_improvement_trajectory": trajectory,
    }


def write_markdown_summary(
    out_path: Path,
    summary: Mapping[str, object],
    evaluation_report: Mapping[str, object],
    contribution_report: Mapping[str, object],
) -> None:
    lines: List[str] = []
    lines.append("# IO Recommender Run Summary")
    lines.append("")
    lines.append("## Core")
    lines.append(f"- runner_mode: `{summary.get('runner_mode', '')}`")
    lines.append(f"- oracle_mode: `{summary.get('oracle_mode', '')}`")
    lines.append(f"- oracle_data_source: `{summary.get('oracle_data_source', '')}`")
    lines.append(f"- total_observations: `{summary.get('total_observations', 0)}`")
    lines.append(f"- replicate_observations: `{summary.get('replicate_observations', 0)}`")
    lines.append("")

    lines.append("## Contribution")
    lines.append(f"- pct_top_k_from_active: `{contribution_report.get('pct_top_k_from_active', 0.0):.2f}%`")
    lines.append("")

    if evaluation_report.get("enabled"):
        agg = evaluation_report.get("aggregate", {})
        lines.append("## Evaluation")
        for key in ["top1_regret", "topk_regret", "hit_at_k", "ndcg_at_k"]:
            val = agg.get(key, {})
            lines.append(
                f"- {key}: mean=`{val.get('mean', 0.0):.4f}` "
                f"CI95=`[{val.get('ci_low', 0.0):.4f}, {val.get('ci_high', 0.0):.4f}]`"
            )
    else:
        lines.append("## Evaluation")
        lines.append(f"- disabled_reason: `{evaluation_report.get('reason', 'disabled')}`")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
