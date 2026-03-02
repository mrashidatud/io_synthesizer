#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WARM_ORCH_DIR = REPO_ROOT / "orchestrators" / "warm_start_sampling_orchestrator"
for _p in (str(REPO_ROOT), str(WARM_ORCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from io_recommender.deploy import DeploymentRecommender, materialize_recommendation_matrix
from io_recommender.model import ConfigEncoder, EnsembleConfig, EnsembleModel, WorkloadEncoder
from io_recommender.pipeline import baseline_from_specs, parse_specs
from io_recommender.types import Observation, ParameterSpec, WorkloadPattern, config_id_from_params
from warm_start_pipeline import append_csv, apply_lustre_knobs, ensure_mpi_binary, run_cmd, ts, workload_key


@dataclass
class ParsedRow:
    pattern_id: str
    config: Dict[str, object]
    config_id: str
    metric: float


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _coerce_spec_value(raw: str, spec: ParameterSpec) -> object:
    s = str(raw).strip()
    for v in spec.values:
        if s == str(v):
            return v
    try:
        f = float(s)
    except Exception:
        f = None
    if f is not None:
        for v in spec.values:
            if isinstance(v, (int, float)) and float(v) == f:
                return v
    raise ValueError(f"Value '{raw}' for '{spec.name}' not in allowed values {spec.values}")


def _coerce_config_from_row(row: Mapping[str, str], specs: Sequence[ParameterSpec]) -> Dict[str, object]:
    cfg: Dict[str, object] = {}
    for spec in specs:
        if spec.name not in row:
            raise KeyError(f"Missing config column '{spec.name}'")
        cfg[spec.name] = _coerce_spec_value(row[spec.name], spec)
    return cfg


def _pattern_from_workload_key(workload_key_value: str) -> str:
    for pid in ("top1_101", "top2_20", "top3_18", "top4_192"):
        if workload_key_value.startswith(pid):
            return pid
    # Fallback for already-short form
    return workload_key_value.split("_cap_")[0]


def _load_rows(csv_path: Path, specs: Sequence[ParameterSpec], has_pattern_id: bool) -> List[ParsedRow]:
    out: List[ParsedRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_raw = (row.get("metric") or "").strip()
            if not metric_raw:
                continue
            try:
                cfg = _coerce_config_from_row(row, specs)
            except Exception:
                continue
            pattern_id = (row.get("pattern_id") or "").strip() if has_pattern_id else ""
            if not pattern_id:
                pattern_id = _pattern_from_workload_key(row.get("workload_key", ""))
            cfg_id = (row.get("config_id") or "").strip() or config_id_from_params(cfg)
            out.append(ParsedRow(pattern_id=pattern_id, config=cfg, config_id=cfg_id, metric=float(metric_raw)))
    return out


def _load_patterns_from_paths(workload_jsons: Sequence[Path], feature_prefix: str = "pct_") -> tuple[List[WorkloadPattern], List[str]]:
    if not workload_jsons:
        raise ValueError("No workload json files")

    first = json.loads(workload_jsons[0].read_text(encoding="utf-8"))
    feature_names = sorted([k for k, v in first.items() if k.startswith(feature_prefix) and isinstance(v, (int, float))])
    if not feature_names:
        feature_names = sorted([k for k, v in first.items() if isinstance(v, (int, float))])
    if not feature_names:
        raise ValueError(f"No numeric features found in {workload_jsons[0]}")

    patterns: List[WorkloadPattern] = []
    for p in workload_jsons:
        obj = json.loads(p.read_text(encoding="utf-8"))
        feat = np.array([float(obj.get(k, 0.0)) for k in feature_names], dtype=float)
        patterns.append(WorkloadPattern(pattern_id=p.stem, features=feat))
    return patterns, feature_names


def _best_match_per_reference(closest_json: Mapping[str, object], references: Sequence[str]) -> Dict[str, Dict[str, object]]:
    ranked = closest_json.get("ranked_matches", [])
    if not isinstance(ranked, list):
        raise ValueError("closest file missing ranked_matches list")

    out: Dict[str, Dict[str, object]] = {}
    for item in ranked:
        if not isinstance(item, dict):
            continue
        ref = str(item.get("closest_reference", ""))
        if ref not in references:
            continue
        dist = float(item.get("weighted_distance", float("inf")))
        prev = out.get(ref)
        if prev is None or dist < float(prev.get("weighted_distance", float("inf"))):
            out[ref] = item

    missing = [r for r in references if r not in out]
    if missing:
        raise RuntimeError(f"Missing best match entries for references: {missing}")
    return out


def _make_output_root(base: Path) -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = base / f"recommendation_quality_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    return out


def _fit_deployment_recommender(
    config_path: Path,
    warm_obs_csv: Path,
    active_obs_csv: Path,
    training_workload_jsons: Sequence[Path],
    ensemble_size: int,
    seed: int,
    use_lightgbm: bool,
    model_mode: str,
    topk_per_pattern: int,
    knn_k: int,
) -> tuple[
    DeploymentRecommender,
    Dict[str, float],
    Dict[str, object],
    Sequence[ParameterSpec],
    Dict[str, object],
    List[str],
]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    specs = parse_specs(cfg)
    baseline_cfg = baseline_from_specs(specs, override=cfg.get("baseline"))
    baseline_id = config_id_from_params(baseline_cfg)

    patterns, feature_names = _load_patterns_from_paths(
        training_workload_jsons,
        feature_prefix=str(cfg.get("data", {}).get("feature_prefix", "pct_")),
    )
    pattern_by_id = {p.pattern_id: p for p in patterns}

    warm_rows = _load_rows(warm_obs_csv, specs, has_pattern_id=False)
    active_rows = _load_rows(active_obs_csv, specs, has_pattern_id=True)

    valid_pids = set(pattern_by_id)
    warm_rows = [r for r in warm_rows if r.pattern_id in valid_pids]
    active_rows = [r for r in active_rows if r.pattern_id in valid_pids]

    baseline_perf: Dict[str, float] = {}
    for pid in valid_pids:
        vals = [r.metric for r in warm_rows if r.pattern_id == pid and r.config_id == baseline_id]
        if not vals:
            raise RuntimeError(f"Baseline config {baseline_id} not found in warm rows for {pid}")
        baseline_perf[pid] = float(statistics.mean(vals))

    workload_encoder = WorkloadEncoder().fit(patterns)
    config_encoder = ConfigEncoder(specs).fit()

    observations: List[Observation] = []
    for r in warm_rows + active_rows:
        pat = pattern_by_id[r.pattern_id]
        base = baseline_perf[r.pattern_id]
        observations.append(
            Observation(
                pattern_id=r.pattern_id,
                config_id=r.config_id,
                config_params=dict(r.config),
                workload_vec=workload_encoder.encode_workload(pat),
                config_vec=config_encoder.encode_config(r.config),
                perf=r.metric,
                baseline_perf=base,
                gain=r.metric - base,
            )
        )

    ensemble = EnsembleModel(
        EnsembleConfig(
            mode=model_mode,
            ensemble_size=ensemble_size,
            seed=seed,
            use_lightgbm=use_lightgbm,
        )
    )
    ensemble.fit(observations)

    rec_matrix = materialize_recommendation_matrix(observations, top_k=topk_per_pattern)
    deploy = DeploymentRecommender(
        workload_encoder=workload_encoder,
        config_encoder=config_encoder,
        ensemble=ensemble,
        patterns=patterns,
        rec_matrix=rec_matrix,
        knn_k=knn_k,
    )

    meta = {
        "baseline_config_id": baseline_id,
        "baseline_config": baseline_cfg,
        "training_pattern_ids": sorted(valid_pids),
        "warm_rows": len(warm_rows),
        "active_rows": len(active_rows),
        "total_observations": len(observations),
    }
    return deploy, baseline_perf, meta, specs, cfg, feature_names


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate recommender quality on closest non-training workloads")
    ap.add_argument("--closest-json", default="/mnt/hasanfs/io_synthesizer/outputs/closest_top25_to_top4_weighted.json")
    ap.add_argument("--inputs-dir", default="/mnt/hasanfs/io_synthesizer/inputs/exemplar_jsons")
    ap.add_argument("--config", default="/mnt/hasanfs/io_synthesizer/io_recommender/config.yaml")
    ap.add_argument("--warm-observations", default="/mnt/hasanfs/samples/warm-start/observations_all.csv")
    ap.add_argument("--active-observations", default="/mnt/hasanfs/samples/active-sampling/observations_all.csv")
    ap.add_argument("--workload-contexts", default="/mnt/hasanfs/samples/active-sampling/workload_contexts.json")
    ap.add_argument("--samples-base", default="/mnt/hasanfs/samples/validation")
    ap.add_argument("--output-root", default="")

    ap.add_argument("--cap", type=float, default=128.0)
    ap.add_argument("--nprocs", type=int, default=1)
    ap.add_argument("--io-api", default="posix")
    ap.add_argument("--meta-api", default="posix")
    ap.add_argument("--coll", default="none")
    ap.add_argument("--meta-scope", default="separate")

    ap.add_argument("--metric-key", default="agg_perf_by_slowest")
    ap.add_argument("--flush-wait-sec", type=float, default=10.0)
    ap.add_argument("--delete-darshan", action="store_true", default=True)
    ap.add_argument("--use-sudo-lustre", action="store_true", default=False)

    ap.add_argument("--ensemble-size", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model-mode", default="ranking")
    ap.add_argument("--use-lightgbm", action="store_true", default=True)
    ap.add_argument("--topk-per-pattern", type=int, default=20)
    ap.add_argument("--knn-k", type=int, default=5)

    ap.add_argument("--bin-dir", default="/mnt/hasanfs/bin")
    ap.add_argument("--force-build", action="store_true", default=False)

    args = ap.parse_args()

    closest_json = Path(args.closest_json)
    inputs_dir = Path(args.inputs_dir)
    cfg_path = Path(args.config)
    warm_obs_csv = Path(args.warm_observations)
    active_obs_csv = Path(args.active_observations)
    contexts_path = Path(args.workload_contexts)

    if args.output_root:
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "logs").mkdir(parents=True, exist_ok=True)
    else:
        output_root = _make_output_root(Path(args.samples_base))

    print(f"{ts()}  Validation output root: {output_root}")

    ensure_mpi_binary(REPO_ROOT, Path(args.bin_dir), args.force_build)

    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    training_pids = sorted(contexts.keys())
    training_jsons = [Path(contexts[pid]["workload_json"]) for pid in training_pids]

    closest_obj = json.loads(closest_json.read_text(encoding="utf-8"))
    best_matches = _best_match_per_reference(closest_obj, training_pids)

    selected = []
    for ref in training_pids:
        item = best_matches[ref]
        cand = str(item["candidate"])
        wjson = inputs_dir / f"{cand}.json"
        if not wjson.exists():
            raise FileNotFoundError(f"Selected workload json not found: {wjson}")
        selected.append(
            {
                "reference_pattern": ref,
                "candidate_pattern": cand,
                "workload_json": str(wjson),
                "weighted_distance": float(item.get("weighted_distance", 0.0)),
                "candidate_rank": int(item.get("candidate_rank", -1)),
            }
        )

    (output_root / "selected_best_matches.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    deploy, baseline_perf_train, train_meta, specs, loaded_cfg, feature_names = _fit_deployment_recommender(
        config_path=cfg_path,
        warm_obs_csv=warm_obs_csv,
        active_obs_csv=active_obs_csv,
        training_workload_jsons=training_jsons,
        ensemble_size=args.ensemble_size,
        seed=args.seed,
        use_lightgbm=args.use_lightgbm,
        model_mode=args.model_mode,
        topk_per_pattern=args.topk_per_pattern,
        knn_k=args.knn_k,
    )

    baseline_cfg = train_meta["baseline_config"]
    baseline_id = str(train_meta["baseline_config_id"])

    global_csv = output_root / "observations_all.csv"
    global_results: List[Dict[str, object]] = []
    workload_reports: List[Dict[str, object]] = []

    for idx, sel in enumerate(selected):
        ref = str(sel["reference_pattern"])
        cand = str(sel["candidate_pattern"])
        workload_json = Path(sel["workload_json"])

        features = json.loads(workload_json.read_text(encoding="utf-8"))
        # DeploymentRecommender expects a vector compatible with training feature order.
        wvec = np.array([float(features.get(name, 0.0)) for name in feature_names], dtype=float)
        recs = deploy.recommend(wvec, top_k=3)
        if len(recs) < 3:
            raise RuntimeError(f"Expected 3 recommendations for {cand}, got {len(recs)}")

        wkey = workload_key(cand, args.cap, args.nprocs, args.io_api, args.meta_api, args.coll)
        workload_dir = output_root / wkey
        workload_dir.mkdir(parents=True, exist_ok=True)

        run_sh = workload_dir / "run_from_features.sh"
        plan_csv = workload_dir / "payload" / "plan.csv"
        if not (run_sh.exists() and plan_csv.exists()):
            cmd = [
                "python3",
                str(REPO_ROOT / "scripts" / "features2synth_opsaware.py"),
                "--features",
                str(workload_json),
                "--cap-total-gib",
                str(args.cap),
                "--io-api",
                args.io_api,
                "--meta-api",
                args.meta_api,
                "--mpi-collective-mode",
                args.coll,
                "--meta-scope",
                args.meta_scope,
                "--nprocs",
                str(args.nprocs),
                "--outdir",
                str(workload_dir),
            ]
            run_cmd(cmd, cwd=REPO_ROOT)

        workload_manifest = {
            "workload_key": wkey,
            "reference_pattern": ref,
            "candidate_pattern": cand,
            "workload_json": str(workload_json),
            "weighted_distance": float(sel["weighted_distance"]),
            "candidate_rank": int(sel["candidate_rank"]),
            "nprocs": args.nprocs,
            "cap_total_gib": args.cap,
            "io_api": args.io_api,
            "meta_api": args.meta_api,
            "collective": args.coll,
            "meta_scope": args.meta_scope,
            "recommendations": recs,
            "baseline_config_id": baseline_id,
            "baseline_config": baseline_cfg,
        }
        (workload_dir / "validation_manifest.json").write_text(json.dumps(workload_manifest, indent=2), encoding="utf-8")

        run_plan = [
            {"label": "baseline", "config_id": baseline_id, "config": baseline_cfg, "score": None},
            {"label": "rec1", "config_id": recs[0]["config_id"], "config": recs[0]["config"], "score": float(recs[0]["score"])},
            {"label": "rec2", "config_id": recs[1]["config_id"], "config": recs[1]["config"], "score": float(recs[1]["score"])},
            {"label": "rec3", "config_id": recs[2]["config_id"], "config": recs[2]["config"], "score": float(recs[2]["score"])},
        ]

        run_rows: List[Dict[str, object]] = []
        for r_idx, item in enumerate(run_plan):
            label = str(item["label"])
            cfg = dict(item["config"])
            cfg_id = str(item["config_id"])
            run_dir = workload_dir / "runs" / f"{r_idx:02d}_{label}_{cfg_id}"
            run_dir.mkdir(parents=True, exist_ok=True)

            darshan_path = run_dir / f"{wkey}__{label}__{cfg_id}.darshan"
            if args.delete_darshan and darshan_path.exists():
                darshan_path.unlink(missing_ok=True)

            print(f"{ts()}  [{idx+1}/{len(selected)}] {cand} ({ref}) -> running {label} ({cfg_id})")
            applied = apply_lustre_knobs(workload_dir, cfg, args.use_sudo_lustre)

            env = os.environ.copy()
            env["DARSHAN_LOGFILE"] = str(darshan_path)
            run_cmd(["bash", str(run_sh)], cwd=REPO_ROOT, env=env)

            if args.flush_wait_sec > 0:
                time.sleep(args.flush_wait_sec)

            run_cmd(
                [
                    "python3",
                    str(REPO_ROOT / "analysis" / "scripts_analysis" / "analyze_darshan_recommender.py"),
                    "--darshan",
                    str(darshan_path),
                    "--outdir",
                    str(run_dir),
                    "--metric-key",
                    args.metric_key,
                ],
                cwd=REPO_ROOT,
            )

            metrics = json.loads((run_dir / "recommender_metrics.json").read_text(encoding="utf-8"))
            perf = float(metrics["selected_metric"])

            row: Dict[str, object] = {
                "timestamp": datetime.utcnow().isoformat(),
                "reference_pattern": ref,
                "candidate_pattern": cand,
                "workload_key": wkey,
                "workload_json": str(workload_json),
                "weighted_distance": float(sel["weighted_distance"]),
                "candidate_rank": int(sel["candidate_rank"]),
                "run_label": label,
                "run_order": r_idx,
                "config_id": cfg_id,
                "predicted_score": item["score"],
                "metric_key": args.metric_key,
                "metric": perf,
                "metric_fallback_mib_per_s": float(metrics.get("fallback_mib_per_s", 0.0)),
                "darshan_file": str(darshan_path),
                "analysis_dir": str(run_dir),
            }
            for k, v in cfg.items():
                row[k] = v
            row.update(applied)

            fieldnames = list(row.keys())
            append_csv(workload_dir / "observations.csv", row, fieldnames)
            append_csv(global_csv, row, fieldnames)

            run_rows.append(row)
            global_results.append(row)

        base_metric = next(r["metric"] for r in run_rows if r["run_label"] == "baseline")
        rec_rows = [r for r in run_rows if r["run_label"].startswith("rec")]
        best_rec = max(rec_rows, key=lambda r: float(r["metric"]))

        report = {
            "reference_pattern": ref,
            "candidate_pattern": cand,
            "workload_key": wkey,
            "weighted_distance": float(sel["weighted_distance"]),
            "candidate_rank": int(sel["candidate_rank"]),
            "baseline": {
                "config_id": baseline_id,
                "metric": float(base_metric),
            },
            "recommendations": [
                {
                    "label": r["run_label"],
                    "config_id": r["config_id"],
                    "metric": float(r["metric"]),
                    "predicted_score": r.get("predicted_score"),
                    "gain_vs_baseline": float(r["metric"] - base_metric),
                    "gain_vs_baseline_pct": float((r["metric"] - base_metric) / base_metric * 100.0 if base_metric else 0.0),
                }
                for r in rec_rows
            ],
            "best_recommendation": {
                "label": best_rec["run_label"],
                "config_id": best_rec["config_id"],
                "metric": float(best_rec["metric"]),
                "gain_vs_baseline": float(best_rec["metric"] - base_metric),
                "gain_vs_baseline_pct": float((best_rec["metric"] - base_metric) / base_metric * 100.0 if base_metric else 0.0),
            },
        }
        (workload_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        workload_reports.append(report)

    summary = {
        "created_at_utc": datetime.utcnow().isoformat(),
        "output_root": str(output_root),
        "closest_json": str(closest_json),
        "config": str(cfg_path),
        "training_meta": train_meta,
        "baseline_perf_train_patterns": baseline_perf_train,
        "selected_best_matches": selected,
        "workload_reports": workload_reports,
        "n_total_runs": len(global_results),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{ts()}  Validation complete")
    print(f"{ts()}  Summary: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
