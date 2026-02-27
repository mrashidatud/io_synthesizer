#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import yaml

from io_recommender.active import ActiveLoopConfig
from io_recommender.active.acquisition import select_hybrid
from io_recommender.active.candidates import generate_candidate_pool
from io_recommender.deploy import DeploymentRecommender, materialize_recommendation_matrix
from io_recommender.eval.metrics import hit_at_3_within, regret_at_3
from io_recommender.model import ConfigEncoder, EnsembleConfig, EnsembleModel, WorkloadEncoder
from io_recommender.pipeline import baseline_from_specs, parse_specs
from io_recommender.types import Observation, ParameterSpec, WorkloadPattern, config_id_from_params
from warm_start_pipeline import (
    append_csv,
    apply_lustre_knobs,
    collect_workload_jsons,
    ensure_mpi_binary,
    load_manifest_meta_scope,
    load_nprocs_from_json,
    parse_bool,
    parse_meta_scope,
    parse_options_csv,
    read_iteration_observation_index,
    run_cmd,
    ts,
    workload_key,
)


@dataclass
class WorkloadContext:
    pattern_id: str
    workload_json: Path
    workload_key: str
    desired_nprocs: int
    workload_dir: Path
    run_sh: Path
    warm_observations_csv: Path


@dataclass
class ParsedObservationRow:
    pattern_id: str
    config: Dict[str, object]
    config_id: str
    metric: float
    iteration: int = 0


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _apply_cfg_override(cfg: Dict[str, object], path: str, value: object) -> None:
    parts = [p for p in path.split(".") if p]
    if not parts:
        return
    cur: Dict[str, object] = cfg
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def apply_cfg_overrides_from_options(cfg: Dict[str, object], opts: Mapping[str, str]) -> Dict[str, object]:
    out = dict(cfg)
    applied: Dict[str, object] = {}
    for k, raw_v in opts.items():
        if not k.startswith("cfg."):
            continue
        path = k[4:].strip()
        if not path:
            continue
        parsed_v = yaml.safe_load(raw_v) if str(raw_v).strip() != "" else ""
        _apply_cfg_override(out, path, parsed_v)
        applied[path] = parsed_v
    if applied:
        print(f"{ts()}  Applied cfg.* overrides from options CSV:")
        for k in sorted(applied):
            print(f"{ts()}    cfg.{k} = {applied[k]!r}")
    return out


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
    raise ValueError(f"Value '{raw}' for '{spec.name}' not present in allowed values {spec.values}")


def _coerce_config_from_row(row: Mapping[str, str], specs: Sequence[ParameterSpec]) -> Dict[str, object]:
    cfg: Dict[str, object] = {}
    for spec in specs:
        if spec.name not in row:
            raise KeyError(f"Missing config column '{spec.name}' in observations row")
        cfg[spec.name] = _coerce_spec_value(row[spec.name], spec)
    return cfg


def _load_observation_rows(
    csv_path: Path,
    specs: Sequence[ParameterSpec],
    default_pattern_id: str | None = None,
) -> List[ParsedObservationRow]:
    out: List[ParsedObservationRow] = []
    if not csv_path.exists():
        return out

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pattern_id = (row.get("pattern_id") or "").strip() or default_pattern_id or ""
            if not pattern_id:
                continue
            metric_raw = (row.get("metric") or "").strip()
            if metric_raw == "":
                continue
            try:
                cfg = _coerce_config_from_row(row, specs)
            except Exception:
                continue
            cfg_id = (row.get("config_id") or "").strip() or config_id_from_params(cfg)
            iteration = int(_safe_float(row.get("iteration", 0), default=0))
            out.append(
                ParsedObservationRow(
                    pattern_id=pattern_id,
                    config=cfg,
                    config_id=cfg_id,
                    metric=float(metric_raw),
                    iteration=iteration,
                )
            )
    return out


def load_patterns_from_paths(
    workload_jsons: Sequence[Path],
    feature_prefix: str = "pct_",
) -> tuple[list[WorkloadPattern], list[str]]:
    if not workload_jsons:
        raise ValueError("No workload json files selected")

    first = json.loads(workload_jsons[0].read_text(encoding="utf-8"))
    feature_names = sorted([k for k, v in first.items() if k.startswith(feature_prefix) and isinstance(v, (int, float))])
    if not feature_names:
        feature_names = sorted([k for k, v in first.items() if isinstance(v, (int, float))])
    if not feature_names:
        raise ValueError(f"No numeric feature keys in {workload_jsons[0]}")

    patterns: list[WorkloadPattern] = []
    for p in workload_jsons:
        obj = json.loads(p.read_text(encoding="utf-8"))
        feat = np.array([float(obj.get(k, 0.0)) for k in feature_names], dtype=float)
        patterns.append(WorkloadPattern(pattern_id=p.stem, features=feat))
    return patterns, feature_names


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Active-sampling orchestration using existing warm-start observations")
    ap.add_argument("--options-csv", default="remote_orchestration/active_sampling_options.csv", help="CSV with option,value rows")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    options_path = Path(args.options_csv)
    if not options_path.is_absolute():
        options_path = (root / options_path).resolve()
    opts = parse_options_csv(options_path)

    recommender_cfg_path = Path(opts.get("recommender_config", "/mnt/hasanfs/io_synthesizer/io_recommender/config.yaml"))
    if not recommender_cfg_path.is_absolute():
        recommender_cfg_path = (root / recommender_cfg_path).resolve()
    cfg = yaml.safe_load(recommender_cfg_path.read_text(encoding="utf-8"))
    cfg = apply_cfg_overrides_from_options(cfg, opts)

    seed = int(opts.get("seed", str(cfg.get("seed", 7))) or str(cfg.get("seed", 7)))
    cfg["seed"] = seed
    specs = parse_specs(cfg)
    baseline_cfg = baseline_from_specs(specs, override=cfg.get("baseline"))
    baseline_id = config_id_from_params(baseline_cfg)

    runner_cfg = cfg.get("runner", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    active_cfg_raw = cfg.get("active", {})
    deploy_cfg = cfg.get("deploy", {})

    io_root = Path(str(runner_cfg.get("io_synth_root", root)))
    inputs_raw = opts.get("inputs", "").strip()
    if inputs_raw:
        inputs_dir = Path(inputs_raw)
    else:
        input_cfg = Path(str(data_cfg.get("input_dir", "inputs/exemplar_jsons")))
        inputs_dir = input_cfg if input_cfg.is_absolute() else (io_root / input_cfg)

    cap_total_gib = float(opts.get("cap", str(runner_cfg.get("cap_total_gib", 512.0))) or str(runner_cfg.get("cap_total_gib", 512.0)))
    nprocs_override = opts.get("nprocs", "").strip()
    nprocs_cap = int(opts.get("nprocs_cap", str(runner_cfg.get("nprocs_cap", 64))) or str(runner_cfg.get("nprocs_cap", 64)))
    io_api = opts.get("io_api", str(runner_cfg.get("io_api", "posix"))) or "posix"
    meta_api = opts.get("meta_api", str(runner_cfg.get("meta_api", "posix"))) or "posix"
    coll = opts.get("mpi_collective_mode", str(runner_cfg.get("mpi_collective_mode", "none"))) or "none"
    meta_scope = parse_meta_scope(opts.get("meta_scope", runner_cfg.get("meta_scope", "separate")))
    flush_wait_sec = float(opts.get("flush_wait_sec", str(runner_cfg.get("flush_wait_sec", 10.0))) or str(runner_cfg.get("flush_wait_sec", 10.0)))
    use_sudo_lustre = parse_bool(opts.get("use_sudo_lustre"), default=bool(runner_cfg.get("use_sudo_for_lustre", False)))
    force_build = parse_bool(opts.get("force_build"), default=False)
    delete_darshan = parse_bool(opts.get("delete_darshan"), default=bool(runner_cfg.get("delete_existing_darshan", True)))
    metric_key = opts.get("metric_key", str(runner_cfg.get("metric_key", "agg_perf_by_slowest"))) or str(
        runner_cfg.get("metric_key", "agg_perf_by_slowest")
    )
    filters_raw = opts.get("filters", "").strip()

    warm_start_root = Path(opts.get("warm_start_root", "/mnt/hasanfs/samples/warm-start"))
    output_root = Path(opts.get("output_root", "/mnt/hasanfs/samples/active-sampling"))
    bin_dir = Path(opts.get("bin_dir", "/mnt/hasanfs/bin"))

    active_cfg = ActiveLoopConfig(
        iterations=int(active_cfg_raw.get("iterations", 8)),
        batch_per_iter=int(active_cfg_raw.get("batch_per_iter", 3)),
        beta_start=float(active_cfg_raw.get("beta_start", 1.2)),
        beta_end=float(active_cfg_raw.get("beta_end", 0.4)),
        lambda_redundancy=float(active_cfg_raw.get("lambda_redundancy", 0.2)),
        ensemble_size=int(active_cfg_raw.get("ensemble_size", 6)),
        explore_mode=str(active_cfg_raw.get("explore_mode", "ucb")),
        model_mode=str(model_cfg.get("mode", "ranking")),
        use_lightgbm=bool(model_cfg.get("use_lightgbm", True)),
        early_stop_rounds=int(active_cfg_raw.get("early_stop_rounds", 0)),
        low_sigma_threshold=float(active_cfg_raw.get("low_sigma_threshold", 0.0)),
        seed=seed,
    )

    print(f"{ts()}  Active-sampling workflow starting")
    print(f"{ts()}  Options CSV: {options_path}")
    print(f"{ts()}  Recommender config: {recommender_cfg_path}")
    print(f"{ts()}  Warm-start root: {warm_start_root}")
    print(f"{ts()}  Active output root: {output_root}")

    ensure_mpi_binary(root, bin_dir, force_build)

    workload_jsons = collect_workload_jsons(inputs_dir, filters_raw)
    if not workload_jsons:
        raise RuntimeError(f"No workload json files selected in {inputs_dir}")

    patterns, feature_names = load_patterns_from_paths(
        workload_jsons,
        feature_prefix=str(data_cfg.get("feature_prefix", "pct_")),
    )
    pattern_by_id = {p.pattern_id: p for p in patterns}
    workload_json_by_pattern = {p.stem: p for p in workload_jsons}
    output_root.mkdir(parents=True, exist_ok=True)

    contexts: Dict[str, WorkloadContext] = {}
    warm_rows: list[ParsedObservationRow] = []

    for workload_json in workload_jsons:
        pattern_id = workload_json.stem
        json_nprocs = load_nprocs_from_json(workload_json)
        if nprocs_override:
            desired_nprocs = min(int(nprocs_override), nprocs_cap)
        elif json_nprocs is not None:
            desired_nprocs = min(json_nprocs, nprocs_cap)
        else:
            desired_nprocs = 1

        wkey = workload_key(pattern_id, cap_total_gib, desired_nprocs, io_api, meta_api, coll)
        warm_workload_dir = warm_start_root / wkey
        warm_obs_csv = warm_workload_dir / "observations.csv"
        if not warm_obs_csv.exists():
            raise FileNotFoundError(f"Missing warm-start observations for {pattern_id}: {warm_obs_csv}")

        workload_dir = output_root / wkey
        workload_dir.mkdir(parents=True, exist_ok=True)
        run_sh = workload_dir / "run_from_features.sh"
        plan_csv = workload_dir / "payload" / "plan.csv"
        plan_exists = run_sh.exists() and plan_csv.exists()
        existing_meta_scope = load_manifest_meta_scope(workload_dir)
        regenerate_for_meta_scope = plan_exists and existing_meta_scope != meta_scope

        if not plan_exists or regenerate_for_meta_scope:
            if regenerate_for_meta_scope:
                print(
                    f"{ts()}  Regenerating plan for {wkey} due to meta_scope change: "
                    f"{existing_meta_scope} -> {meta_scope}"
                )
            cmd = [
                "python3",
                str(root / "scripts" / "features2synth_opsaware.py"),
                "--features",
                str(workload_json),
                "--cap-total-gib",
                str(cap_total_gib),
                "--io-api",
                io_api,
                "--meta-api",
                meta_api,
                "--mpi-collective-mode",
                coll,
                "--meta-scope",
                meta_scope,
                "--nprocs",
                str(desired_nprocs),
                "--outdir",
                str(workload_dir),
            ]
            run_cmd(cmd, cwd=root)
        else:
            print(f"{ts()}  Reusing existing plan/scripts for {wkey}")

        workload_manifest = {
            "workload_key": wkey,
            "workload_json": str(workload_json),
            "workload_dir": str(workload_dir),
            "nprocs": desired_nprocs,
            "cap_total_gib": cap_total_gib,
            "io_api": io_api,
            "meta_api": meta_api,
            "mpi_collective_mode": coll,
            "meta_scope": meta_scope,
        }
        (workload_dir / "workload_manifest.json").write_text(json.dumps(workload_manifest, indent=2), encoding="utf-8")

        contexts[pattern_id] = WorkloadContext(
            pattern_id=pattern_id,
            workload_json=workload_json,
            workload_key=wkey,
            desired_nprocs=desired_nprocs,
            workload_dir=workload_dir,
            run_sh=run_sh,
            warm_observations_csv=warm_obs_csv,
        )
        warm_rows.extend(_load_observation_rows(warm_obs_csv, specs=specs, default_pattern_id=pattern_id))

    baseline_perf_by_pattern: Dict[str, float] = {}
    for pattern in patterns:
        vals = [r.metric for r in warm_rows if r.pattern_id == pattern.pattern_id and r.config_id == baseline_id]
        if not vals:
            raise RuntimeError(
                f"Baseline config ({baseline_id}) not found in warm observations for pattern '{pattern.pattern_id}'. "
                "Ensure warm-start data includes baseline."
            )
        baseline_perf_by_pattern[pattern.pattern_id] = float(statistics.mean(vals))

    workload_encoder = WorkloadEncoder().fit(patterns)
    config_encoder = ConfigEncoder(specs).fit()

    observations: list[Observation] = []
    for r in warm_rows:
        pattern = pattern_by_id[r.pattern_id]
        base = baseline_perf_by_pattern[r.pattern_id]
        observations.append(
            Observation(
                pattern_id=r.pattern_id,
                config_id=r.config_id,
                config_params=dict(r.config),
                workload_vec=workload_encoder.encode_workload(pattern),
                config_vec=config_encoder.encode_config(r.config),
                perf=r.metric,
                baseline_perf=base,
                gain=r.metric - base,
            )
        )

    global_csv = output_root / "observations_all.csv"
    existing_active_rows = _load_observation_rows(global_csv, specs=specs)
    existing_active_rows = [r for r in existing_active_rows if r.pattern_id in pattern_by_id]
    max_existing_iter = max((r.iteration for r in existing_active_rows), default=0)

    for r in existing_active_rows:
        pattern = pattern_by_id[r.pattern_id]
        base = baseline_perf_by_pattern[r.pattern_id]
        observations.append(
            Observation(
                pattern_id=r.pattern_id,
                config_id=r.config_id,
                config_params=dict(r.config),
                workload_vec=workload_encoder.encode_workload(pattern),
                config_vec=config_encoder.encode_config(r.config),
                perf=r.metric,
                baseline_perf=base,
                gain=r.metric - base,
            )
        )

    print(f"{ts()}  Loaded warm observations: {len(warm_rows)}")
    print(f"{ts()}  Loaded existing active observations: {len(existing_active_rows)}")

    ensemble = EnsembleModel(
        EnsembleConfig(
            mode=active_cfg.model_mode,
            ensemble_size=active_cfg.ensemble_size,
            seed=active_cfg.seed,
            use_lightgbm=active_cfg.use_lightgbm,
        )
    )

    history: List[dict] = []
    start_iter = max_existing_iter + 1
    if start_iter > active_cfg.iterations:
        print(f"{ts()}  Active iterations already complete (max existing iteration={max_existing_iter})")
    else:
        for t in range(start_iter, active_cfg.iterations + 1):
            print(f"{ts()}  ---- active iteration {t}/{active_cfg.iterations}")
            ensemble.fit(observations)
            beta = _beta_for_iter(active_cfg, t)
            new_batch: list[Observation] = []

            for p_idx, pattern in enumerate(patterns):
                ctx = contexts[pattern.pattern_id]
                wvec = workload_encoder.encode_workload(pattern)
                pid_obs = [o for o in observations if o.pattern_id == pattern.pattern_id]
                tested_configs = [o.config_params for o in pid_obs]
                top_cfgs = [o.config_params for o in sorted(pid_obs, key=lambda x: x.gain, reverse=True)[:5]]

                candidates = generate_candidate_pool(
                    specs,
                    observations,
                    pattern.pattern_id,
                    top_configs=top_cfgs,
                    seed=active_cfg.seed + 1000 * t + p_idx,
                )
                if not candidates:
                    print(f"{ts()}  No candidates left for {pattern.pattern_id}; skipping")
                    continue

                Xcand = np.vstack([np.concatenate([wvec, config_encoder.encode_config(c)]) for c in candidates])
                mu, sigma = ensemble.predict_mean_std(Xcand)
                picked = select_hybrid(
                    candidates,
                    mu,
                    sigma,
                    tested_configs,
                    specs,
                    b=active_cfg.batch_per_iter,
                    beta=beta,
                    lam=active_cfg.lambda_redundancy,
                    explore_mode=active_cfg.explore_mode,
                    seed=active_cfg.seed + t + p_idx,
                )
                if not picked:
                    continue

                iter_dir = ctx.workload_dir / f"iter_{t:02d}"
                iter_dir.mkdir(parents=True, exist_ok=True)
                iter_csv = iter_dir / "observations.csv"
                workload_csv = ctx.workload_dir / "observations.csv"
                logged_cfg_keys, logged_analysis_dirs = read_iteration_observation_index(iter_csv)

                for cfg_idx, conf in enumerate(picked):
                    cfg_id = config_id_from_params(conf)
                    cfg_key = (str(cfg_idx), cfg_id)
                    cfg_dir = iter_dir / f"cfg_{cfg_idx:03d}_{cfg_id}"
                    cfg_dir_s = str(cfg_dir)
                    cfg_dir_resolved = str(cfg_dir.resolve())
                    if cfg_key in logged_cfg_keys or cfg_dir_s in logged_analysis_dirs or cfg_dir_resolved in logged_analysis_dirs:
                        print(f"{ts()}  SKIP existing active run iteration={t} pattern={pattern.pattern_id} cfg={cfg_idx:03d}")
                        continue

                    if cfg_dir.exists():
                        print(f"{ts()}  Removing stale cfg dir without logged observation: {cfg_dir}")
                        shutil.rmtree(cfg_dir)
                    cfg_dir.mkdir(parents=True, exist_ok=True)

                    darshan_path = cfg_dir / f"{ctx.workload_key}__iter_{t:02d}__{cfg_id}.darshan"
                    if delete_darshan and darshan_path.exists():
                        darshan_path.unlink(missing_ok=True)

                    applied = apply_lustre_knobs(ctx.workload_dir, conf, use_sudo_lustre)
                    env = os.environ.copy()
                    env["DARSHAN_LOGFILE"] = str(darshan_path)
                    run_cmd(["bash", str(ctx.run_sh)], cwd=root, env=env)

                    if flush_wait_sec > 0:
                        time.sleep(flush_wait_sec)

                    run_cmd(
                        [
                            "python3",
                            str(root / "analysis" / "scripts_analysis" / "analyze_darshan_recommender.py"),
                            "--darshan",
                            str(darshan_path),
                            "--outdir",
                            str(cfg_dir),
                            "--metric-key",
                            metric_key,
                        ],
                        cwd=root,
                    )

                    metrics = json.loads((cfg_dir / "recommender_metrics.json").read_text(encoding="utf-8"))
                    perf = float(metrics["selected_metric"])
                    base = baseline_perf_by_pattern[pattern.pattern_id]
                    obs = Observation(
                        pattern_id=pattern.pattern_id,
                        config_id=cfg_id,
                        config_params=dict(conf),
                        workload_vec=wvec,
                        config_vec=config_encoder.encode_config(conf),
                        perf=perf,
                        baseline_perf=base,
                        gain=perf - base,
                    )
                    new_batch.append(obs)

                    row: Dict[str, object] = {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "workload_key": ctx.workload_key,
                        "pattern_id": pattern.pattern_id,
                        "workload_json": str(ctx.workload_json),
                        "iteration": t,
                        "config_index": cfg_idx,
                        "config_id": cfg_id,
                        "metric_key": metric_key,
                        "metric": perf,
                        "metric_fallback_mib_per_s": float(metrics.get("fallback_mib_per_s", 0.0)),
                        "darshan_file": str(darshan_path),
                        "analysis_dir": str(cfg_dir),
                    }
                    for k, v in conf.items():
                        row[k] = v
                    row.update(applied)
                    fieldnames = list(row.keys())
                    append_csv(iter_csv, row, fieldnames)
                    append_csv(workload_csv, row, fieldnames)
                    append_csv(global_csv, row, fieldnames)

            observations.extend(new_batch)
            best_now = max((o.gain for o in observations), default=float("-inf"))
            best3 = _best_by_pattern(observations)
            oracle = dict(best3)
            history.append(
                {
                    "iter": t,
                    "runs": len(observations),
                    "new_runs": len(new_batch),
                    "best_gain_so_far": float(best_now),
                    "regret_at_3": regret_at_3(best3, oracle),
                    "hit_at_3": hit_at_3_within(best3, oracle, tol=0.05),
                    "ndcg_at_3": _ndcg3_from_observed(observations, oracle),
                }
            )
            (output_root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

            if active_cfg.low_sigma_threshold > 0 and new_batch:
                sigma_proxy = float(np.std([o.gain for o in new_batch]))
                if sigma_proxy < active_cfg.low_sigma_threshold:
                    print(f"{ts()}  Early stop: sigma_proxy={sigma_proxy:.6f} < {active_cfg.low_sigma_threshold}")
                    break

    if not observations:
        raise RuntimeError("No observations available for training")

    ensemble.fit(observations)
    rec_matrix = materialize_recommendation_matrix(observations, top_k=int(deploy_cfg.get("topk_per_pattern", 20)))
    deploy = DeploymentRecommender(
        workload_encoder=workload_encoder,
        config_encoder=config_encoder,
        ensemble=ensemble,
        patterns=patterns,
        rec_matrix=rec_matrix,
        knn_k=int(deploy_cfg.get("knn_neighbors", 5)),
    )
    demo = deploy.recommend(patterns[0].features, top_k=int(deploy_cfg.get("return_top_k", 3)))

    summary = {
        "seed": seed,
        "feature_names": feature_names,
        "n_patterns": len(patterns),
        "pattern_ids": [p.pattern_id for p in patterns],
        "warm_observations_loaded": len(warm_rows),
        "active_observations_loaded": len(existing_active_rows),
        "total_observations": len(observations),
        "active_iterations_target": active_cfg.iterations,
        "history": history,
        "deployment_demo": demo,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "recommendation_matrix.json").write_text(json.dumps(rec_matrix.topk_by_pattern, indent=2), encoding="utf-8")
    (output_root / "effective_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    contexts_out = {
        pid: {
            "workload_key": ctx.workload_key,
            "workload_json": str(ctx.workload_json),
            "nprocs": ctx.desired_nprocs,
            "workload_dir": str(ctx.workload_dir),
            "warm_observations_csv": str(ctx.warm_observations_csv),
        }
        for pid, ctx in sorted(contexts.items())
    }
    (output_root / "workload_contexts.json").write_text(json.dumps(contexts_out, indent=2), encoding="utf-8")

    print(f"{ts()}  Active-sampling workflow complete")
    print(f"{ts()}  Output root: {output_root}")
    print(f"{ts()}  Summary: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
