from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import yaml

from io_recommender.active import ActiveLoopConfig, run_active_loop
from io_recommender.deploy import DeploymentRecommender, materialize_recommendation_matrix
from io_recommender.eval import plot_learning_curves
from io_recommender.model import ConfigEncoder, WorkloadEncoder
from io_recommender.runner import StubTestbedRunner
from io_recommender.runner_real import RealSynthRunner
from io_recommender.sampling import build_warm_start_set, enumerate_all_configs, total_space_size
from io_recommender.types import Observation, ParameterSpec, WorkloadPattern, config_id_from_params


@dataclass
class PipelineArtifacts:
    specs: List[ParameterSpec]
    patterns: List[WorkloadPattern]
    warm_configs: List[dict]
    observations: List[Observation]
    history: List[dict]
    rec_matrix: object
    deployment_demo: List[dict]


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_specs(cfg: Mapping) -> List[ParameterSpec]:
    return [
        ParameterSpec(
            name=item["name"],
            values=list(item["values"]),
            is_ordered=bool(item.get("is_ordered", True)),
        )
        for item in cfg["parameters"]
    ]


def baseline_from_specs(specs: Sequence[ParameterSpec], override: Mapping[str, object] | None = None) -> Dict[str, object]:
    baseline = {s.name: s.values[0] for s in specs}
    if override:
        baseline.update(override)
    return baseline


def generate_synthetic_patterns(n_patterns: int, n_features: int, seed: int) -> List[WorkloadPattern]:
    rng = np.random.default_rng(seed)
    patterns: List[WorkloadPattern] = []
    for i in range(n_patterns):
        arr = rng.normal(loc=0.0, scale=1.0, size=n_features)
        patterns.append(WorkloadPattern(pattern_id=f"p_{i:03d}", features=arr))
    return patterns


def _is_numeric(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def load_patterns_from_json_dir(
    input_dir: Path,
    limit: int | None = None,
    feature_prefix: str = "pct_",
) -> tuple[list[WorkloadPattern], list[str]]:
    files = sorted(input_dir.glob("*.json"))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise ValueError(f"No pattern json files found under {input_dir}")

    first = json.loads(files[0].read_text(encoding="utf-8"))
    feature_names = sorted([k for k, v in first.items() if k.startswith(feature_prefix) and _is_numeric(v)])
    if not feature_names:
        feature_names = sorted([k for k, v in first.items() if _is_numeric(v)])
    if not feature_names:
        raise ValueError(f"No numeric feature keys found in {files[0]}")

    patterns: list[WorkloadPattern] = []
    for p in files:
        obj = json.loads(p.read_text(encoding="utf-8"))
        feat = np.array([float(obj.get(k, 0.0)) for k in feature_names], dtype=float)
        patterns.append(WorkloadPattern(pattern_id=p.stem, features=feat))
    return patterns, feature_names


def collect_observations(
    patterns: Sequence[WorkloadPattern],
    configs: Sequence[Mapping[str, object]],
    workload_encoder: WorkloadEncoder,
    config_encoder: ConfigEncoder,
    runner,
    baseline_config: Mapping[str, object],
) -> tuple[list[Observation], dict[str, float]]:
    baseline_perf_by_pattern: Dict[str, float] = {}
    out: List[Observation] = []
    for pattern in patterns:
        wvec = workload_encoder.encode_workload(pattern)
        base_perf = runner.run_testbed(pattern.pattern_id, dict(baseline_config), wvec)
        baseline_perf_by_pattern[pattern.pattern_id] = base_perf
        for conf in configs:
            perf = runner.run_testbed(pattern.pattern_id, dict(conf), wvec)
            out.append(
                Observation(
                    pattern_id=pattern.pattern_id,
                    config_id=config_id_from_params(conf),
                    config_params=dict(conf),
                    workload_vec=wvec,
                    config_vec=config_encoder.encode_config(conf),
                    perf=perf,
                    baseline_perf=base_perf,
                    gain=perf - base_perf,
                )
            )
    return out, baseline_perf_by_pattern


def compute_oracle_best_gain(
    patterns: Sequence[WorkloadPattern],
    specs: Sequence[ParameterSpec],
    baseline_perf_by_pattern: Mapping[str, float],
    workload_encoder: WorkloadEncoder,
    runner,
    enum_threshold: int = 50_000,
    sampled: int = 25_000,
    seed: int = 7,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    if total_space_size(specs) <= enum_threshold:
        candidates = enumerate_all_configs(specs)
    else:
        candidates = []
        seen = set()
        while len(candidates) < sampled:
            c = {s.name: s.values[rng.integers(0, len(s.values))] for s in specs}
            k = tuple(c[s.name] for s in specs)
            if k not in seen:
                seen.add(k)
                candidates.append(c)

    out: Dict[str, float] = {}
    for p in patterns:
        wvec = workload_encoder.encode_workload(p)
        best = float("-inf")
        for c in candidates:
            perf = runner.run_testbed(p.pattern_id, c, wvec)
            best = max(best, perf - baseline_perf_by_pattern[p.pattern_id])
        out[p.pattern_id] = best
    return out


def run_pipeline(cfg: Mapping, output_dir: Path) -> PipelineArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 7))

    specs = parse_specs(cfg)
    baseline_cfg = baseline_from_specs(specs, override=cfg.get("baseline"))

    data_cfg = cfg.get("data", {})
    runner_cfg = cfg.get("runner", {})
    runner_mode = str(runner_cfg.get("mode", "stub")).lower()
    if runner_mode == "real":
        io_root = Path(runner_cfg.get("io_synth_root", "."))
        input_dir_cfg = Path(data_cfg.get("input_dir", "inputs/exemplar_jsons"))
        input_dir = input_dir_cfg if input_dir_cfg.is_absolute() else (io_root / input_dir_cfg)
        patterns, feature_names = load_patterns_from_json_dir(
            input_dir=input_dir,
            limit=int(data_cfg["n_patterns"]) if "n_patterns" in data_cfg else None,
            feature_prefix=str(data_cfg.get("feature_prefix", "pct_")),
        )
    else:
        patterns = generate_synthetic_patterns(
            n_patterns=int(data_cfg.get("n_patterns", 25)),
            n_features=int(data_cfg.get("n_features", 35)),
            seed=seed,
        )
        feature_names = [f"f_{i}" for i in range(int(data_cfg.get("n_features", 35)))]

    workload_encoder = WorkloadEncoder().fit(patterns)
    config_encoder = ConfigEncoder(specs).fit()

    warm_cfg = cfg.get("warm_start", {})
    warm = build_warm_start_set(
        specs,
        baseline=baseline_cfg,
        n_target=int(warm_cfg.get("target_size", 45)),
        seed=seed,
    )

    if runner_mode == "real":
        runner = RealSynthRunner(
            specs=specs,
            io_synth_root=io_root,
            input_dir=input_dir,
            out_root=Path(runner_cfg.get("out_root", "/mnt/hasanfs/out_synth")),
            cap_total_gib=float(runner_cfg.get("cap_total_gib", 512.0)),
            io_api=str(runner_cfg.get("io_api", "posix")),
            meta_api=str(runner_cfg.get("meta_api", "posix")),
            mpi_collective_mode=str(runner_cfg.get("mpi_collective_mode", "none")),
            meta_scope=str(runner_cfg.get("meta_scope", "separate")),
            nprocs_cap=int(runner_cfg.get("nprocs_cap", 64)),
            metric_key=str(runner_cfg.get("metric_key", "POSIX_agg_perf_by_slowest")),
            metric_fallback=str(runner_cfg.get("metric_fallback", "bytes_over_f_time")),
            delete_existing_darshan=bool(runner_cfg.get("delete_existing_darshan", True)),
            flush_wait_sec=float(runner_cfg.get("flush_wait_sec", 10.0)),
            use_sudo_for_lustre=bool(runner_cfg.get("use_sudo_for_lustre", False)),
            dry_run=bool(runner_cfg.get("dry_run", False)),
        )
    else:
        runner = StubTestbedRunner(specs=specs, seed=seed, noise_std=float(runner_cfg.get("noise_std", 0.0)))

    warm_obs, baseline_perf = collect_observations(
        patterns,
        warm.configs,
        workload_encoder,
        config_encoder,
        runner,
        baseline_cfg,
    )

    if runner_mode == "real":
        oracle_best = None
    else:
        oracle_best = compute_oracle_best_gain(
            patterns,
            specs,
            baseline_perf,
            workload_encoder,
            runner,
            seed=seed,
        )

    active_cfg_raw = cfg.get("active", {})
    model_cfg = cfg.get("model", {})
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

    observations, ensemble, history = run_active_loop(
        patterns=patterns,
        specs=specs,
        workload_encoder=workload_encoder,
        config_encoder=config_encoder,
        runner=runner,
        baseline_perf_by_pattern=baseline_perf,
        initial_observations=warm_obs,
        cfg=active_cfg,
        oracle_best_gain_by_pattern=oracle_best,
    )

    deploy_cfg = cfg.get("deploy", {})
    rec_matrix = materialize_recommendation_matrix(observations, top_k=int(deploy_cfg.get("topk_per_pattern", 20)))
    deploy = DeploymentRecommender(
        workload_encoder=workload_encoder,
        config_encoder=config_encoder,
        ensemble=ensemble,
        patterns=patterns,
        rec_matrix=rec_matrix,
        knn_k=int(deploy_cfg.get("knn_neighbors", 5)),
    )
    demo_features = patterns[0].features
    demo = deploy.recommend(demo_features, top_k=int(deploy_cfg.get("return_top_k", 3)))

    plot_learning_curves(history, output_dir / "plots")

    summary = {
        "seed": seed,
        "runner_mode": runner_mode,
        "n_patterns": len(patterns),
        "feature_names": feature_names,
        "warm_size": len(warm.configs),
        "pairwise_coverage_percent": warm.pairwise_coverage_percent,
        "total_observations": len(observations),
        "history": history,
        "deployment_demo": demo,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "recommendation_matrix.json", "w", encoding="utf-8") as f:
        json.dump(rec_matrix.topk_by_pattern, f, indent=2)

    return PipelineArtifacts(
        specs=specs,
        patterns=patterns,
        warm_configs=warm.configs,
        observations=observations,
        history=history,
        rec_matrix=rec_matrix,
        deployment_demo=demo,
    )
