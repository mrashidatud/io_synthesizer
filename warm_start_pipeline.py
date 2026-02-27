#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

from io_recommender.pipeline import baseline_from_specs, parse_specs
from io_recommender.sampling import build_warm_start_set
from io_recommender.types import config_id_from_params


def ts() -> str:
    return datetime.now().strftime("%F %T")


def parse_bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def parse_meta_scope(v: object, default: str = "separate") -> str:
    raw = str(v).strip().lower() if v is not None else default
    scope = raw or default
    if scope not in {"separate", "data_files"}:
        raise ValueError(f"Invalid meta_scope '{v}'. Expected one of: separate, data_files")
    return scope


def parse_options_csv(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip().lower() == "option":
                continue
            if len(row) == 1:
                key = row[0].strip()
                val = ""
            else:
                key = row[0].strip()
                val = row[1].strip()
            if not key:
                continue
            out[key] = val
    return out


def collect_workload_jsons(input_dir: Path, filters_raw: str) -> List[Path]:
    filters = [t.strip() for t in filters_raw.split() if t.strip()]
    seen = set()
    out: List[Path] = []

    def add(p: Path) -> None:
        rp = p.resolve()
        if p.is_file() and rp not in seen:
            seen.add(rp)
            out.append(rp)

    if filters:
        for tok in filters:
            if "-" in tok and tok.replace("-", "").isdigit() and tok.count("-") == 1:
                a_str, b_str = tok.split("-", 1)
                a, b = int(a_str), int(b_str)
                if a > b:
                    a, b = b, a
                for n in range(a, b + 1):
                    for p in sorted(input_dir.glob(f"top{n}_*.json")):
                        add(p)
            elif tok.isdigit():
                n = int(tok)
                for p in sorted(input_dir.glob(f"top{n}_*.json")):
                    add(p)
            else:
                name = tok if tok.endswith(".json") else f"{tok}.json"
                add(input_dir / name)
    else:
        for p in sorted(input_dir.glob("*.json")):
            add(p)
    return out


def load_nprocs_from_json(path: Path) -> int | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    val = obj.get("nprocs")
    if isinstance(val, int):
        return val
    if isinstance(val, str) and val.isdigit():
        return int(val)
    return None


def workload_key(json_base: str, cap_total_gib: float, nprocs: int, io_api: str, meta_api: str, coll: str) -> str:
    return f"{json_base}_cap_{int(cap_total_gib)}_nproc_{nprocs}_io_{io_api}_meta_{meta_api}_coll_{coll}"


def run_cmd(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print(f"{ts()}  RUN: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def lfs_size(value: object) -> str:
    s = str(value).strip().upper()
    if s.endswith(("K", "M", "G")):
        return s
    # Treat raw numeric value as KiB for stripe size compatibility.
    return f"{int(float(s))}K"


def apply_lustre_knobs(workload_dir: Path, cfg: Dict[str, object], use_sudo: bool) -> Dict[str, int]:
    payload = workload_dir / "payload"
    targets = [
        payload / "data" / "ro",
        payload / "data" / "rw",
        payload / "data" / "wo",
        payload / "meta",
    ]
    for t in targets:
        t.mkdir(parents=True, exist_ok=True)

    # Make directories empty so newly created files inherit new stripe policy.
    for t in targets:
        for p in t.rglob("*"):
            if p.is_file():
                p.unlink(missing_ok=True)

    prefix = ["sudo"] if use_sudo else []
    stripe_count = int(cfg["stripe_count"])
    stripe_size = lfs_size(cfg["stripe_size"])

    for t in targets:
        run_cmd(prefix + ["lfs", "setstripe", "-c", str(stripe_count), "-S", stripe_size, str(t)])

    osc_pages = int(cfg.get("osc_max_pages_per_rpc", cfg.get("max_pages_per_rpc", 1)))
    mdc_pages = int(cfg.get("mdc_max_pages_per_rpc", osc_pages))
    osc_rpcs = int(cfg.get("osc_max_rpcs_in_flight", cfg.get("max_rpcs_in_flight", 1)))
    mdc_rpcs = int(cfg.get("mdc_max_rpcs_in_flight", osc_rpcs))

    run_cmd(prefix + ["lctl", "set_param", f"osc.*.max_pages_per_rpc={osc_pages}"])
    run_cmd(prefix + ["lctl", "set_param", f"osc.*.max_rpcs_in_flight={osc_rpcs}"])
    run_cmd(prefix + ["lctl", "set_param", f"mdc.*.max_pages_per_rpc={mdc_pages}"])
    run_cmd(prefix + ["lctl", "set_param", f"mdc.*.max_rpcs_in_flight={mdc_rpcs}"])

    return {
        "osc_max_pages_per_rpc": osc_pages,
        "osc_max_rpcs_in_flight": osc_rpcs,
        "mdc_max_pages_per_rpc": mdc_pages,
        "mdc_max_rpcs_in_flight": mdc_rpcs,
    }


def ensure_mpi_binary(root: Path, bin_dir: Path, force_build: bool) -> None:
    target = bin_dir / "mpi_synthio"
    if target.exists() and not force_build:
        print(f"{ts()}  SKIP build: {target} already exists")
        return
    print(f"{ts()}  BUILD mpi_synthio")
    run_cmd(["make", "clean"], cwd=root / "scripts")
    run_cmd(["make"], cwd=root / "scripts")
    bin_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(root / "scripts" / "mpi_synthio"), str(target))
    print(f"{ts()}  Built {target}")


def append_csv(path: Path, row: Dict[str, object], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        if write_header:
            w.writeheader()
        w.writerow(row)


def read_iteration_observation_index(iter_csv: Path) -> tuple[set[tuple[str, str]], set[str]]:
    cfg_keys: set[tuple[str, str]] = set()
    analysis_dirs: set[str] = set()
    if not iter_csv.exists():
        return cfg_keys, analysis_dirs
    with iter_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cfg_idx = (row.get("config_index") or "").strip()
            cfg_id = (row.get("config_id") or "").strip()
            if cfg_idx and cfg_id:
                cfg_keys.add((cfg_idx, cfg_id))
            adir_raw = (row.get("analysis_dir") or "").strip()
            if adir_raw:
                adir = Path(adir_raw)
                analysis_dirs.add(str(adir))
                try:
                    analysis_dirs.add(str(adir.resolve()))
                except Exception:
                    pass
    return cfg_keys, analysis_dirs


def load_manifest_meta_scope(workload_dir: Path) -> str:
    manifest_path = workload_dir / "workload_manifest.json"
    if not manifest_path.exists():
        return "separate"
    try:
        obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return "separate"
    try:
        return parse_meta_scope(obj.get("meta_scope", "separate"))
    except Exception:
        return "separate"


def build_warm_start_configs(recommender_config: Path, warm_target_override: str, seed_override: str) -> list[dict]:
    cfg = yaml.safe_load(recommender_config.read_text(encoding="utf-8"))
    specs = parse_specs(cfg)
    baseline = baseline_from_specs(specs, override=cfg.get("baseline"))
    n_target = int(warm_target_override) if warm_target_override else int(cfg.get("warm_start", {}).get("target_size", 45))
    seed = int(seed_override) if seed_override else int(cfg.get("seed", 7))

    warm = build_warm_start_set(specs, baseline=baseline, n_target=n_target, seed=seed)
    out = []
    for c in warm.configs:
        out.append({"config_id": config_id_from_params(c), "params": c})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Warm-start data collection pipeline for recommender")
    ap.add_argument("--options-csv", default="remote_orchestration/warm_start_options.csv", help="CSV with option,value rows")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    options_path = Path(args.options_csv)
    if not options_path.is_absolute():
        options_path = (root / options_path).resolve()
    opts = parse_options_csv(options_path)

    cap_total_gib = float(opts.get("cap", "512") or "512")
    nprocs_override = opts.get("nprocs", "").strip()
    nprocs_cap = int(opts.get("nprocs_cap", "64") or "64")
    inputs_dir = Path(opts.get("inputs", "/mnt/hasanfs/io_synthesizer/inputs/exemplar_jsons"))
    force_build = parse_bool(opts.get("force_build"), default=False)
    delete_darshan = parse_bool(opts.get("delete_darshan"), default=True)
    filters_raw = opts.get("filters", "").strip()
    iterations = int(opts.get("iterations", "3") or "3")
    io_api = opts.get("io_api", "posix") or "posix"
    meta_api = opts.get("meta_api", "posix") or "posix"
    coll = opts.get("mpi_collective_mode", "none") or "none"
    meta_scope = parse_meta_scope(opts.get("meta_scope", "separate"))
    flush_wait_sec = float(opts.get("flush_wait_sec", "10") or "10")
    use_sudo_lustre = parse_bool(opts.get("use_sudo_lustre"), default=False)

    output_root = Path(opts.get("output_root", "/mnt/hasanfs/samples/warm-start"))
    recommender_cfg = Path(opts.get("recommender_config", "/mnt/hasanfs/io_synthesizer/io_recommender/config.yaml"))
    warm_target_override = opts.get("warm_start_target", "")
    seed_override = opts.get("seed", "")

    metric_key = opts.get("metric_key", "agg_perf_by_slowest") or "agg_perf_by_slowest"
    bin_dir = Path(opts.get("bin_dir", "/mnt/hasanfs/bin"))

    print(f"{ts()}  Warm-start workflow starting")
    print(f"{ts()}  Options CSV: {options_path}")
    print(f"{ts()}  Output root: {output_root}")

    ensure_mpi_binary(root, bin_dir, force_build)

    workloads = collect_workload_jsons(inputs_dir, filters_raw)
    if not workloads:
        print(f"{ts()}  No workload json files selected in {inputs_dir}")
        return

    warm_configs = build_warm_start_configs(recommender_cfg, warm_target_override, seed_override)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "warm_start_configs.json").write_text(json.dumps(warm_configs, indent=2), encoding="utf-8")
    print(f"{ts()}  Warm-start configs: {len(warm_configs)}")

    global_csv = output_root / "observations_all.csv"

    for workload_json in workloads:
        json_base = workload_json.stem

        json_nprocs = load_nprocs_from_json(workload_json)
        if nprocs_override:
            desired_nprocs = min(int(nprocs_override), nprocs_cap)
        elif json_nprocs is not None:
            desired_nprocs = min(json_nprocs, nprocs_cap)
        else:
            desired_nprocs = 1

        wkey = workload_key(json_base, cap_total_gib, desired_nprocs, io_api, meta_api, coll)
        workload_dir = output_root / wkey
        workload_dir.mkdir(parents=True, exist_ok=True)

        run_sh = workload_dir / "run_from_features.sh"
        plan_csv = workload_dir / "payload" / "plan.csv"
        existing_meta_scope = load_manifest_meta_scope(workload_dir)
        plan_exists = run_sh.exists() and plan_csv.exists()
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
            "iterations": iterations,
            "warm_configs": len(warm_configs),
        }
        (workload_dir / "workload_manifest.json").write_text(json.dumps(workload_manifest, indent=2), encoding="utf-8")

        workload_csv = workload_dir / "observations.csv"

        print(f"{ts()}  >>> workload={wkey}")
        for it in range(1, iterations + 1):
            iter_dir = workload_dir / f"iter_{it:02d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            iter_csv = iter_dir / "observations.csv"
            logged_cfg_keys, logged_analysis_dirs = read_iteration_observation_index(iter_csv)

            print(f"{ts()}  ---- iteration {it}/{iterations}")
            for cfg_idx, cfg_entry in enumerate(warm_configs):
                cfg = cfg_entry["params"]
                cfg_id = cfg_entry["config_id"]
                cfg_dir = iter_dir / f"cfg_{cfg_idx:03d}_{cfg_id}"
                cfg_key = (str(cfg_idx), cfg_id)
                cfg_dir_s = str(cfg_dir)
                cfg_dir_resolved = str(cfg_dir.resolve())

                if cfg_key in logged_cfg_keys or cfg_dir_s in logged_analysis_dirs or cfg_dir_resolved in logged_analysis_dirs:
                    print(f"{ts()}  SKIP existing iteration={it} cfg={cfg_idx:03d} ({cfg_id})")
                    continue

                if cfg_dir.exists():
                    print(
                        f"{ts()}  Removing stale cfg dir without logged observation: {cfg_dir}"
                    )
                    shutil.rmtree(cfg_dir)
                cfg_dir.mkdir(parents=True, exist_ok=True)

                darshan_path = cfg_dir / f"{wkey}__{cfg_id}.darshan"
                if delete_darshan and darshan_path.exists():
                    darshan_path.unlink(missing_ok=True)

                applied = apply_lustre_knobs(workload_dir, cfg, use_sudo_lustre)

                env = os.environ.copy()
                env["DARSHAN_LOGFILE"] = str(darshan_path)
                run_cmd(["bash", str(run_sh)], cwd=root, env=env)

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

                row: Dict[str, object] = {
                    "timestamp": datetime.now().isoformat(),
                    "workload_key": wkey,
                    "workload_json": str(workload_json),
                    "iteration": it,
                    "config_index": cfg_idx,
                    "config_id": cfg_id,
                    "metric_key": metric_key,
                    "metric": metrics["selected_metric"],
                    "metric_fallback_mib_per_s": metrics["fallback_mib_per_s"],
                    "darshan_file": str(darshan_path),
                    "analysis_dir": str(cfg_dir),
                }
                for k, v in cfg.items():
                    row[k] = v
                row.update(applied)

                fieldnames = list(row.keys())
                append_csv(iter_csv, row, fieldnames)
                append_csv(workload_csv, row, fieldnames)
                append_csv(global_csv, row, fieldnames)
                logged_cfg_keys.add(cfg_key)
                logged_analysis_dirs.add(cfg_dir_s)
                logged_analysis_dirs.add(cfg_dir_resolved)

    print(f"{ts()}  Warm-start workflow complete")
    print(f"{ts()}  Global observations: {global_csv}")


if __name__ == "__main__":
    main()
