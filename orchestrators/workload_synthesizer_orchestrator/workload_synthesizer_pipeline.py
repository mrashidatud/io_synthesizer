#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent


def ts() -> str:
    return datetime.now().strftime("%F %T")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            if key.lower() == "option":
                continue
            if not key:
                continue
            val = row[1].strip() if len(row) > 1 else ""
            out[key] = val
    return out


def run_cmd(cmd: List[str], *, cwd: Path | None = None, check: bool = True) -> int:
    print(f"{ts()}  RUN: {shlex.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def collect_workload_jsons(input_dir: Path, filters_raw: str) -> list[Path]:
    filters = [t.strip() for t in filters_raw.split() if t.strip()]
    seen: set[Path] = set()
    out: list[Path] = []

    def add(p: Path) -> None:
        rp = p.resolve()
        if p.is_file() and rp not in seen:
            seen.add(rp)
            out.append(rp)

    if filters:
        for tok in filters:
            m = re.match(r"^(\d+)-(\d+)$", tok)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
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


def extract_darshan_logfile(run_sh: Path) -> Path | None:
    pattern = re.compile(r"^export\s+DARSHAN_LOGFILE=['\"]([^'\"]+)['\"]")
    try:
        for line in run_sh.read_text(encoding="utf-8").splitlines():
            m = pattern.match(line.strip())
            if m:
                return Path(m.group(1))
    except Exception:
        return None
    return None


def cleanup_payload_subdirs(run_root: Path) -> None:
    payload_dir = run_root / "payload"
    if not payload_dir.is_dir():
        return
    print(f"{ts()}  [Cleanup] Removing subdirectories inside {payload_dir}")
    for child in payload_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)


def append_report_rollups(report: Path, json_base: str, master_txt: Path, master_csv: Path) -> None:
    if not report.is_file():
        return

    with master_txt.open("a", encoding="utf-8") as f_out:
        f_out.write("\n")
        f_out.write(f"=== {json_base} ===\n")
        f_out.write(report.read_text(encoding="utf-8"))

    status = ""
    line_re = re.compile(r"^\s*-\s*([^:]+):\s*([^\s]+)\s+vs\s+([^\s]+)\s+\((?:Δ|delta|diff|abs_diff)?=?\s*([^)]+)\)")
    rows: list[tuple[str, str, str, str, str]] = []

    for line in report.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if t.startswith("Within tolerance"):
            status = "within"
            continue
        if t.startswith("Outside tolerance"):
            status = "outside"
            continue

        m = line_re.match(line)
        if m and status:
            rows.append((status, m.group(1), m.group(2), m.group(3), m.group(4)))

    if not rows:
        return

    with master_csv.open("a", encoding="utf-8", newline="") as f_out:
        w = csv.writer(f_out)
        for status, key, inp, produced, diff in rows:
            w.writerow([json_base, status, key, inp, produced, diff])


def ensure_mpi_binary(root: Path, bin_dir: Path, force_build: bool) -> None:
    target = bin_dir / "mpi_synthio"
    if target.exists() and not force_build:
        print(f"{ts()}  SKIP build: {target} already exists")
        return

    print(f"{ts()}  === STEP 0: Build mpi_synthio ===")
    run_cmd(["make", "clean"], cwd=root / "scripts")
    run_cmd(["make"], cwd=root / "scripts")

    built = root / "scripts" / "mpi_synthio"
    if not built.exists():
        raise FileNotFoundError(f"Build completed but binary not found: {built}")

    bin_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(built), str(target))
    print(f"{ts()}  Built {target}")


def resolve_path(raw: str, *, fallback: Path) -> Path:
    if not raw.strip():
        return fallback
    p = Path(raw)
    if p.is_absolute():
        return p
    from_script = (SCRIPT_DIR / p).resolve()
    if from_script.exists():
        return from_script
    return (REPO_ROOT / p).resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description="Workload synthesizer orchestration")
    ap.add_argument("--options-csv", default="pipeline_options.csv", help="CSV with option,value rows")
    args = ap.parse_args()

    options_path = resolve_path(args.options_csv, fallback=SCRIPT_DIR / "pipeline_options.csv")
    if not options_path.is_file():
        raise FileNotFoundError(f"Pipeline options CSV not found: {options_path}")

    opts = parse_options_csv(options_path)

    root = resolve_path(opts.get("io_synth_root", ""), fallback=REPO_ROOT)
    out_root = Path(opts.get("output_root", "/mnt/hasanfs/out_synth"))
    bin_dir = Path(opts.get("bin_dir", "/mnt/hasanfs/bin"))
    input_dir = Path(opts.get("inputs", "/mnt/hasanfs/io_synthesizer/inputs/exemplar_jsons"))
    features_script = root / "scripts" / "features2synth_opsaware.py"
    analyze_script = root / "analysis" / "scripts_analysis" / "analyze_darshan_merged.py"

    cap_total_gib = float(opts.get("cap", "512") or "512")
    nprocs_override = (opts.get("nprocs", "") or "").strip()
    nprocs_cap = int(opts.get("nprocs_cap", "64") or "64")
    force_build = parse_bool(opts.get("force_build"), default=False)
    delete_darshan = parse_bool(opts.get("delete_darshan"), default=False)
    filters_raw = (opts.get("filters", "") or "").strip()
    flush_wait_sec = float(opts.get("flush_wait_sec", "10") or "10")
    meta_scope = parse_meta_scope(opts.get("meta_scope", "separate"))

    if nprocs_override and not nprocs_override.isdigit():
        raise ValueError(f"Invalid nprocs '{nprocs_override}': expected integer")
    if nprocs_cap <= 0:
        raise ValueError("nprocs_cap must be > 0")

    out_root.mkdir(parents=True, exist_ok=True)
    stamp = timestamp()
    master_txt = out_root / f"pct_compare_master_{stamp}.txt"
    master_csv = out_root / f"pct_compare_master_{stamp}.csv"
    master_txt.write_text(f"# pct_* comparison rollup (all cases) @ {stamp}\n", encoding="utf-8")
    with master_csv.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(["json", "status", "key", "input", "produced", "abs_diff"])

    print(f"{ts()}  Options CSV: {options_path}")
    print(f"{ts()}  Repo root: {root}")
    print(f"{ts()}  Input dir: {input_dir}")
    print(f"{ts()}  Output root: {out_root}")
    print(f"{ts()}  Meta scope: {meta_scope}")

    ensure_mpi_binary(root, bin_dir, force_build)

    workloads = collect_workload_jsons(input_dir, filters_raw)
    if not workloads:
        print(f"{ts()}  No JSON inputs selected/found (INPUT_DIR={input_dir}).")
        return

    print(f"{ts()}  === STEP 1..N: Plan, run, validate, analyze for each input ===")
    for workload_json in workloads:
        json_name = workload_json.name
        json_base = workload_json.stem

        print("")
        print(f"{ts()}  ---- Processing: {json_name} ----")

        desired_nprocs: int | None = None
        if nprocs_override:
            desired_nprocs = min(int(nprocs_override), nprocs_cap)
        else:
            json_nprocs = load_nprocs_from_json(workload_json)
            if json_nprocs is not None:
                desired_nprocs = min(json_nprocs, nprocs_cap)

        cmd = [
            "python3",
            str(features_script),
            "--features",
            str(workload_json),
            "--cap-total-gib",
            str(cap_total_gib),
            "--io-api",
            "posix",
            "--meta-api",
            "posix",
            "--mpi-collective-mode",
            "none",
            "--meta-scope",
            meta_scope,
        ]
        if desired_nprocs is not None:
            cmd.extend(["--nprocs", str(desired_nprocs)])

        print(f"{ts()}  [Plan] features2synth_opsaware.py for {json_name}")
        run_cmd(cmd, cwd=root)

        cand1 = out_root / json_base / "run_from_features.sh"
        cand2 = out_root / "run_from_features.sh"
        if cand1.is_file():
            run_sh = cand1
            run_root = cand1.parent
        elif cand2.is_file():
            run_sh = cand2
            run_root = cand2.parent
            print(f"{ts()}  WARN: Using legacy run script location: {run_sh}")
        else:
            print(f"{ts()}  ERROR: Could not find run_from_features.sh in {cand1} or {cand2}")
            continue

        if delete_darshan:
            print(f"{ts()}  [Pre-run] --delete-darshan enabled -> removing existing Darshan files in {run_root}")
            for p in run_root.glob("*.darshan"):
                p.unlink(missing_ok=True)

        expected = extract_darshan_logfile(run_sh)

        if (not delete_darshan) and expected and expected.is_file():
            print(f"{ts()}  SKIP: Found existing Darshan artifact: {expected}")
            print(f"{ts()}  Proceeding directly to analysis")
        else:
            print(f"{ts()}  [Run] {run_sh}")
            run_cmd(["bash", str(run_sh)])

            print(f"{ts()}  [Validate] Sleep {flush_wait_sec}s to allow Darshan to flush")
            time.sleep(flush_wait_sec)

            if expected and expected.is_file():
                print(f"{ts()}  OK: Found Darshan: {expected}")
            elif expected and not expected.is_file():
                print(f"{ts()}  WARN: Expected Darshan not found: {expected}")
                present = sorted(run_root.glob("*.darshan"))
                if present:
                    for p in present:
                        print(f"{ts()}    Present: {p}")
                cleanup_payload_subdirs(run_root)
                continue
            else:
                found = sorted(run_root.glob("*.darshan"))
                if len(found) == 1:
                    expected = found[0]
                    print(f"{ts()}  INFO: Using discovered Darshan file: {expected}")
                else:
                    print(f"{ts()}  ERROR: Could not determine Darshan artifact to analyze")
                    cleanup_payload_subdirs(run_root)
                    continue

        print(f"{ts()}  [Analyze] merged analysis for {json_name}")
        if not analyze_script.is_file():
            analyze_rc = 127
            print(f"{ts()}  ERROR: Missing analyzer script: {analyze_script}")
        else:
            analyze_rc = run_cmd(
                [
                    "python3",
                    str(analyze_script),
                    "--darshan",
                    str(expected),
                    "--input-json",
                    str(workload_json),
                    "--outdir",
                    str(run_root),
                ],
                check=False,
            )

        if analyze_rc != 0:
            print(f"{ts()}  WARN: Analysis returned non-zero ({analyze_rc}) for {json_name}. Continuing.")

        report = run_root / "pct_compare_report.txt"
        append_report_rollups(report, json_base, master_txt, master_csv)

        cleanup_payload_subdirs(run_root)
        print(f"{ts()}  ---- Done: {json_name} ----")

    print("")
    print(f"Master comparison (txt): {master_txt}")
    print(f"Master comparison (csv): {master_csv}")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
