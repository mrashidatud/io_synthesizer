#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import shutil
import subprocess
import sys
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
            # Keep all trailing CSV segments in the value, mirroring shell read behavior.
            val = ",".join(row[1:]).strip() if len(row) > 1 else ""
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


def resolve_darshan_export_value(raw: str, run_root: Path) -> Path | None:
    s = raw.strip()
    for p in (
        re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*:-(.+)\}$"),
        re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*-(.+)\}$"),
    ):
        m = p.match(s)
        if m:
            s = m.group(1).strip()
            break

    if "$" in s:
        return None

    path = Path(s)
    if not path.is_absolute():
        path = (run_root / path).resolve()
    return path


def extract_darshan_logfile(run_sh: Path) -> tuple[Path | None, str | None]:
    pattern = re.compile(r"^export\s+DARSHAN_LOGFILE=(['\"])(.+)\1$")
    try:
        for line in run_sh.read_text(encoding="utf-8").splitlines():
            m = pattern.match(line.strip())
            if m:
                raw = m.group(2).strip()
                return resolve_darshan_export_value(raw, run_sh.parent), raw
    except Exception:
        return None, None
    return None, None


def count_data_rows(plan_csv: Path) -> int:
    if not plan_csv.is_file():
        return 0
    rows = 0
    with plan_csv.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("data,"):
                rows += 1
    return rows


def parse_phase_cap_diagnostics(notes_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not notes_path.is_file():
        return out
    line_re = re.compile(
        r"phase_cap=(\d+)"
        r"(?:\s+\(input=(\d+)\s+adapted=(True|False)\))?"
        r"\s+emitted_rows=(\d+)\s+phase_cap_applied=(True|False)",
        re.IGNORECASE,
    )
    unmet_re = re.compile(r"phase_cap_unmet=(True|False)", re.IGNORECASE)
    try:
        for line in notes_path.read_text(encoding="utf-8").splitlines():
            m = line_re.search(line)
            if m:
                out["phase_cap"] = m.group(1)
                if m.group(2) is not None:
                    out["phase_cap_input"] = m.group(2)
                if m.group(3) is not None:
                    out["phase_cap_adapted"] = m.group(3)
                out["emitted_rows"] = m.group(4)
                out["phase_cap_applied"] = m.group(5)
            u = unmet_re.search(line)
            if u:
                out["phase_cap_unmet"] = u.group(1)
    except Exception:
        return out
    return out


def poll_for_darshan(
    run_root: Path,
    expected: Path | None,
    *,
    timeout_sec: float,
    initial_interval_sec: float,
    max_interval_sec: float,
    backoff: float,
) -> tuple[Path | None, str]:
    deadline = time.time() + max(0.0, timeout_sec)
    interval = max(0.1, initial_interval_sec)
    backoff = max(1.0, backoff)
    max_interval = max(interval, max_interval_sec)

    while True:
        if expected and expected.is_file():
            return expected, "expected_path"

        found = sorted(run_root.glob("*.darshan"))
        if expected is None:
            if len(found) == 1:
                return found[0], "single_discovered"
        else:
            if len(found) == 1:
                return found[0], "expected_missing_single_discovered"

        if time.time() >= deadline:
            return None, "timeout"

        time.sleep(interval)
        interval = min(max_interval, interval * backoff)


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
    src_c = root / "scripts" / "mpi_synthio.c"
    src_make = root / "scripts" / "Makefile"
    if target.exists() and not force_build:
        src_mtimes = [p.stat().st_mtime for p in (src_c, src_make) if p.exists()]
        newest_src = max(src_mtimes) if src_mtimes else 0.0
        if target.stat().st_mtime >= newest_src:
            print(f"{ts()}  SKIP build: {target} already exists and is up-to-date")
            return
        print(f"{ts()}  REBUILD: source newer than {target}")

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
    # Ensure all orchestrator prints are emitted immediately when stdout/stderr
    # are piped through tee into workload_synthesizer_*.log.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

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
    meta_scope = parse_meta_scope(opts.get("meta_scope", "separate"))
    optimizer = (opts.get("optimizer", "lexicographic") or "lexicographic").strip()
    seq_policy = (opts.get("seq_policy", "nonconsec_strict") or "nonconsec_strict").strip()
    alignment_policy = (opts.get("alignment_policy", "structure_preserving") or "structure_preserving").strip()
    phase_cap = int(opts.get("phase_cap", "50000") or "50000")
    data_random_preseek = int(opts.get("data_random_preseek", "0") or "0")
    darshan_poll_timeout_sec = float(
        opts.get("darshan_poll_timeout_sec", opts.get("flush_wait_sec", "120")) or "120"
    )
    darshan_poll_initial_sec = float(opts.get("darshan_poll_initial_sec", "0.5") or "0.5")
    darshan_poll_max_sec = float(opts.get("darshan_poll_max_sec", "5.0") or "5.0")
    darshan_poll_backoff = float(opts.get("darshan_poll_backoff", "1.6") or "1.6")

    if nprocs_override and not nprocs_override.isdigit():
        raise ValueError(f"Invalid nprocs '{nprocs_override}': expected integer")
    if nprocs_cap <= 0:
        raise ValueError("nprocs_cap must be > 0")
    if optimizer != "lexicographic":
        raise ValueError(f"Invalid optimizer '{optimizer}': expected 'lexicographic'")
    if seq_policy != "nonconsec_strict":
        raise ValueError(f"Invalid seq_policy '{seq_policy}': expected 'nonconsec_strict'")
    if alignment_policy not in {"structure_preserving", "strict_per_op"}:
        raise ValueError(
            f"Invalid alignment_policy '{alignment_policy}': expected structure_preserving or strict_per_op"
        )
    if phase_cap < 0:
        raise ValueError("phase_cap must be >= 0")
    if data_random_preseek not in {0, 1}:
        raise ValueError("data_random_preseek must be 0 or 1")
    if darshan_poll_timeout_sec < 0:
        raise ValueError("darshan_poll_timeout_sec must be >= 0")
    if darshan_poll_initial_sec <= 0:
        raise ValueError("darshan_poll_initial_sec must be > 0")
    if darshan_poll_max_sec <= 0:
        raise ValueError("darshan_poll_max_sec must be > 0")
    if darshan_poll_backoff < 1.0:
        raise ValueError("darshan_poll_backoff must be >= 1.0")

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
    print(
        f"{ts()}  Planner controls: optimizer={optimizer} seq_policy={seq_policy} "
        f"alignment_policy={alignment_policy} phase_cap={phase_cap} data_random_preseek={data_random_preseek}"
    )

    ensure_mpi_binary(root, bin_dir, force_build)

    workloads = collect_workload_jsons(input_dir, filters_raw)
    if not workloads:
        print(f"{ts()}  No JSON inputs selected/found (INPUT_DIR={input_dir}).")
        return
    print(f"{ts()}  Selected workloads: {len(workloads)}")

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
            "--optimizer",
            optimizer,
            "--seq-policy",
            seq_policy,
            "--alignment-policy",
            alignment_policy,
            "--phase-cap",
            str(phase_cap),
            "--data-random-preseek",
            str(data_random_preseek),
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

        plan_csv = run_root / "payload" / "plan.csv"
        notes_path = run_root / "run_from_features.sh.notes.txt"
        data_rows = count_data_rows(plan_csv)
        phase_diag = parse_phase_cap_diagnostics(notes_path)
        if phase_diag:
            print(
                f"{ts()}  [PlanDiag] data_rows={data_rows} "
                f"phase_cap={phase_diag.get('phase_cap', 'NA')} "
                f"phase_cap_input={phase_diag.get('phase_cap_input', 'NA')} "
                f"phase_cap_adapted={phase_diag.get('phase_cap_adapted', 'NA')} "
                f"phase_cap_applied={phase_diag.get('phase_cap_applied', 'NA')} "
                f"phase_cap_unmet={phase_diag.get('phase_cap_unmet', 'NA')}"
            )
        else:
            print(f"{ts()}  [PlanDiag] data_rows={data_rows} phase_cap_diagnostics=unavailable")

        if delete_darshan:
            print(f"{ts()}  [Pre-run] --delete-darshan enabled -> removing existing Darshan files in {run_root}")
            for p in run_root.glob("*.darshan"):
                p.unlink(missing_ok=True)

        expected, expected_raw = extract_darshan_logfile(run_sh)
        if expected_raw and expected is None:
            print(f"{ts()}  WARN: Could not fully resolve DARSHAN_LOGFILE export: {expected_raw}")

        if (not delete_darshan) and expected and expected.is_file():
            print(f"{ts()}  SKIP: Found existing Darshan artifact: {expected}")
            print(f"{ts()}  Proceeding directly to analysis")
        else:
            print(f"{ts()}  [Run] {run_sh}")
            run_cmd(["bash", str(run_sh)])
            print(
                f"{ts()}  [Validate] Polling for Darshan logfile "
                f"(timeout={darshan_poll_timeout_sec:.1f}s initial={darshan_poll_initial_sec:.2f}s "
                f"max={darshan_poll_max_sec:.2f}s backoff={darshan_poll_backoff:.2f})"
            )
            resolved, poll_status = poll_for_darshan(
                run_root,
                expected,
                timeout_sec=darshan_poll_timeout_sec,
                initial_interval_sec=darshan_poll_initial_sec,
                max_interval_sec=darshan_poll_max_sec,
                backoff=darshan_poll_backoff,
            )
            if resolved is None:
                present = sorted(run_root.glob("*.darshan"))
                print(f"{ts()}  ERROR: Darshan poll timed out ({poll_status})")
                if present:
                    for p in present:
                        print(f"{ts()}    Present: {p}")
                cleanup_payload_subdirs(run_root)
                continue
            expected = resolved
            print(f"{ts()}  OK: Found Darshan: {expected} ({poll_status})")

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
