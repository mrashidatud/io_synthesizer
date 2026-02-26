#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path


ROOT = Path("/mnt/hasanfs/io_synthesizer")
SCRIPTS = ROOT / "analysis" / "scripts_analysis"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def fallback_perf_mib_per_s(row: dict[str, str]) -> float:
    bytes_total = float(row.get("POSIX_BYTES_READ", 0.0) or 0.0) + float(row.get("POSIX_BYTES_WRITTEN", 0.0) or 0.0)
    total_f_time = (
        float(row.get("POSIX_F_READ_TIME", 0.0) or 0.0)
        + float(row.get("POSIX_F_WRITE_TIME", 0.0) or 0.0)
        + float(row.get("POSIX_F_META_TIME", 0.0) or 0.0)
    )
    if total_f_time <= 1e-12:
        return 0.0
    return bytes_total / total_f_time / (1024.0 * 1024.0)


def parse_perf_metrics_from_darshan(darshan_path: Path) -> dict[str, float]:
    completed = subprocess.run(
        ["darshan-parser", "--perf", str(darshan_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    metrics: dict[str, float] = {}
    line_re = re.compile(r"^\s*#\s*(?:[\w ]+:\s*)?(?P<key>[\w_]+):\s*(?P<val>-?\d+(?:\.\d+)?)\s*$")
    for raw_line in completed.stdout.splitlines():
        m = line_re.match(raw_line)
        if not m:
            continue
        key = m.group("key")
        val = float(m.group("val"))
        if key in {"total_bytes", "agg_perf_by_cumul", "agg_perf_by_open", "agg_perf_by_open_lastio", "agg_perf_by_slowest"}:
            metrics[key] = val
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Darshan analysis for warm-start recommender samples")
    ap.add_argument("--darshan", required=True, help="Path to .darshan file")
    ap.add_argument("--outdir", required=True, help="Output directory for parsed artifacts")
    ap.add_argument("--metric-key", default="agg_perf_by_slowest", help="Preferred metric key (Darshan --perf key or darshan_summary.csv column)")
    ap.add_argument("--txt-name", default=None, help="Optional txt output name")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    darshan_path = Path(args.darshan).resolve()
    if not darshan_path.exists():
        raise FileNotFoundError(f"Missing darshan file: {darshan_path}")

    txt_name = args.txt_name or f"{darshan_path.stem}.txt"
    txt_path = outdir / txt_name
    csv_path = outdir / "darshan_summary.csv"
    metrics_path = outdir / "recommender_metrics.json"

    # 1) darshan -> txt
    with txt_path.open("w", encoding="utf-8") as f:
        subprocess.run(["darshan-parser", "--total", "--all", str(darshan_path)], stdout=f, check=True)

    # 2) txt -> one-row csv
    conv = SCRIPTS / "parse_darshan_txt_to_csv.py"
    run(["python3", str(conv), "--input", str(txt_path), "--output", str(csv_path)])
    perf_metrics = parse_perf_metrics_from_darshan(darshan_path)

    # 3) metric extraction
    with csv_path.open("r", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    if args.metric_key in perf_metrics:
        selected_metric = float(perf_metrics[args.metric_key])
    elif args.metric_key in row and row[args.metric_key] not in ("", None):
        selected_metric = float(row[args.metric_key])
    elif f"POSIX_{args.metric_key}" in row and row[f"POSIX_{args.metric_key}"] not in ("", None):
        selected_metric = float(row[f"POSIX_{args.metric_key}"])
    else:
        selected_metric = fallback_perf_mib_per_s(row)

    metrics = {
        "darshan_file": str(darshan_path),
        "metric_key": args.metric_key,
        "selected_metric": float(selected_metric),
        "fallback_mib_per_s": float(fallback_perf_mib_per_s(row)),
        "darshan_perf_metrics": perf_metrics,
        "agg_perf_by_slowest": float(perf_metrics.get("agg_perf_by_slowest", 0.0)),
        "POSIX_agg_perf_by_slowest": float(row.get("POSIX_agg_perf_by_slowest", 0.0) or 0.0),
        "POSIX_BYTES_READ": float(row.get("POSIX_BYTES_READ", 0.0) or 0.0),
        "POSIX_BYTES_WRITTEN": float(row.get("POSIX_BYTES_WRITTEN", 0.0) or 0.0),
        "POSIX_F_READ_TIME": float(row.get("POSIX_F_READ_TIME", 0.0) or 0.0),
        "POSIX_F_WRITE_TIME": float(row.get("POSIX_F_WRITE_TIME", 0.0) or 0.0),
        "POSIX_F_META_TIME": float(row.get("POSIX_F_META_TIME", 0.0) or 0.0),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
