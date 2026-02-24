#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Darshan analysis for warm-start recommender samples")
    ap.add_argument("--darshan", required=True, help="Path to .darshan file")
    ap.add_argument("--outdir", required=True, help="Output directory for parsed artifacts")
    ap.add_argument("--metric-key", default="POSIX_agg_perf_by_slowest", help="Preferred metric column in darshan_summary.csv")
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

    # 3) metric extraction
    with csv_path.open("r", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    if args.metric_key in row and row[args.metric_key] not in ("", None):
        selected_metric = float(row[args.metric_key])
    else:
        selected_metric = fallback_perf_mib_per_s(row)

    metrics = {
        "darshan_file": str(darshan_path),
        "metric_key": args.metric_key,
        "selected_metric": float(selected_metric),
        "fallback_mib_per_s": float(fallback_perf_mib_per_s(row)),
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
