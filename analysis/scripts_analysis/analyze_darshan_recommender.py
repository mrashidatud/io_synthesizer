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


def _row_float(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def fallback_perf_mib_per_s(row: dict[str, str], module_pref: str = "posix") -> float:
    prefix = "MPIIO" if str(module_pref).strip().lower() == "mpiio" else "POSIX"
    bytes_total = _row_float(row, f"{prefix}_BYTES_READ") + _row_float(row, f"{prefix}_BYTES_WRITTEN")
    total_f_time = (
        _row_float(row, f"{prefix}_F_READ_TIME")
        + _row_float(row, f"{prefix}_F_WRITE_TIME")
        + _row_float(row, f"{prefix}_F_META_TIME")
    )
    if (bytes_total <= 0.0 or total_f_time <= 1e-12) and prefix != "POSIX":
        bytes_total = _row_float(row, "POSIX_BYTES_READ") + _row_float(row, "POSIX_BYTES_WRITTEN")
        total_f_time = (
            _row_float(row, "POSIX_F_READ_TIME")
            + _row_float(row, "POSIX_F_WRITE_TIME")
            + _row_float(row, "POSIX_F_META_TIME")
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


def select_metric(
    row: dict[str, str],
    perf_metrics: dict[str, float],
    metric_key: str,
    io_api: str,
) -> tuple[float, str]:
    # Caller preference first.
    if metric_key in perf_metrics:
        return float(perf_metrics[metric_key]), f"darshan_perf:{metric_key}"
    if metric_key in row and row[metric_key] not in ("", None):
        return float(row[metric_key]), f"summary:{metric_key}"
    if f"MPIIO_{metric_key}" in row and row[f"MPIIO_{metric_key}"] not in ("", None):
        return float(row[f"MPIIO_{metric_key}"]), f"summary:MPIIO_{metric_key}"
    if f"POSIX_{metric_key}" in row and row[f"POSIX_{metric_key}"] not in ("", None):
        return float(row[f"POSIX_{metric_key}"]), f"summary:POSIX_{metric_key}"

    io_api_n = str(io_api).strip().lower()
    if io_api_n == "mpiio":
        if row.get("MPIIO_agg_perf_by_slowest", "") not in ("", None):
            return float(row["MPIIO_agg_perf_by_slowest"]), "summary:MPIIO_agg_perf_by_slowest"
        if "agg_perf_by_slowest" in perf_metrics:
            return float(perf_metrics["agg_perf_by_slowest"]), "darshan_perf:agg_perf_by_slowest"
        if row.get("agg_perf_by_slowest", "") not in ("", None):
            return float(row["agg_perf_by_slowest"]), "summary:agg_perf_by_slowest"
        if row.get("POSIX_agg_perf_by_slowest", "") not in ("", None):
            return float(row["POSIX_agg_perf_by_slowest"]), "summary:POSIX_agg_perf_by_slowest"
        return float(fallback_perf_mib_per_s(row, module_pref="mpiio")), "fallback:bytes_over_f_time:MPIIO"

    if row.get("POSIX_agg_perf_by_slowest", "") not in ("", None):
        return float(row["POSIX_agg_perf_by_slowest"]), "summary:POSIX_agg_perf_by_slowest"
    if "agg_perf_by_slowest" in perf_metrics:
        return float(perf_metrics["agg_perf_by_slowest"]), "darshan_perf:agg_perf_by_slowest"
    if row.get("agg_perf_by_slowest", "") not in ("", None):
        return float(row["agg_perf_by_slowest"]), "summary:agg_perf_by_slowest"
    return float(fallback_perf_mib_per_s(row, module_pref="posix")), "fallback:bytes_over_f_time:POSIX"


def main() -> None:
    ap = argparse.ArgumentParser(description="Darshan analysis for warm-start recommender samples")
    ap.add_argument("--darshan", required=True, help="Path to .darshan file")
    ap.add_argument("--outdir", required=True, help="Output directory for parsed artifacts")
    ap.add_argument("--io-api", choices=["auto", "posix", "mpiio"], default="auto", help="I/O API used for module-aware metric fallback")
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

    io_api = args.io_api
    if io_api == "auto":
        has_mpiio = (_row_float(row, "MPIIO_BYTES_READ") + _row_float(row, "MPIIO_BYTES_WRITTEN")) > 0.0
        io_api = "mpiio" if has_mpiio else "posix"

    selected_metric, metric_source = select_metric(
        row=row,
        perf_metrics=perf_metrics,
        metric_key=args.metric_key,
        io_api=io_api,
    )

    metrics = {
        "darshan_file": str(darshan_path),
        "io_api": io_api,
        "metric_key": args.metric_key,
        "metric_source": metric_source,
        "selected_metric": float(selected_metric),
        "fallback_mib_per_s": float(fallback_perf_mib_per_s(row, module_pref=io_api)),
        "darshan_perf_metrics": perf_metrics,
        "agg_perf_by_slowest": float(perf_metrics.get("agg_perf_by_slowest", 0.0)),
        "POSIX_agg_perf_by_slowest": _row_float(row, "POSIX_agg_perf_by_slowest"),
        "MPIIO_agg_perf_by_slowest": _row_float(row, "MPIIO_agg_perf_by_slowest"),
        "POSIX_BYTES_READ": _row_float(row, "POSIX_BYTES_READ"),
        "POSIX_BYTES_WRITTEN": _row_float(row, "POSIX_BYTES_WRITTEN"),
        "POSIX_F_READ_TIME": _row_float(row, "POSIX_F_READ_TIME"),
        "POSIX_F_WRITE_TIME": _row_float(row, "POSIX_F_WRITE_TIME"),
        "POSIX_F_META_TIME": _row_float(row, "POSIX_F_META_TIME"),
        "MPIIO_BYTES_READ": _row_float(row, "MPIIO_BYTES_READ"),
        "MPIIO_BYTES_WRITTEN": _row_float(row, "MPIIO_BYTES_WRITTEN"),
        "MPIIO_F_READ_TIME": _row_float(row, "MPIIO_F_READ_TIME"),
        "MPIIO_F_WRITE_TIME": _row_float(row, "MPIIO_F_WRITE_TIME"),
        "MPIIO_F_META_TIME": _row_float(row, "MPIIO_F_META_TIME"),
        "MPIIO_INDEP_READS": _row_float(row, "MPIIO_INDEP_READS"),
        "MPIIO_INDEP_WRITES": _row_float(row, "MPIIO_INDEP_WRITES"),
        "MPIIO_COLL_READS": _row_float(row, "MPIIO_COLL_READS"),
        "MPIIO_COLL_WRITES": _row_float(row, "MPIIO_COLL_WRITES"),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
