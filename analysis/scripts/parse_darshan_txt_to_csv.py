#!/usr/bin/env python3
"""
Darshan single-report → CSV (one row)

WHAT THIS DOES
- Reads exactly one Darshan *text* report (e.g., output from `darshan-parser -s <log> > report.txt`)
- Parses job metadata (exe, uid, jobid, start/end times, nprocs, run_time)
- Parses POSIX/MPIIO/STDIO module sections:
    * File-type aggregates (total/read_only/write_only/read_write/unique/shared)
    * "Performance" section key/value pairs (nested & flat)
    * Legacy "total_<module>_*" counters
- Writes a CSV with a single row (one file analyzed), preserving your preferred column ordering

USAGE
  python3 darshan_onefile_to_csv.py --input /path/to/report.txt \
                                    --output /path/to/darshan_summary.csv

NOTES
- This script does *not* walk directories; it processes only the file given by --input.
- If you want to batch, call it multiple times from a shell script or combine CSVs later.
"""

import os
import re
import csv
import logging
import argparse

# ─── CONFIG (defaults; can be overridden via CLI) ──────────────────────────────
file_types    = ["total", "read_only", "write_only", "read_write", "unique", "shared"]
header_keys   = ["exe", "uid", "jobid", "start_time", "start_time_asci", "end_time", "end_time_asci", "nprocs", "run_time"]
modules_order = ["POSIX", "MPIIO", "STDIO"]
# ───────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Parse a single Darshan text report into a one-row CSV.")
    ap.add_argument("--input",  required=True, help="Path to a single Darshan text report (e.g., output of `darshan-parser -s`).")
    ap.add_argument("--output", required=True, help="CSV path to write (will contain exactly one row).")
    ap.add_argument("--log", default=None, help="Optional path for a processing log file.")
    args = ap.parse_args()

    # Logging (console + optional file)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
        fh = logging.FileHandler(args.log)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    in_path = os.path.abspath(args.input)
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # Compile regex patterns (same as your original)
    oex_header_re = re.compile(r"#\s*(exe|uid|jobid|start_time|start_time_asci|end_time|end_time_asci|nprocs|run time):\s*(.*)")
    mod_header_re  = re.compile(r"#\s*([\w-]+)\s+module data")
    file_count_re  = re.compile(
        r"\s*#\s*(?P<ft>" + "|".join(file_types) + r"):\s*"
        r"(?P<count>-?\d+)\s+(?P<bytes>-?\d+)\s+(?P<offset>-?\d+)"
    )
    nested_perf_re = re.compile(
        r"\s*#\s*(?P<section>[\w ]+):\s*(?P<key>[\w_]+):\s*(?P<val>-?\d+(\.\d+)?)"
    )
    simple_perf_re = re.compile(
        r"\s*#\s*(?P<key>[\w_]+):\s*(?P<val>-?\d+(\.\d+)?)"
    )
    def simple_mod_re(m):  # legacy totals per-module
        return re.compile(rf"total_{m}_(?P<cnt>[^:]+):\s*(?P<val>-?\d+(\.\d+)?)")

    # Row we will emit
    row = {"filename": os.path.basename(in_path)}
    for hk in header_keys:
        row[hk] = ""

    all_columns = set()
    current_module = None
    perf_section = False

    logging.info(f"Parsing: {in_path}")
    with open(in_path, "r", encoding="utf-8", errors="ignore") as rf:
        for line in rf:
            # header fields
            m0 = oex_header_re.match(line)
            if m0:
                key, val = m0.group(1), m0.group(2).strip()
                if key == "run time":
                    key = "run_time"
                row[key] = val
                continue

            # module section header
            m1 = mod_header_re.match(line)
            if m1:
                current_module = m1.group(1).replace("-", "").upper()
                perf_section = False
                continue

            if not current_module:
                continue

            # per-file-type aggregates
            m2 = file_count_re.match(line)
            if m2:
                ft   = m2.group("ft")
                cnt  = m2.group("count")
                byts = m2.group("bytes")
                off  = m2.group("offset")
                for suffix, v in [("file_count", cnt), ("total_bytes", byts), ("max_byte_offset", off)]:
                    col = f"{current_module}_file_type_{ft}_{suffix}"
                    row[col] = v
                    all_columns.add(col)
                continue

            # performance section start
            if line.strip().lower().startswith("# performance"):
                perf_section = True
                continue

            # performance section (nested + simple)
            if perf_section:
                m3 = nested_perf_re.match(line)
                if m3:
                    sect = m3.group("section").strip().replace(" ", "_")
                    key  = m3.group("key")
                    val  = m3.group("val")
                    col  = f"{current_module}_{sect}_{key}"
                    row[col] = val
                    all_columns.add(col)
                    continue
                m4 = simple_perf_re.match(line)
                if m4:
                    key = m4.group("key")
                    val = m4.group("val")
                    col = f"{current_module}_{key}"
                    row[col] = val
                    all_columns.add(col)
                    continue

            # legacy total_ counters
            m5 = simple_mod_re(current_module).search(line)
            if m5:
                cnt = m5.group("cnt")
                val = m5.group("val")
                col = f"{current_module}_{cnt}"
                row[col] = val
                all_columns.add(col)
                continue

    # Build CSV header (same ordering rules you used)
    meta_cols = ["filename"] + header_keys
    mod_cols  = []
    for mod in modules_order:
        mod_cols += sorted(c for c in all_columns if c.startswith(f"{mod}_"))
    other_cols = sorted(c for c in all_columns if c not in mod_cols)
    header = meta_cols + mod_cols + other_cols

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', newline='') as wf:
        writer = csv.DictWriter(wf, fieldnames=header)
        writer.writeheader()
        # ensure defaults for any missing columns
        for c in meta_cols:
            row.setdefault(c, "")
        for c in mod_cols + other_cols:
            row.setdefault(c, 0)
        writer.writerow(row)

    logging.info(f"Wrote 1 row to {args.output}")
    print(f"Wrote 1 row to {args.output}")

if __name__ == "__main__":
    main()
