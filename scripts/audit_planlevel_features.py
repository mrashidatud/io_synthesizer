#!/usr/bin/env python3
"""
Plan-level feature audit for exemplar workloads.

This script regenerates synth plans via features2synth_opsaware.py and compares
the resulting plan-derived pct_* features against input JSON pct_* values.

Outputs under --out-root:
  - <workload>/... generated synth artifacts + pct_compare_report_planlevel.txt
  - top25_summary.csv
  - top25_outside.csv
  - top25_within_0p1.csv
  - top25_nonexact_explained.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else float(num) / float(den)


def bin_index(xfer: int) -> int:
    xfer = int(xfer)
    if xfer <= 100:
        return 0
    if xfer <= 1024:
        return 1
    if xfer <= 10240:
        return 2
    if xfer <= 102400:
        return 3
    if xfer <= 1048576:
        return 4
    if xfer <= 4194304:
        return 5
    if xfer <= 10485760:
        return 6
    if xfer <= 104857600:
        return 7
    if xfer <= 1073741824:
        return 8
    return 9


def round2(v: float) -> float:
    return round(float(v) + 1e-12, 2)


def reason_for(feature: str) -> str:
    role_file = {
        "pct_shared_files",
        "pct_unique_files",
        "pct_read_only_files",
        "pct_read_write_files",
        "pct_write_only_files",
        "pct_bytes_shared_files",
        "pct_bytes_unique_files",
        "pct_bytes_read_only_files",
        "pct_bytes_read_write_files",
        "pct_bytes_write_only_files",
    }
    byte_ladder = {
        "pct_byte_reads",
        "pct_byte_writes",
        "pct_read_0_100K",
        "pct_read_100K_10M",
        "pct_read_10M_1G_PLUS",
        "pct_write_0_100K",
        "pct_write_100K_10M",
        "pct_write_10M_1G_PLUS",
    }
    if feature in role_file:
        return "Discrete file-count and integer-op constraints in role/flag layout."
    if feature in byte_ladder:
        return "Integer quantization and fixed transfer ladder constraints."
    return "Integer quantization/rounding constraints."


def intuition_for(feature: str) -> str:
    role = {
        "pct_read_only_files",
        "pct_read_write_files",
        "pct_write_only_files",
        "pct_shared_files",
        "pct_unique_files",
        "pct_bytes_read_only_files",
        "pct_bytes_read_write_files",
        "pct_bytes_write_only_files",
        "pct_bytes_shared_files",
        "pct_bytes_unique_files",
    }
    bins = {
        "pct_read_0_100K",
        "pct_read_100K_10M",
        "pct_write_0_100K",
        "pct_write_100K_10M",
    }
    bytesplit = {"pct_byte_reads", "pct_byte_writes"}
    if feature in role:
        return "Ratios are built from integer file counts/bytes, so only discrete points are representable."
    if feature in bins:
        return "Bin shares come from integer op counts; nearest feasible bin count causes small offsets."
    if feature in bytesplit:
        return "Byte split is constrained by integer ops and fixed transfer-size ladders."
    if feature == "pct_rw_switches":
        return "Switch ratio is based on integer R<->W boundaries in ordered row sequence."
    if feature == "pct_meta_open_access":
        return "Meta fraction uses integer metadata counts over integer total op counts."
    return "Integer count rounding and discrete planning granularity."


def parse_plan(plan_csv: Path) -> Tuple[Dict[str, float], Dict[str, int]]:
    reads = writes = 0
    bytes_read = bytes_write = 0
    consec_reads = consec_writes = 0.0
    seq_reads = seq_writes = 0.0
    file_not_aligned = 0.0
    mem_not_aligned = 0.0

    meta_open = meta_stat = meta_seek = meta_sync = 0
    read_bins = [0] * 10
    write_bins = [0] * 10

    # Group-1 file type accounting includes active data and active meta paths.
    file_group1 = defaultdict(
        lambda: {
            "active_data_ops": 0,
            "meta_ops": 0,
            "read_ops": 0,
            "write_ops": 0,
            "bytes": 0,
            "shared_tag": 0,
            "unique_tag": 0,
        }
    )

    seq_intents: List[Tuple[int, int]] = []

    with plan_csv.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            typ = str(row.get("type", "")).strip().lower()
            path = str(row.get("path", ""))
            flags = str(row.get("flags", ""))

            if typ == "meta":
                m_open = int(float(row.get("meta_open", 0) or 0))
                m_stat = int(float(row.get("meta_stat", 0) or 0))
                m_seek = int(float(row.get("meta_seek", 0) or 0))
                m_sync = int(float(row.get("meta_sync", 0) or 0))
                mops = m_open + m_stat + m_seek + m_sync
                meta_open += m_open
                meta_stat += m_stat
                meta_seek += m_seek
                meta_sync += m_sync

                rec = file_group1[path]
                rec["meta_ops"] += mops
                if "|shared|" in flags:
                    rec["shared_tag"] += 1
                elif "|unique|" in flags:
                    rec["unique_tag"] += 1
                continue

            if typ != "data":
                continue

            ops = int(float(row.get("n_ops", 0) or 0))
            if ops <= 0:
                continue

            xfer = int(float(row.get("xfer", 0) or 0))
            total_bytes = int(float(row.get("total_bytes", 0) or 0))
            is_write = float(row.get("p_write", 0) or 0) >= 0.5

            if is_write:
                writes += ops
                bytes_write += total_bytes
                consec_writes += ops * float(row.get("p_consec_w", 0) or 0)
                seq_writes += ops * float(row.get("p_seq_w", 0) or 0)
                write_bins[bin_index(xfer)] += ops
                seq_intents.append((1, ops))
            else:
                reads += ops
                bytes_read += total_bytes
                consec_reads += ops * float(row.get("p_consec_r", 0) or 0)
                seq_reads += ops * float(row.get("p_seq_r", 0) or 0)
                read_bins[bin_index(xfer)] += ops
                seq_intents.append((0, ops))

            file_not_aligned += ops * float(row.get("p_ua_file", 0) or 0)
            mem_not_aligned += ops * float(row.get("p_ua_mem", 0) or 0)

            rec = file_group1[path]
            rec["active_data_ops"] += ops
            rec["bytes"] += total_bytes
            if is_write:
                rec["write_ops"] += ops
            else:
                rec["read_ops"] += ops
            if "|shared|" in flags:
                rec["shared_tag"] += 1
            elif "|unique|" in flags:
                rec["unique_tag"] += 1

    # Count R/W boundaries from ordered row sequence.
    switches = 0
    prev = None
    for intent, ops in seq_intents:
        if ops <= 0:
            continue
        if prev is not None and intent != prev:
            switches += 1
        prev = intent

    sh_count = uq_count = 0
    sh_bytes = uq_bytes = 0
    ro_count = rw_count = wo_count = 0
    ro_bytes = rw_bytes = wo_bytes = 0

    for rec in file_group1.values():
        active_group1 = rec["active_data_ops"] > 0 or rec["meta_ops"] > 0
        if active_group1:
            is_shared = rec["shared_tag"] >= rec["unique_tag"]
            if is_shared:
                sh_count += 1
                sh_bytes += rec["bytes"]
            else:
                uq_count += 1
                uq_bytes += rec["bytes"]

        if rec["active_data_ops"] > 0:
            r_ops = rec["read_ops"]
            w_ops = rec["write_ops"]
            b = rec["bytes"]
            if r_ops > 0 and w_ops == 0:
                ro_count += 1
                ro_bytes += b
            elif r_ops > 0 and w_ops > 0:
                rw_count += 1
                rw_bytes += b
            elif r_ops == 0 and w_ops > 0:
                wo_count += 1
                wo_bytes += b

    r_0_100k = sum(read_bins[0:4])
    r_100k_10m = sum(read_bins[4:7])
    r_10m_plus = sum(read_bins[7:10])
    w_0_100k = sum(write_bins[0:4])
    w_100k_10m = sum(write_bins[4:7])
    w_10m_plus = sum(write_bins[7:10])

    meta_total = meta_open + meta_stat + meta_seek + meta_sync
    total_acc = reads + writes
    total_bytes = bytes_read + bytes_write

    produced = {
        "pct_file_not_aligned": safe_div(int(round(file_not_aligned)), total_acc),
        "pct_mem_not_aligned": safe_div(int(round(mem_not_aligned)), total_acc),
        "pct_reads": safe_div(reads, total_acc),
        "pct_writes": safe_div(writes, total_acc),
        "pct_consec_reads": safe_div(int(round(consec_reads)), reads),
        "pct_consec_writes": safe_div(int(round(consec_writes)), writes),
        "pct_seq_reads": safe_div(int(round(seq_reads)), reads),
        "pct_seq_writes": safe_div(int(round(seq_writes)), writes),
        "pct_rw_switches": safe_div(switches, total_acc),
        "pct_byte_reads": safe_div(bytes_read, total_bytes),
        "pct_byte_writes": safe_div(bytes_write, total_bytes),
        "pct_io_access": safe_div(total_acc, total_acc + meta_total),
        "pct_meta_open_access": safe_div(meta_open, total_acc + meta_total),
        "pct_meta_stat_access": safe_div(meta_stat, total_acc + meta_total),
        "pct_meta_seek_access": safe_div(meta_seek, total_acc + meta_total),
        "pct_meta_sync_access": safe_div(meta_sync, total_acc + meta_total),
        "pct_read_0_100K": safe_div(r_0_100k, reads),
        "pct_read_100K_10M": safe_div(r_100k_10m, reads),
        "pct_read_10M_1G_PLUS": safe_div(r_10m_plus, reads),
        "pct_write_0_100K": safe_div(w_0_100k, writes),
        "pct_write_100K_10M": safe_div(w_100k_10m, writes),
        "pct_write_10M_1G_PLUS": safe_div(w_10m_plus, writes),
        "pct_shared_files": safe_div(sh_count, sh_count + uq_count),
        "pct_bytes_shared_files": safe_div(sh_bytes, sh_bytes + uq_bytes),
        "pct_unique_files": safe_div(uq_count, sh_count + uq_count),
        "pct_bytes_unique_files": safe_div(uq_bytes, sh_bytes + uq_bytes),
        "pct_read_only_files": safe_div(ro_count, ro_count + rw_count + wo_count),
        "pct_bytes_read_only_files": safe_div(ro_bytes, ro_bytes + rw_bytes + wo_bytes),
        "pct_read_write_files": safe_div(rw_count, ro_count + rw_count + wo_count),
        "pct_bytes_read_write_files": safe_div(rw_bytes, ro_bytes + rw_bytes + wo_bytes),
        "pct_write_only_files": safe_div(wo_count, ro_count + rw_count + wo_count),
        "pct_bytes_write_only_files": safe_div(wo_bytes, ro_bytes + rw_bytes + wo_bytes),
    }

    metrics = {
        "reads": reads,
        "writes": writes,
        "total_acc": total_acc,
        "bytes_read": bytes_read,
        "bytes_write": bytes_write,
        "total_bytes": total_bytes,
        "seq_reads": int(round(seq_reads)),
        "seq_writes": int(round(seq_writes)),
        "switches": switches,
        "meta_open": meta_open,
        "meta_stat": meta_stat,
        "meta_seek": meta_seek,
        "meta_sync": meta_sync,
        "meta_total": meta_total,
        "ops_total": total_acc + meta_total,
        "r_0_100K": r_0_100k,
        "r_100K_10M": r_100k_10m,
        "w_0_100K": w_0_100k,
        "w_100K_10M": w_100k_10m,
        "sh_count": sh_count,
        "uq_count": uq_count,
        "sh_bytes": sh_bytes,
        "uq_bytes": uq_bytes,
        "ro_count": ro_count,
        "rw_count": rw_count,
        "wo_count": wo_count,
        "ro_bytes": ro_bytes,
        "rw_bytes": rw_bytes,
        "wo_bytes": wo_bytes,
    }

    return {k: round2(v) for k, v in produced.items()}, metrics


def evidence_for(feature: str, m: Dict[str, int]) -> str:
    if feature == "pct_read_only_files":
        den = m["ro_count"] + m["rw_count"] + m["wo_count"]
        return f"RO files = {m['ro_count']}/{den} (step={safe_div(1, den):.4f})"
    if feature == "pct_read_write_files":
        den = m["ro_count"] + m["rw_count"] + m["wo_count"]
        return f"RW files = {m['rw_count']}/{den} (step={safe_div(1, den):.4f})"
    if feature == "pct_write_only_files":
        den = m["ro_count"] + m["rw_count"] + m["wo_count"]
        return f"WO files = {m['wo_count']}/{den} (step={safe_div(1, den):.4f})"
    if feature == "pct_shared_files":
        den = m["sh_count"] + m["uq_count"]
        return f"Shared files (data+meta) = {m['sh_count']}/{den} (step={safe_div(1, den):.4f})"
    if feature == "pct_unique_files":
        den = m["sh_count"] + m["uq_count"]
        return f"Unique files (data+meta) = {m['uq_count']}/{den} (step={safe_div(1, den):.4f})"

    if feature == "pct_bytes_read_only_files":
        den = m["ro_bytes"] + m["rw_bytes"] + m["wo_bytes"]
        return f"RO bytes = {m['ro_bytes']}/{den}"
    if feature == "pct_bytes_read_write_files":
        den = m["ro_bytes"] + m["rw_bytes"] + m["wo_bytes"]
        return f"RW bytes = {m['rw_bytes']}/{den}"
    if feature == "pct_bytes_write_only_files":
        den = m["ro_bytes"] + m["rw_bytes"] + m["wo_bytes"]
        return f"WO bytes = {m['wo_bytes']}/{den}"
    if feature == "pct_bytes_shared_files":
        den = m["sh_bytes"] + m["uq_bytes"]
        return f"Shared bytes = {m['sh_bytes']}/{den}"
    if feature == "pct_bytes_unique_files":
        den = m["sh_bytes"] + m["uq_bytes"]
        return f"Unique bytes = {m['uq_bytes']}/{den}"

    if feature == "pct_reads":
        return f"Read ops = {m['reads']}/{m['total_acc']} (step={safe_div(1, m['total_acc']):.6f})"
    if feature == "pct_writes":
        return f"Write ops = {m['writes']}/{m['total_acc']} (step={safe_div(1, m['total_acc']):.6f})"
    if feature == "pct_seq_reads":
        return f"Seq read ops = {m['seq_reads']}/{m['reads']} (step={safe_div(1, m['reads']):.6f})"
    if feature == "pct_rw_switches":
        return f"RW intent boundaries = {m['switches']}/{m['total_acc']} (step={safe_div(1, m['total_acc']):.6f})"

    if feature == "pct_meta_open_access":
        return f"META open ops = {m['meta_open']}/{m['ops_total']} (step={safe_div(1, m['ops_total']):.6f})"

    if feature == "pct_byte_reads":
        return f"Read bytes = {m['bytes_read']}/{m['total_bytes']}"
    if feature == "pct_byte_writes":
        return f"Write bytes = {m['bytes_write']}/{m['total_bytes']}"

    if feature == "pct_read_0_100K":
        return f"Read-bin[0-100K] ops = {m['r_0_100K']}/{m['reads']} (step={safe_div(1, m['reads']):.6f})"
    if feature == "pct_read_100K_10M":
        return f"Read-bin[100K-10M] ops = {m['r_100K_10M']}/{m['reads']} (step={safe_div(1, m['reads']):.6f})"
    if feature == "pct_write_0_100K":
        return f"Write-bin[0-100K] ops = {m['w_0_100K']}/{m['writes']} (step={safe_div(1, m['writes']):.6f})"
    if feature == "pct_write_100K_10M":
        return f"Write-bin[100K-10M] ops = {m['w_100K_10M']}/{m['writes']} (step={safe_div(1, m['writes']):.6f})"

    return "Integer representability constraint under plan discretization."


def workload_sort_key(path: Path) -> Tuple[int, str]:
    m = re.match(r"^top(\d+)_", path.stem)
    rank = int(m.group(1)) if m else 10**9
    return rank, path.stem


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs-dir",
        default=str(repo_root / "inputs" / "exemplar_jsons"),
        help="Directory with workload feature JSON files.",
    )
    ap.add_argument(
        "--glob",
        default="top*.json",
        help="Glob pattern under --inputs-dir to select workloads.",
    )
    ap.add_argument(
        "--out-root",
        default=str(repo_root / "outputs" / "audit_top25_planlevel"),
        help="Output directory for generated plans and reports.",
    )
    ap.add_argument(
        "--features-script",
        default=str(Path(__file__).resolve().parent / "features2synth_opsaware.py"),
        help="Path to features2synth_opsaware.py",
    )
    ap.add_argument(
        "--min-ranks",
        type=int,
        default=2,
        help="Use nprocs=max(input_nprocs, min_ranks) during plan generation.",
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Abs(delta) threshold for within-vs-outside classification.",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="Delete --out-root before running.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-workload progress printing.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    inputs_dir = Path(args.inputs_dir).resolve()
    out_root = Path(args.out_root).resolve()
    features_script = Path(args.features_script).resolve()
    tol = float(args.tolerance)
    min_ranks = max(1, int(args.min_ranks))

    if not inputs_dir.is_dir():
        raise FileNotFoundError(f"Inputs dir not found: {inputs_dir}")
    if not features_script.is_file():
        raise FileNotFoundError(f"features script not found: {features_script}")

    json_paths = sorted(inputs_dir.glob(args.glob), key=workload_sort_key)
    if not json_paths:
        raise RuntimeError(f"No workload JSON files matched {args.glob} in {inputs_dir}")

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    outside_rows = []
    within_rows = []
    nonexact_rows = []

    for idx, json_path in enumerate(json_paths, 1):
        base = json_path.stem
        outdir = out_root / base
        outdir.mkdir(parents=True, exist_ok=True)

        with json_path.open("r", encoding="utf-8") as f:
            input_feats = json.load(f)
        if not isinstance(input_feats, dict):
            raise TypeError(f"Expected JSON object in {json_path}, got {type(input_feats).__name__}")

        nprocs_input = int(input_feats.get("nprocs", 1) or 1)
        nprocs_used = max(min_ranks, nprocs_input)

        cmd = [
            sys.executable,
            str(features_script),
            "--features",
            str(json_path),
            "--outdir",
            str(outdir),
            "--nprocs",
            str(nprocs_used),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

        produced, metrics = parse_plan(outdir / "payload" / "plan.csv")

        pct_keys = sorted(k for k in input_feats.keys() if k.startswith("pct_"))
        compared = len(pct_keys)
        exact = within = outside = 0
        report_lines = []

        for feature in pct_keys:
            inp = float(input_feats.get(feature, 0.0) or 0.0)
            out = float(produced.get(feature, 0.0) or 0.0)
            delta = out - inp
            abs_delta = abs(delta)

            if abs_delta <= 1e-12:
                exact += 1
                continue

            row = {
                "workflow": base,
                "feature": feature,
                "input": inp,
                "produced": out,
                "delta": delta,
                "classification": "within" if abs_delta <= tol + 1e-12 else "limitation",
                "reason": reason_for(feature),
                "evidence": evidence_for(feature, metrics),
                "intuitive_explanation": intuition_for(feature),
            }
            nonexact_rows.append(row)

            if abs_delta <= tol + 1e-12:
                within += 1
                within_rows.append(
                    {
                        k: row[k]
                        for k in [
                            "workflow",
                            "feature",
                            "input",
                            "produced",
                            "delta",
                            "classification",
                            "reason",
                        ]
                    }
                )
            else:
                outside += 1
                outside_rows.append(
                    {
                        k: row[k]
                        for k in [
                            "workflow",
                            "feature",
                            "input",
                            "produced",
                            "delta",
                            "classification",
                            "reason",
                        ]
                    }
                )
                report_lines.append(
                    f"{feature}: input={inp} produced={out} delta={delta} [limitation] {row['reason']}"
                )

        with (outdir / "pct_compare_report_planlevel.txt").open("w", encoding="utf-8") as f:
            f.write(f"Compared={compared} exact={exact} within={within} outside={outside}\n")
            f.write(f"nprocs_input={nprocs_input} nprocs_used={nprocs_used}\n")
            for line in report_lines:
                f.write(line + "\n")

        summary_rows.append(
            {
                "workflow": base,
                "status": "ok",
                "nprocs_input": nprocs_input,
                "nprocs_used": nprocs_used,
                "compared": compared,
                "exact": exact,
                "within_0p1": within,
                "outside_0p1": outside,
                "note": "",
            }
        )

        if not args.quiet:
            print(
                f"[{idx}/{len(json_paths)}] {base}: exact={exact} within={within} outside={outside} "
                f"(nprocs {nprocs_input}->{nprocs_used})"
            )

    with (out_root / "top25_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "workflow",
                "status",
                "nprocs_input",
                "nprocs_used",
                "compared",
                "exact",
                "within_0p1",
                "outside_0p1",
                "note",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)

    with (out_root / "top25_outside.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "workflow",
                "feature",
                "input",
                "produced",
                "delta",
                "classification",
                "reason",
            ],
        )
        w.writeheader()
        w.writerows(outside_rows)

    with (out_root / "top25_within_0p1.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "workflow",
                "feature",
                "input",
                "produced",
                "delta",
                "classification",
                "reason",
            ],
        )
        w.writeheader()
        w.writerows(within_rows)

    with (out_root / "top25_nonexact_explained.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "workflow",
                "feature",
                "input",
                "produced",
                "delta",
                "classification",
                "reason",
                "evidence",
                "intuitive_explanation",
            ],
        )
        w.writeheader()
        w.writerows(nonexact_rows)

    total_outside = sum(int(r["outside_0p1"]) for r in summary_rows)
    total_nonexact = sum(int(r["within_0p1"]) + int(r["outside_0p1"]) for r in summary_rows)
    print(f"Wrote reports under: {out_root}")
    print(
        f"Workloads={len(summary_rows)} compared_per_workload=32 nonexact={total_nonexact} outside={total_outside}"
    )


if __name__ == "__main__":
    main()

