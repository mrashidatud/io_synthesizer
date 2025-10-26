#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merged Darshan analysis pipeline for a single .darshan file:
  .darshan -> .txt (darshan-parser) -> darshan_summary.csv -> darshan_features_updated.json
Then compares pct_* fields vs the input features JSON with a ±0.05 tolerance.

All outputs are written to the provided --outdir, which should be:
  /mnt/hasanfs/out_synth/<json_base>/

Usage:
  python analyze_darshan_merged.py \
      --darshan /mnt/hasanfs/out_synth/<json_base>/<json_base>_cap_..._.darshan \
      --input-json /mnt/hasanfs/io_synthesizer/inputs/exemplar_jsons/<json_base>.json \
      --outdir /mnt/hasanfs/out_synth/<json_base>/
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/mnt/hasanfs/io_synthesizer")
SCRIPTS = ROOT / "analysis" / "scripts_analysis"

# ± tolerance to treat numeric pct_* values as "close enough"
TOL = 0.05


def run(cmd, cwd=None):
    print(f"[exec] {' '.join(map(str, cmd))}")
    subprocess.run(cmd, cwd=cwd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--darshan", required=True, help="Path to the .darshan file to analyze")
    ap.add_argument("--input-json", required=True, help="Path to the original input features JSON")
    ap.add_argument("--outdir", required=True, help="Run folder: /mnt/hasanfs/out_synth/<json_base>/")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    darshan_path = Path(args.darshan).resolve()
    input_json   = Path(args.input_json).resolve()

    # Output paths (all inside run folder)
    txt_path        = outdir / (darshan_path.stem + ".txt")
    csv_path        = outdir / "darshan_summary.csv"
    feats_json_path = outdir / "darshan_features_updated.json"
    report_path     = outdir / "pct_compare_report.txt"

    # 1) .darshan -> .txt
    print(f"[darshan-parser] writing {txt_path}")
    with txt_path.open("w") as outf:
        cmd = ["darshan-parser", "--total", "--all", str(darshan_path)]
        subprocess.run(cmd, stdout=outf, stderr=subprocess.DEVNULL, check=True)

    # 2) .txt -> darshan_summary.csv (call your converter in scripts_analysis/)
    conv = SCRIPTS / "parse_darshan_txt_to_csv.py"
    if not conv.exists():
        print(f"❌ Missing converter: {conv}", file=sys.stderr)
        sys.exit(1)
    run(["python3", str(conv), "--input", str(txt_path), "--output", str(csv_path)])

    if not csv_path.exists():
        print("⚠️  CSV not found after conversion.", file=sys.stderr)
        sys.exit(1)

    # 3) CSV -> features JSON (call your generator with --root=<outdir>)
    gen = SCRIPTS / "generate_features.py"
    if not gen.exists():
        print(f"❌ Missing generator: {gen}", file=sys.stderr)
        sys.exit(1)

    print(f"[features] running generate_features.py in {outdir}")
    run(["python3", str(gen), "--root", str(outdir)], cwd=str(outdir))

    if not feats_json_path.exists():
        print("⚠️  Features JSON not found after generation.", file=sys.stderr)
        sys.exit(1)

    # 4) Compare pct_* fields between input and produced JSON with tolerance
    try:
        with input_json.open() as f:
            input_obj = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read input JSON: {input_json}\n{e}", file=sys.stderr)
        sys.exit(1)

    try:
        with feats_json_path.open() as f:
            produced = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read produced JSON: {feats_json_path}\n{e}", file=sys.stderr)
        sys.exit(1)

    # Allow list or single dict
    if isinstance(produced, list):
        if len(produced) == 0:
            print("⚠️  Produced features list is empty. Writing empty report and continuing.")
            with report_path.open("w") as r:
                r.write("Pct-field comparison report\n")
                r.write(f"Time: {datetime.now().isoformat()}\n")
                r.write(f"Input JSON: {input_json}\n")
                r.write(f"Produced JSON: {feats_json_path}\n")
                r.write("Total pct_* compared: 0\nExact matches: 0\nWithin ±0.05: 0\nDifferences: 0\n")
            print("✅ Analysis complete (empty).")
            print(f"Artifacts:\n - {txt_path}\n - {csv_path}\n - {feats_json_path}\n - {report_path}")
            return
        prod_obj = produced[0]
    elif isinstance(produced, dict):
        prod_obj = produced
    else:
        print("⚠️  Produced features JSON has unexpected structure.", file=sys.stderr)
        sys.exit(1)

    keys = sorted([k for k in input_obj.keys() if k.startswith("pct_") and k in prod_obj])

    exact = []       # exact numeric match (or string equality)
    within = []      # within tolerance but not exact
    diffs = []       # outside tolerance (these will WARN)

    for k in keys:
        iv = input_obj.get(k)
        pv = prod_obj.get(k)

        # Try numeric compare first
        num_comp_done = False
        try:
            fi = float(iv)
            fp = float(pv)
            num_comp_done = True
            if fi == fp:
                exact.append(k)
            else:
                d = abs(fi - fp)
                if d <= TOL:
                    within.append((k, iv, pv, d))
                else:
                    diffs.append((k, iv, pv, d))
        except Exception:
            pass

        # If not numeric, fall back to raw equality
        if not num_comp_done:
            if iv == pv:
                exact.append(k)
            else:
                diffs.append((k, iv, pv, "n/a"))

    # Write the report
    with report_path.open("w") as r:
        r.write("Pct-field comparison report (tolerance ±{:.3f})\n".format(TOL))
        r.write(f"Time: {datetime.now().isoformat()}\n")
        r.write(f"Input JSON: {input_json}\n")
        r.write(f"Produced JSON: {feats_json_path}\n")
        r.write(f"Total pct_* compared: {len(keys)}\n")
        r.write(f"Exact matches: {len(exact)}\n")
        r.write(f"Within ±{TOL}: {len(within)}\n")
        r.write(f"Differences (>|±{TOL}|): {len(diffs)}\n\n")

        if within:
            r.write("Within tolerance (key, input, produced, abs_diff):\n")
            for k, iv, pv, d in within:
                r.write(f"  - {k}: {iv} vs {pv} (Δ={d})\n")
            r.write("\n")

        if diffs:
            r.write("Outside tolerance (key, input, produced, abs_diff):\n")
            for k, iv, pv, d in diffs:
                r.write(f"  - {k}: {iv} vs {pv} (Δ={d})\n")

    # Only warn for differences outside tolerance
    if diffs:
        print(f"⚠️  pct_* differences detected beyond ±{TOL}; see report:", report_path)
        for k, iv, pv, d in diffs:
            print(f"  WARN  {k}: input={iv} produced={pv} Δ={d}")

    # Always print summary to terminal
    print(f"Summary: compared={len(keys)}, exact={len(exact)}, within±{TOL}={len(within)}, outside_tolerance={len(diffs)}")

    print("✅ Analysis complete.")
    print(f"Artifacts:\n - {txt_path}\n - {csv_path}\n - {feats_json_path}\n - {report_path}")

if __name__ == "__main__":
    main()
