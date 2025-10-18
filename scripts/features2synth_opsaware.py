#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features2synth_opsaware.py
Planner: OPS-FIRST, uniform-across-files, single-run.

What it does, briefly:
- Reads a Darshan-derived "features" JSON (list[dict] OR dict).
- Derives minimal file counts from file-type proportions (RO/RW/WO; shared/unique).
- Chooses total IO ops first to match pct_reads/pct_writes & per-bin *op shares*.
- Distributes ops *uniformly across files* designated for each role (RO for reads, WO for epsilon writes).
- Chooses transfer sizes *within* each bin adaptively to fit the IO-byte budget (pct_io_access * cap_total_gib),
  keeping ops proportions exact and adjusting sizes as needed (Sl from 128MiB down to 16–32MiB).
- Implements epsilon writes (8B) when pct_writes>0 but pct_byte_writes=0, with total write bytes ≤ 0.0044%.
- Computes large-op aligned fraction to hit Darshan POSIX_FILE_NOT_ALIGNED target; small ops stay unaligned.
- Emits a single CSV plan + prep/run scripts + notes (single mpiexec execution).

Outputs into /mnt/hasanfs/out_synth:
  - run_prep.sh           (creates/truncates files)
  - run_from_features.sh  (single mpiexec call)
  - run_from_features.sh.notes.txt (human summary)
  - payload/plan.csv
  - payload/data/{ro,rw,wo}/...
  - payload/meta/meta_only.dat
"""

import os
import json
import math
import random
from pathlib import Path

OUT_ROOT = Path("/mnt/hasanfs/out_synth")
PAYLOAD  = OUT_ROOT / "payload"
DATA_RO  = PAYLOAD / "data" / "ro"
DATA_RW  = PAYLOAD / "data" / "rw"
DATA_WO  = PAYLOAD / "data" / "wo"
META_DIR = PAYLOAD / "meta"
PLAN     = PAYLOAD / "plan.csv"
NOTES    = OUT_ROOT / "run_from_features.sh.notes.txt"
PREP     = OUT_ROOT / "run_prep.sh"
RUNNER   = OUT_ROOT / "run_from_features.sh"

LUSTRE_FILE_ALIGN = 1<<20  # 1 MiB
MEM_ALIGN         = 8      # compile-time alignment in harness
EPSILON_BYTE_FRAC_CAP = 0.000044  # 0.0044%

def human_bytes(n):
    g = 1<<30
    m = 1<<20
    k = 1<<10
    if n >= g: return f"{n/g:.2f} GiB"
    if n >= m: return f"{n/m:.2f} MiB"
    if n >= k: return f"{n/k:.2f} KiB"
    return f"{n} B"

def read_features(path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("features JSON is an empty list")
        return data[0]
    elif isinstance(data, dict):
        return data
    else:
        raise TypeError("features JSON must be a dict or list[dict]")

def clamp01(x): return max(0.0, min(1.0, float(x)))

def rational_counts(fracs, max_denom=6, tol=0.02):
    """
    Convert fractions (sum≈1) into small-integer counts with denominator ≤ max_denom.
    Returns list of ints (could include zeros).
    """
    fracs = [max(0.0, f) for f in fracs]
    s = sum(fracs)
    if s <= 0: return [0]*len(fracs)
    fracs = [f/s for f in fracs]
    best = None
    for denom in range(1, max_denom+1):
        counts = [round(denom*f) for f in fracs]
        # ensure nonzero if fraction > 0:
        for i, f in enumerate(fracs):
            if f>0 and counts[i]==0: counts[i]=1
        if sum(counts)==0: continue
        approx = [c/sum(counts) for c in counts]
        err = sum(abs(a-b) for a,b in zip(approx, fracs))
        if best is None or err < best[0]:
            best = (err, counts)
    return best[1] if best else [0]*len(fracs)

def choose_min_file_counts(feats):
    # group2: RO/RW/WO by file count proportions
    ro_f = clamp01(feats.get("pct_read_only_files", 0.0))
    rw_f = clamp01(feats.get("pct_read_write_files", 0.0))
    wo_f = clamp01(feats.get("pct_write_only_files", 0.0))
    if ro_f + rw_f + wo_f == 0:
        # Fallback: at least one RO
        return dict(ro=1, rw=0, wo=0)
    counts = rational_counts([ro_f, rw_f, wo_f], max_denom=6)
    ro_c, rw_c, wo_c = counts
    # We need at least one file for any non-zero fraction:
    if ro_f>0 and ro_c==0: ro_c=1
    if rw_f>0 and rw_c==0: rw_c=1
    if wo_f>0 and wo_c==0: wo_c=1
    # Your case: 0.67/0/0.33 -> [2,0,1]
    return dict(ro=ro_c, rw=rw_c, wo=wo_c)

def plan_from_features(features_path):
    feats = read_features(features_path)
    # ranks and cap
    nprocs = int(feats.get("nprocs", 1))
    cap_total_gib = float(feats.get("cap_total_gib", 512))
    cap_io_bytes = int(cap_total_gib * (1<<30) * clamp01(feats.get("pct_io_access", 0.18)))

    # op shares
    p_reads  = clamp01(feats.get("pct_reads", 0.8))
    p_writes = clamp01(feats.get("pct_writes", 0.2))
    # sequential / consecutive
    consec_r = clamp01(feats.get("pct_consec_reads", 0.0))
    seq_r    = clamp01(feats.get("pct_seq_reads", 0.0))
    if seq_r < consec_r: seq_r = consec_r  # clamp to subset
    consec_w = clamp01(feats.get("pct_consec_writes", 0.0))
    seq_w    = clamp01(feats.get("pct_seq_writes", 0.0))
    if seq_w < consec_w: seq_w = consec_w

    # alignment targets
    p_file_ua = clamp01(feats.get("pct_file_not_aligned", 0.6))
    p_mem_ua  = clamp01(feats.get("pct_mem_not_aligned", 0.4))

    # read/write bin shares by **ops**
    rS = clamp01(feats.get("pct_read_0_100K", 0.5))
    rM = clamp01(feats.get("pct_read_100K_10M", 0.0))
    rL = clamp01(feats.get("pct_read_10M_1G_PLUS", 0.5))
    s = rS + rM + rL
    rS, rM, rL = (rS/s if s>0 else 0.0, rM/s if s>0 else 0.0, rL/s if s>0 else 0.0)

    wS = clamp01(feats.get("pct_write_0_100K", 1.0))
    wM = clamp01(feats.get("pct_write_100K_10M", 0.0))
    wL = clamp01(feats.get("pct_write_10M_1G_PLUS", 0.0))
    s = wS + wM + wL
    wS, wM, wL = (wS/s if s>0 else 0.0, wM/s if s>0 else 0.0, wL/s if s>0 else 0.0)

    # file populations
    fcounts = choose_min_file_counts(feats)  # dict(ro=2, rw=0, wo=1) for your case
    n_ro, n_rw, n_wo = fcounts["ro"], fcounts["rw"], fcounts["wo"]
    if n_ro==0 and p_reads>0:
        # ensure at least one RO or RW exists to place reads
        if n_rw>0: pass
        else: n_ro=1

    # === OPS-FIRST ===
    # Start with a guess for sizes
    Ss = 100*1024          # <=100 KiB
    Sm = 1<<20             # 1 MiB mid-range
    Sl = 128*(1<<20)       # 128 MiB (Lustre-friendly)
    Sw = 8                 # epsilon write

    # Want Br ≈ cap_io_bytes (all bytes are reads in your case)
    # Assume reads only for budget (writes epsilon is negligible).
    denom = rS*Ss + rM*Sm + rL*Sl
    if denom == 0:
        # Degenerate: no reads? create tiny epsilon to avoid division by zero
        Nr = 0
    else:
        Nr = int(cap_io_bytes / denom)

    # Now derive read/write ops
    # Keep op shares: Nr : Nw = p_reads : p_writes
    if p_reads + p_writes == 0:
        Nr, Nw = 0, 0
    else:
        Nio = max(1, int(Nr / p_reads))  # rescale to satisfy shares
        Nr  = int(round(Nio * p_reads))
        Nw  = max(0, Nio - Nr)

    # Split reads by bin **ops**
    Ns = int(round(Nr * rS))
    Nm = int(round(Nr * rM))
    Nl = max(0, Nr - Ns - Nm)

    # Distribute uniformly across files
    ro_paths = [str(DATA_RO / f"ro_shared_{i}.dat") for i in range(n_ro)]
    wo_paths = [str(DATA_WO / f"wo_shared_{i}.dat") for i in range(n_wo)]

    # Epsilon writes: ensure bytes fraction ≤ 0.0044%
    target_w_bytes_cap = int(cap_io_bytes * EPSILON_BYTE_FRAC_CAP)
    if p_writes > 0 and Nw == 0:
        # Ensure at least some Nw; choose from cap on bytes
        Nw = max(1, min(target_w_bytes_cap // max(1, Sw), int((p_writes / max(1e-6,p_reads)) * Nr)))
    # If p_writes==0 → Nw=0
    if p_writes == 0:
        Nw = 0

    # Compute read bytes & adjust sizes if needed
    Br = Ns*Ss + Nm*Sm + Nl*Sl
    if Br == 0 and Nr>0:
        # No sizes? fallback
        Ss, Sl = 64*1024, 32*(1<<20)
        Br = Ns*Ss + Nm*Sm + Nl*Sl

    # Coarse adjust sizes within bins to fit budget
    target = cap_io_bytes
    if Br > 0:
        scale = target / Br
        # Prefer to shrink Sl first if overshooting; grow Ss first if undershooting.
        if scale < 0.9 and Sl >= 16*(1<<20):
            # shrink Sl down to keep within L bin
            Sl = max(16*(1<<20), int(Sl*scale))
            # ensure still ≥10MiB
            Sl = max(Sl, 10*(1<<20))
        elif scale > 1.1:
            # try bump Ss up within 0–100K
            Ss = min(100*1024, int(Ss*scale))
        Br = Ns*Ss + Nm*Sm + Nl*Sl

    # === Alignment: compute required large-op aligned fraction ===
    # small, medium ops contribute unaligned by definition (size<1MiB or not enforced)
    Nr_tot = max(1, Nr)
    frac_small = Ns / Nr_tot
    frac_med   = Nm / Nr_tot
    frac_large = Nl / Nr_tot
    if frac_large > 0:
        # f_large_aligned is the fraction of large ops that should be aligned
        f_large_aligned = 1.0 - max(0.0, min(1.0, (p_file_ua - frac_small - frac_med) / max(1e-9, frac_large)))
        f_large_aligned = max(0.0, min(1.0, f_large_aligned))
    else:
        f_large_aligned = 0.0

    # === Prepare dirs ===
    os.makedirs(DATA_RO, exist_ok=True)
    os.makedirs(DATA_RW, exist_ok=True)
    os.makedirs(DATA_WO, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)

    # === Write plan ===
    random.seed(1)
    lines = []
    header = ("type,path,total_bytes,xfer,p_write,p_rand,p_seq_r,p_consec_r,p_seq_w,p_consec_w,"
              "p_ua_file,p_ua_mem,rw_switch,meta_open,meta_stat,meta_seek,meta_sync,seed,flags,"
              "p_rand_fwd_r,p_rand_fwd_w,p_consec_internal")
    lines.append(header)

    # Split Ns,Nm,Nl across ro files uniformly by ops
    def split_uniform(N, k):
        base, rem = (N // max(1,k)), (N % max(1,k))
        return [base + (1 if i < rem else 0) for i in range(k)] if k>0 else []

    Ns_per = split_uniform(Ns, n_ro)
    Nm_per = split_uniform(Nm, n_ro)
    Nl_per = split_uniform(Nl, n_ro)

    # build data rows
    p_consec_internal = 0.0  # harness uses only to cancel first-op artifact; keep 0
    p_rand_fwd_r = 0.5       # random-forward split internally
    p_rand_fwd_w = 0.0

    def add_data_row(path, Nops, xfer, is_write, force_large_align=False):
        if Nops <= 0: return
        total_bytes = Nops * xfer
        # For reads: p_write=0; For writes: p_write=1
        p_write = 1.0 if is_write else 0.0
        p_rand  = 1.0 - (seq_r if not is_write else seq_w)
        p_con_r = consec_r if not is_write else consec_w
        p_seq_r = seq_r     if not is_write else seq_w
        p_con_w = consec_w
        p_seq_w = seq_w
        # file unaligned
        if is_write:
            p_ua_file_local = p_file_ua
        else:
            if xfer >= LUSTRE_FILE_ALIGN and force_large_align:
                # mark unaligned fraction for large ops
                p_ua_file_local = 1.0 - f_large_aligned
            else:
                p_ua_file_local = 1.0  # small/medium phases unaligned
        seed = random.randint(1, 2**31-1)
        row = [
            "data", path, str(total_bytes), str(xfer),
            f"{p_write:.6f}", f"{p_rand:.6f}",
            f"{p_seq_r:.6f}", f"{p_con_r:.6f}",
            f"{p_seq_w:.6f}", f"{p_con_w:.6f}",
            f"{p_ua_file_local:.6f}", f"{p_mem_ua:.6f}",
            "0.0", "0","0","0","0",
            str(seed), "",
            f"{p_rand_fwd_r:.6f}", f"{p_rand_fwd_w:.6f}",
            f"{p_consec_internal:.6f}"
        ]
        lines.append(",".join(map(str,row)))

    # Reads on RO files
    for i, ro in enumerate(ro_paths):
        if Ns_per:
            add_data_row(ro, Ns_per[i], Ss, is_write=False, force_large_align=False)
        if Nl_per:
            add_data_row(ro, Nl_per[i], Sl, is_write=False, force_large_align=True)
        if Nm_per:
            add_data_row(ro, Nm_per[i], Sm, is_write=False, force_large_align=False)

    # Writes (epsilon) on WO files (uniform)
    if Nw>0 and n_wo>0:
        Nw_per = split_uniform(Nw, n_wo)
        for i, wo in enumerate(wo_paths):
            add_data_row(wo, Nw_per[i], Sw, is_write=True, force_large_align=False)

    # === Meta-only phase (derived from IO ops to satisfy pct_io_access) ===
    p_io   = clamp01(feats.get("pct_io_access", 0.18))
    p_m_o  = clamp01(feats.get("pct_meta_open_access", 0.0))
    p_m_s  = clamp01(feats.get("pct_meta_stat_access", 0.0))
    p_m_k  = clamp01(feats.get("pct_meta_seek_access", 0.0))
    p_m_sy = clamp01(feats.get("pct_meta_sync_access", 0.0))

    io_ops = Nr + Nw  # total planned IO ops

    # Compute required meta ops so that io_ops / (io_ops + meta_ops) == p_io
    if p_io > 0.0:
        meta_total = int(round(io_ops * (1.0 - p_io) / p_io))
    else:
        meta_total = 0

    # Normalize meta proportions and allocate integers with remainder distribution
    sum_meta_p = p_m_o + p_m_s + p_m_k + p_m_sy
    if meta_total > 0 and sum_meta_p > 0.0:
        w_o = p_m_o / sum_meta_p
        w_s = p_m_s / sum_meta_p
        w_k = p_m_k / sum_meta_p
        w_y = p_m_sy / sum_meta_p

        meta_open = int(math.floor(meta_total * w_o))
        meta_stat = int(math.floor(meta_total * w_s))
        meta_seek = int(math.floor(meta_total * w_k))
        meta_sync = int(math.floor(meta_total * w_y))

        rem = meta_total - (meta_open + meta_stat + meta_seek + meta_sync)
        fracs = [
            (meta_total * w_o - meta_open, 'open'),
            (meta_total * w_s - meta_stat, 'stat'),
            (meta_total * w_k - meta_seek, 'seek'),
            (meta_total * w_y - meta_sync, 'sync'),
        ]
        fracs.sort(reverse=True)
        for i in range(rem):
            t = fracs[i % 4][1]
            if   t == 'open': meta_open += 1
            elif t == 'stat': meta_stat += 1
            elif t == 'seek': meta_seek += 1
            elif t == 'sync': meta_sync += 1
    else:
        meta_open = meta_stat = meta_seek = meta_sync = 0

    seed = 777
    meta_row = ["meta", str(META_DIR / "meta_only.dat"), "0","1","0","0","0","0","0","0",
                f"{p_file_ua:.6f}", f"{p_mem_ua:.6f}",
                "0.0",
                str(meta_open), str(meta_stat), str(meta_seek), str(meta_sync),
                str(seed), "meta_only",
                "0.5","0.0","0.0"]
    lines.append(",".join(meta_row))

    # === Write plan.csv ===
    os.makedirs(PAYLOAD, exist_ok=True)
    with open(PLAN, "w") as f:
        f.write("\n".join(lines) + "\n")

    # === Prep script: truncate data files ===
    # We set RO file sizes to ~ half of read bytes each + 1GiB headroom.
    read_bytes_total = Ns*Ss + Nm*Sm + Nl*Sl
    per_ro_bytes = (read_bytes_total // max(1,n_ro)) + (1<<30)

    with open(PREP, "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        f.write(f"mkdir -p {DATA_RO} {DATA_RW} {DATA_WO} {META_DIR}\n")
        for i, ro in enumerate(ro_paths):
            f.write(f"truncate -s {per_ro_bytes} {ro}\n")
        for i, wo in enumerate(wo_paths):
            # set tiny size; writes will extend as needed
            f.write(f"truncate -s {max(4096, Sw)} {wo}\n")
        f.write(f"truncate -s 4096 {META_DIR/'meta_only.dat'}\n")

    os.chmod(PREP, 0o755)

    # === Runner script ===
    with open(RUNNER, "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        f.write(f"bash {PREP}\n")
        f.write("mpiexec -n {n} -genv LD_PRELOAD /mnt/hasanfs/darshan-3.4.7/darshan-runtime/install/lib/libdarshan.so "
                "/mnt/hasanfs/bin/mpi_synthio --plan {plan} --io-api {iapi} --meta-api {mapi} --collective {coll}\n"
                .format(n=nprocs,
                        plan=str(PLAN),
                        iapi=str(feats.get('io_api','posix')),
                        mapi=str(feats.get('meta_api','posix')),
                        coll=str(feats.get('mpi_collective_mode','none'))))
    os.chmod(RUNNER, 0o755)

    # === Notes ===
    with open(NOTES, "w") as f:
        f.write("=== Feature → Execution Mapping (OPS-first, uniform across files, single run) ===\n")
        f.write(f"  cap_total_gib={cap_total_gib:.2f}  pct_io_access={feats.get('pct_io_access',0.18):.2f}  → IO bytes={human_bytes(cap_io_bytes)}\n")
        f.write(f"  Read ops={Nr}  |  Write ops={Nw}\n")
        f.write(f"  Read bin ops: S={Ns}, M={Nm}, L={Nl}\n")
        f.write(f"  Chosen xfer sizes: S={human_bytes(Ss)}, M={human_bytes(Sm)}, L={human_bytes(Sl)}, W={Sw}B\n")
        f.write("\n=== Seq/Consec ===\n")
        f.write(f"  Reads: consec={consec_r:.3f}, seq={seq_r:.3f} (consec⊂seq rule enforced)\n")
        f.write(f"  Writes: consec={consec_w:.3f}, seq={seq_w:.3f}\n")
        f.write("\n=== Alignment ===\n")
        f.write(f"  Target file_unaligned={p_file_ua:.2f}, mem_unaligned={p_mem_ua:.2f}\n")
        f.write(f"  Large-op aligned fraction≈{f_large_aligned:.3f} (small/medium always unaligned)\n")
        f.write("  SITE ALIGNMENT: FILE=1MiB, MEM=8\n")
        f.write("\n=== Files (uniform distribution) ===\n")
        f.write(f"  RO={n_ro}  RW={n_rw}  WO={n_wo}\n")
        for ro in ro_paths: f.write(f"  RO path: {ro}\n")
        for wo in wo_paths: f.write(f"  WO path: {wo}\n")
        f.write("\n=== Meta Ops (derived for pct_io_access) ===\n")
        f.write(f"  open={meta_open}, stat={meta_stat}, seek={meta_seek}, sync={meta_sync}  (total={meta_open+meta_stat+meta_seek+meta_sync}, io_ops={Nr+Nw})\n")
        f.write("\n=== Command ===\n")
        f.write(f"  mpiexec -n {nprocs} -genv LD_PRELOAD /mnt/hasanfs/darshan-3.4.7/darshan-runtime/install/lib/libdarshan.so "
                f"/mnt/hasanfs/bin/mpi_synthio --plan {PLAN} --io-api {feats.get('io_api','posix')} --meta-api {feats.get('meta_api','posix')} --collective {feats.get('mpi_collective_mode','none')}\n")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="path to features JSON")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PAYLOAD.mkdir(parents=True, exist_ok=True)
    plan_from_features(args.features)
    print(f"Wrote {PREP}, {RUNNER}, and {NOTES} to {OUT_ROOT}")

if __name__ == "__main__":
    main()
