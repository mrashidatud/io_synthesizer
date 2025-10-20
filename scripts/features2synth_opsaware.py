#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features2synth_opsaware.py
Planner: OPS-FIRST, fixed S/M sub-sizes, L=16MiB, uniform-across-files,
         deterministic ordering (L→M→S and Consec→SeqRemainder→Random),
         with a post-pass in-bin size rebalancer (largest→smallest) to hit byte%.

USAGE:
  python features2synth_opsaware.py --features /path/to/features.json

OUTPUTS (under /mnt/hasanfs/out_synth):
  - run_prep.sh                  # creates directories & truncates files to planned sizes
  - run_from_features.sh         # mpiexec wrapper with Darshan preload
  - run_from_features.sh.notes.txt
  - payload/plan.csv
  - payload/data/{ro,rw,wo}/...
  - payload/meta/meta_only.dat

Key behaviors:
  * IO byte target = cap_total_gib (meta ops independent).
  * OPS-first: compute Nio from avg per-op sizes (S/M fixed sets, L=16MiB).
  * Fix S and M sizes into sub-categories with equal op counts per file.
  * L is fixed to 16MiB; number of L ops comes from bin shares.
  * Deterministic per-file ordering: for each bin: Consec, then Seq remainder (gap=xfer),
    then Random (descending 32MiB chunks); Random phases instruct harness to pre-seek EOF.
  * Post-pass rebalancer tweaks sizes within the same bin (largest→smallest)
    to match byte% targets (read vs write; optionally bytes per file role),
    without touching op counts nor L size.
  * Align model: large ops can be aligned to 1MiB to hit pct_file_not_aligned; S/M unaligned.
  * Memory misalignment uses pct_mem_not_aligned.
"""

import os
import json
import math
import random
from pathlib import Path
from collections import defaultdict, Counter

# ==== Paths ====
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

# ==== Constants ====
LUSTRE_FILE_ALIGN = 1<<20   # 1 MiB
MEM_ALIGN         = 8       # compile-time alignment in harness
CHUNK_RANDOM      = 32*(1<<20)  # 32 MiB chunking for random offset buckets

# Small/Medium fixed sub-sizes (bytes)
S_SUBS = [100, 1024, 4096, 65536]      # 100 B, 1 KiB, 4 KiB, 64 KiB
M_SUBS = [256*1024, 1<<20, 4*(1<<20)]  # 256 KiB, 1 MiB, 4 MiB
L_SIZE = 16*(1<<20)                    # fixed 16 MiB

TOL_PCT = 0.05          # 5% tolerance on byte proportions
CUSHION = 1<<30         # 1 GiB file-size cushion

def human_bytes(n):
    g = 1<<30; m=1<<20; k=1<<10
    if n >= g: return f"{n/g:.2f} GiB"
    if n >= m: return f"{n/m:.2f} MiB"
    if n >= k: return f"{n/k:.2f} KiB"
    return f"{n} B"

def read_features(path):
    with open(path,"r") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data: raise ValueError("features JSON is empty list")
        return data[0]
    if isinstance(data, dict):
        return data
    raise TypeError("features JSON must be dict or list[dict]")

def clamp01(x): return max(0.0, min(1.0, float(x)))

def rational_counts(fracs, max_denom=6):
    fracs = [max(0.0, f) for f in fracs]
    s = sum(fracs)
    if s <= 0: return [0]*len(fracs)
    fracs = [f/s for f in fracs]
    best = None
    for denom in range(1, max_denom+1):
        counts = [round(denom*f) for f in fracs]
        for i,f in enumerate(fracs):
            if f>0 and counts[i]==0: counts[i]=1
        if sum(counts)==0: continue
        approx = [c/sum(counts) for c in counts]
        err = sum(abs(a-b) for a,b in zip(approx, fracs))
        if best is None or err < best[0]:
            best = (err, counts)
    return best[1]

def choose_min_file_counts(feats):
    ro_f = clamp01(feats.get("pct_read_only_files", 0.0))
    rw_f = clamp01(feats.get("pct_read_write_files", 0.0))
    wo_f = clamp01(feats.get("pct_write_only_files", 0.0))
    if ro_f + rw_f + wo_f == 0:
        return dict(ro=1, rw=0, wo=0)
    ro_c, rw_c, wo_c = rational_counts([ro_f,rw_f,wo_f], max_denom=6)
    if ro_f>0 and ro_c==0: ro_c=1
    if rw_f>0 and rw_c==0: rw_c=1
    if wo_f>0 and wo_c==0: wo_c=1
    return dict(ro=ro_c, rw=rw_c, wo=wo_c)

def split_uniform(n, k):
    if k<=0: return []
    base = n//k; rem = n%k
    return [base + (1 if i<rem else 0) for i in range(k)]

def summarize_bytes(layout):
    total=0
    by_role=Counter()
    by_intent=Counter()
    for row in layout:
        role = row["role"]    # "ro"/"rw"/"wo"
        intent = row["intent"]# "read"/"write"
        b = row["ops"] * row["xfer"]
        total += b
        by_role[role]+=b
        by_intent[intent]+=b
    return total, by_role, by_intent

def rebalancer(layout, target_intent_bytes, target_role_bytes):
    """
    layout: list of dict rows for a single intent ("read" or "write").
      each row: {"role","file","bin","size_class","xfer","ops"}; bin in {"S","M","L"}
    target_intent_bytes: desired bytes for this intent (abs)
    target_role_bytes: dict role-> desired bytes for that role (abs) for this intent (if any)
    Strategy:
      - Never change ops or bin.
      - L stays 16MiB, immutable.
      - Adjust M then S, per (role,file,bin,size_class) — largest→smallest.
      - Round-robin across files to keep uniformity.
    """
    # Compute current
    total = sum(r["ops"]*r["xfer"] for r in layout)
    # Read/write target total for intent; if None, skip
    if target_intent_bytes is not None:
        delta = target_intent_bytes - total
    else:
        delta = 0

    # helper to get adjustable rows (bin in ["M","S"]) grouped by role then bin then size desc
    def rows_by_priority(sign):
        # sign>0 means we need to increase bytes; sign<0 reduce bytes
        # for increase: walk sizes small→large; for decrease: large→small
        pr = []
        for b in ("M","S"):  # adjust M first, then S
            # collect unique sizes in this bin
            sizes = sorted({r["xfer"] for r in layout if r["bin"]==b})
            sizes_sorted = sizes if sign>0 else list(reversed(sizes))
            for sz in sizes_sorted:
                # collect rows with this exact size
                rows = [r for r in layout if r["bin"]==b and r["xfer"]==sz and r["ops"]>0]
                # round-robin order by file to keep uniformity
                rows.sort(key=lambda r:(r["role"], r["file"]))
                pr.extend(rows)
        return pr

    # Size ladders for M and S
    ladder = {
        "M": M_SUBS,
        "S": S_SUBS,
    }

    # Per-role target handling (if provided)
    def role_bytes():
        rb = Counter()
        for r in layout:
            rb[r["role"]] += r["ops"]*r["xfer"]
        return rb

    # Try to fix role-level deviations first (keeps global intent close too)
    if target_role_bytes:
        for round_idx in range(2):  # a couple passes is enough
            rb_now = role_bytes()
            for role, tgt in target_role_bytes.items():
                cur = rb_now.get(role, 0)
                d = tgt - cur
                if abs(d) <= max(1, int(TOL_PCT*max(tgt,1))):  # within abs-tol (coarse guard)
                    continue
                # sign determines direction; adjust rows in that role only
                sign = 1 if d>0 else -1
                for r in rows_by_priority(sign):
                    if r["role"] != role: continue
                    if r["bin"] not in ("M","S"): continue
                    sizes = ladder[r["bin"]]
                    idx = sizes.index(r["xfer"])
                    nxt = None
                    if sign>0 and idx < len(sizes)-1:
                        nxt = sizes[idx+1]  # increase
                    if sign<0 and idx > 0:
                        nxt = sizes[idx-1]  # decrease
                    if nxt is None: continue
                    step = (nxt - r["xfer"]) * r["ops"]
                    # apply cautiously; might overshoot, but later passes dampen
                    r["xfer"] = nxt
                    rb_now[role] += step
                    cur += step
                    if (d>0 and rb_now[role] >= tgt) or (d<0 and rb_now[role] <= tgt):
                        break

    # Fix overall intent delta (if any) by adjusting M then S across all roles/files
    if delta != 0:
        sign = 1 if delta>0 else -1
        remaining = abs(delta)
        for r in rows_by_priority(sign):
            if r["bin"] not in ("M","S"): continue
            sizes = ladder[r["bin"]]
            idx = sizes.index(r["xfer"])
            nxt = None
            if sign>0 and idx < len(sizes)-1:
                nxt = sizes[idx+1]
            if sign<0 and idx > 0:
                nxt = sizes[idx-1]
            if nxt is None: continue
            step_per_op = (nxt - r["xfer"])
            step_total  = step_per_op * r["ops"]
            r["xfer"] = nxt
            if abs(step_total) >= remaining:
                break
            remaining -= abs(step_total)
        # If still not satisfied within tolerance, we leave as-is; planner notes will flag residual.

def plan_from_features(features_path):
    feats = read_features(features_path)

    # ---- Inputs & basic shares ----
    nprocs = int(feats.get("nprocs", 1))
    cap_total_gib = float(feats.get("cap_total_gib", 512))
    io_bytes_target = int(cap_total_gib * (1<<30))  # maximize IO budget

    p_reads  = clamp01(feats.get("pct_reads", 0.8))
    p_writes = clamp01(feats.get("pct_writes", 0.2))
    if p_reads + p_writes == 0:
        p_reads, p_writes = 1.0, 0.0

    consec_r = clamp01(feats.get("pct_consec_reads", 0.0))
    seq_r    = clamp01(feats.get("pct_seq_reads", 0.0))
    if seq_r < consec_r: seq_r = consec_r
    consec_w = clamp01(feats.get("pct_consec_writes", 0.0))
    seq_w    = clamp01(feats.get("pct_seq_writes", 0.0))
    if seq_w < consec_w: seq_w = consec_w

    p_file_ua = clamp01(feats.get("pct_file_not_aligned", 0.6))
    p_mem_ua  = clamp01(feats.get("pct_mem_not_aligned", 0.4))

    # by-bin OP shares
    rS = clamp01(feats.get("pct_read_0_100K", 0.5))
    rM = clamp01(feats.get("pct_read_100K_10M", 0.0))
    rL = clamp01(feats.get("pct_read_10M_1G_PLUS", 0.5))
    rs = rS+rM+rL; rS,rM,rL = (rS/rs, rM/rs, rL/rs) if rs>0 else (0,0,0)

    wS = clamp01(feats.get("pct_write_0_100K", 1.0))
    wM = clamp01(feats.get("pct_write_100K_10M", 0.0))
    wL = clamp01(feats.get("pct_write_10M_1G_PLUS", 0.0))
    ws = wS+wM+wL; wS,wM,wL = (wS/ws, wM/ws, wL/ws) if ws>0 else (0,0,0)

    # file roles
    fcounts = choose_min_file_counts(feats)
    n_ro, n_rw, n_wo = fcounts["ro"], fcounts["rw"], fcounts["wo"]
    if n_ro==0 and p_reads>0 and n_rw==0:
        n_ro=1

    ro_paths = [str(DATA_RO / f"ro_shared_{i}.dat") for i in range(n_ro)]
    rw_paths = [str(DATA_RW / f"rw_shared_{i}.dat") for i in range(n_rw)]
    wo_paths = [str(DATA_WO / f"wo_shared_{i}.dat") for i in range(n_wo)]

    # average size per op for read/write (for OPS-first)
    avgS = sum(S_SUBS)/len(S_SUBS)
    avgM = sum(M_SUBS)/len(M_SUBS)
    avgL = L_SIZE

    avg_read = rS*avgS + rM*avgM + rL*avgL
    avg_write= wS*avgS + wM*avgM + wL*avgL
    denom = p_reads*avg_read + p_writes*avg_write
    if denom == 0:
        Nio = 0
        Nr = Nw = 0
    else:
        Nio = max(1, io_bytes_target // int(denom))
        Nr  = int(round(Nio * p_reads))
        Nw  = max(0, Nio - Nr)

    # per-intent per-bin ops (integerize with residual to L bin)
    R_S = int(round(Nr * rS)); R_M = int(round(Nr * rM)); R_L = max(0, Nr - R_S - R_M)
    W_S = int(round(Nw * wS)); W_M = int(round(Nw * wM)); W_L = max(0, Nw - W_S - W_M)

    # split S/M/L ops into fixed sub-size categories (equal op counts), then across files
    def per_file_subsizes(Nbin, subs, files):
        # equal across subs, then across files
        per_sub = split_uniform(Nbin, len(subs))
        out = []  # list of (file, sub_size, ops)
        for s_idx, ops in enumerate(per_sub):
            per_file = split_uniform(ops, len(files))
            for f_idx, k in enumerate(per_file):
                if k>0:
                    out.append((files[f_idx], subs[s_idx], k))
        return out

    # Build a "layout plan" (before rebalancing) at op granularity:
    # rows are for a single intent; each row is a (role,file,bin,size_class,xfer,ops)
    read_rows = []
    write_rows = []

    # READS go to RO first; if none, RW
    read_files = ro_paths if ro_paths else rw_paths
    if R_S>0: read_rows += [{"role":"ro","file":f,"bin":"S","size_class":sz,"xfer":sz,"ops":k,"intent":"read"}
                            for (f,sz,k) in per_file_subsizes(R_S, S_SUBS, read_files)]
    if R_M>0: read_rows += [{"role":"ro","file":f,"bin":"M","size_class":sz,"xfer":sz,"ops":k,"intent":"read"}
                            for (f,sz,k) in per_file_subsizes(R_M, M_SUBS, read_files)]
    if R_L>0:
        per_file = per_file_subsizes(R_L, [L_SIZE], read_files)
        read_rows += [{"role":"ro","file":f,"bin":"L","size_class":L_SIZE,"xfer":L_SIZE,"ops":k,"intent":"read"}
                      for (f,sz,k) in per_file]

    # WRITES: prefer WO; else RW
    write_files = wo_paths if wo_paths else rw_paths
    if W_S>0: write_rows += [{"role":("wo" if wo_paths else "rw"),"file":f,"bin":"S","size_class":sz,"xfer":sz,"ops":k,"intent":"write"}
                             for (f,sz,k) in per_file_subsizes(W_S, S_SUBS, write_files)]
    if W_M>0: write_rows += [{"role":("wo" if wo_paths else "rw"),"file":f,"bin":"M","size_class":sz,"xfer":sz,"ops":k,"intent":"write"}
                             for (f,sz,k) in per_file_subsizes(W_M, M_SUBS, write_files)]
    if W_L>0:
        per_file = per_file_subsizes(W_L, [L_SIZE], write_files)
        write_rows += [{"role":("wo" if wo_paths else "rw"),"file":f,"bin":"L","size_class":L_SIZE,"xfer":L_SIZE,"ops":k,"intent":"write"}
                       for (f,sz,k) in per_file]

    # ---- Byte share targets ----
    target_read_bytes  = int(io_bytes_target * clamp01(feats.get("pct_byte_reads", 1.0)))
    target_write_bytes = io_bytes_target - target_read_bytes

    # Role-byte targets (optional)
    # For simplicity we only use read-only vs write-only byte targets if specified & non-conflicting.
    target_role_bytes_read  = {}
    target_role_bytes_write = {}
    # Example: if pct_bytes_read_only_files == 1.0 in your case, all read bytes go to RO (already true).

    # ---- Rebalance in-bin sizes (does not change op counts; L immutable) ----
    rebalancer(read_rows,  target_read_bytes,  target_role_bytes_read)
    rebalancer(write_rows, target_write_bytes, target_role_bytes_write)

    # ---- Phase emission (deterministic ordering) ----
    # We emit phases per (file, bin, size_class, intent) in order: L→M→S and Consec→Seq→Random.
    # Each phase is a single xfer with total_bytes = ops*xfer; and exact p_*:
    #   Consec: p_consec=1, p_seq=1, p_rand=0
    #   SeqR  : p_consec=0, p_seq=1, p_rand=0
    #   Random: p_consec=0, p_seq=0, p_rand=1, pre_seek_eof=1
    lines=[]
    header=("type,path,total_bytes,xfer,p_write,p_rand,p_seq_r,p_consec_r,p_seq_w,p_consec_w,"
            "p_ua_file,p_ua_mem,rw_switch,meta_open,meta_stat,meta_seek,meta_sync,seed,flags,"
            "p_rand_fwd_r,p_rand_fwd_w,p_consec_internal,pre_seek_eof")
    lines.append(header)

    def emit_data_phase(path, intent, xfer, ops, phase_kind, p_ua_file, p_ua_mem, pre_seek_eof):
        if ops<=0: return
        total_bytes = ops * xfer
        is_write = (intent=="write")
        if phase_kind=="consec":
            p_con_r = 1.0; p_seq_r = 1.0; p_rand=0.0
        elif phase_kind=="seq":
            p_con_r = 0.0; p_seq_r = 1.0; p_rand=0.0
        else: # random
            p_con_r = 0.0; p_seq_r = 0.0; p_rand=1.0
        seed = random.randint(1,2**31-1)
        row = [
            "data", path, str(total_bytes), str(xfer),
            f"{1.0 if is_write else 0.0:.6f}", f"{p_rand:.6f}",
            f"{p_seq_r:.6f}", f"{p_con_r:.6f}",
            f"{1.0 if is_write else 0.0:.6f}", f"{0.0:.6f}", # p_seq_w, p_consec_w (unused here)
            f"{p_ua_file:.6f}", f"{p_ua_mem:.6f}",
            "0.0","0","0","0","0",
            str(seed), phase_kind,
            "0.0","0.0","0.0",
            "1" if pre_seek_eof else "0"
        ]
        lines.append(",".join(row))

    # compute aligned fraction for LARGE only to hit pct_file_not_aligned
    # Assume S/M unaligned; solve for large aligned fraction f_A:
    # p_file_ua = frac_S + frac_M + frac_L*(1 - f_A)
    def large_aligned_fraction(rows):
        intent_total_ops = sum(r["ops"] for r in rows)
        if intent_total_ops==0: return 0.0
        Ns = sum(r["ops"] for r in rows if r["bin"]=="S")
        Nm = sum(r["ops"] for r in rows if r["bin"]=="M")
        Nl = sum(r["ops"] for r in rows if r["bin"]=="L")
        fracS,fracM,fracL = Ns/intent_total_ops, Nm/intent_total_ops, Nl/intent_total_ops
        if fracL <= 1e-9:
            return 0.0
        fA = 1.0 - max(0.0, min(1.0, (p_file_ua - fracS - fracM)/max(fracL,1e-9)))
        return max(0.0, min(1.0, fA))

    fA_read  = large_aligned_fraction(read_rows)
    fA_write = large_aligned_fraction(write_rows)

    def emit_intent_rows(rows, is_read):
        # Sort by bin order L, M, S
        rows_sorted = sorted(rows, key=lambda r: {"L":0,"M":1,"S":2}[r["bin"]])
        # Group per (file,bin,size)
        grp = defaultdict(list)
        for r in rows_sorted:
            grp[(r["file"], r["bin"], r["xfer"])].append(r)

        # per group emit 3 phases: consec, seq, random
        for (path, bin_name, xfer), parts in grp.items():
            ops = sum(p["ops"] for p in parts)
            if ops<=0: continue
            # p_ua_file: S/M unaligned=1.0; L uses aligned frac
            if bin_name in ("S","M"):
                pua = 1.0
            else:
                # mark phase-level p_ua_file as fraction unaligned. Harness will deterministically align the first chunk.
                fA = fA_read if is_read else fA_write
                pua = 1.0 - (fA if ops>0 else 0.0)

            # split ops into consec/seq/random using global targets
            if is_read:
                pc, ps = consec_r, seq_r
            else:
                pc, ps = consec_w, seq_w
            n_con = int(round(ops * pc))
            n_seq = int(round(ops * max(0.0, ps - pc)))
            n_rnd = max(0, ops - n_con - n_seq)

            # Emit phases (Consec→Seq→Random). Random with pre_seek_eof fence.
            emit_data_phase(path, "read" if is_read else "write", xfer, n_con, "consec", pua, p_mem_ua, pre_seek_eof=False)
            emit_data_phase(path, "read" if is_read else "write", xfer, n_seq, "seq",    pua, p_mem_ua, pre_seek_eof=False)
            emit_data_phase(path, "read" if is_read else "write", xfer, n_rnd, "random", pua, p_mem_ua, pre_seek_eof=True)

    emit_intent_rows(read_rows,  is_read=True)
    emit_intent_rows(write_rows, is_read=False)

    # ---- META phase ----
    # Scale meta counts roughly proportional to IO ops (no byte coupling).
    io_ops = Nr + Nw
    m_open = int(round(clamp01(feats.get("pct_meta_open_access",0.0)) * io_ops / clamp01(feats.get("pct_io_access",0.18))))
    m_stat = int(round(clamp01(feats.get("pct_meta_stat_access",0.0)) * io_ops / clamp01(feats.get("pct_io_access",0.18))))
    m_seek = int(round(clamp01(feats.get("pct_meta_seek_access",0.0)) * io_ops / clamp01(feats.get("pct_io_access",0.18))))
    m_sync = int(round(clamp01(feats.get("pct_meta_sync_access",0.0)) * io_ops / clamp01(feats.get("pct_io_access",0.18))))
    seed = 777
    meta_row = ["meta", str(META_DIR / "meta_only.dat"), "0","1","0","0","0","0","0","0",
                f"{p_file_ua:.6f}", f"{p_mem_ua:.6f}",
                "0.0",
                str(m_open), str(m_stat), str(m_seek), str(m_sync),
                str(seed), "meta_only",
                "0.0","0.0","0.0","0"]
    lines.append(",".join(meta_row))

    # ---- File sizing (truncate) ----
    # For each file, compute worst-case span needed: sum over bins:
    #   consec: n_con * xfer
    #   seqR  : n_seq * (xfer + gap), gap=xfer
    # Random doesn't need contiguous span (sampled anywhere).
    per_file_span = defaultdict(int)
    def add_span(rows, pc, ps):
        grp = defaultdict(lambda: defaultdict(int))  # file -> xfer -> ops
        for r in rows:
            grp[r["file"]][r["xfer"]] += r["ops"]
        for f, sizes in grp.items():
            for xfer, ops in sizes.items():
                n_con = int(round(ops * pc))
                n_seq = int(round(ops * max(0.0, ps - pc)))
                span = n_con * xfer + n_seq * (xfer + xfer)
                per_file_span[f] += span
    add_span(read_rows,  consec_r, seq_r)
    add_span(write_rows, consec_w, seq_w)

    # After span calc, add cushion & ensure >= 4KiB
    file_sizes = {}
    for f in set([r["file"] for r in read_rows+write_rows]):
        sz = per_file_span.get(f, 0) + CUSHION
        file_sizes[f] = max(sz, 4096)

    # ---- Write plan.csv ----
    os.makedirs(PAYLOAD, exist_ok=True)
    with open(PLAN, "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- Prep script ----
    os.makedirs(DATA_RO, exist_ok=True)
    os.makedirs(DATA_RW, exist_ok=True)
    os.makedirs(DATA_WO, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)
    with open(PREP, "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        f.write(f"mkdir -p {DATA_RO} {DATA_RW} {DATA_WO} {META_DIR}\n")
        # truncate data files to computed sizes
        for path, size in sorted(file_sizes.items()):
            f.write(f"truncate -s {size} {path}\n")
        f.write(f"truncate -s 4096 {META_DIR/'meta_only.dat'}\n")
    os.chmod(PREP, 0o755)

    # ---- Runner ----
    with open(RUNNER, "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        f.write(f"bash {PREP}\n")
        f.write("mpiexec -n {n} -genv LD_PRELOAD /mnt/hasanfs/darshan-3.4.7/darshan-runtime/install/lib/libdarshan.so "
                "/mnt/hasanfs/bin/mpi_synthio --plan {plan} --io-api {iapi} --meta-api {mapi} --collective {coll}\n"
                .format(n=nprocs, plan=str(PLAN),
                        iapi=str(feats.get('io_api','posix')),
                        mapi=str(feats.get('meta_api','posix')),
                        coll=str(feats.get('mpi_collective_mode','none'))))
    os.chmod(RUNNER, 0o755)

    # ---- Notes ----
    total_bytes_read  = sum(r["ops"]*r["xfer"] for r in read_rows)
    total_bytes_write = sum(r["ops"]*r["xfer"] for r in write_rows)
    with open(NOTES,"w") as f:
        f.write("=== Feature → Execution Mapping (OPS-first, fixed S/M, L=16MiB, uniform across files) ===\n")
        f.write(f"  cap_total_gib={cap_total_gib:.2f} → IO bytes target={human_bytes(io_bytes_target)}\n")
        f.write(f"  Nio={Nio}  |  Nr={Nr}  |  Nw={Nw}\n")
        f.write(f"  Read bin ops: S={R_S}, M={R_M}, L={R_L}\n")
        f.write(f"  Write bin ops: S={W_S}, M={W_M}, L={W_L}\n")
        f.write(f"  S sizes: 100 B, 1.00 KiB, 4.00 KiB, 64.00 KiB (equal ops)\n")
        f.write(f"  M sizes: 256.00 KiB, 1.00 MiB, 4.00 MiB (equal ops)\n")
        f.write(f"  L size: 16.00 MiB (fixed)\n\n")

        f.write("=== Seq/Consec semantics ===\n")
        f.write(f"  Reads: consec={consec_r:.3f}, seq={seq_r:.3f} (consec⊂seq enforced)\n")
        f.write(f"  Writes: consec={consec_w:.3f}, seq={seq_w:.3f}\n")
        f.write("  Seq remainder gap = xfer (seq-but-not-consec)\n\n")

        f.write("=== Random placement ===\n")
        f.write("  32MiB chunks, descending by chunk start; pre-seek EOF fence before random\n\n")

        f.write("=== Alignment ===\n")
        f.write(f"  Target file_unaligned={p_file_ua:.2f}, mem_unaligned={p_mem_ua:.2f}\n")
        f.write("  S/M unaligned; large ops aligned fraction chosen to hit target\n")
        f.write("  SITE ALIGNMENT: FILE=1MiB, MEM=8B\n\n")

        f.write("=== Files & sizes ===\n")
        f.write(f"  RO={n_ro}  RW={n_rw}  WO={n_wo}\n")
        for p in ro_paths: f.write(f"  RO: {p}  size={human_bytes(file_sizes.get(p,0))}\n")
        for p in rw_paths: f.write(f"  RW: {p}  size={human_bytes(file_sizes.get(p,0))}\n")
        for p in wo_paths: f.write(f"  WO: {p}  size={human_bytes(file_sizes.get(p,0))}\n")
        f.write("\n=== META (from pct shares, independent of IO bytes) ===\n")
        f.write(f"  io_ops={io_ops}, meta≈(open={m_open}, stat={m_stat}, seek={m_seek}, sync={m_sync})\n")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PAYLOAD.mkdir(parents=True, exist_ok=True)
    plan_from_features(args.features)
    print(f"Wrote {PREP}, {RUNNER}, and {NOTES} to {OUT_ROOT}")

if __name__ == "__main__":
    main()
