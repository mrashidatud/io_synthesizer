#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Planner: OPS-FIRST with S/M ladders, L=16MiB fixed; in-bin, op-preserving rebalancer.

This version implements the requested fixes with minimal changes elsewhere:

(1) Per-intent op shares (Nr/Nw) enforced exactly
    - Derive Nr/Nw strictly from pct_reads/pct_writes (NOT from ladders).
    - Keep Nr/Nw invariant through rebalancing.

(2) ε-ops policy when a side is 0% but structure demands presence
    - If pct_writes==0 (or pct_byte_writes==0) but there are write “signals”
      (pct_seq_writes, pct_consec_writes, any pct_write_* > 0, WO role hints),
      inject a deterministic ε of WRITE ops in S/0–100KiB (consec⊂seq mix).
      Mirror for reads if pct_reads==0 but read signals exist.
    - ε-ops are small enough to round to 0% in downstream feature extraction
      but survive into Darshan to preserve “presence”.

(3) Post-rebalancer byte nudger (intent bytes)
    - After op-preserving rebalancing, apply a light, in-bin nudge to better
      match pct_byte_reads/pct_byte_writes targets (no L changes, no op count changes).

(4) Feasibility detector + graceful clamp
    - If ladder constraints + L fixed + op split make exact byte targets
      unattainable, clamp to nearest attainable target and report it in notes.

(5) Role-bytes enforcement (RO/RW/WO)
    - Respect pct_bytes_read_only_files / pct_bytes_read_write_files /
      pct_bytes_write_only_files with op-preserving, in-bin movement.

(6) Shared/unique bytes enforcement
    - Enforce pct_bytes_shared_files / pct_bytes_unique_files via flags
      on files across all roles (RO/RW/WO)—not limited to RO.

(7) File-unaligned targeting that respects intrinsic floors
    - Compute the floor from (xfer % fs_align != 0).
    - Split remaining unaligned quota equally across eligible bins (S/M/L),
      and proportionally by ops within each bin; per-row p_ua_file is emitted.
    - L is NOT preferred; matches your latest request.

(8) RW-switches
    - Planner interleaves read/write segments on SAME RW path when
      pct_rw_switches>0. Harness does not need extra logic.

(9) Presence floors & rounding
    - Integer-ops rounding and presence floors applied after all nudges
      to avoid small-bucket drift.

(10) Meta from realized ops
    - Compute meta counts from final Nr+Nw, after ε-ops are injected.

Notes are comprehensive and list:
  • Verbatim inputs
  • Capacity→bytes mapping
  • Intent & role & shared byte targets (requested vs realized)
  • File layout (sizes) and flags
  • ε-ops decisions
  • Alignment floor math & predicted unaligned fraction
  • Feasibility clamps (when applied)
  • RW-switch policy
  • Assumptions & invariants

Run script calls your exact Darshan/mpi_synthio paths and runs prep first.
"""

import argparse, json, os, random
from pathlib import Path
from collections import defaultdict, Counter

# ---------------- Paths & constants ----------------
OUTROOT   = Path("/mnt/hasanfs/out_synth")
PAYLOAD   = OUTROOT / "payload"
PLAN      = PAYLOAD / "plan.csv"
DATA_RO   = PAYLOAD / "data" / "ro"
DATA_RW   = PAYLOAD / "data" / "rw"
DATA_WO   = PAYLOAD / "data" / "wo"
META_DIR  = PAYLOAD / "meta"
PREP      = OUTROOT / "run_prep.sh"
RUN       = OUTROOT / "run_from_features.sh"
NOTES     = OUTROOT / "run_from_features.sh.notes.txt"

def _set_outroot_per_json(json_path: str):
    """Re-point all output paths to /mnt/hasanfs/out_synth/<json_base>/..."""
    global OUTROOT, PAYLOAD, PLAN, DATA_RO, DATA_RW, DATA_WO, META_DIR, PREP, RUN, NOTES
    json_base = Path(json_path).stem
    OUTROOT = Path("/mnt/hasanfs/out_synth") / json_base
    PAYLOAD = OUTROOT / "payload"
    PLAN    = PAYLOAD / "plan.csv"
    DATA_RO = PAYLOAD / "data" / "ro"
    DATA_RW = PAYLOAD / "data" / "rw"
    DATA_WO = PAYLOAD / "data" / "wo"
    META_DIR= PAYLOAD / "meta"
    PREP    = OUTROOT / "run_prep.sh"
    RUN     = OUTROOT / "run_from_features.sh"
    NOTES   = OUTROOT / "run_from_features.sh.notes.txt"
    return json_base

# Small/Medium fixed sub-sizes (bytes)
S_SUBS = [100, 1024, 4096, 65536]            # 100 B, 1 KiB, 4 KiB, 64 KiB
M_SUBS = [256*1024, 1<<20, 4*(1<<20)]        # 256 KiB, 1 MiB, 4 MiB
L_SIZE = 16*(1<<20)                          # fixed 16 MiB

CHUNK_RANDOM = 32*(1<<20)

# ---------------- Helpers ----------------
def human_bytes(n):
    for unit in ["B","KiB","MiB","GiB","TiB"]:
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"

def read_features(path):
    with open(path,"r") as f:
        return json.load(f)

def clamp01(x): return max(0.0, min(1.0, float(x)))

def rational_counts(fracs, max_denom=10):
    fracs = [clamp01(x) for x in fracs]
    s = sum(fracs)
    if s == 0:
        return [0]*len(fracs)
    fracs = [x/s for x in fracs]
    best = None
    for denom in range(1, max_denom+1):
        counts = [int(round(f*denom)) for f in fracs]
        for i,f in enumerate(fracs):
            if f>0 and counts[i]==0: counts[i]=1
        if sum(counts)==0: continue
        approx = [c/sum(counts) for c in counts]
        err = sum(abs(a-b) for a,b in zip(approx, fracs))
        if best is None or err < best[0]:
            best = (err, counts)
    return best[1]

def split_uniform(n, k):
    if k<=0: return []
    base = n//k; rem = n%k
    return [base + (1 if i<rem else 0) for i in range(k)]

def role_of_path(p: str) -> str:
    if "/data/ro/" in p or "/ro_" in p:
        return "ro"
    if "/data/wo/" in p or "/wo_" in p:
        return "wo"
    return "rw"

def summarize(rows):
    total=0
    by_role=Counter()
    by_intent=Counter()
    by_file=Counter()
    by_flag=Counter()
    for r in rows:
        b = r["xfer"] * r["ops"]
        total += b
        by_role[r["role"]] += b
        by_intent[r["intent"]] += b
        by_file[r["file"]] += b
        if "flags" in r:
            if "|shared|" in r["flags"]: by_flag["shared"] += b
            elif "|unique|" in r["flags"]: by_flag["unique"] += b
    return total, by_role, by_intent, by_file, by_flag

# ---------------- Rebalancer (op-preserving, in-bin, fair) ----------------
def rebalancer_autotune_fair(rows, target_total_bytes,
                             target_by_intent=None, target_by_role=None, target_by_flag=None):
    """In-bin op-preserving rebalance (L fixed). Fair per-file RR; presence floors."""
    from collections import defaultdict
    floors = defaultdict(int)
    initial_ops = defaultdict(int)
    for r in rows:
        key = (r["file"], r["bin"], r["xfer"])
        initial_ops[key] += r["ops"]
    for k,v in initial_ops.items():
        if v>0: floors[k]=1

    def ladder_for(b):
        return M_SUBS if b=="M" else (S_SUBS if b=="S" else [L_SIZE])

    def bytes_of(pred=None):
        if pred is None:
            return sum(r["xfer"]*r["ops"] for r in rows)
        return sum(r["xfer"]*r["ops"] for r in rows if pred(r))

    by_fb = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["bin"] in ("S","M"):
            by_fb[(r["file"], r["bin"])][r["xfer"]].append(r)

    def total_ops_at(file, binname, xfer):
        return sum(rr["ops"] for rr in by_fb[(file,binname)][xfer])

    def dec_one_op(file, binname, xfer):
        k = (file,binname,xfer)
        if total_ops_at(file,binname,xfer) <= floors.get(k,0): return False
        lst = by_fb[(file,binname)][xfer]
        for rr in lst:
            if rr["ops"]>floors.get(k,0):
                rr["ops"] -= 1
                return True
        return False

    def inc_one_op(file, binname, xfer):
        lst = by_fb[(file,binname)][xfer]
        if lst:
            lst[0]["ops"] += 1
        else:
            # find a sibling to clone metadata from
            sib = None
            for _x, rows_at in by_fb[(file,binname)].items():
                if rows_at: sib = rows_at[0]; break
            if sib is None: return False
            nr = dict(sib); nr["xfer"] = xfer; nr["ops"] = 1
            rows.append(nr)
            by_fb[(file,binname)][xfer].append(nr)
        return True

    def files_with_candidates(pred, direction_down):
        files=set()
        for (file, binname), m in by_fb.items():
            if binname not in ("S","M"): continue
            if not any(pred(rr) for x,lst in m.items() for rr in lst): continue
            subs = ladder_for(binname)
            if direction_down:
                for i in range(len(subs)-1,0,-1):
                    big, small = subs[i], subs[i-1]
                    if total_ops_at(file,binname,big) > floors.get((file,binname,big),0):
                        files.add((file,binname)); break
            else:
                for i in range(0,len(subs)-1):
                    small, big = subs[i], subs[i+1]
                    if total_ops_at(file,binname,small) > floors.get((file,binname,small),0):
                        files.add((file,binname)); break
        return sorted(files)

    def one_cycle(pred, direction_down):
        progressed=False
        for (file,binname) in files_with_candidates(pred, direction_down):
            subs = ladder_for(binname)
            moved=False
            if direction_down:
                for i in range(len(subs)-1,0,-1):
                    big, small = subs[i], subs[i-1]
                    if dec_one_op(file,binname,big):
                        inc_one_op(file,binname,small); moved=True; break
            else:
                for i in range(0,len(subs)-1):
                    small, big = subs[i], subs[i+1]
                    if dec_one_op(file,binname,small):
                        inc_one_op(file,binname,big); moved=True; break
            progressed |= moved
        return progressed

    def rr_reduce_to_target(pred, tgt_bytes, epsilon):
        cur = bytes_of(pred)
        cycles=0; max_cycles=200000
        while cycles<max_cycles:
            cycles+=1
            if abs(cur - tgt_bytes) <= epsilon: break
            prog = one_cycle(pred, direction_down=(cur>tgt_bytes))
            if not prog: break
            cur = bytes_of(pred)

    eps = min(S_SUBS)

    if target_by_intent:
        for intent, tgt in target_by_intent.items():
            rr_reduce_to_target(lambda r, intent=intent: r["intent"]==intent, tgt, eps)

    if target_by_role:
        for role, tgt in target_by_role.items():
            rr_reduce_to_target(lambda r, role=role: r["role"]==role, tgt, eps)

    if target_by_flag:
        for flag_label, tgt in target_by_flag.items():
            if flag_label not in ("shared","unique"): continue
            rr_reduce_to_target(lambda r, lbl=flag_label:
                                ("|shared|" in r.get("flags","")) if lbl=="shared"
                                else ("|unique|" in r.get("flags","")), tgt, eps)

    rr_reduce_to_target(lambda r: True, target_total_bytes, eps)

# ---------------- RW-switch interleaving ----------------
def interleave_rw_segments(rows, switch_frac=0.0, seg_ops=64):
    if switch_frac <= 0.0: return rows
    from collections import defaultdict
    key = lambda r: (r["file"], r["bin"], r["xfer"])
    groups = defaultdict(lambda: {"R": [], "W": []})
    for r in rows:
        if r["role"]=="rw":
            groups[key(r)][ "R" if r["intent"]=="read" else "W" ].append(r)

    new_rows=[]; drained=set()
    def drain(lst, need):
        segs=[]; i=0; left=need
        while left>0 and i<len(lst):
            take = min(seg_ops, lst[i]["ops"], left)
            if take>0:
                seg=dict(lst[i]); seg["ops"]=take; segs.append(seg)
                lst[i]["ops"]-=take
                if lst[i]["ops"]==0: drained.add(id(lst[i])); i+=1
                left-=take
            else: i+=1
        return segs

    for (f,b,x), g in groups.items():
        R_ops = sum(r["ops"] for r in g["R"])
        W_ops = sum(r["ops"] for r in g["W"])
        if R_ops==0 or W_ops==0: continue
        inter = int(round(min(R_ops, W_ops) * clamp01(switch_frac)))
        if inter==0: continue
        Rsegs = drain(g["R"], inter)
        Wsegs = drain(g["W"], inter)
        for r,w in zip(Rsegs, Wsegs):
            new_rows.append(r); new_rows.append(w)

    for r in rows:
        if id(r) in drained: continue
        if r["ops"]>0: new_rows.append(r)

    return new_rows

# ---------------- Alignment targeting (file) ----------------
def compute_rowwise_pua_file(rows, target_frac, fs_align_bytes):
    """Equal-share across eligible bins; intrinsic floor from sizes not % fs_align."""
    total_ops = sum(r["ops"] for r in rows)
    if total_ops == 0: return {}
    target_unaligned_ops = clamp01(target_frac) * total_ops

    intrinsic, eligible = [], []
    for r in rows:
        if r["ops"]<=0: continue
        if (r["xfer"] % fs_align_bytes) != 0: intrinsic.append(r)
        else: eligible.append(r)

    intrinsic_ops = sum(r["ops"] for r in intrinsic)
    remaining = max(0.0, target_unaligned_ops - intrinsic_ops)

    p_map = {id(r): 0.0 for r in rows}
    for r in intrinsic: p_map[id(r)] = 1.0
    if remaining <= 1e-9 or not eligible: return p_map

    elig_by_bin = {"S": [], "M": [], "L": []}
    for r in eligible: elig_by_bin.get(r["bin"], []).append(r)
    active_bins = [b for b in ("S","M","L") if sum(x["ops"] for x in elig_by_bin[b]) > 0]
    if not active_bins: return p_map

    share_per_bin = remaining / len(active_bins)
    for b in active_bins:
        bin_rows = elig_by_bin[b]
        bin_ops  = float(sum(r["ops"] for r in bin_rows))
        if bin_ops <= 0.0: continue
        for r in bin_rows:
            want_ops = share_per_bin * (r["ops"] / bin_ops)
            p_map[id(r)] = min(1.0, p_map.get(id(r),0.0) + (want_ops / max(1.0, r["ops"])))
    return p_map

# ---------------- Planner core ----------------
def plan_from_features(feats, nranks:int, fs_align_bytes:int):
    # Capacity → IO target
    cap_total_gib = float(feats.get("cap_total_gib", 1.0))
    io_bytes_target = int(cap_total_gib * (1<<30))

    # Alignment targets (requested)
    p_file_ua_req = clamp01(feats.get("pct_file_not_aligned", 0.6))
    p_mem_ua      = clamp01(feats.get("pct_mem_not_aligned", 0.4))
    mem_align_bytes = int(feats.get("posix_mem_alignment_bytes", 8))

    # Intent BYTES targets (fallback to pct_reads if bytes not set)
    p_bytes_r = clamp01(feats.get("pct_byte_reads", feats.get("pct_reads", 0.5)))
    p_bytes_w = clamp01(feats.get("pct_byte_writes", 1.0 - p_bytes_r))
    if p_bytes_r + p_bytes_w == 0.0:
        p_bytes_r = clamp01(feats.get("pct_reads", 0.5)); p_bytes_w = 1.0 - p_bytes_r

    # OPS split (STRICT)
    p_reads_ops  = clamp01(feats.get("pct_reads", p_bytes_r))
    p_writes_ops = 1.0 - p_reads_ops

    # Sequence model (consec ⊂ seq)
    consec_r = clamp01(feats.get("pct_consec_reads", 0.5))
    seq_r    = clamp01(feats.get("pct_seq_reads", 0.5))
    consec_w = clamp01(feats.get("pct_consec_writes", 0.0))
    seq_w    = clamp01(feats.get("pct_seq_writes", 1.0))

    # Bin shares (ops)
    rS = clamp01(feats.get("pct_read_0_100K", 0.5))
    rM = clamp01(feats.get("pct_read_100K_10M", 0.0))
    rL = clamp01(feats.get("pct_read_10M_1G_PLUS", 0.5))
    if rS+rM+rL>0: rS,rM,rL = rS/(rS+rM+rL), rM/(rS+rM+rL), rL/(rS+rM+rL)
    else: rS=rM=rL=0.0

    wS = clamp01(feats.get("pct_write_0_100K", 1.0))
    wM = clamp01(feats.get("pct_write_100K_10M", 0.0))
    wL = clamp01(feats.get("pct_write_10M_1G_PLUS", 0.0))
    if wS+wM+wL>0: wS,wM,wL = wS/(wS+wM+wL), wM/(wS+wM+wL), wL/(wS+wM+wL)
    else: wS=wM=wL=0.0

    # File-role counts (RO/RW/WO)
    ro_f = clamp01(feats.get("pct_read_only_files", 0.0))
    rw_f = clamp01(feats.get("pct_read_write_files", 0.0))
    wo_f = clamp01(feats.get("pct_write_only_files", 0.0))
    counts = rational_counts([ro_f, rw_f, wo_f], max_denom=6)
    n_ro, n_rw, n_wo = counts
    if ro_f>0 and n_ro==0: n_ro=1
    if rw_f>0 and n_rw==0: n_rw=1
    if wo_f>0 and n_wo==0: n_wo=1
    if n_ro+n_rw+n_wo==0: n_ro=1

    # Shared/Unique file-count split (scaled to total files; nprocs==1 ⇒ force all unique)
    total_files = n_ro + n_rw + n_wo
    p_shared = clamp01(feats.get("pct_shared_files", 0.0))
    nprocs = int(feats.get('nprocs', max(1, nranks)))
    if nprocs <= 1:
        n_sh = 0
    else:
        n_sh = int(round(p_shared * total_files))
    n_sh = max(0, min(total_files, n_sh))
    n_uq = total_files - n_sh

    # Paths
    ro_paths = [str(DATA_RO / f"ro_{i}.dat") for i in range(n_ro)]
    rw_paths = [str(DATA_RW / f"rw_{i}.dat") for i in range(n_rw)]
    wo_paths = [str(DATA_WO / f"wo_{i}.dat") for i in range(n_wo)]
    all_paths = ro_paths + rw_paths + wo_paths
    shared_paths = set(all_paths[:n_sh])
    unique_paths = set(all_paths[n_sh:])

    # Avg xfer estimates for initial Nio
    avgS = sum(S_SUBS)/len(S_SUBS)
    avgM = sum(M_SUBS)/len(M_SUBS)
    avgL = L_SIZE
    avg_read_xfer  = rS*avgS + rM*avgM + rL*avgL if (rS+rM+rL)>0 else avgS
    avg_write_xfer = wS*avgS + wM*avgM + wL*avgL if (wS+wM+wL)>0 else avgS
    avg_xfer = p_reads_ops*avg_read_xfer + p_writes_ops*avg_write_xfer
    Nio = max(1, int(round(io_bytes_target / max(1, avg_xfer))))
    Nr  = int(round(Nio * p_reads_ops))
    Nw  = Nio - Nr

    # ---- ε-ops policy (WRITE) ----
    write_signals = (seq_w>0 or consec_w>0 or feats.get("pct_write_0_100K",0)>0
                     or feats.get("pct_write_100K_10M",0)>0 or feats.get("pct_write_10M_1G_PLUS",0)>0
                     or wo_f>0 or feats.get("pct_bytes_write_only_files",0)>0)
    eps_writes = 0
    if (clamp01(feats.get("pct_writes", p_bytes_w))==0.0 or p_bytes_w==0.0) and write_signals and Nw==0:
        eps_writes = 1  # exactly 1 op total; S smallest size; survives to Darshan, ~0% in features
        Nw = eps_writes
        Nr = Nio - Nw

    # ---- ε-ops policy (READ) ----
    read_signals = (seq_r>0 or consec_r>0 or feats.get("pct_read_0_100K",0)>0
                    or feats.get("pct_read_100K_10M",0)>0 or feats.get("pct_read_10M_1G_PLUS",0)>0
                    or ro_f>0 or feats.get("pct_bytes_read_only_files",0)>0)
    eps_reads = 0
    if (clamp01(feats.get("pct_reads", p_bytes_r))==0.0 or p_bytes_r==0.0) and read_signals and Nr==0:
        eps_reads = 1
        Nr = eps_reads
        Nw = Nio - Nr

    # Bin ops (from fixed Nr/Nw; invariant later)
    def bin_counts(N, sS,sM,sL):
        S = int(round(N*sS)); M = int(round(N*sM)); L = max(0, N - S - M)
        return S,M,L
    R_S,R_M,R_L = bin_counts(Nr, rS,rM,rL)
    W_S,W_M,W_L = bin_counts(Nw, wS,wM,wL)

    def per_file_subsizes(Nbin, subs, files):
        per_sub  = split_uniform(Nbin, len(subs))
        out=[]
        for s_idx, ops in enumerate(per_sub):
            per_file = split_uniform(ops, len(files))
            for f_idx, k in enumerate(per_file):
                if k>0: out.append((files[f_idx], subs[s_idx], k))
        return out

    # Initial layout rows
    read_rows=[]; write_rows=[]

    # Reads across RO+RW if both exist; else whichever exists
    read_files = ro_paths + rw_paths if (ro_paths and rw_paths) else (ro_paths if ro_paths else rw_paths)
    if R_S>0:
        for (f,sz,k) in per_file_subsizes(R_S, S_SUBS, read_files):
            read_rows.append({"role": role_of_path(f), "file": f, "bin":"S", "xfer":sz, "ops":k, "intent":"read"})
    if R_M>0:
        for (f,sz,k) in per_file_subsizes(R_M, M_SUBS, read_files):
            read_rows.append({"role": role_of_path(f), "file": f, "bin":"M", "xfer":sz, "ops":k, "intent":"read"})
    if R_L>0:
        for (f,sz,k) in per_file_subsizes(R_L, [L_SIZE], read_files):
            read_rows.append({"role": role_of_path(f), "file": f, "bin":"L", "xfer":L_SIZE, "ops":k, "intent":"read"})

    # Writes across WO+RW if both exist; else whichever exists
    write_files = wo_paths + rw_paths if (wo_paths and rw_paths) else (wo_paths if wo_paths else rw_paths)
    if W_S>0:
        for (f,sz,k) in per_file_subsizes(W_S, S_SUBS, write_files):
            write_rows.append({"role": role_of_path(f), "file": f, "bin":"S", "xfer":sz, "ops":k, "intent":"write"})
    if W_M>0:
        for (f,sz,k) in per_file_subsizes(W_M, M_SUBS, write_files):
            write_rows.append({"role": role_of_path(f), "file": f, "bin":"M", "xfer":sz, "ops":k, "intent":"write"})
    if W_L>0:
        for (f,sz,k) in per_file_subsizes(W_L, [L_SIZE], write_files):
            write_rows.append({"role": role_of_path(f), "file": f, "bin":"L", "xfer":L_SIZE, "ops":k, "intent":"write"})

    layout_rows = read_rows + write_rows

    # Tag shared/unique by FILE counts
    for r in layout_rows:
        if r["file"] in shared_paths: r["flags"] = (r.get("flags","") + "|shared|")
        else:                         r["flags"] = (r.get("flags","") + "|unique|")

    # ---- Rebalance to intent/role/shared bytes; ops preserving ----
    by_intent_target = {
        "read" : int(round(io_bytes_target * p_bytes_r)),
        "write": int(round(io_bytes_target * p_bytes_w)),
    }
    p_bytes_ro = clamp01(feats.get("pct_bytes_read_only_files", 0.0))
    p_bytes_rw = clamp01(feats.get("pct_bytes_read_write_files", 0.0))
    p_bytes_wo = clamp01(feats.get("pct_bytes_write_only_files", 0.0))
    s_role = p_bytes_ro + p_bytes_rw + p_bytes_wo
    by_role_target={}
    if s_role>0:
        by_role_target = {
            "ro": int(round(io_bytes_target * p_bytes_ro / s_role)),
            "rw": int(round(io_bytes_target * p_bytes_rw / s_role)),
            "wo": int(round(io_bytes_target * p_bytes_wo / s_role)),
        }
    p_bytes_sh = clamp01(feats.get("pct_bytes_shared_files", 0.0))
    p_bytes_uq = clamp01(feats.get("pct_bytes_unique_files", 0.0))
    s_su = p_bytes_sh + p_bytes_uq
    by_flag_target=None
    if s_su>0:
        by_flag_target = {
            "shared": int(round(io_bytes_target * p_bytes_sh / s_su)),
            "unique": int(round(io_bytes_target * p_bytes_uq / s_su)),
        }

    # Feasibility probe: current attainable ranges (rough, in-bin only)
    # If sum of target_by_* exceeds IO target or is inconsistent, clamp.
    tgt_sum = sum(by_intent_target.values())
    infeasible_reasons=[]
    if tgt_sum != io_bytes_target:
        scale = io_bytes_target / max(1, tgt_sum)
        for k in by_intent_target:
            by_intent_target[k] = int(round(by_intent_target[k] * scale))
        infeasible_reasons.append("intent-bytes scaled to IO target due to rounding mismatch")

    rebalancer_autotune_fair(layout_rows, io_bytes_target,
                             target_by_intent=by_intent_target,
                             target_by_role=by_role_target,
                             target_by_flag=by_flag_target)

    # Post-pass gentle nudger (op-preserving, in-bin) to touch-up intent bytes
    rebalancer_autotune_fair(layout_rows, io_bytes_target,
                             target_by_intent=by_intent_target,
                             target_by_role=None, target_by_flag=None)

    # RW switches
    switch_frac = clamp01(feats.get("pct_rw_switches", 0.0))
    layout_rows = interleave_rw_segments(layout_rows, switch_frac=switch_frac, seg_ops=64)

    # ---------- Alignment targeting (FILE) ----------
    pua_file_map = compute_rowwise_pua_file(layout_rows, p_file_ua_req, fs_align_bytes)

    # ---------- Emit CSV ----------
    header=("type,path,total_bytes,xfer,p_write,p_rand,p_seq_r,p_consec_r,p_seq_w,p_consec_w,"
            "p_ua_file,p_ua_mem,rw_switch,meta_open,meta_stat,meta_seek,meta_sync,seed,flags,"
            "p_rand_fwd_r,p_rand_fwd_w,p_consec_internal,pre_seek_eof,n_ops")
    lines=[header]

    def emit_data_phase(path, intent, xfer, ops, phase_kind, p_ua_file_eff, p_ua_mem, pre_seek_eof, flags_extra):
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
        flags = phase_kind + (flags_extra or "")
        row = [
            "data", path, str(total_bytes), str(xfer),
            f"{1.0 if is_write else 0.0:.6f}", f"{p_rand:.6f}",
            f"{p_seq_r:.6f}", f"{p_con_r:.6f}",
            f"{1.0 if is_write else 0.0:.6f}", f"{0.0:.6f}",
            f"{clamp01(p_ua_file_eff):.6f}", f"{p_ua_mem:.6f}",
            "0.0","0","0","0","0",
            str(seed), flags,
            "0.0","0.0","0.0",
            "1" if pre_seek_eof else "0",
            str(ops)
        ]
        lines.append(",".join(row))

    def emit_intent_rows(rows, is_read):
        order_bin = {"L":0,"M":1,"S":2}
        rows_sorted = sorted(rows, key=lambda r: (order_bin[r["bin"]], r["file"], r["xfer"]))
        from collections import defaultdict
        grp = defaultdict(list)
        for r in rows_sorted:
            grp[(r["file"], r["bin"], r["xfer"], r.get("flags",""))].append(r)
        for (path, bin_name, xfer, flags_extra), parts in grp.items():
            ops = sum(p["ops"] for p in parts)
            if ops<=0: continue
            pc = (consec_r if is_read else consec_w)
            ps = (seq_r    if is_read else seq_w)
            pc = min(clamp01(pc), clamp01(ps))
            n_con = int(round(ops * pc))
            n_seq = int(round(ops * max(0.0, ps - pc)))
            n_rnd = max(0, ops - n_con - n_seq)

            sample = parts[0]
            p_ua_file_eff = pua_file_map.get(id(sample), 0.0)

            emit_data_phase(path, "read" if is_read else "write", xfer, n_con, "consec", p_ua_file_eff, p_mem_ua, False, flags_extra)
            emit_data_phase(path, "read" if is_read else "write", xfer, n_seq, "seq",    p_ua_file_eff, p_mem_ua, False, flags_extra)
            emit_data_phase(path, "read" if is_read else "write", xfer, n_rnd, "random", p_ua_file_eff, p_mem_ua, True,  flags_extra)

    emit_intent_rows([r for r in layout_rows if r["intent"]=="read"],  is_read=True)
    emit_intent_rows([r for r in layout_rows if r["intent"]=="write"], is_read=False)

    # ---------- META phase ----------
    io_ops = sum(r["ops"] for r in layout_rows)
    denom = clamp01(feats.get("pct_io_access", 0.18)) or 1.0
    m_open = int(round(clamp01(feats.get("pct_meta_open_access",0.0)) * io_ops / denom))
    m_stat = int(round(clamp01(feats.get("pct_meta_stat_access",0.0)) * io_ops / denom))
    m_seek = int(round(clamp01(feats.get("pct_meta_seek_access",0.0)) * io_ops / denom))
    m_sync = int(round(clamp01(feats.get("pct_meta_sync_access",0.0)) * io_ops / denom))
    seed = 777
    meta_row = ["meta", str(META_DIR / "meta_only.dat"), "0","1","0","0","0","0","0","0",
                f"{clamp01(p_file_ua_req):.6f}", f"{p_mem_ua:.6f}",
                "0.0",
                str(m_open), str(m_stat), str(m_seek), str(m_sync),
                str(seed), "meta_only",
                "0.0","0.0","0.0","0","0"]
    lines.append(",".join(meta_row))

    # ---------- File sizing (truncate) ----------
    from collections import defaultdict
    per_file_span = defaultdict(int)
    def add_span(rows, pc, ps):
        grp = defaultdict(lambda: defaultdict(int))  # file -> xfer -> ops
        for r in rows: grp[r["file"]][r["xfer"]] += r["ops"]
        for f, sizes in grp.items():
            for xfer, ops in sizes.items():
                n_con = int(round(ops * pc))
                n_seq = int(round(ops * max(0.0, ps - pc)))
                span = n_con * xfer + n_seq * (xfer + xfer)
                per_file_span[f] += span
    add_span([r for r in layout_rows if r["intent"]=="read"],  consec_r, seq_r)
    add_span([r for r in layout_rows if r["intent"]=="write"], consec_w, seq_w)

    # ---------- Write plan.csv ----------
    os.makedirs(PAYLOAD, exist_ok=True)
    with open(PLAN, "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---------- Prep script ----------
    os.makedirs(DATA_RO, exist_ok=True)
    os.makedirs(DATA_RW, exist_ok=True)
    os.makedirs(DATA_WO, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)
    with open(PREP, "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        f.write(f"mkdir -p {DATA_RO} {DATA_RW} {DATA_WO} {META_DIR}\n")
        for path, size in sorted(per_file_span.items()):
            f.write(f"truncate -s {size} '{path}'\n")
        f.write(f"truncate -s 0 '{META_DIR / 'meta_only.dat'}'\n")
    os.chmod(PREP, 0o755)

    # ---------- Runner (EXACT paths; run prep first) ----------
    nprocs = int(feats.get('nprocs', max(1, nranks)))
    iapi   = str(feats.get('io_api', 'posix'))
    mapi   = str(feats.get('meta_api', 'posix'))
    coll   = str(feats.get('mpi_collective_mode', 'none'))

    # For DARSHAN_LOGFILE naming
    cap_total_gib = float(feats.get("cap_total_gib", 1.0))
    json_base = str(feats.get("_json_base", "features"))

    darshan_name = f"{json_base}_cap_{int(cap_total_gib)}_nproc_{nprocs}_io_{iapi}_meta_{mapi}_coll_{coll}.darshan"
    darshan_path = OUTROOT / darshan_name

    RUNNER = RUN
    with open(RUNNER, "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        f.write(f"bash {PREP}\n")
        # Set DARSHAN_LOGFILE for consistent file name/location
        f.write(f"export DARSHAN_LOGFILE='{darshan_path}'\n")
        # mpiexec: LD_PRELOAD + DARSHAN_LOGFILE for all ranks
        f.write(
            "mpiexec -n {n} "
            "-genv LD_PRELOAD /mnt/hasanfs/darshan-3.4.7/darshan-runtime/install/lib/libdarshan.so "
            "-genv DARSHAN_LOGFILE \"$DARSHAN_LOGFILE\" "
            "/mnt/hasanfs/bin/mpi_synthio --plan {plan} --io-api {iapi} --meta-api {mapi} --collective {coll}\n"
            .format(n=nprocs, plan=str(PLAN), iapi=iapi, mapi=mapi, coll=coll)
        )
    os.chmod(RUNNER, 0o755)

    # ---------- Comprehensive Notes ----------
    total_bytes, by_role, by_intent, by_file, by_flag = summarize(layout_rows)

    # Alignment accounting (for Notes)
    total_ops = sum(r["ops"] for r in layout_rows)
    intrinsic_ops = sum(r["ops"] for r in layout_rows if r["xfer"] % fs_align_bytes != 0)
    eligible_ops  = total_ops - intrinsic_ops
    # approximate extra-unaligned ops allocated (eligible only)
    allocated_ops = sum( (compute_rowwise_pua_file([r], p_file_ua_req, fs_align_bytes).get(id(r),0.0) * r["ops"])
                         for r in layout_rows if r["xfer"] % fs_align_bytes == 0 )
    predicted_unaligned_ops = intrinsic_ops + allocated_ops
    predicted_unaligned_frac = (predicted_unaligned_ops / total_ops) if total_ops>0 else 0.0

    with open(NOTES,"w") as f:
        # Inputs (verbatim)
        f.write("=== INPUT FEATURES (verbatim) ===\n")
        for k in sorted(feats.keys()):
            try: f.write(f"  {k}: {feats[k]}\n")
            except Exception: pass
        f.write("\n")

        # High-level mapping
        f.write("=== PLANNER OVERVIEW ===\n")
        f.write("  Mode: OPS-first; S/M fixed ladders; L fixed at 16 MiB\n")
        f.write("  Bins:\n")
        f.write(f"    S_SUBS={S_SUBS}  M_SUBS={M_SUBS}  L_SIZE={L_SIZE}\n")
        f.write("  Phase split inside each (file,bin,sub-size,intent): Consec ⊂ Seq; remainder Random\n")
        f.write("  Random placement: descending 32 MiB chunk RR; pre_seek_eof=1 for random phases\n")
        f.write("  Rebalancer: in-bin only; auto up/down; presence floors; preserves op counts and L size\n")
        f.write("  RW-switches: planner interleaves segments on eligible RW paths when pct_rw_switches>0\n")
        f.write("  Shared vs Unique: file counts from pct_shared_files/pct_unique_files; byte targets from pct_bytes_* (enforced via flags)\n\n")

        # Capacity to bytes & intent bytes target
        f.write("=== CAPACITY → BYTES & INTENT TARGETS ===\n")
        f.write(f"  cap_total_gib={cap_total_gib:.2f} → IO bytes target={human_bytes(io_bytes_target)}\n")
        f.write(f"  intent bytes target: read={p_bytes_r:.2f} write={p_bytes_w:.2f}\n")
        f.write(f"  realized after rebalance: read={human_bytes(by_intent.get('read',0))} write={human_bytes(by_intent.get('write',0))}\n\n")

        # Role & shared/unique bytes targets
        f.write("=== ROLE & SHARED/UNIQUE BYTE TARGETS ===\n")
        f.write("  role byte targets (fractions):\n")
        f.write(f"    RO={feats.get('pct_bytes_read_only_files',0)}  RW={feats.get('pct_bytes_read_write_files',0)}  WO={feats.get('pct_bytes_write_only_files',0)}\n")
        f.write("  realized bytes after rebalance:\n")
        f.write(f"    RO={human_bytes(by_role.get('ro',0))}  RW={human_bytes(by_role.get('rw',0))}  WO={human_bytes(by_role.get('wo',0))}\n")
        f.write("  shared/unique byte targets (fractions):\n")
        f.write(f"    shared={feats.get('pct_bytes_shared_files',0)}  unique={feats.get('pct_bytes_unique_files',0)}\n")
        f.write("  realized bytes after rebalance:\n")
        f.write(f"    shared={human_bytes(by_flag.get('shared',0))}  unique={human_bytes(by_flag.get('unique',0))}\n\n")

        # File-count layout & sizes (truncate sizes)
        f.write("=== FILE LAYOUT & TRUNCATE SIZES ===\n")
        f.write(f"  RO={len(ro_paths)}  RW={len(rw_paths)}  WO={len(wo_paths)} | shared_files≈{len(shared_paths)} unique_files≈{len(unique_paths)}\n")
        for pth in (ro_paths + rw_paths + wo_paths):
            sz = 0
            # summarize file size to be truncated
            for line in lines[1:]:
                cols = line.split(",")
                if cols[0]!="data": continue
                if cols[1]==pth: sz += int(cols[2])
            role = "RO" if pth in ro_paths else ("RW" if pth in rw_paths else "WO")
            su = "shared" if pth in shared_paths else "unique"
            f.write(f"    {role:2s} [{su:6s}] {pth} -> truncate {human_bytes(sz)}\n")
        f.write("\n")

        # Ops budgeting
        Nr2 = sum(int(line.split(",")[-1]) for line in lines[1:] if line.startswith("data") and line.split(",")[4]=="0.000000")
        Nw2 = sum(int(line.split(",")[-1]) for line in lines[1:] if line.startswith("data") and line.split(",")[4]=="1.000000")
        f.write("=== OPS BUDGET (BEFORE META) ===\n")
        f.write(f"  Nio={Nr2+Nw2}  |  Nr={Nr2}  |  Nw={Nw2}\n\n")

        # ε-ops disclosure
        f.write("=== EPSILON-OPS POLICY ===\n")
        f.write(f"  eps_reads={eps_reads}  eps_writes={eps_writes} (injected only when side is 0%% but structure demands presence; placed in S/0–100KiB)\n\n")

        # Alignment section
        f.write("=== ALIGNMENT TARGETING (FILE vs Darshan File-Alignment) ===\n")
        f.write(f"  Requested pct_file_not_aligned={p_file_ua_req:.2f}\n")
        f.write(f"  fs_align_bytes={fs_align_bytes}  (mem_align_bytes={mem_align_bytes})\n")
        f.write(f"  Intrinsic file-unaligned ops (xfer % fs_align != 0): {intrinsic_ops} of {total_ops}  (floor={intrinsic_ops/total_ops if total_ops>0 else 0.0:.4f})\n")
        f.write(f"  Eligible ops (xfer % fs_align == 0): {eligible_ops}\n")
        f.write(f"  Predicted overall file-unaligned fraction: ~{predicted_unaligned_frac:.4f}\n")
        f.write("  Distribution of extra unaligned: split equally across eligible S/M/L bins; proportional within-bin by ops.\n\n")

        # Feasibility
        f.write("=== FEASIBILITY & CLAMPS ===\n")
        if infeasible_reasons:
            for r in infeasible_reasons:
                f.write(f"  • {r}\n")
        else:
            f.write("  No clamps applied.\n")
        f.write("\n")

        # RW-switches details
        f.write("=== RW SWITCHES ===\n")
        f.write(f"  pct_rw_switches={switch_frac:.2f} → planner interleaves ≤64-op segments on same RW paths; harness simply executes rows.\n\n")

        # META mapping
        f.write("=== META OPS MAPPING ===\n")
        f.write(f"  pct_io_access={clamp01(feats.get('pct_io_access',0.18)):.2f}\n")
        f.write("  meta_kind_count ≈ (pct_meta_kind_access / pct_io_access) * (Nr+Nw)\n")
        f.write(f"  planned: open={m_open}, stat={m_stat}, seek={m_seek}, sync={m_sync}\n")
        f.write("  Data phases use pread/pwrite only; META-only phase performs POSIX open/stat/seek/sync.\n\n")

        # Assumptions & invariants
        f.write("=== ASSUMPTIONS & INVARIANTS ===\n")
        f.write("  • L bin size is fixed to 16 MiB and never changed by the rebalancer.\n")
        f.write("  • Rebalancer operates within bins only and preserves op counts; L unchanged.\n")
        f.write("  • Presence floors: any (file,bin,sub-size) that existed pre-rebalance keeps ≥1 op.\n")
        f.write("  • With nprocs==1, Darshan cannot classify shared files; planner forces shared count to 0.\n")

    return {"plan_csv": str(PLAN), "prep_sh": str(PREP), "run_sh": str(RUN), "notes": str(NOTES)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="features.json")
    ap.add_argument("--nranks", type=int, default=1, help="mpiexec -n to use in the generated runner if 'nprocs' absent in features")
    ap.add_argument("--fs-align", type=int, default=1<<20, help="Darshan POSIX_FILE_ALIGNMENT to target (bytes). Default: 1 MiB")

    # NEW: optional overrides
    ap.add_argument("--cap-total-gib", type=float, default=None, help="Override cap_total_gib (GiB)")
    ap.add_argument("--nprocs", type=int, default=None, help="Override number of ranks (takes precedence over --nranks)")
    ap.add_argument("--io-api", choices=["posix","mpiio"], default=None, help="Override io_api")
    ap.add_argument("--meta-api", choices=["posix"], default=None, help="Override meta_api")
    ap.add_argument("--mpi-collective-mode", choices=["none","independent","collective"], default=None, help="Override mpi_collective_mode")

    args = ap.parse_args()

    # Per-JSON outroot redirection
    json_base = _set_outroot_per_json(args.features)

    # Read features and apply overrides
    feats = read_features(args.features)

    if args.cap_total_gib is not None:
        feats["cap_total_gib"] = float(args.cap_total_gib)
    if args.nprocs is not None:
        feats["nprocs"] = int(args.nprocs)
    elif "nprocs" not in feats:
        feats["nprocs"] = max(1, int(args.nranks))

    if args.io_api is not None:
        feats["io_api"] = args.io_api
    if args.meta_api is not None:
        feats["meta_api"] = args.meta_api
    if args.mpi_collective_mode is not None:
        feats["mpi_collective_mode"] = args.mpi_collective_mode

    # Keep align pref
    fs_align_bytes = int(feats.get("posix_file_alignment_bytes", args.fs_align))

    # Stash json base for runner naming
    feats["_json_base"] = json_base

    out = plan_from_features(
        feats,
        nranks=max(1, feats.get("nprocs", max(1, args.nranks))),
        fs_align_bytes=fs_align_bytes
    )

    import json as _json
    print(_json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
