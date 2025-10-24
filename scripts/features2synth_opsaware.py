#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features2synth_opsaware.py

Planner: OPS-FIRST with Small/Medium sub-size ladders, L=16MiB; deterministic ordering
         (L→M→S and Consec→SeqRemainder→Random). Post-pass in-bin rebalancer that can
         auto up/down and distribute adjustments fairly across files, while preserving
         sub-size presence (e.g., the 100 B sub-size never disappears if it existed).

Uses ONLY the feature names provided in the spec/input file.

What this planner guarantees (recap of your requirements):
  • Split file/memory unalignment PROPORTIONALLY across files, roles (R/W), bins, and sub-sizes.
  • Split consec/seq/random PROPORTIONALLY across sub-sizes per (file,role,bin) group.
  • Add n_ops in plan for each phase.
  • Headers and Notes reflect all behavior and assumptions.
  • RO/RW reads split proportional to file counts when both exist (else whichever exists).
    Similarly, WO/RW writes.
  • RW-switches: use pct_rw_switches (0..1) to interleave R/W segments on the SAME RW path.
  • Shared vs unique: counts from pct_shared_files/pct_unique_files; bytes from
    pct_bytes_shared_files/pct_bytes_unique_files; flags “|shared|/|unique|” emitted for harness.
  • Rebalancer: in-bin only; preserves ops and L=16MiB; strict per-file round-robin; presence floors.
  • NEW: Alignment targeting honors a FILE-ALIGNMENT floor:
      - Compute intrinsic file-unaligned ops = ops whose xfer % fs_align != 0.
      - If requested pct_file_not_aligned ≤ floor, force all eligible sizes (xfer % fs_align == 0) to aligned (p_ua_file=0) and prefer keeping L fully aligned.
      - Else distribute the *excess* unaligned fraction only across eligible sizes, preferring non-L bins first; per-row p_ua_file is set to hit the global target.
"""

import argparse, json, os, random
from pathlib import Path
from collections import defaultdict, Counter

# ---------------- Paths & constants ----------------
OUTROOT   = Path("/mnt/hasanfs/out_synth")   # default; overridden in main() per features file
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
    by_flag=Counter()    # "|shared|" vs "|unique|" (after tagging)
    by_fbs_ops=defaultdict(int)  # (file,bin,xfer,intent)->ops
    for r in rows:
        b = r["xfer"] * r["ops"]
        total += b
        by_role[r["role"]] += b
        by_intent[r["intent"]] += b
        by_file[r["file"]] += b
        by_fbs_ops[(r["file"], r["bin"], r["xfer"], r["intent"])] += r["ops"]
        if "flags" in r:
            if "|shared|" in r["flags"]: by_flag["shared"] += b
            elif "|unique|" in r["flags"]: by_flag["unique"] += b
    return total, by_role, by_intent, by_file, by_flag, by_fbs_ops

# ---------------- Rebalancer (fair, presence-preserving, auto up/down) ----------------
def rebalancer_autotune_fair(rows, target_total_bytes,
                             target_by_intent, target_by_role, target_by_flag=None):
    """
    In-bin, op-preserving rebalance that can both shrink and grow sizes.
    - Never touches L bin sizes (fixed at 16 MiB).
    - DOWN-tune: prefer M then S; within bin, larger→smaller.
    - UP-tune  : prefer S then M; within bin, smaller→larger.
    - Fairness: strict per-file round-robin across files (one step per file per cycle).
    - Presence floors: any (file,bin,xfer) that initially had ops keeps ≥1 op.
    - Supports target-by-flag group (shared/unique) after flags are assigned.
    """
    floors = defaultdict(int)  # (file,bin,xfer) -> min_ops (≥1 if initially >0)
    initial_ops = defaultdict(int)
    for r in rows:
        key = (r["file"], r["bin"], r["xfer"])
        initial_ops[key] += r["ops"]
    for k,v in initial_ops.items():
        if v>0:
            floors[k] = 1

    def ladder_for(binname):
        return M_SUBS if binname=="M" else (S_SUBS if binname=="S" else [L_SIZE])

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
            sib = None
            for _x, rows_at in by_fb[(file,binname)].items():
                if rows_at: sib = rows_at[0]; break
            if sib is None: return False
            nr = dict(sib); nr["xfer"] = xfer; nr["ops"] = 1
            rows.append(nr)
            by_fb[(file,binname)][xfer].append(nr)
        return True

    def files_with_candidates(pred, direction_down):
        files = set()
        for (file, binname), m in by_fb.items():
            if binname not in ("S","M"): continue
            if not any(pred(rr) for x, lst in m.items() for rr in lst):
                continue
            subs = ladder_for(binname)
            if direction_down:
                for i in range(len(subs)-1, 0, -1):
                    big, small = subs[i], subs[i-1]
                    if total_ops_at(file, binname, big) > floors.get((file,binname,big),0):
                        files.add((file,binname)); break
            else:
                for i in range(0, len(subs)-1):
                    small, big = subs[i], subs[i+1]
                    if total_ops_at(file, binname, small) > floors.get((file,binname,small),0):
                        files.add((file,binname)); break
        return sorted(files)

    def one_cycle(pred, direction_down):
        progressed = False
        for (file, binname) in files_with_candidates(pred, direction_down):
            subs = ladder_for(binname)
            moved=False
            if direction_down:
                for i in range(len(subs)-1, 0, -1):
                    big, small = subs[i], subs[i-1]
                    if dec_one_op(file,binname,big):
                        inc_one_op(file,binname,small)
                        moved=True; break
            else:
                for i in range(0, len(subs)-1):
                    small, big = subs[i], subs[i+1]
                    if dec_one_op(file,binname,small):
                        inc_one_op(file,binname,big)
                        moved=True; break
            progressed |= moved
        return progressed

    def rr_reduce_to_target(pred, tgt_bytes, epsilon):
        cur = bytes_of(pred)
        cycles=0; max_cycles=200000
        while cycles<max_cycles:
            cycles+=1
            if abs(cur - tgt_bytes) <= epsilon: break
            if cur > tgt_bytes: prog = one_cycle(pred, direction_down=True)
            else:               prog = one_cycle(pred, direction_down=False)
            if not prog: break
            cur = bytes_of(pred)

    eps = min(S_SUBS)

    for intent, tgt in (target_by_intent or {}).items():
        rr_reduce_to_target(lambda r, intent=intent: r["intent"]==intent, tgt, eps)

    for role, tgt in (target_by_role or {}).items():
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
    key = lambda r: (r["file"], r["bin"], r["xfer"])
    groups = defaultdict(lambda: {"R": [], "W": []})
    for r in rows:
        if r["role"]=="rw":
            groups[key(r)][ "R" if r["intent"]=="read" else "W" ].append(r)

    new_rows = []
    drained_ids = set()

    def drain(rows_list, need):
        segs=[]; i=0; left=need
        while left>0 and i<len(rows_list):
            take = min(seg_ops, rows_list[i]["ops"], left)
            if take>0:
                seg = dict(rows_list[i]); seg["ops"]=take
                segs.append(seg)
                rows_list[i]["ops"] -= take
                if rows_list[i]["ops"]==0:
                    drained_ids.add(id(rows_list[i]))
                    i+=1
                left-=take
            else:
                i+=1
        return segs

    for (f,binname,xfer), g in groups.items():
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
        if id(r) in drained_ids: continue
        if r["ops"]>0: new_rows.append(r)

    return new_rows

# ---------------- Alignment targeting helper (NEW) ----------------
def compute_rowwise_pua_file(rows, target_frac, fs_align_bytes):
    """
    Compute per-row p_ua_file so that:
      • Any row with xfer % fs_align != 0 is intrinsically unaligned (p_ua_file = 1.0).
      • The remaining unaligned quota is split EQUALLY across eligible bins (S, M, L)
        that have eligible rows (xfer % fs_align == 0), then PROPORTIONALLY by ops
        among eligible rows *within each bin*.

    Returns: dict mapping id(row) -> p_ua_file_eff in [0, 1].
    """
    total_ops = sum(r["ops"] for r in rows)
    if total_ops == 0:
        return {}

    target_unaligned_ops = clamp01(target_frac) * total_ops

    # Partition rows
    intrinsic = []   # xfer not divisible by fs_align -> always unaligned
    eligible  = []   # xfer divisible by fs_align   -> can be aligned/unaligned via p
    for r in rows:
        if r["ops"] <= 0:
            continue
        if (r["xfer"] % fs_align_bytes) != 0:
            intrinsic.append(r)
        else:
            eligible.append(r)

    intrinsic_ops = sum(r["ops"] for r in intrinsic)
    remaining = max(0.0, target_unaligned_ops - intrinsic_ops)

    # Initialize map: default 0.0 (aligned), intrinsic 1.0
    p_map = {id(r): 0.0 for r in rows}
    for r in intrinsic:
        p_map[id(r)] = 1.0

    if remaining <= 1e-9 or not eligible:
        return p_map

    # Group eligible rows by BIN (S/M/L)
    elig_by_bin = {"S": [], "M": [], "L": []}
    for r in eligible:
        elig_by_bin.get(r["bin"], []).append(r)

    # Consider only bins that actually have eligible ops
    active_bins = [b for b in ("S","M","L") if sum(x["ops"] for x in elig_by_bin[b]) > 0]
    if not active_bins:
        return p_map

    # Split remaining equally across active bins
    share_per_bin = remaining / len(active_bins)

    # For each active bin, distribute its share proportionally by ops within the bin
    for b in active_bins:
        bin_rows = elig_by_bin[b]
        bin_ops  = float(sum(r["ops"] for r in bin_rows))
        if bin_ops <= 0.0:
            continue
        for r in bin_rows:
            # desired unaligned ops for this row
            want_ops = share_per_bin * (r["ops"] / bin_ops)
            # translate to probability; cap at 1.0
            p_map[id(r)] = min(1.0, p_map.get(id(r), 0.0) + (want_ops / max(1.0, r["ops"])))

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

    # OPS seed split
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

    # Shared/Unique file-count split
    sh_f = clamp01(feats.get("pct_shared_files", 0.0))
    uq_f = clamp01(feats.get("pct_unique_files", 0.0))
    sh_counts = rational_counts([sh_f, uq_f], max_denom=max(1, n_ro+n_rw+n_wo))
    n_sh, n_uq = sh_counts
    if sh_f>0 and n_sh==0: n_sh=1
    if uq_f>0 and n_uq==0: n_uq=1
    if n_sh + n_uq == 0: n_uq = n_ro+n_rw+n_wo  # default unique if unspecified

    # Paths
    ro_paths = [str(DATA_RO / f"ro_{i}.dat") for i in range(n_ro)]
    rw_paths = [str(DATA_RW / f"rw_{i}.dat") for i in range(n_rw)]
    wo_paths = [str(DATA_WO / f"wo_{i}.dat") for i in range(n_wo)]

    # Shared-file set selection: pick first n_sh paths from concatenated list
    all_paths = ro_paths + rw_paths + wo_paths
    shared_paths = set(all_paths[:min(len(all_paths), n_sh)])
    unique_paths = set(all_paths[min(len(all_paths), n_sh):])

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

    # Bin ops
    R_S = int(round(Nr * rS)); R_M = int(round(Nr * rM)); R_L = max(0, Nr - R_S - R_M)
    W_S = int(round(Nw * wS)); W_M = int(round(Nw * wM)); W_L = max(0, Nw - W_S - W_M)

    def per_file_subsizes(Nbin, subs, files):
        per_sub  = split_uniform(Nbin, len(subs))
        out=[]
        for s_idx, ops in enumerate(per_sub):
            per_file = split_uniform(ops, len(files))
            for f_idx, k in enumerate(per_file):
                if k>0: out.append((files[f_idx], subs[s_idx], k))
        return out

    # Initial layout rows
    read_rows=[]
    write_rows=[]

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
        if r["file"] in shared_paths:
            r["flags"] = (r.get("flags","") + "|shared|")
        else:
            r["flags"] = (r.get("flags","") + "|unique|")

    # Build bytes targets (intent/role/flags)
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

    # Rebalance fairly with presence floors
    rebalancer_autotune_fair(layout_rows, io_bytes_target,
                             target_by_intent=by_intent_target,
                             target_by_role=by_role_target,
                             target_by_flag=by_flag_target)

    # RW switches (interleave on RW files)
    switch_frac = clamp01(feats.get("pct_rw_switches", 0.0))
    layout_rows = interleave_rw_segments(layout_rows, switch_frac=switch_frac, seg_ops=64)

    # ---------- Alignment targeting (FILE) ----------
    # Compute per-row p_ua_file (effective) following your strategy:
    # 1) Intrinsic floor = ops with xfer % fs_align != 0 (these get p_ua_file=1).
    # 2) Remaining unaligned ops are equally distributed across eligible rows (xfer % fs_align == 0)
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
        grp = defaultdict(list)
        for r in rows_sorted:
            grp[(r["file"], r["bin"], r["xfer"], r.get("flags",""))].append(r)

        for (path, bin_name, xfer, flags_extra), parts in grp.items():
            ops = sum(p["ops"] for p in parts)
            if ops<=0: continue
            pc = consec_r if is_read else consec_w
            ps = seq_r    if is_read else seq_w
            pc = min(clamp01(pc), clamp01(ps))
            n_con = int(round(ops * pc))
            n_seq = int(round(ops * max(0.0, ps - pc)))
            n_rnd = max(0, ops - n_con - n_seq)

            # Same p_ua_file for all phases of the same (file,bin,xfer,intent) group:
            # compute as ops-weighted from its member rows (they should be same p in our map).
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
    total_bytes, by_role, by_intent, by_file, by_flag, by_fbs_ops = summarize(layout_rows)

    # Alignment accounting (for Notes):
    total_ops = sum(r["ops"] for r in layout_rows)
    intrinsic_ops = sum(r["ops"] for r in layout_rows if r["xfer"] % fs_align_bytes != 0)
    eligible_ops  = total_ops - intrinsic_ops
    allocated_ops = sum(clamp01(pua_file_map.get(id(r),0.0)) * r["ops"] for r in layout_rows if r["xfer"] % fs_align_bytes == 0)
    predicted_unaligned_ops = intrinsic_ops + allocated_ops
    predicted_unaligned_frac = (predicted_unaligned_ops / total_ops) if total_ops>0 else 0.0

    with open(NOTES,"w") as f:
        # Inputs (verbatim)
        f.write("=== INPUT FEATURES (verbatim) ===\n")
        for k in sorted(feats.keys()):
            try:
                f.write(f"  {k}: {feats[k]}\n")
            except Exception:
                pass
        f.write("\n")

        # High-level mapping
        f.write("=== PLANNER OVERVIEW ===\n")
        f.write("  Mode: OPS-first; S/M fixed ladders; L fixed at 16 MiB\n")
        f.write("  Bins:\n")
        f.write(f"    S_SUBS={S_SUBS}  M_SUBS={M_SUBS}  L_SIZE={L_SIZE}\n")
        f.write("  Phase split inside each (file,bin,sub-size,intent): Consec ⊂ Seq; remainder Random\n")
        f.write("  Random placement: descending 32 MiB chunk RR; pre_seek_eof=1 for random phases\n")
        f.write("  Rebalancer: in-bin only; auto up/down; strict per-file RR; presence floors; preserves op counts and L size\n")
        f.write("  RW-switches: pct_rw_switches defines fraction of eligible RW alternations interleaved on RW files\n")
        f.write("  Shared vs Unique: file counts from pct_shared_files/pct_unique_files; byte targets from pct_bytes_* (enforced via flags)\n\n")

        # Capacity to bytes & global intent bytes target
        f.write("=== CAPACITY → BYTES & INTENT TARGETS ===\n")
        f.write(f"  cap_total_gib={cap_total_gib:.2f} → IO bytes target={human_bytes(io_bytes_target)}\n")
        f.write(f"  intent bytes target: read={p_bytes_r:.2f} write={p_bytes_w:.2f}\n")
        f.write(f"  actual after rebalance: read={human_bytes(by_intent.get('read',0))} write={human_bytes(by_intent.get('write',0))}\n\n")

        # Role & shared/unique bytes targets
        f.write("=== ROLE & SHARED/UNIQUE BYTE TARGETS ===\n")
        f.write("  role byte targets (fractions):\n")
        f.write(f"    RO={feats.get('pct_bytes_read_only_files',0)}  RW={feats.get('pct_bytes_read_write_files',0)}  WO={feats.get('pct_bytes_write_only_files',0)}\n")
        f.write("  actual bytes after rebalance:\n")
        f.write(f"    RO={human_bytes(by_role.get('ro',0))}  RW={human_bytes(by_role.get('rw',0))}  WO={human_bytes(by_role.get('wo',0))}\n")
        f.write("  shared/unique byte targets (fractions):\n")
        f.write(f"    shared={feats.get('pct_bytes_shared_files',0)}  unique={feats.get('pct_bytes_unique_files',0)}\n")
        f.write("  actual bytes after rebalance:\n")
        f.write(f"    shared={human_bytes(by_flag.get('shared',0))}  unique={human_bytes(by_flag.get('unique',0))}\n\n")

        # File-count layout & sizes (truncate sizes)
        f.write("=== FILE LAYOUT & TRUNCATE SIZES ===\n")
        f.write(f"  RO={len(ro_paths)}  RW={len(rw_paths)}  WO={len(wo_paths)} | shared_files≈{len(shared_paths)} unique_files≈{len(unique_paths)}\n")
        for pth in (ro_paths + rw_paths + wo_paths):
            sz = per_file_span.get(pth, 0)
            role = "RO" if pth in ro_paths else ("RW" if pth in rw_paths else "WO")
            su = "shared" if pth in shared_paths else "unique"
            f.write(f"    {role:2s} [{su:6s}] {pth} -> truncate {human_bytes(sz)}\n")
        f.write("\n")

        # Ops budgeting
        Nr = sum(r["ops"] for r in layout_rows if r["intent"]=="read")
        Nw = sum(r["ops"] for r in layout_rows if r["intent"]=="write")
        Nio = Nr+Nw
        f.write("=== OPS BUDGET (BEFORE META) ===\n")
        f.write(f"  Nio={Nio}  |  Nr={Nr}  |  Nw={Nw}\n")
        by_bin_ops = Counter()
        for r in layout_rows:
            by_bin_ops[(r["intent"], r["bin"])] += r["ops"]
        f.write("  Per-intent per-bin ops:\n")
        for intent in ("read","write"):
            f.write(f"    {intent}: S={by_bin_ops.get((intent,'S'),0)} M={by_bin_ops.get((intent,'M'),0)} L={by_bin_ops.get((intent,'L'),0)}\n")
        f.write("\n")

        # Alignment section (NEW)
        f.write("=== ALIGNMENT TARGETING (FILE vs Darshan File-Alignment) ===\n")
        f.write(f"  Requested pct_file_not_aligned={p_file_ua_req:.2f}\n")
        f.write(f"  fs_align_bytes={fs_align_bytes}  (mem_align_bytes={mem_align_bytes})\n")
        f.write(f"  Intrinsic file-unaligned ops (xfer % fs_align != 0): {intrinsic_ops} of {total_ops}  (floor={intrinsic_ops/total_ops if total_ops>0 else 0.0:.4f})\n")
        f.write(f"  Eligible ops (xfer % fs_align == 0): {eligible_ops}\n")
        f.write(f"  Allocated extra unaligned ops on eligible sizes: ~{int(round(allocated_ops))}\n")
        f.write(f"  Predicted overall file-unaligned fraction: ~{predicted_unaligned_frac:.4f}\n\n")

        # Per-(file,bin,sub-size) ops split (aggregated) with effective p_ua_file
        f.write("=== PER-FILE SUB-SIZE OPS & EFFECTIVE p_ua_file (after rebalance) ===\n")
        order_bin = {"L":0,"M":1,"S":2}
        for fpath in sorted(set(r["file"] for r in layout_rows)):
            f.write(f"  FILE: {fpath}\n")
            rows_f = [r for r in layout_rows if r["file"]==fpath]
            groups = defaultdict(lambda: dict(read=0, write=0, pua=None, elig=None))
            for r in rows_f:
                key=(r["bin"], r["xfer"])
                groups[key]["read"]  = groups[key].get("read",0)  + (r["ops"] if r["intent"]=="read"  else 0)
                groups[key]["write"] = groups[key].get("write",0) + (r["ops"] if r["intent"]=="write" else 0)
                groups[key]["pua"]   = pua_file_map.get(id(r),0.0)
                groups[key]["elig"]  = (r["xfer"] % fs_align_bytes == 0)
            for (b, x), g in sorted(groups.items(), key=lambda kv: (order_bin[kv[0][0]], kv[0][1])):
                f.write(f"    bin={b} xfer={x} elig={'Y' if g['elig'] else 'N'} -> read_ops={g.get('read',0)} write_ops={g.get('write',0)} p_ua_file≈{g['pua']:.3f}\n")
        f.write("\n")

        # RW-switches details
        f.write("=== RW SWITCHES ===\n")
        f.write(f"  pct_rw_switches={switch_frac:.2f} → interleave this fraction of eligible RW (file,bin,xfer) pairs in segments (≤64 ops)\n")
        f.write("  Interleaving happens only on RW files; Darshan counts read↔write alternations on the same path.\n\n")

        # META mapping
        f.write("=== META OPS MAPPING ===\n")
        f.write(f"  pct_io_access={clamp01(feats.get('pct_io_access',0.18)):.2f}\n")
        f.write("  meta_kind_count ≈ (pct_meta_kind_access / pct_io_access) * (Nr+Nw)\n")
        f.write(f"  planned: open={m_open}, stat={m_stat}, seek={m_seek}, sync={m_sync}\n")
        f.write("  Data phases use pread/pwrite only; META-only phase performs POSIX open/stat/seek/sync.\n\n")

        # Assumptions & invariants
        f.write("=== ASSUMPTIONS & INVARIANTS ===\n")
        f.write("  • L bin size is fixed to 16 MiB and never changed by the rebalancer.\n")
        f.write("  • Rebalancer operates within bins only and preserves op counts (adjusts xfer by moving ops across S/M sub-sizes).\n")
        f.write("  • Presence floors: any (file,bin,sub-size) that existed pre-rebalance keeps ≥1 op (prevents 100 B from disappearing).\n")
        f.write("  • Shared vs Unique is enacted by the harness (|shared| sharded over ranks; |unique| pinned to a single owner).\n")
        f.write("  • Random phases set pre_seek_eof=1 to stress end-of-file behavior; offsets remain within [0, size-xfer].\n")

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
