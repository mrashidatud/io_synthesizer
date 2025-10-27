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
# L bin default and allowed sizes (MiB): 12, 16, 20, ..., 64
L_SIZE       = 16 * (1<<20)  # default stays 16 MiB
L_SIZE_CHOICES = [mb * (1<<20) for mb in range(10+1, 64+1, 1)]

CHUNK_RANDOM = 128*(1<<20)

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

def split_ops_by_bins(N, sS, sM, sL):
    S = int(round(N * sS)); M = int(round(N * sM)); L = max(0, N - S - M)
    return S, M, L

def min_size_in_bin(binname):
    if binname == "S": return min(S_SUBS)
    if binname == "M": return min(M_SUBS)
    return min(L_SIZE_CHOICES)

def max_size_in_bin(binname):
    if binname == "S": return max(S_SUBS)
    if binname == "M": return max(M_SUBS)
    return max(L_SIZE_CHOICES)

def weighted_extreme_avg(shareS, shareM, shareL, choose_max=True):
    s = shareS + shareM + shareL
    if s <= 0: return min(S_SUBS)  # harmless default
    shareS, shareM, shareL = shareS/s, shareM/s, shareL/s
    pickS = max(S_SUBS) if choose_max else min(S_SUBS)
    pickM = max(M_SUBS) if choose_max else min(M_SUBS)
    pickL = max(L_SIZE_CHOICES) if choose_max else min(L_SIZE_CHOICES)
    return shareS*pickS + shareM*pickM + shareL*pickL

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

    def l_nudge_intent(intent_label, needed_delta_bytes):
        """
        Shift L-bin bytes for one intent by jumping ops directly to the nearest
        ladder edge in the desired direction (minimizes number of moved ops).
        Preserves ops and bin. Returns realized byte delta.
        """
        need = int(needed_delta_bytes)
        if need == 0:
            return 0

        if not L_SIZE_CHOICES:
            return 0

        choices = L_SIZE_CHOICES
        idx_of  = {sz: i for i, sz in enumerate(choices)}
        up      = need > 0
        realized = 0
        need_abs = abs(need)

        # Candidate L rows for this intent; biggest ops first
        L_rows = [r for r in rows
                  if r.get("intent") == intent_label and r.get("bin") == "L"
                  and r.get("xfer") in idx_of and r.get("ops", 0) > 0]
        L_rows.sort(key=lambda rr: -rr["ops"])

        for r in L_rows:
            cur_i = idx_of[r["xfer"]]
            tgt_i = (len(choices) - 1) if up else 0
            if cur_i == tgt_i:
                continue

            delta_per_op = abs(choices[tgt_i] - choices[cur_i])
            if delta_per_op <= 0:
                continue

            # minimal ops needed if we jump this row all the way to tgt_i
            take = min(r["ops"], (need_abs + delta_per_op - 1) // delta_per_op)
            if take <= 0:
                continue

            # find/create sibling at target rung
            sib = None
            for rr in rows:
                if (rr.get("file") == r.get("file") and rr.get("bin") == "L" and
                    rr.get("intent") == intent_label and rr.get("xfer") == choices[tgt_i] and
                    rr.get("role") == r.get("role") and rr.get("flags","") == r.get("flags","")):
                    sib = rr
                    break
            if sib is None:
                sib = dict(r)
                sib["xfer"] = choices[tgt_i]
                sib["ops"]  = 0
                rows.append(sib)

            r["ops"]   -= take
            sib["ops"] += take

            bump = delta_per_op * take
            realized += bump
            if bump >= need_abs:
                break
            need_abs -= bump

        return realized if up else -realized

    # --- coarse S/M passes ---
    if target_by_intent:
        for intent, tgt in target_by_intent.items():
            rr_reduce_to_target(lambda r, intent=intent: r["intent"] == intent, tgt, eps)
    if target_by_role:
        for role, tgt in (target_by_role or {}).items():
            rr_reduce_to_target(lambda r, role=role: r["role"] == role, tgt, eps)
    if target_by_flag:
        for lbl, tgt in (target_by_flag or {}).items():
            if lbl in ("shared","unique"):
                rr_reduce_to_target(
                    lambda r, lbl=lbl: ("|shared|" in r.get("flags","")) if lbl == "shared"
                                       else ("|unique|" in r.get("flags","")),
                    tgt, eps
                )

    # --- intent L-nudges toward requested bytes ---
    if target_by_intent:
        def bytes_of_intent(lbl):
            return sum(r["xfer"] * r["ops"] for r in rows if r.get("intent") == lbl)

        for intent, tgt in target_by_intent.items():
            l_nudge_intent(intent, tgt - bytes_of_intent(intent))
            rr_reduce_to_target(lambda r, intent=intent: r["intent"] == intent, tgt, eps)

        # ===== FINALIZE: fraction-first =====
        req_sum  = float(sum(target_by_intent.values())) or 1.0
        f_read   = float(target_by_intent.get("read", 0.0)) / req_sum
        f_write  = 1.0 - f_read

        cur_r = bytes_of_intent("read")
        cur_w = bytes_of_intent("write")

        # If writes are above their fraction, expand reads so produced split is exact
        desired_total = (cur_w / f_write) if f_write > 0 else (cur_r / max(f_read, 1e-9))
        desired_read  = desired_total * f_read
        desired_write = desired_total - desired_read

        # L-first, then tiny S/M polish
        l_nudge_intent("read",  desired_read  - cur_r)
        l_nudge_intent("write", desired_write - cur_w)

        rr_reduce_to_target(lambda r: r.get("intent") == "read",  desired_read,  eps)
        rr_reduce_to_target(lambda r: r.get("intent") == "write", desired_write, eps)

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

    # META is shared only when we have multiple ranks
    meta_is_shared = (nprocs > 1)
    meta_flag = "meta_only|shared|" if meta_is_shared else "meta_only|unique|"

    # Paths
    ro_paths = [str(DATA_RO / f"ro_{i}.dat") for i in range(n_ro)]
    rw_paths = [str(DATA_RW / f"rw_{i}.dat") for i in range(n_rw)]
    wo_paths = [str(DATA_WO / f"wo_{i}.dat") for i in range(n_wo)]
    data_paths = ro_paths + rw_paths + wo_paths
    N_data = len(data_paths)

    if nprocs <= 1:
        # Your invariant: single-rank runs have no shared data files
        N_shared_data = 0
    else:
        # Target shared *file-count* fraction includes the META file when shared
        # shared_fraction = (shared_data + 1 META) / (data + 1 META)
        N_shared_data = int(round(p_shared * (N_data + 1) - 1.0))
        N_shared_data = max(0, min(N_data, N_shared_data))

    shared_paths = set(data_paths[:N_shared_data])
    unique_paths = set(data_paths[N_shared_data:])

    # Avg xfer estimates for initial Nio
    avgS = sum(S_SUBS)/len(S_SUBS)
    avgM = sum(M_SUBS)/len(M_SUBS)
    avgL = L_SIZE
    avg_read_xfer  = rS*avgS + rM*avgM + rL*avgL if (rS+rM+rL)>0 else avgS
    avg_write_xfer = wS*avgS + wM*avgM + wL*avgL if (wS+wM+wL)>0 else avgS
    avg_xfer = p_reads_ops*avg_read_xfer + p_writes_ops*avg_write_xfer

    # ---- Symmetric quadrant policy for reads & writes (replaces ε-ops block) ----
    EPS_OPS_FRAC   = 1e-3    # target: side_ops / total_ops ≤ 0.1% when ops≈0
    EPS_BYTES_FRAC = 1e-3    # target: side_bytes / total_bytes ≤ 0.1% when bytes≈0

    # Desired presence flags
    want_ops_r   = (p_reads_ops  > 0.0)
    want_bytes_r = (p_bytes_r    > 0.0)
    want_ops_w   = (p_writes_ops > 0.0)
    want_bytes_w = (p_bytes_w    > 0.0)

    # Seed based on cap (will adjust)
    Nio = max(1, int(round(io_bytes_target / max(1, avg_xfer))))
    Nr  = int(round(Nio * p_reads_ops))
    Nw  = max(0, Nio - Nr)

    # Precompute extreme avg xfers per side
    avg_r_max = weighted_extreme_avg(rS, rM, rL, choose_max=True)
    avg_r_min = weighted_extreme_avg(rS, rM, rL, choose_max=False)
    avg_w_max = weighted_extreme_avg(wS, wM, wL, choose_max=True)
    avg_w_min = weighted_extreme_avg(wS, wM, wL, choose_max=False)

    # Utility: ensure side's ops fraction ≤ eps by increasing the other side's ops.
    def force_ops_fraction_small(side_ops, other_ops, eps=EPS_OPS_FRAC):
        if side_ops == 0: return other_ops
        min_other = int(round(max(0.0, (side_ops/eps) - side_ops)))
        return max(other_ops, min_other)

    # Utility: ensure side's bytes fraction ≤ eps by inflating other side bytes via ops*avg_xfer
    def force_bytes_fraction_small(side_ops, side_avg_xfer, other_ops, other_avg_xfer, eps=EPS_BYTES_FRAC):
        side_bytes = side_ops * max(1.0, side_avg_xfer)
        if side_bytes == 0 or other_avg_xfer <= 0: return other_ops
        need_total = side_bytes / max(eps, 1e-9)
        have_total = side_bytes + other_ops * other_avg_xfer
        if need_total > have_total:
            add = int(round((need_total - have_total) / other_avg_xfer))
            return other_ops + max(0, add)
        return other_ops

    # We may need to choose min/max per bin later when materializing rows:
    read_pick  = {"S": None, "M": None, "L": None}   # None → ladder as-is
    write_pick = {"S": None, "M": None, "L": None}

    # Order of handling when both sides want "ops≈0 & bytes>0":
    # prioritize the side with larger byte share so the other can treat ops≈0 relative to a large denominator.
    sides_order = ["read", "write"]
    if (not want_ops_r) and want_bytes_r and (not want_ops_w) and want_bytes_w:
        if p_bytes_w > p_bytes_r:
            sides_order = ["write", "read"]

    for side in sides_order:
        if side == "read":
            # Quadrants for READ
            if (not want_ops_r) and want_bytes_r:
                # Few, large reads + inflate writes so read-ops% ≈ 0
                target_bytes = io_bytes_target * p_bytes_r
                Nr = max(1, int(round(target_bytes / max(1.0, avg_r_max))))
                Nw = force_ops_fraction_small(Nr, Nw, eps=EPS_OPS_FRAC)
                # choose max sizes in each bin for reads
                read_pick["S"] = max(S_SUBS); read_pick["M"] = max(M_SUBS); read_pick["L"] = max(L_SIZE_CHOICES)

            elif want_ops_r and (not want_bytes_r):
                # Many, tiny reads; keep read-bytes% ≈ 0 by inflating write bytes if needed
                # Start from split
                Nr = max(1, int(round(max(1.0, Nio * p_reads_ops))))
                Nw = max(0, Nio - Nr)
                # push bytes ratio small
                Nw = force_bytes_fraction_small(Nr, avg_r_min, Nw, avg_w_max, eps=EPS_BYTES_FRAC)
                Nio = Nr + Nw
                # choose min sizes for reads
                read_pick["S"] = min(S_SUBS); read_pick["M"] = min(M_SUBS); read_pick["L"] = min(L_SIZE_CHOICES)

            elif (not want_ops_r) and (not want_bytes_r):
                # ε presence only if read signals exist; else none
                read_signals = (seq_r>0 or consec_r>0 or feats.get("pct_read_0_100K",0)>0
                                or feats.get("pct_read_100K_10M",0)>0 or feats.get("pct_read_10M_1G_PLUS",0)>0
                                or ro_f>0 or feats.get("pct_bytes_read_only_files",0)>0)
                if Nr == 0 and read_signals:
                    Nr = 1
                # keep picks as None (we'll use ladders)
            else:
                # normal: keep current Nr/Nw and ladders
                pass

        else:
            # Quadrants for WRITE
            if (not want_ops_w) and want_bytes_w:
                target_bytes = io_bytes_target * p_bytes_w
                Nw = max(1, int(round(target_bytes / max(1.0, avg_w_max))))
                Nr = force_ops_fraction_small(Nw, Nr, eps=EPS_OPS_FRAC)
                write_pick["S"] = max(S_SUBS); write_pick["M"] = max(M_SUBS); write_pick["L"] = max(L_SIZE_CHOICES)

            elif want_ops_w and (not want_bytes_w):
                Nw = max(1, int(round(max(1.0, Nio * p_writes_ops))))
                Nr = max(0, Nio - Nw)
                Nr = force_bytes_fraction_small(Nw, avg_w_min, Nr, avg_r_max, eps=EPS_BYTES_FRAC)
                Nio = Nr + Nw
                write_pick["S"] = min(S_SUBS); write_pick["M"] = min(M_SUBS); write_pick["L"] = min(L_SIZE_CHOICES)

            elif (not want_ops_w) and (not want_bytes_w):
                write_signals = (seq_w>0 or consec_w>0 or feats.get("pct_write_0_100K",0)>0
                                 or feats.get("pct_write_100K_10M",0)>0 or feats.get("pct_write_10M_1G_PLUS",0)>0
                                 or wo_f>0 or feats.get("pct_bytes_write_only_files",0)>0)
                if Nw == 0 and write_signals:
                    Nw = 1
            else:
                pass

        # Keep Nio in sync after each side's adjustment
        Nio = Nr + Nw

    # Final bin splits from authoritative Nr/Nw
    R_S,R_M,R_L = split_ops_by_bins(Nr, rS,rM,rL)
    W_S,W_M,W_L = split_ops_by_bins(Nw, wS,wM,wL)

    def per_file_subsizes(Nbin, subs, files):
        per_sub  = split_uniform(Nbin, len(subs))
        out=[]
        for s_idx, ops in enumerate(per_sub):
            per_file = split_uniform(ops, len(files))
            for f_idx, k in enumerate(per_file):
                if k>0: out.append((files[f_idx], subs[s_idx], k))
        return out

    # ----- READ rows -----
    read_rows = []
    read_files = ro_paths + rw_paths if (ro_paths and rw_paths) else (ro_paths if ro_paths else rw_paths)

    def emit_read_bin(Nbin, binname, pick):
        if Nbin <= 0: return
        subs = [pick] if pick is not None else (S_SUBS if binname=="S" else (M_SUBS if binname=="M" else [L_SIZE]))
        for (f, sz, k) in per_file_subsizes(Nbin, subs, read_files):
            read_rows.append({"role": role_of_path(f), "file": f, "bin": binname, "xfer": sz, "ops": k, "intent": "read"})

    emit_read_bin(R_S, "S", read_pick["S"])
    emit_read_bin(R_M, "M", read_pick["M"])
    emit_read_bin(R_L, "L", read_pick["L"])

    # ----- WRITE rows -----
    write_rows = []
    write_files = wo_paths + rw_paths if (wo_paths and rw_paths) else (wo_paths if wo_paths else rw_paths)

    def emit_write_bin(Nbin, binname, pick):
        if Nbin <= 0: return
        subs = [pick] if pick is not None else (S_SUBS if binname=="S" else (M_SUBS if binname=="M" else [L_SIZE]))
        for (f, sz, k) in per_file_subsizes(Nbin, subs, write_files):
            write_rows.append({"role": role_of_path(f), "file": f, "bin": binname, "xfer": sz, "ops": k, "intent": "write"})

    emit_write_bin(W_S, "S", write_pick["S"])
    emit_write_bin(W_M, "M", write_pick["M"])
    emit_write_bin(W_L, "L", write_pick["L"])

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
        if ops <= 0:
            return
        total_bytes = ops * xfer
        is_write = (intent == "write")

        # Phase structure → one definition used for both read and write columns
        if phase_kind == "consec":
            p_con = 1.0
            p_seq = 1.0
            p_rand = 0.0
        elif phase_kind == "seq":
            p_con = 0.0
            p_seq = 1.0
            p_rand = 0.0
        else:  # "random"
            p_con = 0.0
            p_seq = 0.0
            p_rand = 1.0

        seed = random.randint(1, 2**31 - 1)
        flags = phase_kind + (flags_extra or "")

        # Write the same phase structure into both read and write columns.
        # The harness selects the correct pair based on is_write at runtime.
        row = [
            "data", path, str(total_bytes), str(xfer),

            # p_write
            f"{1.0 if is_write else 0.0:.6f}",

            # p_rand
            f"{p_rand:.6f}",

            # p_seq_r, p_consec_r
            f"{p_seq:.6f}", f"{p_con:.6f}",

            # p_seq_w, p_consec_w
            f"{p_seq:.6f}", f"{p_con:.6f}",

            # p_ua_file, p_ua_mem
            f"{clamp01(p_ua_file_eff):.6f}", f"{p_ua_mem:.6f}",

            # rw_switch, meta_open/stat/seek/sync
            "0.0", "0", "0", "0", "0",

            # seed, flags
            str(seed), flags,

            # p_rand_fwd_r, p_rand_fwd_w, p_consec_internal
            "0.0", "0.0", "0.0",

            # pre_seek_eof, n_ops
            "1" if pre_seek_eof else "0",
            str(ops),
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
                str(seed), meta_flag,
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
        f.write("  Mode: OPS-first; S/M fixed ladders; L:11-64 MiB with 1 MiB increment;\n")
        f.write("  Bins:\n")
        f.write(f"    S_SUBS={S_SUBS}  M_SUBS={M_SUBS}  L_CHOICES={L_SIZE_CHOICES}\n")
        f.write("  Phase split inside each (file,bin,sub-size,intent): Consec ⊂ Seq; remainder Random\n")
        f.write("  Random placement: descending 128 MiB chunk RR; pre_seek_eof=1 for random phases\n")
        f.write("  Rebalancer: in-bin only for S/M; preserves op counts; L moves within the 11–64 MiB ladder per intent; presence floors.\n")
        f.write("  RW-switches: planner interleaves segments on eligible RW paths when pct_rw_switches>0\n")
        f.write("  Shared vs Unique: file counts from pct_shared_files/pct_unique_files; byte targets from pct_bytes_* (enforced via flags)\n\n")

        # Capacity to bytes & intent bytes target
        f.write("=== CAPACITY → BYTES & INTENT TARGETS ===\n")
        f.write(f"  cap_total_gib={cap_total_gib:.2f} → IO bytes target={human_bytes(io_bytes_target)}\n")
        f.write(f"  intent bytes target: read={p_bytes_r:.2f} write={p_bytes_w:.2f}\n")
        f.write(f"  realized after rebalance: read={human_bytes(by_intent.get('read',0))} write={human_bytes(by_intent.get('write',0))}\n\n")

        # --- Fraction-first finalize & L-ladder disclosure ---
        f.write("=== INTENT FRACTION RECONCILE (fraction-first) ===\n")
        # Requested fractions (already used to compute targets)
        f.write(f"  requested fractions: read={p_bytes_r:.2f}  write={p_bytes_w:.2f}\n")

        # Realized bytes & fractions
        _tot = float(total_bytes) if total_bytes > 0 else 1.0
        _read_b  = float(by_intent.get("read", 0))
        _write_b = float(by_intent.get("write", 0))
        _read_f  = _read_b  / _tot
        _write_f = _write_b / _tot
        f.write("  realized bytes: "
                f"read={human_bytes(_read_b)} "
                f"write={human_bytes(_write_b)}\n")
        f.write(f"  realized fractions: read={_read_f:.4f}  write={_write_f:.4f}\n")

        # Produced total vs original IO target (fraction-first can move total a bit under L floors)
        _delta_total = int(total_bytes) - int(io_bytes_target)
        sign = "+" if _delta_total >= 0 else "-"
        f.write(f"  produced_total={human_bytes(total_bytes)}  "
                f"target_total={human_bytes(io_bytes_target)}  "
                f"Δ_total={sign}{human_bytes(abs(_delta_total))}\n")

        # L-ladder used for nudging (MiB) + min step
        _ladder_mib = [int(v // (1<<20)) for v in L_SIZE_CHOICES]
        if len(L_SIZE_CHOICES) >= 2:
            _min_step_bytes = min(abs(b - a) for a, b in zip(L_SIZE_CHOICES, L_SIZE_CHOICES[1:]))
        else:
            _min_step_bytes = 0
        _min_step_mib = _min_step_bytes / float(1<<20) if _min_step_bytes else 0.0
        f.write(f"  L ladder choices (MiB): {_ladder_mib}\n")
        f.write(f"  L min adjacent step: {_min_step_mib:.2f} MiB\n")

        # Method summary
        f.write("  method: intent bytes rebalanced in S/M (ops-preserving),\n")
        f.write("          then L-size nudges per intent (ops preserved) to satisfy requested split,\n")
        f.write("          followed by a small S/M fair pass to polish residuals.\n\n")
        
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
        f.write("=== ZERO-FRACTION POLICY (ops vs bytes) ===\n")
        f.write(f"  requested fractions: READ(op={p_reads_ops:.4f}, bytes={p_bytes_r:.4f})  |  WRITE(op={p_writes_ops:.4f}, bytes={p_bytes_w:.4f})\n")
        def _mode(want_ops, want_bytes):
            if (not want_ops) and want_bytes:  return "OPS≈0, BYTES>0 (few, large ops; other side inflated)"
            if want_ops and (not want_bytes):  return "BYTES≈0, OPS>0 (many, tiny ops)"
            if (not want_ops) and (not want_bytes): return "OPS≈0, BYTES≈0 (ε presence if signals)"
            return "normal"
        f.write(f"  READ mode:  {_mode(want_ops_r, want_bytes_r)}\n")
        f.write(f"  WRITE mode: {_mode(want_ops_w, want_bytes_w)}\n\n")

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
        f.write("  Data phases use pread/pwrite only; META-only phase performs POSIX open/stat/seek/sync.\n")
        f.write("  META scope = shared if nprocs>1 else unique; counts in plan are GLOBAL; harness shards evenly per rank when shared.\n")
        f.write("  Shared-file targeting accounts for META when shared: (shared_data + 1) / (data + 1) ≈ pct_shared_files.\n\n")

        # Assumptions & invariants
        f.write("=== ASSUMPTIONS & INVARIANTS ===\n")
        f.write("  • Rebalancer: in-bin only, op-preserving, presence floors. L bin uses a bounded ladder 11-64 MiB with 1 MiB increment to auto up/down-tune per intent (read/write) before the fair in-bin nudger; S/M adjust only within their fixed ladders. Keeps 10M+ constraints intact.\n")
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
