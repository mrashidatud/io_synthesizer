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

import argparse, json, os, random, math
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
# L bin default and allowed sizes (MiB): 11, 12, 16, 20, ..., 64
L_SIZE       = 16 * (1<<20)  # default stays 16 MiB
L_SIZE_CHOICES = [mb * (1<<20) for mb in range(10+1, 64+1, 1)]

CHUNK_RANDOM = 128*(1<<20)

# --- Tiny-footprint policy knobs ---
EPS_OPS_ABS_MIN   = 64          # minimum ops to realize structure (>=64 avoids rounding collapses)
EPS_OPS_PER_FILE  = 4           # tiny ops per participating file (RW/WO for writes; RO/RW for reads)
EPS_BYTES_FRAC_CAP = 5e-4       # ≤ 0.05% of total bytes when minimizing bytes
EPS_OPS_FRAC_CAP   = 5e-4       # ≤ 0.05% of total ops   when minimizing ops

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

def rational_counts(fracs, max_denom=64):
    """
    Map fractional shares to small integer counts, preferring higher-denominator
    fits when error ties (so we can realize ratios like 7/1/6).
    Guarantees: if frac_i>0 → count_i>=1; if all fracs=0 → all counts=0.
    """
    fracs = [clamp01(float(x)) for x in fracs]
    s = sum(fracs)
    if s <= 0:
        return [0] * len(fracs)
    fracs = [x/s for x in fracs]

    best = None
    for denom in range(1, max_denom + 1):
        counts = [int(round(f * denom)) for f in fracs]
        # ensure presence for positive fracs
        for i, f in enumerate(fracs):
            if f > 0 and counts[i] == 0:
                counts[i] = 1
        total = sum(counts)
        if total == 0:
            continue
        approx = [c/total for c in counts]
        # primary metric: L1 error; tie-break 1) smaller L∞, 2) larger denom, 3) larger total
        l1 = sum(abs(a - b) for a, b in zip(approx, fracs))
        linf = max(abs(a - b) for a, b in zip(approx, fracs))
        cand = (l1, linf, -denom, -total, counts)
        if best is None or cand < best:
            best = cand
    return best[-1]

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

def ensure_rw_has_both(layout_rows, nprocs):
    """
    For any file with role 'rw', ensure both intents exist.
    If one intent is missing, split ops from the other intent
    (prefer the largest-ops row) to create a new row with >= min_seed ops.
    """
    from collections import defaultdict

    per_file = defaultdict(lambda: {"read": [], "write": []})
    for idx, r in enumerate(layout_rows):
        if r["role"] == "rw":
            per_file[r["file"]][r["intent"]].append((idx, r))

    # Walk each RW file and fix missing side
    inserts = []
    decrements = []
    for f, sides in per_file.items():
        has_r = len(sides["read"]) > 0
        has_w = len(sides["write"]) > 0
        if has_r and has_w:
            continue

        missing = "read" if not has_r else "write"
        donor_side = "write" if missing == "read" else "read"
        donors = sides[donor_side]
        if not donors:
            # nothing to do; this file wasn't emitted as RW after all
            continue

        # pick the biggest donor row for this file
        donors_sorted = sorted(donors, key=lambda t: t[1]["ops"], reverse=True)
        di, drow = donors_sorted[0]
        if drow["ops"] <= 1:
            # can't steal anything meaningful
            continue

        seed_ops = max(1, min(nprocs, drow["ops"] // 2))
        # clone donor row, flip intent, set ops to seed_ops
        newrow = dict(drow)
        newrow["intent"] = missing
        newrow["ops"] = seed_ops
        if newrow.get("bin") in ("S","M","L"):
            newrow["total_bytes"] = newrow["xfer"] * newrow["ops"]

        # schedule: insert right next to donor so intra-file ordering stays tight
        inserts.append((di + 1, newrow))
        # decrement donor
        drow = dict(drow)
        drow["ops"] -= seed_ops
        if drow.get("bin") in ("S","M","L"):
            drow["total_bytes"] = drow["xfer"] * drow["ops"]
        decrements.append((di, drow))

    # apply edits (indices from original, so do decrements first)
    for di, drow in decrements:
        layout_rows[di] = drow
    offset = 0
    for ins_idx, newrow in sorted(inserts, key=lambda t: t[0]):
        layout_rows.insert(ins_idx + offset, newrow)
        offset += 1

    return layout_rows

def make_file_weights_for_intent(feats, ro_paths, rw_paths, wo_paths, intent):
    """
    Return per-file weights for the given intent ('read' or 'write'),
    based on *byte* fractions across file roles. We normalize only over
    roles that actually have files, then divide each role's weight evenly
    across its files.
    """
    if intent == "read":
        # Read bytes can land on RO or RW files
        p_ro = clamp01(feats.get("pct_bytes_read_only_files", 0.0))
        p_rw = clamp01(feats.get("pct_bytes_read_write_files", 0.0))
        role_specs = [("RO", ro_paths, p_ro), ("RW", rw_paths, p_rw)]
    else:
        # Write bytes can land on RW or WO files
        p_rw = clamp01(feats.get("pct_bytes_read_write_files", 0.0))
        p_wo = clamp01(feats.get("pct_bytes_write_only_files", 0.0))
        role_specs = [("RW", rw_paths, p_rw), ("WO", wo_paths, p_wo)]

    # Keep only roles that actually have files
    avail = [(role, paths, frac) for (role, paths, frac) in role_specs if paths]

    # If nothing is available, return empty (caller will fallback)
    if not avail:
        return {}

    # Normalize fractions across available roles; if all zeros, split evenly
    s = sum(frac for _, _, frac in avail)
    if s > 0.0:
        w_roles = {role: (frac / s) for (role, _, frac) in avail}
    else:
        eq = 1.0 / float(len(avail))
        w_roles = {role: eq for (role, _, _) in avail}

    # Divide role weights evenly across that role's files
    weights = {}
    for role, paths, _ in avail:
        per = w_roles[role] / float(len(paths))
        for p in paths:
            weights[p] = per

    # Light guard: if something went wrong and all zeros, fall back to uniform
    if sum(weights.values()) <= 0.0:
        uni = 1.0 / float(sum(len(paths) for _, paths, _ in avail))
        for _, paths, _ in avail:
            for p in paths:
                weights[p] = uni

    return weights

def normalize_weights(paths, wdict):
    # return weights normalized to sum to 1.0 over the given paths
    vals = [max(0.0, float(wdict.get(p, 0.0))) for p in paths]
    s = sum(vals)
    if s <= 0:
        # uniform fallback
        return {p: 1.0/len(paths) for p in paths} if paths else {}
    return {p: v/s for p, v in zip(paths, vals)}

def choose_shared_paths_bytes_aware(
    data_paths, read_file_w, write_file_w,
    B_read_total, B_write_total,
    K, B_target_shared
):
    """
    Pick exactly K paths to mark as shared so that the sum of expected bytes
    is as close as possible to B_target_shared.
    """
    if K <= 0 or not data_paths:
        return set(), set(data_paths)
    if K >= len(data_paths):
        return set(data_paths), set()

    # Normalize weights on the full data_paths universe
    rw = normalize_weights(data_paths, read_file_w)
    ww = normalize_weights(data_paths, write_file_w)

    # Per-path expected bytes
    exp_bytes = {p: B_read_total * rw.get(p, 0.0) + B_write_total * ww.get(p, 0.0)
                 for p in data_paths}

    # --- Greedy initializer: take K items with largest exp_bytes if target is large,
    #     else K items closest to per-file target (B_target_shared / K).
    per_file_goal = (B_target_shared / K) if K > 0 else 0.0

    # Heuristic switch: if target is closer to taking the bulk, use top-K;
    # otherwise use "closest to per-file-goal".
    total_exp = sum(exp_bytes.values())
    if B_target_shared >= 0.5 * total_exp:
        cand = sorted(data_paths, key=lambda p: exp_bytes[p], reverse=True)[:K]
    else:
        cand = sorted(data_paths, key=lambda p: abs(exp_bytes[p] - per_file_goal))[:K]

    S = set(cand)                         # shared set
    U = set(data_paths) - S               # unique set
    sum_S = sum(exp_bytes[p] for p in S)

    # --- Small 2-opt improvement: swap one in S with one in U if it reduces error
    target = B_target_shared
    best_err = abs(sum_S - target)
    improved = True
    # Limit passes to keep this cheap (K * |U| can be large)
    passes = 0
    MAX_PASSES = 3
    while improved and passes < MAX_PASSES:
        improved = False
        passes += 1
        # Try a few best candidates only (prune)
        S_sorted = sorted(S, key=lambda p: exp_bytes[p], reverse=True)[:min(len(S), 64)]
        U_sorted = sorted(U, key=lambda p: exp_bytes[p])[:min(len(U), 64)]
        for ps in S_sorted:
            for pu in U_sorted:
                new_sum = sum_S - exp_bytes[ps] + exp_bytes[pu]
                err = abs(new_sum - target)
                if err + 1e-9 < best_err:
                    # perform swap
                    S.remove(ps); U.add(ps)
                    U.remove(pu); S.add(pu)
                    sum_S = new_sum
                    best_err = err
                    improved = True
                    break
            if improved:
                break

    return S, U

def pick_bin_sizes_for_intent(intent, feats, S_SUBS, M_SUBS, L_CHOICES, minimize_bytes=True):
    """
    Return [(bin_label, xfer_bytes, bin_weight)] (weights sum to 1).
    Uses the pct_* bin signals. For minimize_bytes pick the smallest xfer in each
    signaled bin; otherwise pick the largest.
    """
    if intent == "write":
        bS = feats.get("pct_write_0_100K", 0.0)
        bM = feats.get("pct_write_100K_10M", 0.0)
        bL = feats.get("pct_write_10M_1G_PLUS", 0.0)
    else:
        bS = feats.get("pct_read_0_100K", 0.0)
        bM = feats.get("pct_read_100K_10M", 0.0)
        bL = feats.get("pct_read_10M_1G_PLUS", 0.0)

    vec = [("S", bS), ("M", bM), ("L", bL)]
    total = sum(v for _, v in vec)
    if total <= 0.0:
        # default to S if nothing is signaled
        vec = [("S", 1.0)]
        total = 1.0
    vec = [(k, v/total) for k, v in vec if v > 0.0]

    picks = []
    for k, w in vec:
        if k == "S":
            xfer = min(S_SUBS) if minimize_bytes else max(S_SUBS)
        elif k == "M":
            xfer = min(M_SUBS) if minimize_bytes else max(M_SUBS)
        else:
            xfer = min(L_CHOICES) if minimize_bytes else max(L_CHOICES)
        picks.append((k, int(xfer), w))
    return picks

def structure_counts(N, p_consec, p_seq):
    """
    Enforce Consec ⊂ Seq. Return (N_con, N_seq_only, N_rand).
    """
    p_con = max(0.0, min(1.0, float(p_consec)))
    p_sq  = max(0.0, min(1.0, float(p_seq)))
    if p_sq < p_con: p_sq = p_con
    N_con = int(round(N * p_con))
    N_seq = int(round(N * (p_sq - p_con)))
    if N_con > N: N_con = N
    if N_seq > (N - N_con): N_seq = N - N_con
    N_rand = N - N_con - N_seq
    # ε safeguard (donate one from rand if rounding killed a requested class)
    if p_sq > p_con and N_seq == 0 and N_rand > 0:
        N_seq += 1; N_rand -= 1
    if p_con > 0 and N_con == 0 and N_rand > 0:
        N_con += 1; N_rand -= 1
    return N_con, N_seq, N_rand

def plan_tiny_intent(intent, feats, Nio, target_total_bytes,
                     S_SUBS, M_SUBS, L_CHOICES,
                     file_roles_by_intent,
                     want_ops, want_bytes,
                     p_ops_target, p_bytes_target, file_weights, nprocs):
    """
    Build a tiny-but-structured footprint for one intent ('read'|'write').
    - If want_ops=False and want_bytes=False: minimize both (tiny N, smallest xfers)
    - If want_ops=False and want_bytes=True: hit bytes with few ops (large xfers, tiny N)
    - If want_ops=True and want_bytes=False: hit ops with tiny bytes (small xfers, larger N)
    - If want_ops=True and want_bytes=True: not used here; caller should use the normal path.

    Returns: (tiny_items, N_ops_total)
      where tiny_items is a list of dicts:
        { "path": str, "xfer": int, "con":int, "seq":int, "rnd":int, "role":"RO|RW|WO" }
    """
    # 0) structure targets
    if intent == "write":
        p_con, p_seq = feats.get("pct_consec_writes", 0.0), feats.get("pct_seq_writes", 0.0)
        pool_main = file_roles_by_intent["write"].get("RW", [])
        pool_alt  = file_roles_by_intent["write"].get("WO", [])
    else:
        p_con, p_seq = feats.get("pct_consec_reads", 0.0), feats.get("pct_seq_reads", 0.0)
        pool_main = file_roles_by_intent["read"].get("RO", [])
        pool_alt  = file_roles_by_intent["read"].get("RW", [])

    pool = (pool_main or []) + (pool_alt or [])
    if not pool:
        # planner guarantees files exist, but keep a safe guard
        pool = ["/dev/null"]

    # 1) decide bin sizes
    if not want_ops and want_bytes:
        # bytes-priority → choose largest sizes in signaled bins
        picks = pick_bin_sizes_for_intent(intent, feats, S_SUBS, M_SUBS, L_CHOICES, minimize_bytes=False)
    else:
        # ops-priority (or minimize both) → choose smallest sizes in signaled bins
        picks = pick_bin_sizes_for_intent(intent, feats, S_SUBS, M_SUBS, L_CHOICES, minimize_bytes=True)

    # 2) decide N_ops_tiny
    if want_ops and not want_bytes:
        # realize requested op fraction, bytes ≈ 0 thanks to tiny xfers
        N_ops_tiny = max(EPS_OPS_ABS_MIN, int(round(p_ops_target * float(Nio))))
    elif not want_ops and want_bytes:
        # realize requested bytes with minimal ops: pick a representative "large" xfer
        # Use the largest xfer among chosen picks as our dominant grain.
        xfer_max = max(x for (_, x, _) in picks)
        desired_bytes = int(round(p_bytes_target * float(target_total_bytes)))
        N_ops_tiny = max(1, int(math.ceil(desired_bytes / float(xfer_max))))
        # soft-cap ops (prefer increasing xfer via picks rather than growing ops)
        base_cap = int(math.floor(EPS_OPS_FRAC_CAP * float(Nio)))
        # ensure at least nprocs, but don't exceed Nio
        N_cap = max(1, min(Nio, max(nprocs, base_cap)))
        if N_ops_tiny > N_cap:
            N_ops_tiny = N_cap
    else:
        # minimize both (0/0 case) or both false → small but nonzero to keep structure alive
        base_cap = int(math.floor(EPS_OPS_FRAC_CAP * float(Nio)))
        # ensure at least nprocs, but don't exceed Nio
        N_cap = max(1, min(Nio, max(nprocs, base_cap)))
        # touch at least a few files
        n_files_tiny = max(1, min(len(pool), int(math.ceil(N_cap / max(EPS_OPS_PER_FILE,1)))))
        N_ops_tiny = min(max(EPS_OPS_ABS_MIN, EPS_OPS_PER_FILE * n_files_tiny), N_cap)

    # 3) split structure at whole-intent level
    N_con, N_seq, N_rand = structure_counts(N_ops_tiny, p_con, p_seq)

    # 4) apportion per bin by weight; last bin absorbs remainder
    rem_con, rem_seq, rem_rand = N_con, N_seq, N_rand
    bin_plan = []
    for j, (_, xfer, w) in enumerate(picks):
        if j < len(picks)-1:
            c = min(rem_con, int(round(N_con * w)))
            s = min(rem_seq, int(round(N_seq * w)))
            r = min(rem_rand, int(round(N_rand * w)))
        else:
            c, s, r = rem_con, rem_seq, rem_rand
        rem_con -= c; rem_seq -= s; rem_rand -= r
        if c+s+r > 0:
            bin_plan.append((xfer, c, s, r))

    # inside plan_tiny_intent, after pool is set:
    if file_weights:
        w_raw = [(p, max(0.0, float(file_weights.get(p, 0.0)))) for p in pool]
        s = sum(w for _, w in w_raw)
        if s <= 0:
            w_norm = [(p, 1.0/len(pool)) for p in pool]
        else:
            w_norm = [(p, w/s) for p, w in w_raw]
        # alias-free sampler: expand a schedule proportional to weights * total tiny ops
        # ensures at least 1 for any file with positive weight
        k_sched = [max(1 if w>0 else 0, int(round(w * (N_con+N_seq+N_rand)))) for _, w in w_norm]
        # adjust to match the exact total
        delta = (N_con+N_seq+N_rand) - sum(k_sched)
        if delta != 0:
            order = sorted(range(len(w_norm)), key=lambda i: w_norm[i][1], reverse=True)
            for i in (order if delta>0 else reversed(order)):
                if delta == 0: break
                if delta>0: k_sched[i]+=1; delta-=1
                elif k_sched[i]>0: k_sched[i]-=1; delta+=1
        weighted_cycle = []
        for (p,_), k in zip(w_norm, k_sched):
            weighted_cycle.extend([p]*k)
    else:
        weighted_cycle = pool[:]  # fallback round-robin
    if not weighted_cycle:
        weighted_cycle = pool[:]  # uniform fallback

    # 5) spread across files round-robin, preferring main pool first
    items = []
    fi = 0
    # Precompute a fast role lookup once (outside the loops):
    role_map = {}
    if intent == "read":
        for p in file_roles_by_intent["read"].get("RO", []): role_map[p] = "RO"
        for p in file_roles_by_intent["read"].get("RW", []): role_map[p] = "RW"
    else:
        for p in file_roles_by_intent["write"].get("RW", []): role_map[p] = "RW"
        for p in file_roles_by_intent["write"].get("WO", []): role_map[p] = "WO"
    for (xfer, c, s, r) in bin_plan:
        total = c+s+r
        for _ in range(total):
            path = weighted_cycle[fi % len(weighted_cycle)]
            role = role_map.get(path, ("RW" if intent=="read" else "WO")) 
            if c > 0:
                items.append({"path": path, "xfer": xfer, "con":1, "seq":0, "rnd":0, "role":role}); c -= 1
            elif s > 0:
                items.append({"path": path, "xfer": xfer, "con":0, "seq":1, "rnd":0, "role":role}); s -= 1
            else:
                items.append({"path": path, "xfer": xfer, "con":0, "seq":0, "rnd":1, "role":role}); r -= 1
            fi += 1

    # ---- soft-cap tiny bytes when we are minimizing bytes on this intent ----
    tiny_bytes = sum(it["xfer"] * (it["con"] + it["seq"] + it["rnd"]) for it in items)
    bytes_cap  = int(EPS_BYTES_FRAC_CAP * float(target_total_bytes))

    # We only enforce the hard cap when BOTH ops and bytes are being minimized.
    # If ops must be honored (want_ops=True, want_bytes=False), the cap is *soft*:
    # we try to trim, but if that would break structure/ops floors we stop early.
    need_hard_cap = (not want_ops) and (not want_bytes)

    if tiny_bytes > bytes_cap and (need_hard_cap or tiny_bytes > 2 * bytes_cap):
        # proportional downscale across items, preferring to remove from largest-xfer first
        items.sort(key=lambda z: z["xfer"], reverse=True)

        def take_one_from(it):
            # drop in the order: rnd -> seq -> con (preserve structure as much as possible)
            if it["rnd"] > 0: it["rnd"] -= 1; return it["xfer"]
            if it["seq"] > 0: it["seq"] -= 1; return it["xfer"]
            if it["con"] > 0: it["con"] -= 1; return it["xfer"]
            return 0

        # minimum structure floors: keep at least 1 op in any class that’s requested (>0 in features)
        want_con = (p_con > 0.0)
        want_seq = (p_seq > p_con)
        floors = {"con": 1 if want_con else 0,
                  "seq": 1 if want_seq else 0,
                  "rnd": 0}

        # enforce floors per (path,xfer) bucket
        from collections import defaultdict
        per_bucket = defaultdict(lambda: {"con":0,"seq":0,"rnd":0})
        for it in items:
            k = (it["path"], it["xfer"])
            per_bucket[k]["con"] += it["con"]
            per_bucket[k]["seq"] += it["seq"]
            per_bucket[k]["rnd"] += it["rnd"]

        # trimming loop
        target = bytes_cap if need_hard_cap else 2 * bytes_cap
        while tiny_bytes > target:
            progressed = False
            for it in items:
                k = (it["path"], it["xfer"])
                # check floors
                if it["con"] <= 0 and it["seq"] <= 0 and it["rnd"] <= 0:
                    continue
                # do not violate per-bucket floors
                if it["con"]   <= floors["con"] and it["seq"] == 0 and it["rnd"] == 0: continue
                if it["seq"]   <= floors["seq"] and it["con"] == 0 and it["rnd"] == 0: continue

                removed = take_one_from(it)
                if removed:
                    tiny_bytes -= removed
                    per_bucket[k]  # (not read—but kept for symmetry/future)
                    progressed = True
                    if tiny_bytes <= target:
                        break
            if not progressed:
                break  # stop early—structure/ops floors reached

        # drop empty items
        items = [it for it in items if (it["con"] + it["seq"] + it["rnd"]) > 0]

    return items, N_ops_tiny

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
    counts = rational_counts([ro_f, rw_f, wo_f], max_denom=64)
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

    roles_dict = {
        "read":  {"RO": list(ro_paths), "RW": list(rw_paths)},
        "write": {"RW": list(rw_paths), "WO": list(wo_paths)},
    }

    read_file_w  = make_file_weights_for_intent(feats, ro_paths, rw_paths, wo_paths, "read")
    write_file_w = make_file_weights_for_intent(feats, ro_paths, rw_paths, wo_paths, "write")

    # Shared file count target (respect your meta-is-shared rule)
    if nprocs <= 1:
        N_shared_data = 0
    else:
        # p_shared is the desired fraction of shared *files* (data+meta)
        # We want: (N_shared_data + (1 if meta_is_shared else 0)) / (N_data + 1) ≈ p_shared
        N_data = len(data_paths)
        N_shared_data = int(round(p_shared * (N_data + 1) - (1 if meta_is_shared else 0)))
        N_shared_data = max(0, min(N_data, N_shared_data))

    # Compute the actual totals you planned for each intent (before rebalancer tweaks)
    B_read_total  = int(round(io_bytes_target * p_bytes_r))
    B_write_total = int(round(io_bytes_target * p_bytes_w))

    # Shared *bytes* target over data files only (meta bytes ~ 0)
    B_total_data = B_read_total + B_write_total
    B_target_shared = clamp01(feats.get("pct_bytes_shared_files", 0.0)) * B_total_data

    shared_paths, unique_paths = choose_shared_paths_bytes_aware(
        data_paths,
        read_file_w, write_file_w,
        B_read_total, B_write_total,
        N_shared_data,
        B_target_shared
    )

    # Build flags map consumed by emit_* functions
    path_flags = {p: ("|shared|" if p in shared_paths else "|unique|") for p in data_paths}

    # Avg xfer estimates for initial Nio
    avgS = sum(S_SUBS)/len(S_SUBS)
    avgM = sum(M_SUBS)/len(M_SUBS)
    avgL = L_SIZE
    avg_read_xfer  = rS*avgS + rM*avgM + rL*avgL if (rS+rM+rL)>0 else avgS
    avg_write_xfer = wS*avgS + wM*avgM + wL*avgL if (wS+wM+wL)>0 else avgS
    avg_xfer = p_reads_ops*avg_read_xfer + p_writes_ops*avg_write_xfer

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

    # We may need to choose min/max per bin later when materializing rows:
    read_pick  = {"S": None, "M": None, "L": None}   # None → ladder as-is
    write_pick = {"S": None, "M": None, "L": None}

    # Order of handling when both sides want "ops≈0 & bytes>0":
    # prioritize the side with larger byte share so the other can treat ops≈0 relative to a large denominator.
    sides_order = ["read", "write"]
    extra_tiny_rows_read = []
    extra_tiny_rows_write = []
    if (not want_ops_r) and want_bytes_r and (not want_ops_w) and want_bytes_w:
        if p_bytes_w > p_bytes_r:
            sides_order = ["write", "read"]

    for side in sides_order:
        if side == "read":
            # Quadrants for READ
            if want_ops_r and want_bytes_r:
                # normal path (your existing logic)
                pass
            else:
                tiny_read_rows, Nr = plan_tiny_intent(
                    "read", feats, Nio, io_bytes_target,
                    S_SUBS, M_SUBS, L_SIZE_CHOICES,
                    roles_dict,
                    want_ops=want_ops_r,
                    want_bytes=want_bytes_r,
                    p_ops_target=p_reads_ops,         # feats.get("pct_reads")
                    p_bytes_target=p_bytes_r,          # feats.get("pct_byte_reads")
                    file_weights=read_file_w,
                    nprocs=nprocs
                )
                # Stash for coalescing later:
                extra_tiny_rows_read = tiny_read_rows

        else:
            # Quadrants for WRITE
            if want_ops_w and want_bytes_w:
                # normal path (your existing logic)
                pass
            else:
                tiny_write_rows, Nw = plan_tiny_intent(
                    "write", feats, Nio, io_bytes_target,
                    S_SUBS, M_SUBS, L_SIZE_CHOICES,
                    roles_dict,
                    want_ops=want_ops_w,
                    want_bytes=want_bytes_w,
                    p_ops_target=p_writes_ops,        # feats.get("pct_writes")
                    p_bytes_target=p_bytes_w,          # feats.get("pct_byte_writes")
                    file_weights=write_file_w,
                    nprocs=nprocs
                )
                extra_tiny_rows_write = tiny_write_rows

        # Keep Nio in sync after each side's adjustment
        Nio = Nr + Nw

    # Final bin splits from authoritative Nr/Nw
    R_S,R_M,R_L = split_ops_by_bins(Nr, rS,rM,rL)
    W_S,W_M,W_L = split_ops_by_bins(Nw, wS,wM,wL)

    def _int_round_allocation(total, weights_seq):
        # Proportional allocation with largest-remainder rounding
        raw = [total * w for w in weights_seq]
        base = [int(math.floor(x)) for x in raw]
        rem  = total - sum(base)
        if rem > 0:
            # assign remaining to the largest fractional parts
            order = sorted(range(len(raw)), key=lambda i: (raw[i] - base[i]), reverse=True)
            for i in order[:rem]:
                base[i] += 1
        return base

    def per_file_subsizes_weighted(Nbin, subs, files, file_weights):
        """
        Split Nbin ops across 'subs' (sizes) first (uniform by sub),
        then across files proportionally to 'file_weights' (same for each sub).
        Returns list of (file, sub_size, ops) triples (ops may be 0).
        """
        if Nbin <= 0 or not files:
            return []
        per_sub = split_uniform(Nbin, len(subs))
        # normalize weights for the participating files
        w = [max(0.0, float(file_weights.get(f, 0.0))) for f in files]
        s = sum(w)
        if s <= 0:
            w = [1.0/len(files)] * len(files)
        else:
            w = [x/s for x in w]
        out = []
        for sub_ops, sz in zip(per_sub, subs):
            per_file_ops = _int_round_allocation(sub_ops, w)
            for f, k in zip(files, per_file_ops):
                if k > 0:
                    out.append((f, sz, k))
        return out

    # ----- READ rows -----
    read_rows = []
    read_files = ro_paths + rw_paths if (ro_paths and rw_paths) else (ro_paths if ro_paths else rw_paths)

    def emit_read_bin(Nbin, binname, pick):
        if Nbin <= 0: return
        subs = [pick] if pick is not None else (S_SUBS if binname=="S" else (M_SUBS if binname=="M" else [L_SIZE]))
        files = list(read_file_w.keys())  # files that actually have weights
        triples = per_file_subsizes_weighted(Nbin, subs, files, read_file_w)
        for (f, sz, k) in triples:
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
        files = list(write_file_w.keys())
        triples = per_file_subsizes_weighted(Nbin, subs, files, write_file_w)
        for (f, sz, k) in triples:
            write_rows.append({"role": role_of_path(f), "file": f, "bin": binname, "xfer": sz, "ops": k, "intent": "write"})

    emit_write_bin(W_S, "S", write_pick["S"])
    emit_write_bin(W_M, "M", write_pick["M"])
    emit_write_bin(W_L, "L", write_pick["L"])

    layout_rows = read_rows + write_rows
    layout_rows = ensure_rw_has_both(layout_rows, nprocs=nprocs)

    # Tag shared/unique by FILE counts
    for r in layout_rows:
        if r["file"] in shared_paths: r["flags"] = (r.get("flags","") + "|shared|")
        else:                         r["flags"] = (r.get("flags","") + "|unique|")

    # ---------- Tiny bytes accounting (exclude tiny rows from rebalancer) ----------
    from collections import defaultdict

    def _tiny_stats(tiny_items, intent_label, path_flags):
        """Return tiny bytes split by intent/role/flag for subtraction from targets."""
        by_intent = defaultdict(int)
        by_role   = defaultdict(int)   # keys: 'ro','rw','wo'
        by_flag   = defaultdict(int)   # keys: 'shared','unique'
        total_b   = 0
        if not tiny_items:
            return total_b, by_intent, by_role, by_flag

        for it in tiny_items:
            ops = int(it.get("con",0)) + int(it.get("seq",0)) + int(it.get("rnd",0))
            if ops <= 0: 
                continue
            b = int(it["xfer"]) * ops
            total_b += b
            by_intent[intent_label] += b

            role_key = str(it.get("role","")).lower()  # "RO|RW|WO" → "ro|rw|wo"
            if role_key in ("ro","rw","wo"):
                by_role[role_key] += b

            fl = path_flags.get(it["path"], "|unique|")
            if "|shared|" in fl:
                by_flag["shared"] += b
            else:
                by_flag["unique"] += b

        return total_b, by_intent, by_role, by_flag

    # Tiny rows you collected earlier (may be empty lists)
    tiny_read_items  = locals().get("extra_tiny_rows_read",  [])
    tiny_write_items = locals().get("extra_tiny_rows_write", [])

    # Bytes already “spent” by tiny presence
    tr_tot, tr_int, tr_role, tr_flag = _tiny_stats(tiny_read_items,  "read",  path_flags)
    tw_tot, tw_int, tw_role, tw_flag = _tiny_stats(tiny_write_items, "write", path_flags)

    tiny_total_bytes = tr_tot + tw_tot

    # Original targets (you already compute these above)
    by_intent_target = {
        "read":  int(round(io_bytes_target * p_bytes_r)),
        "write": int(round(io_bytes_target * p_bytes_w)),
    }

    p_bytes_ro = clamp01(feats.get("pct_bytes_read_only_files", 0.0))
    p_bytes_rw = clamp01(feats.get("pct_bytes_read_write_files", 0.0))
    p_bytes_wo = clamp01(feats.get("pct_bytes_write_only_files", 0.0))
    s_role = p_bytes_ro + p_bytes_rw + p_bytes_wo
    by_role_target = {}
    if s_role > 0:
        by_role_target = {
            "ro": int(round(io_bytes_target * p_bytes_ro / s_role)),
            "rw": int(round(io_bytes_target * p_bytes_rw / s_role)),
            "wo": int(round(io_bytes_target * p_bytes_wo / s_role)),
        }

    p_bytes_sh = clamp01(feats.get("pct_bytes_shared_files", 0.0))
    p_bytes_uq = clamp01(feats.get("pct_bytes_unique_files", 0.0))
    s_su = p_bytes_sh + p_bytes_uq
    by_flag_target = None
    if s_su > 0:
        by_flag_target = {
            "shared": int(round(io_bytes_target * p_bytes_sh / s_su)),
            "unique": int(round(io_bytes_target * p_bytes_uq / s_su)),
        }

    # ----- Build residual targets (subtract tiny bytes; clamp at 0) -----
    res_intent = {
        "read":  max(0, by_intent_target.get("read", 0)  - tr_int.get("read", 0)),
        "write": max(0, by_intent_target.get("write", 0) - tw_int.get("write", 0)),
    }

    res_role = None
    if by_role_target:
        tiny_by_role = defaultdict(int)
        for k, v in tr_role.items(): tiny_by_role[k] += v
        for k, v in tw_role.items(): tiny_by_role[k] += v
        res_role = {k: max(0, by_role_target.get(k,0) - tiny_by_role.get(k,0)) for k in by_role_target.keys()}

    res_flag = None
    if by_flag_target:
        tiny_by_flag = defaultdict(int)
        for k, v in tr_flag.items(): tiny_by_flag[k] += v
        for k, v in tw_flag.items(): tiny_by_flag[k] += v
        res_flag = {k: max(0, by_flag_target.get(k,0) - tiny_by_flag.get(k,0)) for k in by_flag_target.keys()}

    # Optional: if the sum of residual intent targets drifts from the available normal bytes,
    # scale them proportionally (keeps rebalancer well-conditioned).
    normal_bytes = sum(r["xfer"]*r["ops"] for r in layout_rows)
    sum_res_intent = sum(res_intent.values())
    if sum_res_intent > 0 and normal_bytes > 0 and sum_res_intent != normal_bytes:
        scale = normal_bytes / float(sum_res_intent)
        res_intent = {k: int(round(v * scale)) for k, v in res_intent.items()}

    # ----- Rebalance ONLY normal rows to residual targets -----
    rebalancer_autotune_fair(
        layout_rows,
        target_total_bytes=normal_bytes,            # informational in your implementation
        target_by_intent=res_intent,
        target_by_role=res_role,
        target_by_flag=res_flag
    )

    # RW switches
    switch_frac = clamp01(feats.get("pct_rw_switches", 0.0))
    layout_rows = interleave_rw_segments(layout_rows, switch_frac=switch_frac, seg_ops=max(64, nprocs))
    
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

    def emit_tiny_items(tiny_items, intent, path_flags, p_ua_file_default, p_mem_ua):
        """
        tiny_items: list of {"path","xfer","con","seq","rnd","role"} from plan_tiny_intent()
        intent: "read"|"write"
        path_flags: dict path->flags suffix like "|unique|" or "|shared|"
        p_ua_file_default: float to use for p_ua_file (you already compute p_file_ua_req earlier)
        p_mem_ua: float (you already have p_mem_ua)
        """
        from collections import defaultdict
        # Coalesce identical (path,xfer,flags) to keep CSV compact
        agg = defaultdict(lambda: {"con":0, "seq":0, "rnd":0})
        for it in tiny_items:
            path = it["path"]
            xfer = int(it["xfer"])
            flags_extra = path_flags.get(path, "|unique|")  # fallback sensible default
            key = (path, xfer, flags_extra)
            a = agg[key]
            a["con"] += int(it.get("con", 0))
            a["seq"] += int(it.get("seq", 0))
            a["rnd"] += int(it.get("rnd", 0))

        # Emit three phases per (path,xfer,flags): consec, seq, random
        # Use pre_seek_eof=True for the random leg to match your convention
        n_emitted_ops = 0
        for (path, xfer, flags_extra), a in agg.items():
            if a["con"] > 0:
                emit_data_phase(path, intent, xfer, a["con"], "consec",
                                p_ua_file_default, p_mem_ua, False, flags_extra)
                n_emitted_ops += a["con"]
            if a["seq"] > 0:
                emit_data_phase(path, intent, xfer, a["seq"], "seq",
                                p_ua_file_default, p_mem_ua, False, flags_extra)
                n_emitted_ops += a["seq"]
            if a["rnd"] > 0:
                emit_data_phase(path, intent, xfer, a["rnd"], "random",
                                p_ua_file_default, p_mem_ua, True, flags_extra)
                n_emitted_ops += a["rnd"]
        return n_emitted_ops

    # Optional tiny rows collected earlier per your quadrant logic:
    # (initialize these to [] earlier in the script if they might be absent)
    extra_tiny_rows_read  = locals().get("extra_tiny_rows_read",  [])
    extra_tiny_rows_write = locals().get("extra_tiny_rows_write", [])

    # Emit tiny read/write items with exact consec/seq/rand ratios
    tiny_ops_total = 0
    if extra_tiny_rows_read:
        tiny_ops_total += emit_tiny_items(
            extra_tiny_rows_read, intent="read",
            path_flags=path_flags,
            p_ua_file_default=clamp01(p_file_ua_req),  # same alignment you use elsewhere
            p_mem_ua=p_mem_ua
        )

    if extra_tiny_rows_write:
        tiny_ops_total += emit_tiny_items(
            extra_tiny_rows_write, intent="write",
            path_flags=path_flags,
            p_ua_file_default=clamp01(p_file_ua_req),
            p_mem_ua=p_mem_ua
        )

    # after the layout_rows (data rows) are finalized
    Nr = sum(r["ops"] for r in layout_rows if r["intent"]=="read")
    Nw = sum(r["ops"] for r in layout_rows if r["intent"]=="write")
    Nio_realized = Nr + Nw + tiny_ops_total

    # desired meta totals (counts), using *features* fractions and the user’s requested io/meta split
    p_io   = clamp01(feats.get("pct_io_access", 1))
    p_open = clamp01(feats.get("pct_meta_open_access", 0.0))
    p_stat = clamp01(feats.get("pct_meta_stat_access", 0.0))
    p_seek = clamp01(feats.get("pct_meta_seek_access", 0.0))
    p_sync = clamp01(feats.get("pct_meta_sync_access", 0.0))

    # If p_io == 0, avoid divide-by-zero; otherwise scale meta so that:
    # meta_total / (meta_total + Nio_realized) ≈ (1 - p_io)
    if p_io > 0.0:
        scale = (1.0 - p_io) / p_io
    else:
        scale = 0.0  # no IO requested → keep meta at 0

    meta_total_target = int(round(scale * Nio_realized))

    # allocate meta_total_target across kinds according to p_open:p_stat:p_seek:p_sync
    k_fracs = [p_open, p_stat, p_seek, p_sync]
    s_k = sum(k_fracs) or 1.0
    k_fracs = [k/s_k for k in k_fracs]

    m_open = int(round(meta_total_target * k_fracs[0]))
    m_stat = int(round(meta_total_target * k_fracs[1]))
    m_seek = int(round(meta_total_target * k_fracs[2]))
    m_sync = int(round(meta_total_target * k_fracs[3]))

    # emit one META row using these finalized counts
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
    def add_span(rows, pc, ps, pr):
        """
        rows: iterable of {"file","xfer","ops"}
        pc, ps, pr: fractions for consec, seq, random *for this intent* (already clamped [0,1])
        """
        # aggregate ops per (file, xfer)
        by_fx = defaultdict(lambda: defaultdict(int))  # file -> xfer -> ops
        for r in rows:
            by_fx[r["file"]][r["xfer"]] += int(r["ops"])

        for f, sizes in by_fx.items():
            for xfer, ops in sizes.items():
                if ops <= 0:
                    continue

                # split ops by structure
                n_con = int(round(ops * pc))
                n_seq = int(round(ops * max(0.0, ps - pc)))
                n_rand = max(0, ops - n_con - n_seq)

                # coverage for consec/seq:
                # - consec: each op advances by xfer
                # - seq:    same stride; previous code doubled, but coverage is ~n_seq*xfer
                #   (If you intentionally want “headroom” for seq, you can add +xfer once.)
                cov_struct = n_con * xfer + n_seq * xfer

                # coverage for random:
                # harness uses 128MiB random "chunks" and RR across chunks.
                # If we try to avoid reusing offsets inside a chunk, the most
                # new space one chunk can “cover” before reuse is CHUNK_RANDOM.
                # Number of chunks "needed" if we wanted no reuse:
                ops_per_chunk = max(1, CHUNK_RANDOM // max(1, xfer))
                chunks_needed = int(math.ceil(n_rand / float(ops_per_chunk))) if n_rand > 0 else 0
                cov_random = chunks_needed * CHUNK_RANDOM

                # final required span for this (file,xfer) bucket is the larger of
                # structured coverage and random-chunk coverage.
                span = max(cov_struct, cov_random)

                # accumulate (multiple xfer buckets for the same file combine)
                per_file_span[f] += span

    # NOTE: pass the per-intent random fraction too (usually 1 - seq when pc=0 for random legs)
    # For reads:
    add_span([r for r in layout_rows if r["intent"] == "read"],  consec_r, seq_r, pr=1.0 - min(1.0, seq_r))
    # For writes:
    add_span([r for r in layout_rows if r["intent"] == "write"], consec_w, seq_w, pr=1.0 - min(1.0, seq_w))

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
