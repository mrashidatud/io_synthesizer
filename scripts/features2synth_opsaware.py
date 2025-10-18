#!/usr/bin/env python3
"""
Synthesizer (single-row -> scripts)

Emits:
  - run_prep.sh          : creates target files ONLY (no Darshan)
  - run_from_features.sh : runs IOR (read-only), mdtest (metadata),
                           and mpi_synthio harness (read-corrector + write-only),
                           with Darshan preloaded (MPICH: -genv LD_PRELOAD <lib>).

What this script enforces
-------------------------
• Read-size bins (op-based):
    small  (0–100K)   → 64 KiB
    medium (100K–10M) → 1 MiB  (as requested)
    large  (10M–1G+)  → 128 MiB
• Layout: honors pct_shared_files / pct_unique_files (“FPP”) and runs I/O on those files.
• Global shares: pct_reads / pct_writes; seq/consec; alignment (harness implements misalignment).
• Feasibility: reduces IOR read share if needed so the harness can realize seq/con & alignment.
• File-type buckets (***NEW***): matches both **file counts** and **bytes** for
  read-only (RO), read-write (RW), write-only (WO) across the file set.
  - RO files are read by IOR + harness-READ.
  - RW files are read (harness-READ) **and** written (harness-RW) to carry byte share.
  - WO files are only written (harness-WO).
• Meta as fraction of **TOTAL** ops: open/stat/seek/sync derived from total_ops and rounded.
• Caps (***aggregate-aware***): --cap-per-file-gib and/or --cap-total-gib set **global**
  byte regions per (layout,bin). We then derive **per-rank** budgets for FPP and **global**
  budgets for shared so that SUM across all files/ranks ≤ cap.
• Robust notes: shows targets, planned ops, file layout (counts & bytes target vs planned),
  per-rank/global budgets for IOR (-b) and harness (-B), and meta as fraction of total ops.
"""

import json
import os
from pathlib import Path
import argparse
import math
from collections import defaultdict

# ---- paths & constants --------------------------------------------------------
IOR      = "/mnt/hasanfs/bin/ior"
MDTEST   = "/mnt/hasanfs/bin/mdtest"
HARNESS  = "/mnt/hasanfs/bin/mpi_synthio"
DARSHAN  = "/mnt/hasanfs/darshan-3.4.7/darshan-runtime/install/lib/libdarshan.so"

OUT_ROOT = "/mnt/hasanfs/synth_from_features"
PREP_SH  = "run_prep.sh"
RUN_SH   = "run_from_features.sh"
NOTES_TXT= "run_from_features.sh.notes.txt"

KiB = 1024
MiB = 1024 * KiB
GiB = 1024 * MiB

# Read bin representatives
XS_SMALL  = 64 * KiB      # 0–100K bin
XS_MEDIUM = 1 * MiB       # 100K–10M bin
XS_LARGE  = 128 * MiB     # 10M–1G+ bin
TSIZE = {'small': XS_SMALL, 'medium': XS_MEDIUM, 'large': XS_LARGE}

OPS_PER_RANK_BASE = 2000  # global op budget basis

def mpiexec(nranks, cmd, with_darshan=False, hostfile="~/hfile"):
    env = f"-hostfile {hostfile} "
    if with_darshan:
        env += f"-genv LD_PRELOAD {DARSHAN} "
    return f"mpiexec {env}-n {nranks} {cmd}"

def bytes2gib(b): return f"{b/float(GiB):.2f} GiB"
def clamp01(x):   return max(0.0, min(1.0, float(x)))

def align_down(v, q): 
    if q <= 0: return 0
    return (v // q) * q

def align_up(v, q):
    if q <= 0: return 0
    return ((v + q - 1) // q) * q

# ---- helpers -----------------------------------------------------------------
def _solve_harness_p(target_frac, ior_n, h_n):
    if h_n <= 0:
        return (0.0, ior_n == 0 or abs(target_frac-1.0) < 1e-12)
    p = (target_frac * (ior_n + h_n) - ior_n) / float(h_n)
    return (p, 0.0 <= p <= 1.0)

def _scale_down_ior_for_seq_con(ior_reads, target_seq, target_con, total_reads):
    curr = int(ior_reads)
    for _ in range(30):
        h_reads = max(0, total_reads - curr)
        p_seq, ok_s = _solve_harness_p(target_seq, curr, h_reads)
        p_con, ok_c = _solve_harness_p(target_con, curr, h_reads)
        if ok_s and ok_c:
            return curr
        curr = int(curr * 0.8)
        if curr <= 0:
            return 0
    return max(0, curr)

def _ensure_alignment_possible(io_ops_total, ior_reads, harness_writes, target_file_ua, target_mem_ua):
    curr = int(ior_reads)
    for _ in range(30):
        h_reads = max(0, io_ops_total - (curr + harness_writes))
        harness_io = h_reads + harness_writes
        if harness_io <= 0:
            if target_file_ua == 0 and target_mem_ua == 0:
                return curr
            curr = 0
            continue
        need_file = (target_file_ua * io_ops_total) / float(harness_io)
        need_mem  = (target_mem_ua  * io_ops_total) / float(harness_io)
        if need_file <= 1.0 + 1e-12 and need_mem <= 1.0 + 1e-12:
            return curr
        curr = int(curr * 0.8)
        if curr <= 0:
            return 0
    return max(0, curr)

# integer splitter that preserves sum and ensures any positive fraction→≥1 file
def split_counts(total, fractions):
    raw = [f * total for f in fractions]
    ints = [int(round(x)) for x in raw]
    # guarantee non-zero where fraction>0
    for i, f in enumerate(fractions):
        if f > 0 and ints[i] == 0:
            ints[i] = 1
    d = total - sum(ints)
    # fix drift
    while d != 0:
        # add/remove from the largest bin by raw fractional remainder
        rema = [r - int(r) for r in raw]
        idx = max(range(len(rema)), key=lambda i: rema[i]) if d > 0 else min(range(len(rema)), key=lambda i: rema[i])
        if d > 0:
            ints[idx] += 1; d -= 1
        else:
            if ints[idx] > 0:
                ints[idx] -= 1; d += 1
            else:
                # find someone to borrow from
                for j in range(len(ints)):
                    if ints[j] > 0:
                        ints[j] -= 1; d += 1; break
    return ints

# ---- main planner -------------------------------------------------------------
def plan_from_features(feat, total_ranks, cap_per_file_gib=None, cap_total_gib=None):
    # A) Inputs
    p_reads  = clamp01(feat.get("pct_reads", 0.5))
    p_writes = clamp01(feat.get("pct_writes", 0.5))
    if p_reads + p_writes == 0:
        p_reads = 1.0; p_writes = 0.0
    else:
        s = p_reads + p_writes
        p_reads, p_writes = p_reads/s, p_writes/s

    p_io = clamp01(feat.get("pct_io_access", 1.0))

    # meta as fractions of TOTAL ops
    _mo = clamp01(feat.get("pct_meta_open_access", 0.0))
    _ms = clamp01(feat.get("pct_meta_stat_access", 0.0))
    _mk = clamp01(feat.get("pct_meta_seek_access", 0.0))
    _my = clamp01(feat.get("pct_meta_sync_access", 0.0))

    # alignment targets (global, fraction of IO ops)
    p_file_ua = clamp01(feat.get("pct_file_not_aligned", 0.0))
    p_mem_ua  = clamp01(feat.get("pct_mem_not_aligned", 0.0))

    # sequential / consecutive targets
    p_seq_r = clamp01(feat.get("pct_seq_reads", 0.0))
    p_seq_w = clamp01(feat.get("pct_seq_writes", 0.0))
    p_con_r = clamp01(feat.get("pct_consec_reads", 0.0))
    p_con_w = clamp01(feat.get("pct_consec_writes", 0.0))

    # RW-switches probability (harness)
    rw_prob = clamp01(feat.get("pct_rw_switches", 0.0))

    # read-size bin splits (3-bin)
    r0 = clamp01(feat.get("pct_read_0_100K", 1.0))
    r1 = clamp01(feat.get("pct_read_100K_10M", 0.0))
    r2 = clamp01(feat.get("pct_read_10M_1G_PLUS", 0.0))
    rs = r0 + r1 + r2
    if rs == 0: r0, r1, r2 = 1.0, 0.0, 0.0
    else: r0, r1, r2 = r0/rs, r1/rs, r2/rs

    # write-size bin splits (used for write-only & RW writes)
    w0 = clamp01(feat.get("pct_write_0_100K", 1.0))
    w1 = clamp01(feat.get("pct_write_100K_10M", 0.0))
    w2 = clamp01(feat.get("pct_write_10M_1G_PLUS", 0.0))
    ws = w0 + w1 + w2
    if ws == 0: w0, w1, w2 = 1.0, 0.0, 0.0
    else: w0, w1, w2 = w0/ws, w1/ws, w2/ws

    # layout fractions
    p_shared = clamp01(feat.get("pct_shared_files", 1.0))
    p_fpp    = clamp01(feat.get("pct_unique_files", 0.0))
    if p_shared + p_fpp == 0:
        p_shared, p_fpp = 1.0, 0.0
    else:
        s = p_shared + p_fpp
        p_shared, p_fpp = p_shared/s, p_fpp/s

    # file-type fractions (counts & bytes)
    f_ro = clamp01(feat.get("pct_read_only_files", 1.0))
    f_rw = clamp01(feat.get("pct_read_write_files", 0.0))
    f_wo = clamp01(feat.get("pct_write_only_files", 0.0))
    fs = f_ro + f_rw + f_wo
    if fs == 0: f_ro, f_rw, f_wo = 1.0, 0.0, 0.0
    else: f_ro, f_rw, f_wo = f_ro/fs, f_rw/fs, f_wo/fs

    b_ro = clamp01(feat.get("pct_bytes_read_only_files", 1.0))
    b_rw = clamp01(feat.get("pct_bytes_read_write_files", 0.0))
    b_wo = clamp01(feat.get("pct_bytes_write_only_files", 0.0))
    bs = b_ro + b_rw + b_wo
    if bs == 0: b_ro, b_rw, b_wo = 1.0, 0.0, 0.0
    else: b_ro, b_rw, b_wo = b_ro/bs, b_rw/bs, b_wo/bs

    # B) Global budgets
    total_ops_global = OPS_PER_RANK_BASE * total_ranks
    io_ops_target   = int(round(total_ops_global * p_io))
    meta_ops_target = total_ops_global - io_ops_target
    read_ops_target  = int(round(io_ops_target * p_reads))
    write_ops_target = io_ops_target - read_ops_target

    # meta absolute counts from TOTAL ops (honor your definition)
    mo_i = int(round(total_ops_global * _mo))
    ms_i = int(round(total_ops_global * _ms))
    mk_i = int(round(total_ops_global * _mk))
    my_i = int(round(total_ops_global * _my))
    # nudge to sum exactly to meta_ops_target
    delta = meta_ops_target - (mo_i + ms_i + mk_i + my_i)
    if delta != 0:
        my_i = max(0, my_i + delta)

    # C) Choose IOR read share; ensure seq/con & alignment are feasible
    ior_reads_init = int(round(read_ops_target * 0.60))
    ior_reads = _scale_down_ior_for_seq_con(ior_reads_init, p_seq_r, p_con_r, total_reads=read_ops_target)
    ior_reads = _ensure_alignment_possible(
        io_ops_total=io_ops_target,
        ior_reads=ior_reads,
        harness_writes=write_ops_target,
        target_file_ua=p_file_ua,
        target_mem_ua=p_mem_ua
    )
    h_reads = max(0, read_ops_target - ior_reads)

    # harness probabilities for reads (IOR reads are seq+con)
    p_seq_read_h, _ = _solve_harness_p(p_seq_r, ior_reads, h_reads)
    p_con_read_h, _ = _solve_harness_p(p_con_r, ior_reads, h_reads)
    p_con_read_h = min(p_con_read_h, p_seq_read_h)

    # writes (for RW+WO phases)
    p_seq_write_h = p_seq_w
    p_con_write_h = min(p_con_w, p_seq_write_h)

    # Alignment inside harness
    harness_io_ops = h_reads + write_ops_target
    if harness_io_ops <= 0:
        p_ua_file_h = 0.0
        p_ua_mem_h  = 0.0
    else:
        p_ua_file_h = clamp01((p_file_ua * io_ops_target) / float(harness_io_ops))
        p_ua_mem_h  = clamp01((p_mem_ua  * io_ops_target) / float(harness_io_ops))

    # D) Split IOR reads across 3 bins (proportional to r0,r1,r2), all seq+con
    ior_r0 = int(round(ior_reads * r0))
    ior_r1 = int(round(ior_reads * r1))
    ior_r2 = ior_reads - ior_r0 - ior_r1

    # Remaining reads to harness across bins
    h_r0 = max(0, int(round(read_ops_target * r0)) - ior_r0)
    h_r1 = max(0, int(round(read_ops_target * r1)) - ior_r1)
    h_r2 = max(0, int(round(read_ops_target * r2)) - ior_r2)
    drift = h_reads - (h_r0 + h_r1 + h_r2)
    while drift != 0:
        if drift > 0:
            if h_r0 >= h_r1 and h_r0 >= h_r2: h_r0 += 1
            elif h_r1 >= h_r0 and h_r1 >= h_r2: h_r1 += 1
            else: h_r2 += 1
            drift -= 1
        else:
            if h_r0 > 0 and h_r0 >= h_r1 and h_r0 >= h_r2: h_r0 -= 1
            elif h_r1 > 0 and h_r1 >= h_r0 and h_r1 >= h_r2: h_r1 -= 1
            elif h_r2 > 0: h_r2 -= 1
            drift += 1

    # E) Layout split (shared vs FPP) for each bin
    def split_layout(nops):
        sh = int(round(nops * p_shared))
        up = nops - sh
        return sh, up

    ior_r0_sh, ior_r0_up = split_layout(ior_r0)
    ior_r1_sh, ior_r1_up = split_layout(ior_r1)
    ior_r2_sh, ior_r2_up = split_layout(ior_r2)

    h_r0_sh,  h_r0_up  = split_layout(h_r0)
    h_r1_sh,  h_r1_up  = split_layout(h_r1)
    h_r2_sh,  h_r2_up  = split_layout(h_r2)

    # F) Planned GLOBAL bytes per (layout,bin) (for capping)
    plan_read_bytes = {
      ('shared','small'): (ior_r0_sh + h_r0_sh) * XS_SMALL,
      ('shared','medium'): (ior_r1_sh + h_r1_sh) * XS_MEDIUM,
      ('shared','large'):  (ior_r2_sh + h_r2_sh) * XS_LARGE,
      ('fpp','small'):     (ior_r0_up + h_r0_up) * XS_SMALL,
      ('fpp','medium'):    (ior_r1_up + h_r1_up) * XS_MEDIUM,
      ('fpp','large'):     (ior_r2_up + h_r2_up) * XS_LARGE,
    }

    per_file_cap = int(cap_per_file_gib * GiB) if cap_per_file_gib is not None else None
    total_cap    = int(cap_total_gib * GiB)    if cap_total_gib is not None else None

    # apply per-file cap
    capped_global = {k: (min(v, per_file_cap) if per_file_cap is not None else v) for k,v in plan_read_bytes.items()}

    # apply total cap proportionally
    sum_capped_reads = sum(capped_global.values())
    if total_cap is not None and sum_capped_reads > 0 and sum_capped_reads > total_cap:
        scale = total_cap / float(sum_capped_reads)
        capped_global = {k: int(v*scale) for k,v in capped_global.items()}

    # ---- Per-tool byte budgets ----------------------------------------------
    # IOR: -b is PER-RANK blocksize; Harness -B: SHARED uses GLOBAL; FPP uses PER-RANK.
    def ior_b_per_rank(total_ops_bin, nranks, tsize):
        ops_pr = (total_ops_bin + nranks - 1)//nranks
        return ops_pr * tsize, ops_pr

    def harness_B(layout, binname, nranks):
        tsize = TSIZE[binname]
        capG  = capped_global[(layout,binname)]
        if capG <= 0:
            return 0, 0
        if layout == 'shared':
            B_used = align_down(capG, tsize)    # GLOBAL
            return B_used, 0
        else:
            per_rank = align_down(capG // nranks, tsize)   # PER-RANK
            return per_rank, per_rank

    # G) File-type realization (counts & bytes)
    # Choose total logical file count baseline: keep it modest but large enough for splits.
    # For FPP we’ll have ≥ total_ranks files automatically, but we also need explicit counts per type.
    # Use 3 * max(1,total_ranks) to smooth rounding; minimum 6.
    base_files = max(6, 3 * max(1, total_ranks))
    N_ro, N_rw, N_wo = split_counts(base_files, [f_ro, f_rw, f_wo])

    # Planned read bytes (global) before RW/WO writes are added:
    total_read_bytes_global = sum(capped_global.values())

    # Target bytes by file-type (of ALL bytes):
    # We approximate total bytes as read_bytes (from read targets) + write_bytes (RW+WO phases).
    # Decide write bytes so that read_only/read_write/write_only byte shares are met.
    target_bytes_total = total_read_bytes_global / max(1e-9, b_ro) if b_ro > 0 else total_read_bytes_global
    target_bytes_ro = int(round(target_bytes_total * b_ro))
    target_bytes_rw = int(round(target_bytes_total * b_rw))
    target_bytes_wo = int(round(target_bytes_total * b_wo))

    # Now compute how much extra write bytes we need to add (beyond the reads already counted).
    # RO bytes should be *read-only bytes on RO files* → we will assign all READ bins onto RO and RW files
    # in proportion to file-count split; writes will go to RW and WO to carry their byte shares.
    # First, allocate a fraction of the read bytes to RW files so RW has some reads too.
    # Use a stable rule: let RW read-bytes proportion match its file-count proportion among (RO+RW).
    rw_read_fraction = (N_rw / max(1, (N_ro + N_rw))) if (N_ro + N_rw) > 0 else 0.0
    rw_read_bytes = int(round(total_read_bytes_global * rw_read_fraction))
    ro_read_bytes = total_read_bytes_global - rw_read_bytes

    # Given targets, remaining bytes needed on RW and WO must be supplied by WRITES:
    need_rw_write_bytes = max(0, target_bytes_rw - rw_read_bytes)
    need_wo_write_bytes = max(0, target_bytes_wo - 0)

    # If RO target is less than the RO read bytes (can happen), we’ll still keep RO reads;
    # Darshan will show RO bytes >= target; but in typical use b_ro≥b_rw+b_wo this is fine.

    # Convert desired write bytes to op counts per bin using your write bin fractions (w0/w1/w2).
    def bytes_to_ops_by_bin(total_bytes):
        ops0 = int(round((total_bytes * w0) / float(XS_SMALL))) if total_bytes > 0 else 0
        ops1 = int(round((total_bytes * w1) / float(XS_MEDIUM))) if total_bytes > 0 else 0
        # remainder to large:
        used = ops0*XS_SMALL + ops1*XS_MEDIUM
        rem_bytes = max(0, total_bytes - used)
        ops2 = rem_bytes // XS_LARGE
        return max(0, ops0), max(0, ops1), max(0, ops2)

    rw_w0_ops, rw_w1_ops, rw_w2_ops = bytes_to_ops_by_bin(need_rw_write_bytes)
    wo_w0_ops, wo_w1_ops, wo_w2_ops = bytes_to_ops_by_bin(need_wo_write_bytes)

    # Write-only tiny region choice if bytes target is ~0
    def tiny_B_for_writes(tsize, nranks):
        per_rank = 8 * tsize
        return per_rank * nranks

    # H) Filenames
    outdir = OUT_ROOT
    Path(outdir).mkdir(parents=True, exist_ok=True)

    def fname(layout, binname):
        base = "shared" if layout=="shared" else "fpp"
        return os.path.join(outdir, f"ior_{base}_{binname}.dat")

    files_read = {
      ('shared','small'):  fname('shared','small'),
      ('shared','medium'): fname('shared','medium'),
      ('shared','large'):  fname('shared','large'),
      ('fpp','small'):     fname('fpp','small'),
      ('fpp','medium'):    fname('fpp','medium'),
      ('fpp','large'):     fname('fpp','large'),
    }

    # file-type paths (RO, RW, WO): we will simply “tag” read-target files as RO or RW,
    # and keep dedicated WO files for write-only.
    # For RO/RW tagging we’ll distribute per layout using the splits p_shared/p_fpp.
    # We only need the **counts** manifest; the actual I/O follows the tag rules.

    # Create WO files (three sizes)
    write_only_files = {
        'small': os.path.join(outdir, "write_only_small.dat"),
        'medium':os.path.join(outdir, "write_only_medium.dat"),
        'large': os.path.join(outdir, "write_only_large.dat"),
    }

    # I) Build PREP (create read-target files; -k to keep)
    prep = ["#!/usr/bin/env bash","set -euo pipefail","", f'mkdir -p "{os.path.join(outdir,"mdtree")}"']

    def prep_ior(layout, binname, total_ops_bin):
        if total_ops_bin <= 0:
            return
        tsize = TSIZE[binname]
        b_pr, ops_pr = ior_b_per_rank(total_ops_bin, total_ranks, tsize)
        path  = files_read[(layout,binname)]
        flagF = "" if layout=="shared" else "-F"
        prep.append(mpiexec(total_ranks, f"{IOR} -a MPIIO -w -c -k {flagF} -b {b_pr} -t {tsize} -o {path}"))

    # IOR PREP per class/bin
    prep_ior('shared','small',  ior_r0_sh)
    prep_ior('shared','medium', ior_r1_sh)
    prep_ior('shared','large',  ior_r2_sh)
    prep_ior('fpp','small',     ior_r0_up)
    prep_ior('fpp','medium',    ior_r1_up)
    prep_ior('fpp','large',     ior_r2_up)

    # Always create the WO files; their sizes will be controlled by harness -B
    prep += [
        "",
        "# create the write-only files (sizes are governed later by harness -B)",
        mpiexec(1, f"{IOR} -a MPIIO -w -c -k -b {64*KiB} -t {64*KiB} -o {write_only_files['small']}"),
        mpiexec(1, f"{IOR} -a MPIIO -w -c -k -b {64*KiB} -t {64*KiB} -o {write_only_files['medium']}"),
        mpiexec(1, f"{IOR} -a MPIIO -w -c -k -b {64*KiB} -t {64*KiB} -o {write_only_files['large']}"),
    ]

    # J) RUN: IOR reads (Darshan), mdtest, harness-READ (RO+RW reads), harness-RW writes, harness-WO writes
    run = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Global ops={total_ops_global}, IO={io_ops_target}, META={meta_ops_target}",
        f"# Read bins target: small={r0:.2f}, medium={r1:.2f}, large={r2:.2f}",
        f"# IOR reads chosen: total={ior_reads}, harness reads={h_reads}",
        f"# Seq/Con READ target={p_seq_r:.2f}/{p_con_r:.2f} → harness p={p_seq_read_h:.3f}/{p_con_read_h:.3f}",
        f"# Seq/Con WRITE target={p_seq_w:.2f}/{p_con_w:.2f} → harness p={p_seq_write_h:.3f}/{p_con_write_h:.3f}",
        f"# Alignment target: file={p_file_ua:.2f}, mem={p_mem_ua:.2f} → harness p={p_ua_file_h:.3f}/{p_ua_mem_h:.3f}",
        f"# RW-switch prob (harness)={rw_prob:.3f}",
        f"# Layout target: shared={p_shared:.2f}, fpp={p_fpp:.2f}",
        ""
    ]

    def run_ior(layout, binname, total_ops_bin):
        if total_ops_bin <= 0:
            return
        tsize = TSIZE[binname]
        b_pr, _ops_pr = ior_b_per_rank(total_ops_bin, total_ranks, tsize)
        path  = files_read[(layout,binname)]
        flagF = "" if layout=="shared" else "-F"
        run.append(
            mpiexec(total_ranks,
                    f"{IOR} -a MPIIO -c -r {flagF} -b {b_pr} -t {tsize} -o {path}",
                    with_darshan=True)
        )

    # IOR READ phases
    run_ior('shared','small',  ior_r0_sh)
    run_ior('shared','medium', ior_r1_sh)
    run_ior('shared','large',  ior_r2_sh)
    run_ior('fpp','small',     ior_r0_up)
    run_ior('fpp','medium',    ior_r1_up)
    run_ior('fpp','large',     ior_r2_up)

    # mdtest (metadata only)
    run.append(mpiexec(total_ranks, f"{MDTEST} -F -C -T -r -n 100 -d {os.path.join(outdir,'mdtree')}", with_darshan=True))

    # Harness READ (follow each bin; 0 writes) — this covers RO + the RW read portion.
    def harness_read(layout, binname, ops_read):
        if ops_read <= 0:
            return
        tsize = TSIZE[binname]
        path  = files_read[(layout,binname)]
        B_used, per_rank_B = harness_B(layout, binname, total_ranks)
        layout_arg = "shared" if layout=='shared' else "fpp"
        run.append(
            mpiexec(
                total_ranks,
                " ".join([
                    f"{HARNESS} -o {path} --layout {layout_arg}",
                    f"-t {tsize}",
                    f"-B {B_used}",
                    "--p-write 0.0",
                    "--p-rand 0.0",
                    f"--p-unaligned-file {p_ua_file_h:.6f}",
                    f"--p-unaligned-mem {p_ua_mem_h:.6f}",
                    f"--rw-switch-prob {rw_prob:.6f}",
                    f"--p-seq-read {p_seq_read_h:.6f}",
                    f"--p-seq-write {p_seq_write_h:.6f}",
                    f"--p-consec-read {p_con_read_h:.6f}",
                    f"--p-consec-write {p_con_write_h:.6f}",
                    f"--meta-open {mo_i} --meta-stat {ms_i} --meta-seek {mk_i} --meta-sync {my_i}",
                    "--seed 1"
                ]),
                with_darshan=True
            ) + f"  # harness READ {layout}/{binname}"
        )

    harness_read('shared','small',  h_r0_sh)
    harness_read('shared','medium', h_r1_sh)
    harness_read('shared','large',  h_r2_sh)
    harness_read('fpp','small',     h_r0_up)
    harness_read('fpp','medium',    h_r1_up)
    harness_read('fpp','large',     h_r2_up)

    # Harness WRITE phases to realize RW and WO byte shares.
    # Strategy:
    #   • RW writes: direct writes onto the same *read* files to mark them RW (not RO).
    #   • WO writes: write to dedicated WO files.
    # Use -B regions sized by the requested bytes (aligned).
    def harness_write_rw(binname, ops_w, global_bytes):
        if ops_w <= 0 or global_bytes <= 0:
            return
        tsize = TSIZE[binname]
        # Distribute across layout by p_shared/p_fpp
        for layout in ['shared','fpp']:
            share_frac = p_shared if layout=='shared' else p_fpp
            gbytes = align_down(int(global_bytes * share_frac), tsize)
            if gbytes <= 0: 
                continue
            layout_arg = "shared" if layout=='shared' else "fpp"
            path  = files_read[(layout,binname)]
            B_used = gbytes if layout=='shared' else align_down(gbytes // total_ranks, tsize) * total_ranks
            run.append(
                mpiexec(
                    total_ranks,
                    " ".join([
                        f"{HARNESS} -o {path} --layout {layout_arg}",
                        f"-t {tsize}",
                        f"-B {B_used}",
                        "--p-write 1.0",
                        "--p-rand 0.0",
                        f"--p-unaligned-file {p_ua_file_h:.6f}",
                        f"--p-unaligned-mem {p_ua_mem_h:.6f}",
                        f"--rw-switch-prob {rw_prob:.6f}",
                        "--p-seq-read 0.0",
                        f"--p-seq-write {p_seq_write_h:.6f}",
                        "--p-consec-read 0.0",
                        f"--p-consec-write {p_con_write_h:.6f}",
                        "--meta-open 0 --meta-stat 0 --meta-seek 0 --meta-sync 0",
                        "--seed 3"
                    ]),
                    with_darshan=True
                ) + f"  # harness RW-WRITE {layout}/{binname}"
            )

    def harness_write_wo(binname, ops_w, global_bytes):
        if ops_w <= 0 or global_bytes <= 0:
            return
        tsize = TSIZE[binname]
        path  = write_only_files[binname]
        B_used = align_down(global_bytes, tsize)
        if B_used <= 0:
            B_used = tiny_B_for_writes(tsize, total_ranks)
        run.append(
            mpiexec(
                total_ranks,
                " ".join([
                    f"{HARNESS} -o {path} --layout shared",
                    f"-t {tsize}",
                    f"-B {B_used}",
                    "--p-write 1.0",
                    "--p-rand 0.0",
                    f"--p-unaligned-file {p_ua_file_h:.6f}",
                    f"--p-unaligned-mem {p_ua_mem_h:.6f}",
                    f"--rw-switch-prob {rw_prob:.6f}",
                    "--p-seq-read 0.0",
                    f"--p-seq-write {p_seq_write_h:.6f}",
                    "--p-consec-read 0.0",
                    f"--p-consec-write {p_con_write_h:.6f}",
                    "--meta-open 0 --meta-stat 0 --meta-seek 0 --meta-sync 0",
                    "--seed 4"
                ]),
                with_darshan=True
            ) + f"  # harness WO-WRITE {binname}"
        )

    # Decide RW vs WO write bytes by bin following w0/w1/w2.
    rw_write_bytes_total = need_rw_write_bytes
    wo_write_bytes_total = need_wo_write_bytes

    def split_bytes_by_bin(total_bytes):
        b0 = int(round(total_bytes * w0))
        b1 = int(round(total_bytes * w1))
        b2 = max(0, total_bytes - b0 - b1)
        return b0, b1, b2

    rw_b0, rw_b1, rw_b2 = split_bytes_by_bin(rw_write_bytes_total)
    wo_b0, wo_b1, wo_b2 = split_bytes_by_bin(wo_write_bytes_total)

    # Emit RW writes
    harness_write_rw('small',  rw_w0_ops, rw_b0)
    harness_write_rw('medium', rw_w1_ops, rw_b1)
    harness_write_rw('large',  rw_w2_ops, rw_b2)

    # Emit WO writes
    harness_write_wo('small',  wo_w0_ops, wo_b0)
    harness_write_wo('medium', wo_w1_ops, wo_b1)
    harness_write_wo('large',  wo_w2_ops, wo_b2)

    # ---- Write scripts --------------------------------------------------------
    Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(OUT_ROOT, PREP_SH), "w") as f:
        f.write("\n".join(prep) + "\n")
    os.chmod(os.path.join(OUT_ROOT, PREP_SH), 0o755)

    with open(os.path.join(OUT_ROOT, RUN_SH), "w") as f:
        f.write("\n".join(run) + "\n")
    os.chmod(os.path.join(OUT_ROOT, RUN_SH), 0o755)

    # ---- Notes (robust) -------------------------------------------------------
    ior_read_ops_total = ior_r0 + ior_r1 + ior_r2
    h_read_ops_total   = h_r0 + h_r1 + h_r2
    h_write_ops_total  = write_ops_target

    exp_seq_reads_global    = ior_read_ops_total + int(round(p_seq_read_h    * h_read_ops_total))
    exp_consec_reads_global = ior_read_ops_total + int(round(p_con_read_h    * h_read_ops_total))
    exp_seq_writes_global   = int(round(p_seq_write_h    * h_write_ops_total))
    exp_consec_writes_global= int(round(p_con_write_h    * h_write_ops_total))

    # Per-rank IOR -b report
    def ior_b_pr_report(total_ops_bin, tsize):
        ops_pr = (total_ops_bin + total_ranks - 1)//total_ranks
        return ops_pr * tsize, ops_pr

    ior_b_report = {
        ('shared','small'):  ior_b_pr_report(ior_r0_sh, XS_SMALL),
        ('shared','medium'): ior_b_pr_report(ior_r1_sh, XS_MEDIUM),
        ('shared','large'):  ior_b_pr_report(ior_r2_sh, XS_LARGE),
        ('fpp','small'):     ior_b_pr_report(ior_r0_up, XS_SMALL),
        ('fpp','medium'):    ior_b_pr_report(ior_r1_up, XS_MEDIUM),
        ('fpp','large'):     ior_b_pr_report(ior_r2_up, XS_LARGE),
    }

    # Harness -B report (shared=GLOBAL, fpp=PER-RANK)
    hB_report = {}
    agg_global_read_bytes = 0
    for key in capped_global.keys():
        layout, binname = key
        tsize = TSIZE[binname]
        B_used, per_rank_B = harness_B(layout, binname, total_ranks)
        if layout == 'shared':
            hB_report[key] = f"GLOBAL {bytes2gib(B_used)}"
            agg_global_read_bytes += B_used
        else:
            hB_report[key] = f"PER-RANK {bytes2gib(per_rank_B)} (agg≈{bytes2gib(per_rank_B*total_ranks)})"
            agg_global_read_bytes += per_rank_B * total_ranks

    # WO/RW write byte summaries
    rw_bytes_bins = {'small': rw_b0, 'medium': rw_b1, 'large': rw_b2}
    wo_bytes_bins = {'small': wo_b0, 'medium': wo_b1, 'large': wo_b2}
    rw_bytes_total = sum(rw_bytes_bins.values())
    wo_bytes_total = sum(wo_bytes_bins.values())

    # File-type counts (reported)
    file_type_counts = {'RO': N_ro, 'RW': N_rw, 'WO': N_wo}

    # Byte targets vs planned summary
    planned_bytes_ro = ro_read_bytes                   # (we only read on RO)
    planned_bytes_rw = rw_read_bytes + rw_bytes_total  # (read + write)
    planned_bytes_wo = wo_bytes_total                  # (write only)

    total_planned_bytes = planned_bytes_ro + planned_bytes_rw + planned_bytes_wo
    # (Total planned bytes ≈ read regions + write regions; the harness may revisit, but budgets
    #  are capped by -B and IOR -b as planned above.)

    notes = []
    notes.append(f"Bins → xfer: small={XS_SMALL//KiB}KiB, medium={XS_MEDIUM//KiB}KiB, large={XS_LARGE//MiB}MiB")
    notes.append(
        "Targets: "
        f"R={p_reads:.2f}, W={p_writes:.2f}, "
        f"seqR={p_seq_r:.2f}, consecR={p_con_r:.2f}, "
        f"seqW={p_seq_w:.2f}, consecW={p_con_w:.2f}, rwSwitch={rw_prob:.2f}"
    )
    notes.append(f"Alignment(harness): file={p_ua_file_h:.3f}, mem={p_ua_mem_h:.3f}")
    notes.append(f"Layout target: shared={p_shared:.2f}, fpp={p_fpp:.2f}")

    # Read ops
    notes.append(
        "IOR read ops: "
        f"r0={ior_r0} (sh={ior_r0_sh},up={ior_r0_up}), "
        f"r1={ior_r1} (sh={ior_r1_sh},up={ior_r1_up}), "
        f"r2={ior_r2} (sh={ior_r2_sh},up={ior_r2_up})"
    )
    notes.append(
        "Harness READ ops: "
        f"r0={h_r0} (sh={h_r0_sh},up={h_r0_up}), "
        f"r1={h_r1} (sh={h_r1_sh},up={h_r1_up}), "
        f"r2={h_r2} (sh={h_r2_sh},up={h_r2_up})"
    )

    # Expected seq/con
    notes.append(
        "Expected global seq/consec: "
        f"seqR≈{exp_seq_reads_global}, consecR≈{exp_consec_reads_global}, "
        f"seqW≈{exp_seq_writes_global}, consecW≈{exp_consec_writes_global}"
    )

    # Meta as fraction of TOTAL
    notes.append(
        f"Meta as fraction of TOTAL ops: open={mo_i}/{total_ops_global}={mo_i/total_ops_global:.2f}, "
        f"stat={ms_i/total_ops_global:.2f}, seek={mk_i/total_ops_global:.2f}, "
        f"sync={my_i/total_ops_global:.2f}; meta_total={(mo_i+ms_i+mk_i+my_i)}/{total_ops_global}="
        f"{(mo_i+ms_i+mk_i+my_i)/total_ops_global:.2f}"
    )

    # IOR per-rank -b
    lines = []
    for key, val in ior_b_report.items():
        layout, binname = key
        bpr, opspr = val
        lines.append(f"{key}: -b per-rank={bytes2gib(bpr)} ({bpr} B), ops/rank≈{opspr}")
    notes.append("IOR per-rank -b by (layout,bin): " + "; ".join(lines))

    # Harness -B
    lines = []
    for key, text in hB_report.items():
        lines.append(f"{key}: {text}")
    notes.append("Harness -B by (layout,bin): " + "; ".join(lines))

    notes.append("GLOBAL read caps per (layout,bin): " + str({k: bytes2gib(v) for k,v in capped_global.items()}))
    notes.append(f"Aggregate GLOBAL read size (sum across read files) = {bytes2gib(agg_global_read_bytes)}")

    # ---- File-type layout summary (counts & bytes target vs planned) ---------
    notes.append("File-type counts (target fractions → counts): "
                 f"RO={f_ro:.2f}→{N_ro}, RW={f_rw:.2f}→{N_rw}, WO={f_wo:.2f}→{N_wo}")

    notes.append("File-type bytes (target vs planned): " +
        f"RO: target≈{b_ro:.2f} of total, planned≈{bytes2gib(planned_bytes_ro)}; "
        f"RW: target≈{b_rw:.2f} of total, planned≈{bytes2gib(planned_bytes_rw)}; "
        f"WO: target≈{b_wo:.2f} of total, planned≈{bytes2gib(planned_bytes_wo)}; "
        f"total≈{bytes2gib(total_planned_bytes)}")

    with open(os.path.join(OUT_ROOT, NOTES_TXT), "w") as f:
        f.write("\n".join(notes) + "\n")

# ---- CLI ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="path to one features.json")
    ap.add_argument("--total-ranks", type=int, required=True)
    ap.add_argument("--cap-per-file-gib", type=float, default=None,
                    help="Hard cap (GiB) for each read-target file (per layout+bin).")
    ap.add_argument("--cap-total-gib", type=float, default=None,
                    help="Hard cap (GiB) on the sum across all read-target files.")
    args = ap.parse_args()

    with open(args.features) as jf:
        feat = json.load(jf)

    plan_from_features(
        feat,
        total_ranks=args.total_ranks,
        cap_per_file_gib=args.cap_per_file_gib,
        cap_total_gib=args.cap_total_gib
    )
    print(f"Wrote {PREP_SH}, {RUN_SH}, and {NOTES_TXT} to {OUT_ROOT}")

if __name__ == "__main__":
    main()
