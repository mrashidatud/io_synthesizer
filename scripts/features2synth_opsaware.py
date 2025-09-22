#!/usr/bin/env python3
# =============================================================================
# Synthesizer / Launcher (features2synth_opsaware.py)
#
# What this does
#  - Reads a features.json row (percentages by BYTES and by OPERATIONS).
#  - Plans IOR subjobs to realize the BYTES mix (shared vs FPP, RO/RW/WO bytes,
#    size bins, seq/random by bytes).
#  - Estimates IOR operation counts each subjob will perform given its -t.
#  - Solves the minimal harness ops so global OP-LEVEL targets are feasible with
#    per-op probabilities ≤ 1:
#       * pct_file_not_aligned, pct_mem_not_aligned
#       * pct_rw_switches
#       * pct_seq_reads / pct_seq_writes  (Darshan POSIX_SEQ_*) 
#       * pct_consec_reads / pct_consec_writes (Darshan POSIX_CONSEC_*)
#  - Distributes those harness ops across subjobs proportional to IOR ops so the
#    harness mirrors IOR files, layouts, -t, and patterns.
#  - Emits scripts in this order:
#       1) **PREP**: single MPMD mpiexec (NO Darshan) to create/init all files
#          (IO500-style) and build the mdtest tree.
#       2) **RUN**:  IOR phases (all ranks; Darshan on) →
#                    mdtest (all ranks; Darshan on) →
#                    harness phases (all ranks; Darshan on).
#
# Defaults (override via CLI if you want)
#  - IOR:      /mnt/hasanfs/bin/ior
#  - mdtest:   /mnt/hasanfs/bin/mdtest
#  - harness:  /mnt/hasanfs/bin/mpi_synthio
#  - mpiexec prefix (Darshan on): 
#       mpiexec -hostfile ~/hfile env LD_PRELOAD=/mnt/hasanfs/darshan-3.4.7/darshan-runtime/install/lib/libdarshan.so
# =============================================================================
import argparse, json, os, math, sys
from typing import Dict, List

# ---------- helpers ----------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def pct(key: str, d: dict, default: float = 0.0) -> float:
    v = d.get(key, default)
    try:
        v = float(v)
    except Exception:
        v = default
    if v > 1.0: v = v/100.0
    return clamp01(v)

def norm(parts: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in parts.values())
    if total <= 0:
        n = len(parts)
        return {k: 1.0/n for k in parts} if n else {}
    return {k: max(0.0, v)/total for k, v in parts.items()}

def human_bytes(n: float) -> str:
    units = ['B','KiB','MiB','GiB','TiB']
    i = 0; x = float(n)
    while x >= 1024.0 and i < len(units)-1:
        x /= 1024.0; i += 1
    return f"{x:.2f} {units[i]}"

def round_bytes(n: float, gran: int = 4096) -> int:
    if n <= 0: return 0
    return int(max(gran, int(n // gran * gran)))

def parse_size_token(tok: str) -> int:
    t = tok.strip().lower()
    if t.endswith('k'): return int(float(t[:-1]) * 1024)
    if t.endswith('m'): return int(float(t[:-1]) * 1024*1024)
    if t.endswith('g'): return int(float(t[:-1]) * 1024*1024*1024)
    if t.endswith('b'): return int(float(t[:-1]))
    return int(float(t))

def build_mpiexec_prefix(with_darshan: bool, extra: str=""):
    base = "mpiexec -hostfile ~/hfile"
    if with_darshan:
        # MPICH/Hydra: propagate LD_PRELOAD to all ranks
        return f'{base} -genv LD_PRELOAD /mnt/hasanfs/darshan-3.4.7/darshan-runtime/install/lib/libdarshan.so {extra}'.strip()
    else:
        return f"{base} {extra}".strip()

def build_ior_cmd(ior, mode, layout, pattern, tsize, per_rank_b, out_path, extra=""):
    """
    Emit an IOR command for MPIIO with the correct collective/independent choice:
      - shared + sequential -> collective (-c)
      - random              -> independent (no -c)
      - FPP (-F)            -> independent
    Always add -k (keep files) so IOR doesn't delete them afterward.
    """
    flags = ["-a", "MPIIO", "-k"]   # <-- keep files

    # access types
    if mode in ("w", "rw"):
        flags.append("-w")
    if mode in ("r", "rw"):
        flags.append("-r")

    # layout
    if layout == "fpp":
        flags.append("-F")           # FPP => independent
    else:
        # shared file: collective only if NOT random
        if pattern != "random":
            flags.append("-c")

    # pattern
    if pattern == "random":
        flags.append("-z")           # random offsets => independent (no -c)

    # sizes & target
    flags += ["-b", str(per_rank_b), "-t", tsize, "-o", out_path]

    if extra:
        flags += extra.split()

    # no '-H'
    return f"{ior} {' '.join(flags)}"

def build_mdtest_cmd(mdtest, files_per_proc, out_dir, flags="-F -C -T -r"):
    return f"{mdtest} {flags} -n {files_per_proc} -d {out_dir}"

def build_harness_cmd(harness, out_path, layout, tsize, per_rank_b, p_write, p_rand,
                      p_file_ua, p_mem_ua, p_rws, p_seq_r, p_seq_w, p_con_r, p_con_w, extra_flags=""):
    lay = f"--layout {layout}"
    base = f"{harness} -o {out_path} {lay} -t {tsize} -B {per_rank_b} --p-write {p_write:.6f} --p-rand {p_rand:.6f}"
    base += f" --p-unaligned-file {p_file_ua:.6f} --p-unaligned-mem {p_mem_ua:.6f} --rw-switch-prob {p_rws:.6f}"
    base += f" --p-seq-read {p_seq_r:.6f} --p-seq-write {p_seq_w:.6f} --p-consec-read {p_con_r:.6f} --p-consec-write {p_con_w:.6f}"
    if extra_flags.strip():
        base += " " + extra_flags.strip()
    return base

def plan_ior_subjobs(features, total_bytes, canon, sharing, modesB, read_bins, write_bins, seq_reads, seq_writes, out_root):
    def pick_tsize(bin_name: str) -> str:
        b = bin_name.lower()
        if "small" in b: return canon["small"]
        if "medium" in b: return canon["medium"]
        if "large" in b: return canon["large"]
        return canon["medium"]

    shared_bytes = total_bytes * sharing["shared"]
    fpp_bytes    = total_bytes * sharing["fpp"]

    subs = []
    def add_job(bytes_bud, mode, layout, pattern, bin_name):
        if bytes_bud <= 0: return
        tsize = pick_tsize(bin_name)
        fname = f"{out_root}/ior_{layout}_{pattern}_{mode}_{bin_name}.dat"
        subs.append({"bytes": bytes_bud, "mode": mode, "layout": layout, "pattern": pattern,
                     "bin": bin_name, "tsize": tsize, "out": fname})

    def expand(layout, layout_bytes):
        if layout_bytes <= 0: return
        roB = layout_bytes * modesB.get("ro", 0.0)
        for patt, pP in seq_reads.items():
            for bin_name, pBin in read_bins.items():
                add_job(roB * pP * pBin, "r", layout, patt, bin_name)
        woB = layout_bytes * modesB.get("wo", 0.0)
        for patt, pP in seq_writes.items():
            for bin_name, pBin in write_bins.items():
                add_job(woB * pP * pBin, "w", layout, patt, bin_name)
        rwB = layout_bytes * modesB.get("rw", 0.0)
        rB = rwB * 0.5; wB = rwB * 0.5
        for patt, pP in seq_reads.items():
            for bin_name, pBin in read_bins.items():
                add_job(rB * pP * pBin, "r", layout, patt, bin_name)
        for patt, pP in seq_writes.items():
            for bin_name, pBin in write_bins.items():
                add_job(wB * pP * pBin, "w", layout, patt, bin_name)

    expand("shared", shared_bytes)
    expand("fpp",    fpp_bytes)
    return subs

def sum_read_write_ops(subjobs, ops_list):
    Oi_r = Oi_w = Si_r = Si_w = Ci_r = Ci_w = 0
    for sj, ops in zip(subjobs, ops_list):
        if ops <= 0: continue
        if sj["mode"] == "r":
            Oi_r += ops
            if sj["pattern"] == "sequential":
                Si_r += ops; Ci_r += ops
        elif sj["mode"] == "w":
            Oi_w += ops
            if sj["pattern"] == "sequential":
                Si_w += ops; Ci_w += ops
    return Oi_r, Oi_w, Si_r, Si_w, Ci_r, Ci_w

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="features.json -> scripts with IO500-style prep + Darshan-run + micro-harness.")
    ap.add_argument("--features", required=True)
    ap.add_argument("--out-script", default="/mnt/hasanfs/synth_from_features/run_from_features.sh")
    ap.add_argument("--prep-script", default="/mnt/hasanfs/synth_from_features/run_prep.sh")
    ap.add_argument("--ior", default="/mnt/hasanfs/bin/ior")
    ap.add_argument("--mdtest", default="/mnt/hasanfs/bin/mdtest")
    ap.add_argument("--micro-harness", dest="micro_harness", default="/mnt/hasanfs/bin/mpi_synthio")
    ap.add_argument("--out-root", default="/mnt/hasanfs/synth_from_features")
    ap.add_argument("--total-ranks", type=int, default=60)
    ap.add_argument("--total-bytes", type=float, default=64*1024*1024*1024)
    ap.add_argument("--tsize-small", default="64k")
    ap.add_argument("--tsize-medium", default="1m")
    ap.add_argument("--tsize-large", default="128m")
    ap.add_argument("--mdtest-n", type=int, default=100)
    ap.add_argument("--mdtest-flags", default="-F -C -T -r")
    ap.add_argument("--harness-flags", default="")
    ap.add_argument("--single-mpmd-job", type=int, default=0, help="If 1, emit all phases under a SINGLE mpiexec MPMD (Darshan on).")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    with open(args.features,'r') as f:
        F = json.load(f)

    # Splits
    frac_io   = clamp01(F.get("pct_io_access", 1.0))
    sharing = norm({"shared": pct("pct_bytes_shared_files", F, F.get("pct_shared_files", 0.5)),
                    "fpp":    pct("pct_bytes_unique_files", F, F.get("pct_unique_files", 0.5))})
    modesB = norm({
        "ro": pct("pct_bytes_read_only_files", F, F.get("pct_read_only_files", 0.33)),
        "rw": pct("pct_bytes_read_write_files", F, F.get("pct_read_write_files", 0.34)),
        "wo": pct("pct_bytes_write_only_files", F, F.get("pct_write_only_files", 0.33)),
    })
    read_bins  = norm({"small": pct("pct_read_0_100K", F, 0.0),
                       "medium": pct("pct_read_100K_10M", F, 0.0),
                       "large": pct("pct_read_10M_1G_PLUS", F, 0.0)})
    write_bins = norm({"small": pct("pct_write_0_100K", F, 0.0),
                       "medium": pct("pct_write_100K_10M", F, 0.0),
                       "large": pct("pct_write_10M_1G_PLUS", F, 0.0)})
    seq_reads  = norm({"sequential": pct("pct_seq_reads",  F, 1.0),
                       "random": 1.0 - pct("pct_seq_reads",  F, 1.0)})
    seq_writes = norm({"sequential": pct("pct_seq_writes", F, 1.0),
                       "random": 1.0 - pct("pct_seq_writes", F, 1.0)})

    canon = {"small": args.tsize_small, "medium": args.tsize_medium, "large": args.tsize_large}

    data_bytes_total = args.total_bytes * frac_io
    subjobs = plan_ior_subjobs(F, data_bytes_total, canon, sharing, modesB, read_bins, write_bins, seq_reads, seq_writes, args.out_root)

    # Per-subjob op counts if run with all ranks
    R = args.total_ranks
    ops_ior = []
    prb_ior = []
    for sj in subjobs:
        tsz = parse_size_token(sj["tsize"])
        chunks_per_rank = max(1, int(math.ceil(sj["bytes"] / (R * tsz))))
        prb = round_bytes(chunks_per_rank * tsz, 4096)
        prb_ior.append(prb)
        ops_ior.append(R * chunks_per_rank)
    O_i = sum(ops_ior)
    Oi_r, Oi_w, Si_r, Si_w, Ci_r, Ci_w = sum_read_write_ops(subjobs, ops_ior)

    # OP targets
    p_file_ua = pct("pct_file_not_aligned", F, 0.0)
    p_mem_ua  = pct("pct_mem_not_aligned",  F, 0.0)
    p_rws     = pct("pct_rw_switches",      F, 0.0)
    pW        = pct("pct_writes", F, 1.0 - pct("pct_reads", F, 0.5))
    pR        = 1.0 - pW
    p_seq_r   = pct("pct_seq_reads",   F, 1.0)
    p_seq_w   = pct("pct_seq_writes",  F, 1.0)
    p_con_r   = pct("pct_consec_reads",  F, min(p_seq_r, 1.0))
    p_con_w   = pct("pct_consec_writes", F, min(p_seq_w, 1.0))

    def min_Oh_for_ratio(target, I_count, Oi_type, w_share):
        if target <= 0.0: return 0
        if w_share <= 0.0: return int(1e9)
        if target >= 1.0:
            return 0 if I_count >= Oi_type else int(1e9)
        num = target * Oi_type - I_count
        den = (1.0 - target) * w_share
        if num <= 0: return 0
        return int(math.ceil(num / den))

    Oh_file = 0 if p_file_ua <= 0 else int(math.ceil(O_i * p_file_ua / (1.0 - p_file_ua)))
    Oh_mem  = 0 if p_mem_ua  <= 0 else int(math.ceil(O_i * p_mem_ua  / (1.0 - p_mem_ua)))
    Oh_rws  = 0 if p_rws     <= 0 else int(math.ceil(O_i * p_rws     / (1.0 - p_rws)))
    Oh_seq_r = min_Oh_for_ratio(p_seq_r, Si_r, Oi_r, pR)
    Oh_seq_w = min_Oh_for_ratio(p_seq_w, Si_w, Oi_w, pW)
    Oh_con_r = min_Oh_for_ratio(p_con_r, Ci_r, Oi_r, pR)
    Oh_con_w = min_Oh_for_ratio(p_con_w, Ci_w, Oi_w, pW)
    O_h = max(Oh_file, Oh_mem, Oh_rws, Oh_seq_r, Oh_seq_w, Oh_con_r, Oh_con_w)

    def q_global(p): return 0.0 if p<=0 else min(1.0, p * (O_i + O_h) / max(1, O_h))
    q_file = q_global(p_file_ua)
    q_mem  = q_global(p_mem_ua)
    q_rws  = q_global(p_rws)

    def q_type(target, I_count, Oi_type, w_share):
        if target <= 0.0 or w_share <= 0: return 0.0
        num = target * (Oi_type + w_share * O_h) - I_count
        den = w_share * O_h
        if den <= 0: return 0.0
        return max(0.0, min(1.0, num / den))

    q_seq_r = q_type(p_seq_r, Si_r, Oi_r, pR)
    q_seq_w = q_type(p_seq_w, Si_w, Oi_w, pW)
    q_con_r = q_type(p_con_r, Ci_r, Oi_r, pR)
    q_con_w = q_type(p_con_w, Ci_w, Oi_w, pW)
    q_con_r = min(q_con_r, q_seq_r)
    q_con_w = min(q_con_w, q_seq_w)

    # Distribute harness ops across subjobs
    if O_h > 0 and sum(ops_ior) > 0:
        raw = [O_h * w / sum(ops_ior) for w in ops_ior]
        O_h_j = [int(math.floor(x)) for x in raw]
        used = sum(O_h_j); rem = O_h - used
        fr = sorted(list(enumerate([x - math.floor(x) for x in raw])), key=lambda kv: kv[1], reverse=True)
        i=0
        while rem>0 and i < len(fr):
            O_h_j[fr[i][0]] += 1; rem -= 1; i += 1
    else:
        O_h_j = [0]*len(subjobs)

    prb_h = []
    for sj, Oh in zip(subjobs, O_h_j):
        tsz = parse_size_token(sj["tsize"])
        if Oh <= 0: prb_h.append(0); continue
        ops_per_rank = max(1, int(math.ceil(Oh / R)))
        prb_h.append(round_bytes(ops_per_rank * tsz, 4096))

    # ------------- emit PREP script (sequential, NO Darshan) -------------
    prep_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    mpiprep = build_mpiexec_prefix(with_darshan=False)
    md_dir = os.path.join(args.out_root, "mdtree")
    prep_lines.append(f'mkdir -p "{md_dir}"')

    # For every READ subjob, create the file using IOR write so aggregate size matches read phase.
    # IMPORTANT: prep always writes SEQUENTIAL (no -z), even if that subjob is "random" later.
    for sj, prb in zip(subjobs, prb_ior):
        if sj["mode"] != "r":
            continue
        prep_lines.append(f'mkdir -p "$(dirname {sj["out"]})"')

        layout = "shared" if sj["layout"] == "shared" else "fpp"

        # Force sequential pattern for prep (no -z). We'll pass pattern="sequential" here so build_ior_cmd adds -c for shared.
        ior_prep_cmd = build_ior_cmd(
            args.ior, mode="w",
            layout=layout,
            pattern="sequential",     # <-- force sequential for prep
            tsize=sj["tsize"],
            per_rank_b=prb,
            out_path=sj["out"]
        )
        prep_lines.append(f"{mpiprep} -n {R} {ior_prep_cmd}")

    with open(args.prep_script, "w") as f:
        f.write("\n".join(prep_lines) + "\n")
    os.chmod(args.prep_script, 0o755)

    # ------------- emit RUN script (either all-in-one MPMD or sequential) -------------
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    mpirun = build_mpiexec_prefix(with_darshan=True)

    if args.single_mpmd_job:
        parts = []
        # IOR phases
        for sj, prb in zip(subjobs, prb_ior):
            cmd = build_ior_cmd(
                args.ior,
                sj["mode"],
                "shared" if sj["layout"] == "shared" else "fpp",
                sj["pattern"],
                sj["tsize"],
                prb,
                sj["out"]
            )
            parts.append(f'-n {R} {cmd}')
        # mdtest
        md_cmd = build_mdtest_cmd(args.mdtest, args.mdtest_n, md_dir, flags=args.mdtest_flags)
        parts.append(f'-n {R} {md_cmd}')
        # harness
        for sj, prb in zip(subjobs, prb_h):
            if prb <= 0: continue
            p_rand = 1.0 if sj["pattern"]=="random" else 0.0
            cmdh = build_harness_cmd(args.micro_harness, sj["out"],
                                     layout=("shared" if sj["layout"]=="shared" else "fpp"),
                                     tsize=sj["tsize"], per_rank_b=prb,
                                     p_write=pW, p_rand=p_rand,
                                     p_file_ua=q_file, p_mem_ua=q_mem, p_rws=q_rws,
                                     p_seq_r=q_seq_r, p_seq_w=q_seq_w, p_con_r=q_con_r, p_con_w=q_con_w,
                                     extra_flags=args.harness_flags)
            parts.append(f'-n {R} {cmdh}')
        lines.append(f'{mpirun} ' + "  :  ".join(parts))
    else:
        # sequential (one mpiexec per phase)
        for sj, prb in zip(subjobs, prb_ior):
            cmd = build_ior_cmd(
                args.ior,
                sj["mode"],
                "shared" if sj["layout"] == "shared" else "fpp",
                sj["pattern"],
                sj["tsize"],
                prb,
                sj["out"]
            )
            lines.append(f'{mpirun} -n {R} {cmd}  # IOR {sj["layout"]} {sj["pattern"]} {sj["mode"]} {sj["bin"]}')
        md_cmd = build_mdtest_cmd(args.mdtest, args.mdtest_n, md_dir, flags=args.mdtest_flags)
        lines.append(f'{mpirun} -n {R} {md_cmd}  # metadata')
        for sj, prb in zip(subjobs, prb_h):
            if prb <= 0: continue
            p_rand = 1.0 if sj["pattern"]=="random" else 0.0
            cmdh = build_harness_cmd(args.micro_harness, sj["out"],
                                     layout=("shared" if sj["layout"]=="shared" else "fpp"),
                                     tsize=sj["tsize"], per_rank_b=prb,
                                     p_write=pW, p_rand=p_rand,
                                     p_file_ua=q_file, p_mem_ua=q_mem, p_rws=q_rws,
                                     p_seq_r=q_seq_r, p_seq_w=q_seq_w, p_con_r=q_con_r, p_con_w=q_con_w,
                                     extra_flags=args.harness_flags)
            lines.append(f'{mpirun} -n {R} {cmdh}  # HARNESS follows {sj["layout"]} {sj["pattern"]} {sj["bin"]}')

    with open(args.out_script, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(args.out_script, 0o755)

    # Notes
    notes = []
    notes.append(f"Prep script: {args.prep_script}  (NO Darshan)")
    notes.append(f"Run script : {args.out_script}   (Darshan enabled via LD_PRELOAD)")
    notes.append(f"Data bytes target: {human_bytes(data_bytes_total)}")
    notes.append(f"IOR subjobs: {len(subjobs)}; est IOR ops: {O_i}")
    notes.append(f"Targets: fileUA={p_file_ua:.3f}, memUA={p_mem_ua:.3f}, rwSwitch={p_rws:.3f}, "
                 f"seqR={p_seq_r:.3f}, seqW={p_seq_w:.3f}, consecR={p_con_r:.3f}, consecW={p_con_w:.3f}")
    notes.append(f"Chosen O_h={O_h} → q_file={q_file:.3f}, q_mem={q_mem:.3f}, q_rws={q_rws:.3f}, "
                 f"q_seq_r={q_seq_r:.3f}, q_seq_w={q_seq_w:.3f}, q_con_r={q_con_r:.3f}, q_con_w={q_con_w:.3f}")
    with open(args.out_script + ".notes.txt","w") as nf:
        nf.write("\n".join(notes) + "\n")

    print("Wrote", args.prep_script)
    print("Wrote", args.out_script)
    print("Wrote", args.out_script + ".notes.txt")

if __name__ == "__main__":
    main()
