#!/usr/bin/env python3
"""
Generate machine-learning features from the Darshan summary CSV, with aggregation by jobid.

Reads darshan_summary.csv in the root log directory,
filters rows by total_bytes, computes POSIX-based features,
and writes an array of JSON feature objects to darshan_features_updated.json.

Notes:
- The filtered intermediate CSV (darshan_summary_filtered.csv) is still written for QC.
- The JSON output mirrors your "features.json" style:
  { filename, jobid, nprocs, pct_* feature fields, ... }
"""
import os
import json
import pandas as pd
import numpy as np
import argparse   # ← ADD THIS

# ─── CONFIG (dynamic path) ─────────────────────────────────────────────────────
# --root/--outdir points to the directory that CONTAINS darshan_summary.csv.
# All outputs (filtered CSV + features JSON) will be written to this same dir.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root", "--outdir",
    dest="root",
    default=os.getcwd(),
    help="Directory containing darshan_summary.csv; outputs will be written here (default: current working dir)",
)
parser.add_argument(
    "--module-basis",
    choices=["auto", "posix", "mpiio"],
    default="auto",
    help="Counter basis for pct_* derivation (auto selects MPIIO when MPIIO byte counters are active).",
)
args = parser.parse_args()
root_dir     = os.path.abspath(args.root)
input_csv    = os.path.join(root_dir, "darshan_summary.csv")
filtered_csv = os.path.join(root_dir, "darshan_summary_filtered.csv")
output_json  = os.path.join(root_dir, "darshan_features_updated.json")
# filter threshold (bytes)
# threshold    = 100 * 1024 * 1024  # 100 MB
threshold    = 0
# ────────────────────────────────────────────────────────────────────────────────

# 1) Load raw data
df_raw = pd.read_csv(
    input_csv,
    engine='python',
    on_bad_lines='skip'
)
initial_count = len(df_raw)
print(f"Loaded {initial_count} rows from {input_csv}")

# 2) Filter on total_bytes per row
module_basis = str(args.module_basis).strip().lower()
if module_basis == "auto":
    posix_read = df_raw["POSIX_BYTES_READ"] if "POSIX_BYTES_READ" in df_raw.columns else pd.Series(0, index=df_raw.index)
    posix_write = df_raw["POSIX_BYTES_WRITTEN"] if "POSIX_BYTES_WRITTEN" in df_raw.columns else pd.Series(0, index=df_raw.index)
    mpiio_read = df_raw["MPIIO_BYTES_READ"] if "MPIIO_BYTES_READ" in df_raw.columns else pd.Series(0, index=df_raw.index)
    mpiio_write = df_raw["MPIIO_BYTES_WRITTEN"] if "MPIIO_BYTES_WRITTEN" in df_raw.columns else pd.Series(0, index=df_raw.index)
    posix_total = (posix_read + posix_write).fillna(0)
    mpiio_total = (mpiio_read + mpiio_write).fillna(0)
    module_basis = "mpiio" if float(mpiio_total.sum()) > 0.0 else "posix"
MODULE = "MPIIO" if module_basis == "mpiio" else "POSIX"
print(f"Feature module basis: {MODULE}")

def mcol(name: str) -> str:
    return f"{MODULE}_{name}"

df_raw['total_bytes'] = df_raw.get(mcol('BYTES_READ'), 0) + df_raw.get(mcol('BYTES_WRITTEN'), 0)
stats = df_raw[df_raw['total_bytes'] >= threshold].reset_index(drop=True)
processed_count = len(stats)
print(f"Filtered to {processed_count} rows with total_bytes >= {threshold}")
print(f"Input samples: {initial_count}, Processed samples: {processed_count}")
stats.to_csv(filtered_csv, index=False)

# helper functions
def get_series(col):
    return stats[col] if col in stats.columns else pd.Series(0, index=stats.index)

def get_series_sum(cols):
    if isinstance(cols, str):
        cols = [cols]
    out = pd.Series(0.0, index=stats.index)
    for c in cols:
        if c in stats.columns:
            out = out + pd.to_numeric(stats[c], errors="coerce").fillna(0.0)
    return out

def choose_series(primary, fallback=None, allow_fallback=True):
    """
    Return the sum of primary columns when present; otherwise fallback columns.
    """
    if fallback is None:
        fallback = []
    if isinstance(primary, str):
        primary = [primary]
    if isinstance(fallback, str):
        fallback = [fallback]

    primary_present = [c for c in primary if c in stats.columns]
    if primary_present:
        return get_series_sum(primary_present)

    if allow_fallback:
        fallback_present = [c for c in fallback if c in stats.columns]
        if fallback_present:
            return get_series_sum(fallback_present)

    return pd.Series(0.0, index=stats.index)

def ratio_from_counter_pair(primary_num, primary_den, fallback_num=None, fallback_den=None):
    """
    Compute ratio from a preferred numerator/denominator pair, falling back to
    another pair when preferred numerator counters are unavailable.
    """
    if fallback_num is None:
        fallback_num = []
    if fallback_den is None:
        fallback_den = pd.Series(0.0, index=stats.index)

    if isinstance(primary_num, str):
        primary_num = [primary_num]
    if isinstance(fallback_num, str):
        fallback_num = [fallback_num]

    primary_present = [c for c in primary_num if c in stats.columns]
    if primary_present:
        return safe_div(get_series_sum(primary_present), primary_den)

    fallback_present = [c for c in fallback_num if c in stats.columns]
    if fallback_present:
        return safe_div(get_series_sum(fallback_present), fallback_den)

    return safe_div(pd.Series(0.0, index=stats.index), primary_den)

def mpiio_collective_activity() -> pd.Series:
    return (
        get_series_sum("MPIIO_COLL_READS")
        + get_series_sum("MPIIO_COLL_WRITES")
        + get_series_sum("MPIIO_SPLIT_READS")
        + get_series_sum("MPIIO_SPLIT_WRITES")
    )

def module_reads() -> pd.Series:
    if MODULE == "MPIIO":
        return (
            get_series("MPIIO_INDEP_READS")
            + get_series("MPIIO_COLL_READS")
            + get_series("MPIIO_SPLIT_READS")
            + get_series("MPIIO_NB_READS")
        )
    return get_series(mcol("READS"))

def module_writes() -> pd.Series:
    if MODULE == "MPIIO":
        return (
            get_series("MPIIO_INDEP_WRITES")
            + get_series("MPIIO_COLL_WRITES")
            + get_series("MPIIO_SPLIT_WRITES")
            + get_series("MPIIO_NB_WRITES")
        )
    return get_series(mcol("WRITES"))

def safe_div(num, den):
    num = num.astype(float)
    den = den.astype(float)
    return (num / den).fillna(0).replace([np.inf, -np.inf], 0)

# prepare feature frame
feats = pd.DataFrame(index=stats.index)

# ─── metadata operation categories ─────────────────────────────────────────────
if MODULE == "MPIIO":
    # Darshan MPIIO does not expose POSIX-like META category keys directly
    # (e.g., MPIIO_OPENS/STATS/SEEKS/FSYNCS/FDSYNCS). For synthesized runs,
    # planner metadata phases currently execute through POSIX calls, so use
    # POSIX counters as primary metadata signal and MPIIO open/sync counters
    # only as fallback.
    meta_counts = {
        "open": choose_series(
            ["POSIX_OPENS"],
            ["MPIIO_OPENS", "MPIIO_INDEP_OPENS", "MPIIO_COLL_OPENS"],
        ),
        "stat": choose_series(["POSIX_STATS"], ["MPIIO_STATS"]),
        "seek": choose_series(["POSIX_SEEKS"], ["MPIIO_SEEKS"]),
        "sync": choose_series(
            ["POSIX_FSYNCS", "POSIX_FDSYNCS"],
            ["MPIIO_SYNCS", "MPIIO_FSYNCS", "MPIIO_FDSYNCS"],
        ),
    }
else:
    meta_groups = {
        # opening/closing file descriptors
        'open': [mcol('OPENS')],
        # querying file attributes
        'stat': [mcol('STATS')],
        # repositioning file pointers
        'seek': [mcol('SEEKS')],
        # explicit on-disk flushes
        'sync': [mcol('FSYNCS'), mcol('FDSYNCS')],
    }

    # count per metadata category
    meta_counts = {
        cat: sum(get_series(c) for c in cols)
        for cat, cols in meta_groups.items()
    }

# total metadata ops (will be used to split out pct_io vs pct_meta)
meta_count = sum(meta_counts.values())

# derived totals
stats['total_accesses'] = module_reads() + module_writes()
total_acc = stats['total_accesses']
total_bytes = stats['total_bytes']

# ─── ratio features ───────────────────────────────────────────────────────────
# alignment percentages
if MODULE == "MPIIO":
    coll_active = mpiio_collective_activity() > 0
    mpi_file_na = choose_series("MPIIO_FILE_NOT_ALIGNED", allow_fallback=False)
    posix_file_na = choose_series("POSIX_FILE_NOT_ALIGNED", allow_fallback=False)
    # Collective MPIIO may generate ROMIO-internal POSIX activity that does not
    # map to planner file-alignment intent; only fallback to POSIX for non-collective.
    file_na_num = mpi_file_na.where((mpi_file_na > 0) | coll_active, posix_file_na)

    mpi_mem_na = choose_series("MPIIO_MEM_NOT_ALIGNED", allow_fallback=False)
    posix_mem_na = choose_series("POSIX_MEM_NOT_ALIGNED", allow_fallback=False)
    mem_na_num = mpi_mem_na.where((mpi_mem_na > 0) | coll_active, posix_mem_na)
else:
    file_na_num = get_series(mcol('FILE_NOT_ALIGNED'))
    mem_na_num = get_series(mcol('MEM_NOT_ALIGNED'))

feats['pct_file_not_aligned'] = safe_div(file_na_num, total_acc)
feats['pct_mem_not_aligned']  = safe_div(mem_na_num, total_acc)
# read/write percentages
reads_total = module_reads()
writes_total = module_writes()
feats['pct_reads']         = safe_div(reads_total, total_acc)
feats['pct_writes']        = safe_div(writes_total, total_acc)
if MODULE == "MPIIO":
    rw_switches = choose_series("MPIIO_RW_SWITCHES", ["POSIX_RW_SWITCHES"])
    posix_reads_total = get_series_sum("POSIX_READS")
    posix_writes_total = get_series_sum("POSIX_WRITES")
    pct_consec_reads = ratio_from_counter_pair(
        "MPIIO_CONSEC_READS",
        reads_total,
        "POSIX_CONSEC_READS",
        posix_reads_total,
    )
    pct_consec_writes = ratio_from_counter_pair(
        "MPIIO_CONSEC_WRITES",
        writes_total,
        "POSIX_CONSEC_WRITES",
        posix_writes_total,
    )
    pct_seq_reads = ratio_from_counter_pair(
        "MPIIO_SEQ_READS",
        reads_total,
        "POSIX_SEQ_READS",
        posix_reads_total,
    )
    pct_seq_writes = ratio_from_counter_pair(
        "MPIIO_SEQ_WRITES",
        writes_total,
        "POSIX_SEQ_WRITES",
        posix_writes_total,
    )
    # Avoid counting ROMIO/internal POSIX reads for write-only MPIIO jobs.
    feats['pct_consec_reads'] = pct_consec_reads.where(reads_total > 0, 0.0)
    feats['pct_seq_reads'] = pct_seq_reads.where(reads_total > 0, 0.0)
    feats['pct_consec_writes'] = pct_consec_writes.where(writes_total > 0, 0.0)
    feats['pct_seq_writes'] = pct_seq_writes.where(writes_total > 0, 0.0)
else:
    rw_switches = get_series(mcol('RW_SWITCHES'))
    feats['pct_consec_reads']  = safe_div(get_series(mcol('CONSEC_READS')), reads_total)
    feats['pct_consec_writes'] = safe_div(get_series(mcol('CONSEC_WRITES')), writes_total)
    feats['pct_seq_reads']     = safe_div(get_series(mcol('SEQ_READS')), reads_total)
    feats['pct_seq_writes']    = safe_div(get_series(mcol('SEQ_WRITES')), writes_total)

feats['pct_rw_switches']   = safe_div(rw_switches, total_acc)
# byte traffic percentages
feats['pct_byte_reads']    = safe_div(get_series(mcol('BYTES_READ')), total_bytes)
feats['pct_byte_writes']   = safe_div(get_series(mcol('BYTES_WRITTEN')), total_bytes)
# IO vs metadata operations
ops_total = total_acc + meta_count
feats['pct_io_access']     = safe_div(total_acc, ops_total)
for cat, cnt in meta_counts.items():
    feats[f'pct_meta_{cat}_access'] = safe_div(cnt, ops_total)

# ─── condensed size-range features ────────────────────────────────────────────
# Read size bins
if MODULE == "MPIIO":
    read_bins_0_100K      = ['MPIIO_SIZE_READ_AGG_0_100','MPIIO_SIZE_READ_AGG_100_1K','MPIIO_SIZE_READ_AGG_1K_10K','MPIIO_SIZE_READ_AGG_10K_100K']
    read_bins_100K_10M    = ['MPIIO_SIZE_READ_AGG_100K_1M','MPIIO_SIZE_READ_AGG_1M_4M','MPIIO_SIZE_READ_AGG_4M_10M']
    read_bins_10M_1G_PLUS = ['MPIIO_SIZE_READ_AGG_10M_100M','MPIIO_SIZE_READ_AGG_100M_1G','MPIIO_SIZE_READ_AGG_1G_PLUS']
else:
    read_bins_0_100K      = ['POSIX_SIZE_READ_0_100','POSIX_SIZE_READ_100_1K','POSIX_SIZE_READ_1K_10K','POSIX_SIZE_READ_10K_100K']
    read_bins_100K_10M    = ['POSIX_SIZE_READ_100K_1M','POSIX_SIZE_READ_1M_4M','POSIX_SIZE_READ_4M_10M']
    read_bins_10M_1G_PLUS = ['POSIX_SIZE_READ_10M_100M','POSIX_SIZE_READ_100M_1G','POSIX_SIZE_READ_1G_PLUS']
sum_read_0_100K      = sum(get_series(c) for c in read_bins_0_100K)
sum_read_100K_10M    = sum(get_series(c) for c in read_bins_100K_10M)
sum_read_10M_1G_PLUS = sum(get_series(c) for c in read_bins_10M_1G_PLUS)
feats['pct_read_0_100K']      = safe_div(sum_read_0_100K, reads_total)
feats['pct_read_100K_10M']    = safe_div(sum_read_100K_10M, reads_total)
feats['pct_read_10M_1G_PLUS'] = safe_div(sum_read_10M_1G_PLUS, reads_total)

# Write size bins
if MODULE == "MPIIO":
    write_bins_0_100K      = ['MPIIO_SIZE_WRITE_AGG_0_100','MPIIO_SIZE_WRITE_AGG_100_1K','MPIIO_SIZE_WRITE_AGG_1K_10K','MPIIO_SIZE_WRITE_AGG_10K_100K']
    write_bins_100K_10M    = ['MPIIO_SIZE_WRITE_AGG_100K_1M','MPIIO_SIZE_WRITE_AGG_1M_4M','MPIIO_SIZE_WRITE_AGG_4M_10M']
    write_bins_10M_1G_PLUS = ['MPIIO_SIZE_WRITE_AGG_10M_100M','MPIIO_SIZE_WRITE_AGG_100M_1G','MPIIO_SIZE_WRITE_AGG_1G_PLUS']
else:
    write_bins_0_100K      = ['POSIX_SIZE_WRITE_0_100','POSIX_SIZE_WRITE_100_1K','POSIX_SIZE_WRITE_1K_10K','POSIX_SIZE_WRITE_10K_100K']
    write_bins_100K_10M    = ['POSIX_SIZE_WRITE_100K_1M','POSIX_SIZE_WRITE_1M_4M','POSIX_SIZE_WRITE_4M_10M']
    write_bins_10M_1G_PLUS = ['POSIX_SIZE_WRITE_10M_100M','POSIX_SIZE_WRITE_100M_1G','POSIX_SIZE_WRITE_1G_PLUS']
sum_write_0_100K      = sum(get_series(c) for c in write_bins_0_100K)
sum_write_100K_10M    = sum(get_series(c) for c in write_bins_100K_10M)
sum_write_10M_1G_PLUS = sum(get_series(c) for c in write_bins_10M_1G_PLUS)
write_den = writes_total
if MODULE == "MPIIO":
    # In collective mode we may pad collective call counts with zero-length ops
    # so all ranks participate in MPI_File_*_at_all. Darshan counts those as
    # 0-100B writes and they can bias size-bin percentages. Remove this artifact
    # when access-size counters indicate no genuine sub-100KiB accesses.
    coll_writes = get_series_sum(["MPIIO_COLL_WRITES", "MPIIO_SPLIT_WRITES"])
    acc_small_counts = pd.Series(0.0, index=stats.index)
    for i in range(1, 5):
        access = get_series_sum(f"MPIIO_ACCESS{i}_ACCESS")
        count = get_series_sum(f"MPIIO_ACCESS{i}_COUNT")
        acc_small_counts = acc_small_counts + count.where((access > 0) & (access < 100 * 1024), 0.0)

    pad_like_small = sum_write_0_100K.where((coll_writes > 0) & (acc_small_counts == 0), 0.0)
    sum_write_0_100K = (sum_write_0_100K - pad_like_small).clip(lower=0.0)
    write_den = (write_den - pad_like_small).clip(lower=0.0)

feats['pct_write_0_100K']      = safe_div(sum_write_0_100K, write_den)
feats['pct_write_100K_10M']    = safe_div(sum_write_100K_10M, write_den)
feats['pct_write_10M_1G_PLUS'] = safe_div(sum_write_10M_1G_PLUS, write_den)

# ─── POSIX file-type features ─────────────────────────────────────────────────
# group 1: shared vs unique
ft_group1   = ['shared', 'unique']
file_sum_g1 = sum(get_series(f'{MODULE}_file_type_{ft}_file_count')   for ft in ft_group1).clip(lower=0)
byte_sum_g1 = sum(get_series(f'{MODULE}_file_type_{ft}_total_bytes') for ft in ft_group1).clip(lower=0)
for ft in ft_group1:
    feats[f'pct_{ft}_files']       = safe_div(get_series(f'{MODULE}_file_type_{ft}_file_count'), file_sum_g1)
    feats[f'pct_bytes_{ft}_files'] = safe_div(get_series(f'{MODULE}_file_type_{ft}_total_bytes'), byte_sum_g1)

# group 2: read_only / read_write / write_only
ft_group2   = ['read_only', 'read_write', 'write_only']
file_sum_g2 = sum(get_series(f'{MODULE}_file_type_{ft}_file_count')   for ft in ft_group2).clip(lower=0)
byte_sum_g2 = sum(get_series(f'{MODULE}_file_type_{ft}_total_bytes') for ft in ft_group2).clip(lower=0)
for ft in ft_group2:
    feats[f'pct_{ft}_files']       = safe_div(get_series(f'{MODULE}_file_type_{ft}_file_count'), file_sum_g2)
    feats[f'pct_bytes_{ft}_files'] = safe_div(get_series(f'{MODULE}_file_type_{ft}_total_bytes'), byte_sum_g2)

# ─── Assemble output objects ──────────────────────────────────────────────────
out = pd.concat(
    [stats[['filename','jobid','nprocs']].reset_index(drop=True),
     feats.reset_index(drop=True)],
    axis=1
)

# round to 2 decimals for stability
out = out.round(2)

# Replace NaNs with 0 for JSON compatibility
out = out.fillna(0)

# Convert each row to a dict “features.json”-style
records = out.to_dict(orient='records')
for rec in records:
    rec["feature_module_basis"] = MODULE

# Write a single JSON array file
with open(output_json, "w") as jf:
    json.dump(records, jf, indent=2)

print(f"Wrote features for {len(records)} jobs to {output_json}")
