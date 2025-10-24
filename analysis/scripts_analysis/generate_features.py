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
args = parser.parse_args()
root_dir     = os.path.abspath(args.root)
input_csv    = os.path.join(root_dir, "darshan_summary.csv")
filtered_csv = os.path.join(root_dir, "darshan_summary_filtered.csv")
output_json  = os.path.join(root_dir, "darshan_features_updated.json")
# filter threshold (bytes)
threshold    = 100 * 1024 * 1024  # 100 MB
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
df_raw['total_bytes'] = df_raw.get('POSIX_BYTES_READ', 0) + df_raw.get('POSIX_BYTES_WRITTEN', 0)
stats = df_raw[df_raw['total_bytes'] >= threshold].reset_index(drop=True)
processed_count = len(stats)
print(f"Filtered to {processed_count} rows with total_bytes >= {threshold}")
print(f"Input samples: {initial_count}, Processed samples: {processed_count}")
stats.to_csv(filtered_csv, index=False)

# helper functions
def get_series(col):
    return stats[col] if col in stats.columns else pd.Series(0, index=stats.index)

def safe_div(num, den):
    num = num.astype(float)
    den = den.astype(float)
    return (num / den).fillna(0).replace([np.inf, -np.inf], 0)

# prepare feature frame
feats = pd.DataFrame(index=stats.index)

# ─── metadata operation categories ─────────────────────────────────────────────
meta_groups = {
    # opening/closing file descriptors
    'open': ['POSIX_OPENS'],
    # querying file attributes
    'stat': ['POSIX_STATS'],
    # repositioning file pointers
    'seek': ['POSIX_SEEKS'],
    # explicit on-disk flushes
    'sync': ['POSIX_FSYNCS', 'POSIX_FDSYNCS'],
}

# count per metadata category
meta_counts = {
    cat: sum(get_series(c) for c in cols)
    for cat, cols in meta_groups.items()
}

# total metadata ops (will be used to split out pct_io vs pct_meta)
meta_count = sum(meta_counts.values())

# derived totals
stats['total_accesses'] = get_series('POSIX_READS') + get_series('POSIX_WRITES')
total_acc = stats['total_accesses']
total_bytes = stats['total_bytes']

# ─── ratio features ───────────────────────────────────────────────────────────
# alignment percentages
feats['pct_file_not_aligned'] = safe_div(get_series('POSIX_FILE_NOT_ALIGNED'), total_acc)
feats['pct_mem_not_aligned']  = safe_div(get_series('POSIX_MEM_NOT_ALIGNED'), total_acc)
# read/write percentages
feats['pct_reads']         = safe_div(get_series('POSIX_READS'), total_acc)
feats['pct_writes']        = safe_div(get_series('POSIX_WRITES'), total_acc)
feats['pct_consec_reads']  = safe_div(get_series('POSIX_CONSEC_READS'), get_series('POSIX_READS'))
feats['pct_consec_writes'] = safe_div(get_series('POSIX_CONSEC_WRITES'), get_series('POSIX_WRITES'))
feats['pct_seq_reads']     = safe_div(get_series('POSIX_SEQ_READS'), get_series('POSIX_READS'))
feats['pct_seq_writes']    = safe_div(get_series('POSIX_SEQ_WRITES'), get_series('POSIX_WRITES'))
feats['pct_rw_switches']   = safe_div(get_series('POSIX_RW_SWITCHES'), total_acc)
# byte traffic percentages
feats['pct_byte_reads']    = safe_div(get_series('POSIX_BYTES_READ'), total_bytes)
feats['pct_byte_writes']   = safe_div(get_series('POSIX_BYTES_WRITTEN'), total_bytes)
# IO vs metadata operations
ops_total = total_acc + meta_count
feats['pct_io_access']     = safe_div(total_acc, ops_total)
for cat, cnt in meta_counts.items():
    feats[f'pct_meta_{cat}_access'] = safe_div(cnt, ops_total)

# ─── condensed size-range features ────────────────────────────────────────────
# Read size bins
read_bins_0_100K      = ['POSIX_SIZE_READ_0_100','POSIX_SIZE_READ_100_1K','POSIX_SIZE_READ_1K_10K','POSIX_SIZE_READ_10K_100K']
read_bins_100K_10M    = ['POSIX_SIZE_READ_100K_1M','POSIX_SIZE_READ_1M_4M','POSIX_SIZE_READ_4M_10M']
read_bins_10M_1G_PLUS = ['POSIX_SIZE_READ_10M_100M','POSIX_SIZE_READ_100M_1G','POSIX_SIZE_READ_1G_PLUS']
sum_read_0_100K      = sum(get_series(c) for c in read_bins_0_100K)
sum_read_100K_10M    = sum(get_series(c) for c in read_bins_100K_10M)
sum_read_10M_1G_PLUS = sum(get_series(c) for c in read_bins_10M_1G_PLUS)
feats['pct_read_0_100K']      = safe_div(sum_read_0_100K, get_series('POSIX_READS'))
feats['pct_read_100K_10M']    = safe_div(sum_read_100K_10M, get_series('POSIX_READS'))
feats['pct_read_10M_1G_PLUS'] = safe_div(sum_read_10M_1G_PLUS, get_series('POSIX_READS'))

# Write size bins
write_bins_0_100K      = ['POSIX_SIZE_WRITE_0_100','POSIX_SIZE_WRITE_100_1K','POSIX_SIZE_WRITE_1K_10K','POSIX_SIZE_WRITE_10K_100K']
write_bins_100K_10M    = ['POSIX_SIZE_WRITE_100K_1M','POSIX_SIZE_WRITE_1M_4M','POSIX_SIZE_WRITE_4M_10M']
write_bins_10M_1G_PLUS = ['POSIX_SIZE_WRITE_10M_100M','POSIX_SIZE_WRITE_100M_1G','POSIX_SIZE_WRITE_1G_PLUS']
sum_write_0_100K      = sum(get_series(c) for c in write_bins_0_100K)
sum_write_100K_10M    = sum(get_series(c) for c in write_bins_100K_10M)
sum_write_10M_1G_PLUS = sum(get_series(c) for c in write_bins_10M_1G_PLUS)
feats['pct_write_0_100K']      = safe_div(sum_write_0_100K, get_series('POSIX_WRITES'))
feats['pct_write_100K_10M']    = safe_div(sum_write_100K_10M, get_series('POSIX_WRITES'))
feats['pct_write_10M_1G_PLUS'] = safe_div(sum_write_10M_1G_PLUS, get_series('POSIX_WRITES'))

# ─── POSIX file-type features ─────────────────────────────────────────────────
# group 1: shared vs unique
ft_group1   = ['shared', 'unique']
file_sum_g1 = sum(get_series(f'POSIX_file_type_{ft}_file_count')   for ft in ft_group1).clip(lower=0)
byte_sum_g1 = sum(get_series(f'POSIX_file_type_{ft}_total_bytes') for ft in ft_group1).clip(lower=0)
for ft in ft_group1:
    feats[f'pct_{ft}_files']       = safe_div(get_series(f'POSIX_file_type_{ft}_file_count'), file_sum_g1)
    feats[f'pct_bytes_{ft}_files'] = safe_div(get_series(f'POSIX_file_type_{ft}_total_bytes'), byte_sum_g1)

# group 2: read_only / read_write / write_only
ft_group2   = ['read_only', 'read_write', 'write_only']
file_sum_g2 = sum(get_series(f'POSIX_file_type_{ft}_file_count')   for ft in ft_group2).clip(lower=0)
byte_sum_g2 = sum(get_series(f'POSIX_file_type_{ft}_total_bytes') for ft in ft_group2).clip(lower=0)
for ft in ft_group2:
    feats[f'pct_{ft}_files']       = safe_div(get_series(f'POSIX_file_type_{ft}_file_count'), file_sum_g2)
    feats[f'pct_bytes_{ft}_files'] = safe_div(get_series(f'POSIX_file_type_{ft}_total_bytes'), byte_sum_g2)

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

# Write a single JSON array file
with open(output_json, "w") as jf:
    json.dump(records, jf, indent=2)

print(f"Wrote features for {len(records)} jobs to {output_json}")
