#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


META_TIME_COLS = [
    "POSIX_F_META_TIME",
    "MPIIO_F_META_TIME",
    "STDIO_F_META_TIME",
]
IO_TIME_COLS = [
    "POSIX_F_READ_TIME",
    "POSIX_F_WRITE_TIME",
    "MPIIO_F_READ_TIME",
    "MPIIO_F_WRITE_TIME",
    "STDIO_F_READ_TIME",
    "STDIO_F_WRITE_TIME",
]

# Percent-of-ops features from exemplar extraction.
OPS_IO_COL = "pct_io_access"
OPS_META_COLS = [
    "pct_meta_open_access",
    "pct_meta_stat_access",
    "pct_meta_seek_access",
    "pct_meta_sync_access",
]
FILE_COUNT_COL = "POSIX_file_type_total_file_count"


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _clip01(s: pd.Series) -> pd.Series:
    return s.clip(lower=0.0, upper=1.0)


def _fmt_value(v: float, *, is_int: bool = False) -> str:
    if is_int:
        return f"{int(round(v)):,}"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:,.1f}"
    return f"{v:.2f}"


def build_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["run_time"] = _num(df, "run_time")

    raw_meta_t = sum((_num(df, c) for c in META_TIME_COLS), start=pd.Series(np.zeros(len(df)), index=df.index))
    raw_io_t = sum((_num(df, c) for c in IO_TIME_COLS), start=pd.Series(np.zeros(len(df)), index=df.index))
    raw_total_t = raw_meta_t + raw_io_t

    io_ops = _clip01(_num(df, OPS_IO_COL))
    meta_ops = _clip01(sum((_num(df, c) for c in OPS_META_COLS), start=pd.Series(np.zeros(len(df)), index=df.index)))

    ops_total = io_ops + meta_ops
    need_norm = ops_total > 0
    io_ops_norm = io_ops.copy()
    meta_ops_norm = meta_ops.copy()
    io_ops_norm.loc[need_norm] = io_ops.loc[need_norm] / ops_total.loc[need_norm]
    meta_ops_norm.loc[need_norm] = meta_ops.loc[need_norm] / ops_total.loc[need_norm]

    # Keep whole bar equal to end-to-end runtime.
    # Prefer Darshan time split when available; fallback to ops split when missing.
    meta_share = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    has_time = raw_total_t > 0
    meta_share.loc[has_time] = raw_meta_t.loc[has_time] / raw_total_t.loc[has_time]

    no_time = ~has_time
    meta_share.loc[no_time] = meta_ops_norm.loc[no_time]
    meta_share = _clip01(meta_share)

    out["meta_time"] = out["run_time"] * meta_share
    out["io_time"] = out["run_time"] - out["meta_time"]
    out["meta_time_pct"] = meta_share * 100.0
    out["io_time_pct"] = 100.0 - out["meta_time_pct"]

    out["io_ops_pct"] = io_ops_norm * 100.0
    out["meta_ops_pct"] = meta_ops_norm * 100.0
    out["total_file_count"] = _num(df, FILE_COUNT_COL)

    # Friendly workload labels while preserving CSV order.
    if "label" in df.columns:
        labels = [f"top{i+1} (L{int(v)})" for i, v in enumerate(pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int))]
    else:
        labels = [f"top{i+1}" for i in range(len(df))]
    out["workload"] = labels

    return out


def plot_stacked(plot_df: pd.DataFrame, output_path: Path, title: str | None = None) -> None:
    x = np.arange(len(plot_df))
    width = 0.82

    fig, axes = plt.subplots(2, 1, figsize=(24, 13), sharex=True)

    # Subplot 1: normalized runtime composition + total runtime line.
    axes[0].bar(x, plot_df["io_time_pct"], width=width, label="I/O Time %", color="#17becf")
    axes[0].bar(x, plot_df["meta_time_pct"], width=width, bottom=plot_df["io_time_pct"], label="Meta Time %", color="#bcbd22")
    axes[0].set_ylabel("Percent of Runtime")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Normalized I/O Time % vs Meta Time % (bars) + Total Runtime (line)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    ax0r = axes[0].twinx()
    ax0r.plot(x, plot_df["run_time"], color="#222222", marker="o", linewidth=1.8, label="Total Runtime (s)")
    ax0r.set_ylabel("Total Runtime (s)")
    for xi, yi in zip(x, plot_df["run_time"]):
        ax0r.annotate(_fmt_value(float(yi)), (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color="#111111")

    h0, l0 = axes[0].get_legend_handles_labels()
    h0r, l0r = ax0r.get_legend_handles_labels()
    axes[0].legend(h0 + h0r, l0 + l0r, loc="upper right")

    # Subplot 2: op-share composition + total file count line.
    axes[1].bar(x, plot_df["io_ops_pct"], width=width, label="I/O Ops %", color="#2ca02c")
    axes[1].bar(x, plot_df["meta_ops_pct"], width=width, bottom=plot_df["io_ops_pct"], label="Meta Ops %", color="#d62728")
    axes[1].set_ylabel("Percent of Total Ops")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("% I/O Ops vs % Meta Ops (bars) + Total File Count (line)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    ax1r = axes[1].twinx()
    ax1r.plot(x, plot_df["total_file_count"], color="#111111", marker="D", linewidth=1.8, label="Total File Count")
    ax1r.set_ylabel("Total File Count")
    for xi, yi in zip(x, plot_df["total_file_count"]):
        ax1r.annotate(_fmt_value(float(yi), is_int=True), (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color="#111111")

    h1, l1 = axes[1].get_legend_handles_labels()
    h1r, l1r = ax1r.get_legend_handles_labels()
    axes[1].legend(h1 + h1r, l1 + l1r, loc="upper right")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_df["workload"], rotation=45, ha="right")
    axes[1].set_xlabel("Workloads (CSV order)")

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.992)

    details = (
        "How to read this figure\n"
        "• Top subplot (runtime composition): stacked bars show I/O time % + Meta time % (each bar = 100%).\n"
        "• Top subplot line (right axis): total runtime in seconds from 'run_time'; each marker label is exact.\n"
        "• Bottom subplot (operation composition): stacked bars show I/O ops % + Meta ops % "
        "(pct_io_access vs sum of pct_meta_*; each bar = 100%).\n"
        f"• Bottom subplot line (right axis): total file count from '{FILE_COUNT_COL}'; each marker label is exact."
    )
    fig.text(
        0.5,
        0.93,
        details,
        ha="center",
        va="top",
        multialignment="center",
        fontsize=10,
        linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f7f7", edgecolor="#bfbfbf", alpha=0.97),
    )
    fig.subplots_adjust(top=0.76, bottom=0.12, hspace=0.30)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stacked bars for top25 exemplar runtime/ops composition")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/mnt/hasanfs/io_synthesizer/inputs/cluster_top25_exemplars_with_darshan.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/hasanfs/io_synthesizer/outputs/top25_meta_io_ops_stacked.png"),
        help="Output figure path",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Top-25 Exemplar Workloads: Runtime and Operation Composition",
        help="Optional overall figure title",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    plot_df = build_plot_df(df)
    plot_stacked(plot_df, args.output, title=args.title)
    print(f"Wrote figure: {args.output}")


if __name__ == "__main__":
    main()
