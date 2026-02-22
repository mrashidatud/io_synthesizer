from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_learning_curves(history: Sequence[Mapping[str, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [h["runs"] for h in history]

    for key, ylabel in [
        ("regret_at_3", "Regret@3"),
        ("hit_at_3", "Hit@3 (within 5%)"),
        ("ndcg_at_3", "NDCG@3"),
    ]:
        y = [h.get(key, 0.0) for h in history]
        plt.figure(figsize=(6, 4))
        plt.plot(runs, y, marker="o")
        plt.xlabel("# Runs")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{key}.png", dpi=160)
        plt.close()
