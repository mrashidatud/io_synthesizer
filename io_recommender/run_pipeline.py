#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep CLI stable on constrained systems.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

from io_recommender.pipeline import load_config, run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Warm-start + active learning IO recommender pipeline")
    p.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to YAML config",
    )
    p.add_argument("--output-dir", type=str, default="artifacts", help="Where to write outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    art = run_pipeline(cfg, Path(args.output_dir))

    print(f"Warm-start configs: {len(art.warm_configs)}")
    print(f"Total observations: {len(art.observations)}")
    print("Deployment top-3 demo:")
    for row in art.deployment_demo:
        print(f"  {row['config_id']} score={row['score']:.4f} config={row['config']}")


if __name__ == "__main__":
    main()
