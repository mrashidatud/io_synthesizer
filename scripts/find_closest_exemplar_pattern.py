#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


RANK_RE = re.compile(r"^top(?P<rank>\d+)_")

HIGH_PRIORITY = {
    "pct_rw_switches",
    "pct_file_not_aligned",
    "pct_mem_not_aligned",
    "pct_io_access",
    "pct_seq_reads",
    "pct_seq_writes",
    "pct_consec_reads",
    "pct_consec_writes",
}

MEDIUM_PRIORITY = {
    "pct_byte_reads",
    "pct_byte_writes",
    "pct_reads",
    "pct_writes",
    "pct_read_0_100K",
    "pct_read_100K_10M",
    "pct_read_10M_1G_PLUS",
    "pct_write_0_100K",
    "pct_write_100K_10M",
    "pct_write_10M_1G_PLUS",
}


@dataclass(frozen=True)
class Pattern:
    name: str
    path: Path
    rank: int
    payload: dict


def is_numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_rank(path: Path) -> int | None:
    match = RANK_RE.match(path.stem)
    if not match:
        return None
    return int(match.group("rank"))


def load_patterns(input_dir: Path, top_n: int) -> list[Pattern]:
    patterns: list[Pattern] = []
    for path in sorted(input_dir.glob("top*.json")):
        rank = parse_rank(path)
        if rank is None or rank > top_n:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        patterns.append(Pattern(name=path.stem, path=path, rank=rank, payload=payload))
    patterns.sort(key=lambda p: p.rank)
    return patterns


def infer_pct_features(patterns: list[Pattern]) -> list[str]:
    if not patterns:
        return []
    keys = {
        k
        for k, v in patterns[0].payload.items()
        if k.startswith("pct_") and is_numeric(v)
    }
    for pat in patterns[1:]:
        pat_keys = {
            k
            for k, v in pat.payload.items()
            if k.startswith("pct_") and is_numeric(v)
        }
        keys &= pat_keys
    return sorted(keys)


def feature_vector(pattern: Pattern, features: list[str]) -> list[float]:
    return [float(pattern.payload.get(k, 0.0)) for k in features]


def priority_for_feature(feature: str) -> str:
    if feature in HIGH_PRIORITY:
        return "high"
    if feature in MEDIUM_PRIORITY:
        return "medium"
    return "low"


def weighted_l2_distance(
    a: list[float],
    b: list[float],
    features: list[str],
    priority_weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    totals = {"high": 0.0, "medium": 0.0, "low": 0.0}
    weighted_sum = 0.0
    for x, y, feat in zip(a, b, features):
        prio = priority_for_feature(feat)
        w = priority_weights[prio]
        sq = (x - y) ** 2
        totals[prio] += sq
        weighted_sum += w * sq
    return math.sqrt(weighted_sum), totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find which top-N exemplar pattern is closest to any of the first K patterns "
            "using numeric pct_* features."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("inputs/exemplar_jsons"),
        help="Directory containing top*.json exemplars (default: inputs/exemplar_jsons).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Only consider patterns with rank <= top-n (default: 25).",
    )
    parser.add_argument(
        "--reference-top-k",
        type=int,
        default=4,
        help="Use top1..topK as reference patterns (default: 4).",
    )
    parser.add_argument(
        "--include-references",
        action="store_true",
        help="If set, allow top1..topK to appear in candidate set too.",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=25,
        help="How many ranked candidate rows to print (default: 25).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write detailed ranking JSON.",
    )
    parser.add_argument(
        "--high-weight",
        type=float,
        default=4.0,
        help="Weight for high-priority features (default: 4.0).",
    )
    parser.add_argument(
        "--medium-weight",
        type=float,
        default=2.0,
        help="Weight for medium-priority features (default: 2.0).",
    )
    parser.add_argument(
        "--low-weight",
        type=float,
        default=1.0,
        help="Weight for low-priority features (default: 1.0).",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if args.reference_top_k < 1:
        raise ValueError("--reference-top-k must be >= 1")
    if args.top_n < args.reference_top_k:
        raise ValueError("--top-n must be >= --reference-top-k")
    if args.high_weight < 0 or args.medium_weight < 0 or args.low_weight < 0:
        raise ValueError("All weights must be >= 0")

    patterns = load_patterns(input_dir=input_dir, top_n=args.top_n)
    if len(patterns) < args.reference_top_k:
        raise ValueError(
            f"Need at least {args.reference_top_k} patterns, but found {len(patterns)} under {input_dir}"
        )

    refs = [p for p in patterns if p.rank <= args.reference_top_k]
    if args.include_references:
        candidates = patterns
    else:
        candidates = [p for p in patterns if p.rank > args.reference_top_k]

    if not candidates:
        raise ValueError("No candidate patterns available. Try --include-references or increase --top-n.")

    features = infer_pct_features(patterns)
    if not features:
        raise ValueError("No shared numeric pct_* features found across selected patterns.")

    priority_weights = {
        "high": float(args.high_weight),
        "medium": float(args.medium_weight),
        "low": float(args.low_weight),
    }

    feature_counts = {
        "high": sum(1 for f in features if priority_for_feature(f) == "high"),
        "medium": sum(1 for f in features if priority_for_feature(f) == "medium"),
        "low": sum(1 for f in features if priority_for_feature(f) == "low"),
    }

    vectors = {p.name: feature_vector(p, features) for p in patterns}

    ranked: list[dict] = []
    for cand in candidates:
        best_ref = None
        best_dist = float("inf")
        best_unweighted = float("inf")
        best_contrib = {"high": 0.0, "medium": 0.0, "low": 0.0}
        for ref in refs:
            dist, contrib = weighted_l2_distance(
                vectors[cand.name],
                vectors[ref.name],
                features=features,
                priority_weights=priority_weights,
            )
            if dist < best_dist:
                best_dist = dist
                best_ref = ref
                best_contrib = contrib
                best_unweighted = math.sqrt(sum((x - y) ** 2 for x, y in zip(vectors[cand.name], vectors[ref.name])))
        ranked.append(
            {
                "candidate": cand.name,
                "candidate_rank": cand.rank,
                "closest_reference": best_ref.name,
                "closest_reference_rank": best_ref.rank,
                "weighted_distance": best_dist,
                "unweighted_distance": best_unweighted,
                "priority_squared_error": best_contrib,
            }
        )

    ranked.sort(key=lambda row: row["weighted_distance"])
    best = ranked[0]

    print(f"Input dir: {input_dir}")
    print(f"Using {len(features)} shared numeric pct_* features")
    print(
        "Priority feature counts: "
        f"high={feature_counts['high']}, medium={feature_counts['medium']}, low={feature_counts['low']}"
    )
    print(
        f"Weights: high={priority_weights['high']}, "
        f"medium={priority_weights['medium']}, low={priority_weights['low']}"
    )
    print(
        f"Closest candidate: {best['candidate']} (top{best['candidate_rank']}) "
        f"-> {best['closest_reference']} (top{best['closest_reference_rank']}), "
        f"weighted_distance={best['weighted_distance']:.6f}, "
        f"unweighted_distance={best['unweighted_distance']:.6f}"
    )
    print("")
    print("Ranked candidates by minimum weighted distance to any reference:")
    limit = max(1, min(args.show_top, len(ranked)))
    for idx, row in enumerate(ranked[:limit], start=1):
        print(
            f"{idx:2d}. {row['candidate']:>10s} (top{row['candidate_rank']:>2d})  "
            f"nearest={row['closest_reference']:>10s} (top{row['closest_reference_rank']:>2d})  "
            f"weighted={row['weighted_distance']:.6f}  "
            f"unweighted={row['unweighted_distance']:.6f}"
        )

    if args.output_json is not None:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "input_dir": str(input_dir),
            "top_n": args.top_n,
            "reference_top_k": args.reference_top_k,
            "include_references": bool(args.include_references),
            "metric": "weighted_euclidean_l2",
            "priority_weights": priority_weights,
            "priority_feature_counts": feature_counts,
            "feature_count": len(features),
            "features": features,
            "best_match": best,
            "ranked_matches": ranked,
        }
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("")
        print(f"Wrote detailed report: {output_path}")


if __name__ == "__main__":
    main()
