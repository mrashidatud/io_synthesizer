from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Sequence

from io_recommender.types import Config, ParameterSpec


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _parse_darshan_from_run_script(run_sh: Path) -> Path | None:
    for line in run_sh.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export DARSHAN_LOGFILE="):
            payload = line.split("=", 1)[1].strip().strip("'").strip('"')
            if payload:
                return Path(payload)
    return None


def _kib_or_size_to_lfs(value: object) -> str:
    # Accept preformatted values ("64K", "1M", "256M"), or numeric KiB.
    if isinstance(value, str):
        s = value.strip().upper()
        if s.endswith(("K", "M", "G")):
            return s
        v = int(float(s))
    else:
        v = int(value)
    if v % (1024 * 1024) == 0:
        return f"{v // (1024 * 1024)}G"
    if v % 1024 == 0:
        return f"{v // 1024}M"
    return f"{v}K"


@dataclass
class RealSynthRunner:
    specs: Sequence[ParameterSpec]
    io_synth_root: Path
    input_dir: Path
    out_root: Path
    cap_total_gib: float = 512.0
    io_api: str = "posix"
    meta_api: str = "posix"
    mpi_collective_mode: str = "none"
    nprocs_cap: int = 64
    metric_key: str = "POSIX_agg_perf_by_slowest"
    metric_fallback: str = "bytes_over_f_time"
    delete_existing_darshan: bool = True
    flush_wait_sec: float = 10.0
    use_sudo_for_lustre: bool = False
    dry_run: bool = False
    feature_script_relpath: str = "scripts/features2synth_opsaware.py"
    analyze_script_relpath: str = "analysis/scripts_analysis/analyze_darshan_merged.py"
    pattern_file_by_id: Dict[str, Path] = field(default_factory=dict)
    plan_cache: Dict[str, Dict[str, Path]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.io_synth_root = Path(self.io_synth_root)
        self.input_dir = Path(self.input_dir)
        self.out_root = Path(self.out_root)
        self._discover_pattern_files()

    def _discover_pattern_files(self) -> None:
        if self.pattern_file_by_id:
            return
        mapping: Dict[str, Path] = {}
        for p in sorted(self.input_dir.glob("*.json")):
            mapping[p.stem] = p
        self.pattern_file_by_id = mapping

    def _cmd_lctl_set_param(self, key: str, value: object) -> list[str]:
        base = ["lctl", "set_param", f"osc.*.{key}={value}"]
        return ["sudo"] + base if self.use_sudo_for_lustre else base

    def _cmd_lfs_setstripe(self, stripe_count: object, stripe_size: object, path: Path) -> list[str]:
        base = [
            "lfs",
            "setstripe",
            "-c",
            str(int(stripe_count)),
            "-S",
            _kib_or_size_to_lfs(stripe_size),
            str(path),
        ]
        return ["sudo"] + base if self.use_sudo_for_lustre else base

    def _generate_plan(self, pattern_id: str) -> Dict[str, Path]:
        if pattern_id in self.plan_cache:
            return self.plan_cache[pattern_id]

        pattern_json = self.pattern_file_by_id.get(pattern_id)
        if pattern_json is None:
            raise KeyError(f"Pattern '{pattern_id}' not found in {self.input_dir}")

        feat_script = self.io_synth_root / self.feature_script_relpath
        pattern_obj = json.loads(pattern_json.read_text(encoding="utf-8"))
        nprocs_val = int(pattern_obj.get("nprocs", self.nprocs_cap))
        desired_nprocs = min(max(nprocs_val, 1), self.nprocs_cap)
        cmd = [
            "python3",
            str(feat_script),
            "--features",
            str(pattern_json),
            "--cap-total-gib",
            str(self.cap_total_gib),
            "--io-api",
            self.io_api,
            "--meta-api",
            self.meta_api,
            "--mpi-collective-mode",
            self.mpi_collective_mode,
            "--nprocs",
            str(desired_nprocs),
        ]
        if self.dry_run:
            print("[dry-run]", " ".join(cmd))
            generated = {
                "plan_csv": self.out_root / pattern_id / "payload/plan.csv",
                "prep_sh": self.out_root / pattern_id / "run_prep.sh",
                "run_sh": self.out_root / pattern_id / "run_from_features.sh",
                "notes": self.out_root / pattern_id / "run_from_features.sh.notes.txt",
            }
            self.plan_cache[pattern_id] = generated
            return generated

        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        text = completed.stdout.strip()
        # Last JSON object printed by features2synth script.
        start = text.rfind("{")
        if start < 0:
            raise RuntimeError(f"Failed to parse planner output for pattern '{pattern_id}': {text}")
        payload = json.loads(text[start:])
        generated = {k: Path(v) for k, v in payload.items()}
        self.plan_cache[pattern_id] = generated
        return generated

    def _apply_knobs_before_run(self, run_root: Path, config: Config) -> None:
        payload = run_root / "payload"
        data_ro = payload / "data_ro"
        data_rw = payload / "data_rw"
        data_wo = payload / "data_wo"
        meta = payload / "meta"
        targets = [payload, data_ro, data_rw, data_wo, meta]

        for t in targets:
            t.mkdir(parents=True, exist_ok=True)

        # Ensure new files are created with the new stripe policy.
        for t in [data_ro, data_rw, data_wo, meta]:
            for p in t.rglob("*"):
                if p.is_file():
                    p.unlink(missing_ok=True)

        # New stripe policy must be applied before run_prep.sh creates files.
        for t in targets:
            _run(self._cmd_lfs_setstripe(config["stripe_count"], config["stripe_size"], t))

        _run(self._cmd_lctl_set_param("max_pages_per_rpc", config["max_pages_per_rpc"]))
        _run(self._cmd_lctl_set_param("max_rpcs_in_flight", config["max_rpcs_in_flight"]))

    @staticmethod
    def _metric_from_csv(csv_path: Path, metric_key: str, metric_fallback: str) -> float:
        with csv_path.open("r", encoding="utf-8") as f:
            row = next(csv.DictReader(f))
        if metric_key in row and row[metric_key] not in ("", None):
            return float(row[metric_key])

        if metric_fallback == "bytes_over_f_time":
            bytes_rw = float(row.get("POSIX_BYTES_READ", 0.0)) + float(row.get("POSIX_BYTES_WRITTEN", 0.0))
            total_f_time = (
                float(row.get("POSIX_F_READ_TIME", 0.0))
                + float(row.get("POSIX_F_WRITE_TIME", 0.0))
                + float(row.get("POSIX_F_META_TIME", 0.0))
            )
            if total_f_time <= 1e-12:
                return 0.0
            return bytes_rw / total_f_time / (1024.0 * 1024.0)
        raise KeyError(f"Metric key '{metric_key}' missing in {csv_path}")

    def run_testbed(self, pattern_id: str, config: Config, workload_vec=None) -> float:
        generated = self._generate_plan(pattern_id)
        run_sh = generated["run_sh"]
        if self.dry_run:
            return 0.0
        run_root = run_sh.parent
        run_root.mkdir(parents=True, exist_ok=True)

        if self.delete_existing_darshan:
            for d in run_root.glob("*.darshan"):
                d.unlink(missing_ok=True)

        self._apply_knobs_before_run(run_root, config)

        _run(["bash", str(run_sh)], cwd=self.io_synth_root)
        if self.flush_wait_sec > 0:
            time.sleep(self.flush_wait_sec)

        expected_darshan = _parse_darshan_from_run_script(run_sh)
        if expected_darshan is None or not expected_darshan.exists():
            found = sorted(run_root.glob("*.darshan"))
            if len(found) != 1:
                raise RuntimeError(f"Could not resolve darshan artifact in {run_root}")
            expected_darshan = found[0]

        analyze_script = self.io_synth_root / self.analyze_script_relpath
        pattern_json = self.pattern_file_by_id[pattern_id]
        _run(
            [
                "python3",
                str(analyze_script),
                "--darshan",
                str(expected_darshan),
                "--input-json",
                str(pattern_json),
                "--outdir",
                str(run_root),
            ],
            cwd=self.io_synth_root,
        )
        summary_csv = run_root / "darshan_summary.csv"
        if not summary_csv.exists():
            raise RuntimeError(f"Missing analysis output: {summary_csv}")
        return self._metric_from_csv(summary_csv, self.metric_key, self.metric_fallback)
