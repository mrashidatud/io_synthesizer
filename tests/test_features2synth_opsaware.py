import csv
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "features2synth_opsaware.py"
SPEC = importlib.util.spec_from_file_location("features2synth_opsaware", SCRIPT_PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MOD)


def _base_features():
    return {
        "_json_base": "unit_case",
        "cap_total_gib": 0.01,
        "nprocs": 2,
        "io_api": "posix",
        "meta_api": "posix",
        "mpi_collective_mode": "none",
        "meta_scope": "data_files",
        "pct_reads": 0.5,
        "pct_writes": 0.5,
        "pct_byte_reads": 0.5,
        "pct_byte_writes": 0.5,
        "pct_read_0_100K": 1.0,
        "pct_read_100K_10M": 0.0,
        "pct_read_10M_1G_PLUS": 0.0,
        "pct_write_0_100K": 1.0,
        "pct_write_100K_10M": 0.0,
        "pct_write_10M_1G_PLUS": 0.0,
        "pct_read_only_files": 0.5,
        "pct_read_write_files": 0.5,
        "pct_write_only_files": 0.0,
        "pct_consec_reads": 0.0,
        "pct_seq_reads": 0.0,
        "pct_consec_writes": 0.0,
        "pct_seq_writes": 0.0,
        "pct_rw_switches": 0.2,
        "pct_shared_files": 0.5,
        "pct_bytes_shared_files": 0.5,
        "pct_bytes_unique_files": 0.5,
        "pct_io_access": 0.9,
        "pct_meta_open_access": 1.0,
        "pct_meta_stat_access": 0.0,
        "pct_meta_seek_access": 0.0,
        "pct_meta_sync_access": 0.0,
        "optimizer": "lexicographic",
        "seq_policy": "nonconsec_strict",
        "alignment_policy": "structure_preserving",
        "phase_cap": 50000,
        "data_random_preseek": 0,
    }


def _run_plan(tmp_path, overrides=None):
    outdir = tmp_path / "out"
    MOD._set_outroot_per_json(str(tmp_path / "input.json"), outdir=str(outdir))
    feats = _base_features()
    if overrides:
        feats.update(overrides)
    out = MOD.plan_from_features(feats, nranks=2, fs_align_bytes=4096)
    plan_csv = Path(out["plan_csv"])
    notes = Path(out["notes"])
    return plan_csv, notes


def test_lexicographic_alignment_does_not_change_ordering_error():
    rows = [
        {"file": "f0", "intent": "read", "xfer": 4096, "ops": 12, "phase_kind": "consec", "flags": "|shared|", "role": "rw", "bin": "S"},
        {"file": "f0", "intent": "read", "xfer": 4096, "ops": 8, "phase_kind": "seq", "flags": "|shared|", "role": "rw", "bin": "S"},
        {"file": "f0", "intent": "write", "xfer": 4096, "ops": 10, "phase_kind": "random", "flags": "|shared|", "role": "rw", "bin": "S"},
    ]
    targets = {
        "pct_seq_reads": 1.0,
        "pct_seq_writes": 0.0,
        "pct_consec_reads": 0.6,
        "pct_consec_writes": 0.0,
        "pct_rw_switches": 0.0,
    }
    m1 = MOD.estimate_ordering_metrics(rows, nprocs=2)
    e1 = MOD.compute_ordering_error(m1, targets)
    _ = MOD.compute_rowwise_pua_file(rows, target_frac=0.3, fs_align_bytes=4096, alignment_policy="structure_preserving")
    m2 = MOD.estimate_ordering_metrics(rows, nprocs=2)
    e2 = MOD.compute_ordering_error(m2, targets)
    assert e2 == e1


def test_seq_rows_stay_nonconsecutive_in_generated_plan(tmp_path):
    plan_csv, _ = _run_plan(
        tmp_path,
        {
            "pct_seq_reads": 0.8,
            "pct_consec_reads": 0.2,
            "pct_seq_writes": 0.8,
            "pct_consec_writes": 0.2,
        },
    )
    with plan_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    seq_rows = [r for r in rows if r["type"] == "data" and r["flags"].startswith("seq")]
    assert seq_rows
    for r in seq_rows:
        assert r["p_seq_r"] == "1.000000"
        assert r["p_consec_r"] == "0.000000"
        assert r["p_seq_w"] == "1.000000"
        assert r["p_consec_w"] == "0.000000"


def test_phase_cap_enforced_when_enabled():
    rows = []
    for i in range(8):
        rows.append({"file": f"rw_{i}", "intent": "read", "xfer": 4096, "ops": 200, "phase_kind": "random", "flags": "|shared|", "role": "rw", "bin": "S"})
        rows.append({"file": f"rw_{i}", "intent": "write", "xfer": 4096, "ops": 200, "phase_kind": "random", "flags": "|shared|", "role": "rw", "bin": "S"})
    capped, _ = MOD.interleave_rw_segments(rows, switch_frac=0.9, seg_ops=1, phase_cap=50)
    assert len(capped) <= 50


def test_phase_cap_zero_disables_capping():
    rows = []
    for i in range(8):
        rows.append({"file": f"rw_{i}", "intent": "read", "xfer": 4096, "ops": 200, "phase_kind": "random", "flags": "|shared|", "role": "rw", "bin": "S"})
        rows.append({"file": f"rw_{i}", "intent": "write", "xfer": 4096, "ops": 200, "phase_kind": "random", "flags": "|shared|", "role": "rw", "bin": "S"})
    capped, _ = MOD.interleave_rw_segments(rows, switch_frac=0.9, seg_ops=1, phase_cap=50)
    uncapped, _ = MOD.interleave_rw_segments(rows, switch_frac=0.9, seg_ops=1, phase_cap=0)
    assert len(uncapped) >= len(capped)
    assert len(uncapped) > 50


def test_clamp_risk_mitigation_reduces_predicted_clamps():
    row = {
        "file": "/tmp/f0.dat",
        "intent": "write",
        "xfer": 4096,
        "ops": 16,
        "phase_kind": "seq",
        "flags": "|unique|",
        "role": "wo",
        "bin": "S",
    }
    per_file_span = {"/tmp/f0.dat": 8192}
    pua_map = {id(row): 0.0}
    diag = MOD.apply_adaptive_sparse_span_growth(
        per_file_span,
        [row],
        pua_map,
        fs_align_bytes=4096,
        tiny_mode_intents={"write"},
    )
    assert diag["clamp_before"] > 0
    assert diag["clamp_after"] < diag["clamp_before"]
    assert per_file_span["/tmp/f0.dat"] > 8192


def test_data_random_preseek_zero_avoids_meta_seek_inflation_path(tmp_path):
    plan_csv, _ = _run_plan(
        tmp_path,
        {
            "data_random_preseek": 0,
            "pct_seq_reads": 0.0,
            "pct_consec_reads": 0.0,
            "pct_seq_writes": 0.0,
            "pct_consec_writes": 0.0,
            "pct_meta_seek_access": 0.0,
            "pct_meta_open_access": 1.0,
            "pct_io_access": 0.9,
        },
    )
    with plan_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    random_data_rows = [r for r in rows if r["type"] == "data" and r["p_rand"] == "1.000000"]
    assert random_data_rows
    assert all(r["pre_seek_eof"] == "0" for r in random_data_rows)

    meta_rows = [r for r in rows if r["type"] == "meta"]
    assert meta_rows
    assert all(int(r["meta_seek"]) == 0 for r in meta_rows)
