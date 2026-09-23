"""Controls and provenance for the bounded route-collapse experiment."""

import hashlib
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_exp0011 import build_parser, condition_spec, route_statistics, run


def test_predeclared_conditions_and_parameter_control():
    assert condition_spec("sparse_baseline") == ("sparse", 512, 0.01)
    assert condition_spec("sparse_balanced") == ("sparse", 512, 1.0)
    assert condition_spec("dense_active") == ("dense", 512, 0.0)
    assert condition_spec("dense_parameters") == ("dense", 4096, 0.0)
    with pytest.raises(ValueError, match="condition"):
        condition_spec("unknown")


def test_route_statistics_distinguish_collapse_from_balanced_use():
    collapsed = route_statistics([0, 0, 8, 0])
    assert collapsed["effective_experts"] == pytest.approx(1)
    assert collapsed["max_route_share"] == 1
    assert collapsed["experts_at_least_one_percent"] == 1
    balanced = route_statistics([2, 2, 2, 2])
    assert balanced["effective_experts"] == pytest.approx(4)
    assert balanced["max_route_share"] == 0.25
    assert balanced["experts_at_least_one_percent"] == 4
    assert math.isfinite(balanced["effective_experts"])


def test_paired_tiny_runs_keep_windows_and_results_separate(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(bytes(range(256)) * 8)
    checksum = hashlib.sha256(corpus.read_bytes()).hexdigest()
    results = []
    for condition in ("sparse_balanced", "dense_active"):
        output = tmp_path / condition
        args = build_parser().parse_args([
            "--corpus", str(corpus), "--output", str(output),
            "--expected-corpus-sha256", checksum, "--condition", condition,
            "--seed", "41", "--device", "cpu", "--steps", "1",
            "--width", "16", "--layers", "1", "--heads", "4",
            "--shared-ff", "32", "--groups", "2",
            "--experts-per-group", "2", "--batch-size", "1", "--window", "8",
            "--context", "16", "--val-batches", "1", "--runtime-tokens", "1",
            "--warmup-tokens", "1", "--repeats", "1", "--generation-tokens", "1",
        ])
        result = run(args)
        assert result["status"] == "completed"
        assert result["experiment"] == "EXP-0011"
        assert result["condition"] == condition
        assert result["corpus_sha256"] == checksum
        assert result["training"]["route_intervals"] if condition == "sparse_balanced" else not result["training"]["route_intervals"]
        assert (output / "checkpoint.pt").exists()
        assert (output / "summary.json").exists()
        results.append(result)
        with pytest.raises(FileExistsError):
            run(args)
    assert results[0]["training"]["windows_sha256"] == results[1]["training"]["windows_sha256"]
    assert results[0]["evaluation"]["validation_windows_sha256"] == results[1]["evaluation"]["validation_windows_sha256"]
    assert len(results[0]["training"]["losses"]) == 1


def test_corpus_mismatch_fails_before_output_directory(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(bytes(range(256)) * 4)
    output = tmp_path / "run"
    args = build_parser().parse_args([
        "--corpus", str(corpus), "--output", str(output),
        "--condition", "dense_active", "--seed", "41",
        "--expected-corpus-sha256", "0" * 64,
    ])
    with pytest.raises(ValueError, match="corpus SHA-256"):
        run(args)
    assert not output.exists()
