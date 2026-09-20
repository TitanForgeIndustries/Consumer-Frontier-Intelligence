from pathlib import Path

from cfi_experiment_logger.core import ExperimentLogger
from cfi_experiment_logger.registry import build_registry, export_registry_csv


def test_experiment_lifecycle(tmp_path: Path):
    root = tmp_path / "experiments"
    logger = ExperimentLogger("EXP-TEST-0001", root)
    logger.initialize(
        {
            "model": "Qwen3-4B-Base",
            "method": "QLoRA",
            "dataset": "DAPO-Math-17k-Processed",
            "hypothesis": "Baseline training establishes the resource/capability reference point.",
            "training": {"max_steps": 10, "context_length": 2048},
        }
    )

    logger.record_training(step=1, loss=2.8, learning_rate=5e-6, tokens=1000)
    logger.record_training(step=10, loss=2.1, learning_rate=1e-6, tokens=10000)
    logger.record_hardware(
        gpu_name="RTX 3070",
        gpu_memory_used_mb=6100,
        system_ram_used_mb=8000,
        gpu_utilization_pct=70,
        gpu_power_w=125,
        gpu_temperature_c=58,
    )
    logger.record_evaluation(math_accuracy=0.42)

    summary = logger.finalize()

    assert summary["final_step"] == 10
    assert summary["final_loss"] == 2.1
    assert summary["total_tokens"] == 10000.0
    assert summary["peak_vram_gb"] is not None
    assert (root / "EXP-TEST-0001" / "report.md").exists()
    assert (root / "EXP-TEST-0001" / "summary.json").exists()

    rows = build_registry(root)
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "EXP-TEST-0001"

    output = tmp_path / "registry.csv"
    export_registry_csv(root, output)
    assert output.exists()


def test_cli_export_without_experiment_id(tmp_path: Path, monkeypatch):
    from cfi_experiment_logger.cli import main

    root = tmp_path / "experiments"
    results = tmp_path / "results"
    monkeypatch.chdir(tmp_path)

    assert main([
        "export",
        "--root", str(root),
        "--results", str(results),
        "--no-xlsx",
    ]) == 0
    assert (results / "CFI_Experiment_Registry.csv").exists()
