import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cfi_paths import data_path


def test_default_data_root_is_repo_local(monkeypatch):
    monkeypatch.delenv("CFI_DATA_ROOT", raising=False)
    assert data_path("Results", "run") == ROOT / ".cfi-data" / "Results" / "run"


def test_external_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv("CFI_DATA_ROOT", str(tmp_path))
    assert data_path("Datasets", "input.txt") == tmp_path / "Datasets" / "input.txt"
