"""Local data paths for experiment runners, independent of checkout location."""

from __future__ import annotations

import os
from pathlib import Path


def data_path(*parts: str) -> Path:
    root = os.environ.get("CFI_DATA_ROOT")
    base = (
        Path(root).expanduser()
        if root
        else Path(__file__).resolve().parents[1] / ".cfi-data"
    )
    return base.joinpath(*parts)
