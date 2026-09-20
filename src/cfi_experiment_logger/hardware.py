"""Hardware sampling for CFI experiments."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _try_psutil() -> Any:
    try:
        import psutil
        return psutil
    except ImportError:
        return None


def collect_gpu_snapshots() -> list[dict[str, Any]]:
    """Collect one snapshot per NVIDIA GPU using nvidia-smi when available."""
    if shutil.which("nvidia-smi") is None:
        return []

    query = (
        "index,name,utilization.gpu,memory.used,memory.total,"
        "power.draw,temperature.gpu"
    )
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return []

    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        columns = [item.strip() for item in line.split(",")]
        if len(columns) != 7:
            continue
        try:
            rows.append(
                {
                    "gpu_index": int(columns[0]),
                    "gpu_name": columns[1],
                    "gpu_utilization_pct": float(columns[2]),
                    "gpu_memory_used_mb": float(columns[3]),
                    "gpu_memory_total_mb": float(columns[4]),
                    "gpu_power_w": float(columns[5]),
                    "gpu_temperature_c": float(columns[6]),
                }
            )
        except ValueError:
            continue
    return rows


def collect_hardware_snapshot() -> dict[str, Any]:
    """Return a point-in-time hardware snapshot."""
    snapshot: dict[str, Any] = {
        "timestamp": _utc_now(),
        "gpus": collect_gpu_snapshots(),
    }

    psutil = _try_psutil()
    if psutil is not None:
        virtual = psutil.virtual_memory()
        snapshot["system_ram_used_mb"] = round(virtual.used / (1024 * 1024), 2)
        snapshot["system_ram_total_mb"] = round(virtual.total / (1024 * 1024), 2)
        snapshot["cpu_utilization_pct"] = psutil.cpu_percent(interval=None)

    # Preserve the primary GPU as flat fields for easy aggregation.
    if snapshot["gpus"]:
        primary = snapshot["gpus"][0]
        for key in (
            "gpu_index",
            "gpu_name",
            "gpu_utilization_pct",
            "gpu_memory_used_mb",
            "gpu_memory_total_mb",
            "gpu_power_w",
            "gpu_temperature_c",
        ):
            snapshot[key] = primary[key]
    return snapshot


def monitor_to_file(
    path: str | Path,
    interval_seconds: float = 2.0,
    stop_after_seconds: float | None = None,
) -> int:
    """Sample hardware until interrupted or an optional duration expires."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    count = 0
    with output.open("a", encoding="utf-8") as handle:
        while True:
            snapshot = collect_hardware_snapshot()
            handle.write(json.dumps(snapshot) + "\n")
            handle.flush()
            count += 1
            if stop_after_seconds is not None:
                if time.monotonic() - started >= stop_after_seconds:
                    break
            time.sleep(interval_seconds)
    return count
