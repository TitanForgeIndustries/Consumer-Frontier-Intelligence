"""Core experiment record and normalization logic."""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if isinstance(value, dict):
                events.append(value)
    return events


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _last_metric(events: Iterable[dict[str, Any]], key: str) -> Any:
    value = None
    for event in events:
        if key in event and event[key] is not None:
            value = event[key]
    return value


def _max_metric(events: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [_as_float(event.get(key)) for event in events]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _avg_metric(events: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [_as_float(event.get(key)) for event in events]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


class ExperimentLogger:
    """Create and maintain one machine-readable experiment record."""

    def __init__(self, experiment_id: str, root: str | Path = "experiments") -> None:
        if not experiment_id or any(character in experiment_id for character in "/\\"):
            raise ValueError("experiment_id must be a non-empty directory-safe identifier")
        self.experiment_id = experiment_id
        self.root = Path(root)
        self.path = self.root / experiment_id
        self.config_path = self.path / "config.json"
        self.events_path = self.path / "events.jsonl"
        self.hardware_path = self.path / "hardware_samples.jsonl"
        self.evaluation_path = self.path / "evaluation.json"
        self.summary_path = self.path / "summary.json"
        self.report_path = self.path / "report.md"

    def initialize(self, config: dict[str, Any] | None = None) -> Path:
        if self.config_path.exists():
            raise FileExistsError(f"Experiment already exists: {self.path}")
        self.path.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment_id": self.experiment_id,
            "schema_version": 1,
            "status": "running",
            "started_at": utc_now(),
            "completed_at": None,
            "hypothesis": None,
            "change": None,
            "baseline": None,
            "model": None,
            "method": None,
            "dataset": None,
            "hardware": {},
            "software": {},
            "training": {},
            "inference": {},
            "notes": None,
        }
        if config:
            payload.update(config)
        write_json(self.config_path, payload)
        # Create the complete record skeleton immediately so every experiment
        # has the same predictable on-disk structure even before metrics arrive.
        self.events_path.touch(exist_ok=True)
        self.hardware_path.touch(exist_ok=True)
        if not self.evaluation_path.exists():
            write_json(self.evaluation_path, {})
        return self.path

    def read_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Missing config: {self.config_path}")
        return load_json(self.config_path)

    def update_config(self, **updates: Any) -> None:
        config = self.read_config()
        config.update(updates)
        write_json(self.config_path, config)

    def record(self, kind: str, **data: Any) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Experiment is not initialized: {self.path}")
        event = {"timestamp": utc_now(), "kind": kind, **data}
        append_jsonl(self.events_path, event)

    def record_training(self, **metrics: Any) -> None:
        self.record("training", **metrics)

    def record_evaluation(self, **metrics: Any) -> None:
        self.record("evaluation", **metrics)
        current = {}
        if self.evaluation_path.exists():
            current = load_json(self.evaluation_path)
        current.update(metrics)
        write_json(self.evaluation_path, current)

    def record_hardware(self, **metrics: Any) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Experiment is not initialized: {self.path}")
        append_jsonl(
            self.hardware_path,
            {"timestamp": utc_now(), **metrics},
        )

    def import_metrics(self, records: list[dict[str, Any]]) -> int:
        """Import normalized training/evaluation records from external logs."""
        count = 0
        for record in records:
            payload = dict(record)
            kind = str(payload.pop("kind", "training"))
            self.record(kind, **payload)
            count += 1
        return count

    def finalize(self, status: str = "completed", notes: str | None = None) -> dict[str, Any]:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError("status must be completed, failed, or cancelled")
        config = self.read_config()
        config["status"] = status
        config["completed_at"] = utc_now()
        if notes is not None:
            config["notes"] = notes
        write_json(self.config_path, config)

        events = load_events(self.events_path)
        training_events = [event for event in events if event.get("kind") == "training"]
        evaluation_events = [event for event in events if event.get("kind") == "evaluation"]
        hardware_events = load_jsonl(self.hardware_path)

        started = _parse_datetime(config.get("started_at"))
        completed = _parse_datetime(config.get("completed_at"))
        duration_seconds = _last_metric(training_events, "runtime_seconds")
        if duration_seconds is None and started and completed:
            duration_seconds = max(0.0, (completed - started).total_seconds())

        final_step = _last_metric(training_events, "step")
        final_loss = _last_metric(training_events, "loss")
        final_learning_rate = _last_metric(training_events, "learning_rate")
        final_tokens = _last_metric(training_events, "tokens")
        final_tokens_per_sec = _last_metric(training_events, "tokens_per_sec")

        summary: dict[str, Any] = {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "status": status,
            "started_at": config.get("started_at"),
            "completed_at": config.get("completed_at"),
            "duration_seconds": duration_seconds,
            "model": config.get("model"),
            "method": config.get("method"),
            "dataset": config.get("dataset"),
            "hypothesis": config.get("hypothesis"),
            "change": config.get("change"),
            "baseline": config.get("baseline"),
            "max_steps": config.get("training", {}).get("max_steps"),
            "context_length": config.get("training", {}).get("context_length"),
            "final_step": final_step,
            "final_loss": _as_float(final_loss),
            "final_learning_rate": _as_float(final_learning_rate),
            "total_tokens": _as_float(final_tokens),
            "tokens_per_second": _as_float(final_tokens_per_sec),
            "peak_vram_gb": _gb_from_mb(_max_metric(hardware_events, "gpu_memory_used_mb")),
            "peak_system_ram_gb": _gb_from_mb(_max_metric(hardware_events, "system_ram_used_mb")),
            "avg_gpu_utilization_pct": _avg_metric(hardware_events, "gpu_utilization_pct"),
            "peak_gpu_power_w": _max_metric(hardware_events, "gpu_power_w"),
            "peak_gpu_temperature_c": _max_metric(hardware_events, "gpu_temperature_c"),
            "evaluation": load_json(self.evaluation_path) if self.evaluation_path.exists() else (
                evaluation_events[-1] if evaluation_events else {}
            ),
            "training_event_count": len(training_events),
            "hardware_sample_count": len(hardware_events),
            "notes": config.get("notes"),
        }
        write_json(self.summary_path, summary)
        write_markdown_report(config, summary, self.report_path)
        return summary


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _gb_from_mb(value: float | None) -> float | None:
    return round(value / 1024, 3) if value is not None else None


def write_markdown_report(
    config: dict[str, Any],
    summary: dict[str, Any],
    path: Path,
) -> None:
    evaluation = summary.get("evaluation") or {}
    rows = [
        ("Status", summary.get("status")),
        ("Model", summary.get("model")),
        ("Method", summary.get("method")),
        ("Dataset", summary.get("dataset")),
        ("Baseline", summary.get("baseline")),
        ("Final step", summary.get("final_step")),
        ("Final loss", summary.get("final_loss")),
        ("Total tokens", summary.get("total_tokens")),
        ("Tokens/sec", summary.get("tokens_per_second")),
        ("Duration (s)", summary.get("duration_seconds")),
        ("Peak VRAM (GB)", summary.get("peak_vram_gb")),
        ("Peak system RAM (GB)", summary.get("peak_system_ram_gb")),
        ("Avg GPU utilization (%)", summary.get("avg_gpu_utilization_pct")),
        ("Peak GPU power (W)", summary.get("peak_gpu_power_w")),
        ("Peak GPU temperature (C)", summary.get("peak_gpu_temperature_c")),
        ("Training events", summary.get("training_event_count")),
        ("Hardware samples", summary.get("hardware_sample_count")),
    ]
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## Research definition",
        "",
        f"**Hypothesis:** {config.get('hypothesis') or ''}",
        "",
        f"**Change:** {config.get('change') or ''}",
        "",
        f"**Next step:** {config.get('next_experiment') or ''}",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines.extend(f"| {label} | {value if value is not None else ''} |" for label, value in rows)
    lines.extend([
        "",
        "## Evaluation",
        "",
        "| Key | Value |",
        "|---|---|",
    ])
    if evaluation:
        lines.extend(f"| {key} | {value} |" for key, value in sorted(evaluation.items()))
    else:
        lines.append("| No evaluation recorded | |")
    lines.extend([
        "",
        "## Notes",
        "",
        config.get("notes") or "",
        "",
        "This report is generated from the machine-readable experiment record.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def read_tabular_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".jsonl", ".ndjson"}:
        return load_jsonl(path)
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            # Hugging Face Trainer/Unsloth trainer_state.json stores step-level
            # metrics under log_history and run-level throughput at top level.
            log_history = value.get("log_history")
            if isinstance(log_history, list):
                records = [row for row in log_history if isinstance(row, dict)]
                run_metadata = {
                    key: value[key]
                    for key in ("train_runtime", "train_tokens_per_second")
                    if key in value
                }
                if run_metadata:
                    records.append(run_metadata)
                return records
            for key in ("history", "metrics", "records", "events"):
                nested = value.get(key)
                if isinstance(nested, list):
                    return [row for row in nested if isinstance(row, dict)]
            return [value]
    raise ValueError(f"Unsupported metrics file: {path}")


_KEY_ALIASES = {
    "step": "step",
    "global_step": "step",
    "current_step": "step",
    "loss": "loss",
    "train_loss": "loss",
    "training_loss": "loss",
    "learning_rate": "learning_rate",
    "lr": "learning_rate",
    "grad_norm": "grad_norm",
    "gradient_norm": "grad_norm",
    "epoch": "epoch",
    "tokens": "tokens",
    "num_tokens": "tokens",
    "total_tokens": "tokens",
    "tokens_per_second": "tokens_per_sec",
    "tok_per_sec": "tokens_per_sec",
    "throughput": "tokens_per_sec",
    "elapsed_seconds": "elapsed_seconds",
    "elapsed": "elapsed_seconds",
    "runtime_seconds": "runtime_seconds",
    "train_runtime": "runtime_seconds",
    "train_tokens_per_second": "tokens_per_sec",
}

def normalize_metric_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, value in record.items():
        key = _KEY_ALIASES.get(str(raw_key).strip().lower())
        if key:
            normalized[key] = value
    if "kind" in record:
        normalized["kind"] = record["kind"]
    return normalized
