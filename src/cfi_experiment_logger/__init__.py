"""Experiment tracking utilities for Consumer Frontier Intelligence."""

from .core import ExperimentLogger, load_events
from .hardware import collect_hardware_snapshot
from .registry import build_registry, export_registry_csv, export_registry_xlsx

__all__ = [
    "ExperimentLogger",
    "load_events",
    "collect_hardware_snapshot",
    "build_registry",
    "export_registry_csv",
    "export_registry_xlsx",
]

__version__ = "0.1.0"
