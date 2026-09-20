"""Command-line interface for CFI experiment tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import ExperimentLogger, normalize_metric_record, read_tabular_file
from .hardware import monitor_to_file
from .registry import export_registry_csv, export_registry_xlsx


def _parse_value(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfi-experiment",
        description="Track, measure, and export Consumer Frontier Intelligence experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create an experiment record.")
    init.add_argument("experiment_id")
    init.add_argument("--root", default="experiments")
    init.add_argument("--model")
    init.add_argument("--method")
    init.add_argument("--dataset")
    init.add_argument("--hypothesis")
    init.add_argument("--change")
    init.add_argument("--baseline")
    init.add_argument("--max-steps", type=int)
    init.add_argument("--context-length", type=int)
    init.add_argument("--notes")

    record = subparsers.add_parser("record", help="Record one training or evaluation event.")
    record.add_argument("experiment_id")
    record.add_argument("--root", default="experiments")
    record.add_argument("--kind", choices=["training", "evaluation", "hardware"], default="training")
    record.add_argument("--data", required=True, help="JSON object containing metrics.")

    import_metrics = subparsers.add_parser(
        "import-metrics",
        help="Import CSV, JSON, or JSONL training logs.",
    )
    import_metrics.add_argument("experiment_id")
    import_metrics.add_argument("metrics_file")
    import_metrics.add_argument("--root", default="experiments")

    monitor = subparsers.add_parser("monitor", help="Sample NVIDIA GPU/system metrics.")
    monitor.add_argument("experiment_id")
    monitor.add_argument("--root", default="experiments")
    monitor.add_argument("--interval", type=float, default=2.0)
    monitor.add_argument("--duration", type=float)

    finalize = subparsers.add_parser("finalize", help="Finalize an experiment and update the registry.")
    finalize.add_argument("experiment_id")
    finalize.add_argument("--root", default="experiments")
    finalize.add_argument("--status", choices=["completed", "failed", "cancelled"], default="completed")
    finalize.add_argument("--notes")

    export = subparsers.add_parser("export", help="Regenerate registry CSV and Excel files.")
    export.add_argument("--root", default="experiments")
    export.add_argument("--results", default="results")
    export.add_argument("--no-xlsx", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "init":
        config = {
            "model": args.model,
            "method": args.method,
            "dataset": args.dataset,
            "hypothesis": args.hypothesis,
            "change": args.change,
            "baseline": args.baseline,
            "notes": args.notes,
            "training": {
                "max_steps": args.max_steps,
                "context_length": args.context_length,
            },
        }
        logger = ExperimentLogger(args.experiment_id, args.root)
        logger.initialize(config)
        print(logger.path)
        return 0

    logger = ExperimentLogger(args.experiment_id, args.root)

    if args.command == "record":
        data = json.loads(args.data)
        if not isinstance(data, dict):
            raise SystemExit("--data must be a JSON object")
        if args.kind == "training":
            logger.record_training(**data)
        elif args.kind == "evaluation":
            logger.record_evaluation(**data)
        else:
            logger.record_hardware(**data)
        return 0

    if args.command == "import-metrics":
        records = [
            normalize_metric_record(record)
            for record in read_tabular_file(Path(args.metrics_file))
        ]
        logger.import_metrics(records)
        print(f"Imported {len(records)} records into {logger.events_path}")
        return 0

    if args.command == "monitor":
        try:
            count = monitor_to_file(
                logger.hardware_path,
                interval_seconds=args.interval,
                stop_after_seconds=args.duration,
            )
            print(f"Captured {count} hardware samples into {logger.hardware_path}")
        except KeyboardInterrupt:
            print(f"Hardware monitoring stopped. Samples are in {logger.hardware_path}")
        return 0

    if args.command == "finalize":
        summary = logger.finalize(args.status, notes=args.notes)
        results_dir = Path(args.root).parent / "results"
        csv_path = export_registry_csv(args.root, results_dir / "CFI_Experiment_Registry.csv")
        print(json.dumps(summary, indent=2))
        print(f"Registry CSV: {csv_path}")
        try:
            xlsx_path = export_registry_xlsx(args.root, results_dir / "CFI_Experiment_Registry.xlsx")
            print(f"Registry XLSX: {xlsx_path}")
        except RuntimeError as exc:
            print(f"Registry XLSX skipped: {exc}")
        return 0

    if args.command == "export":
        results = Path(args.results)
        csv_path = export_registry_csv(args.root, results / "CFI_Experiment_Registry.csv")
        print(f"Registry CSV: {csv_path}")
        if not args.no_xlsx:
            try:
                xlsx_path = export_registry_xlsx(args.root, results / "CFI_Experiment_Registry.xlsx")
                print(f"Registry XLSX: {xlsx_path}")
            except RuntimeError as exc:
                print(f"Registry XLSX skipped: {exc}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
