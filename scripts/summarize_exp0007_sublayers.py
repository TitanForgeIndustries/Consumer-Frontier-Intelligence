"""Rebuild the EXP-0007 sublayer summary from an existing JSONL run."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0007-Instrumentation\sublayer_metrics.jsonl"
)
DEFAULT_OUTPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0007-Instrumentation\sublayer_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize existing CFI EXP-0007 sublayer JSONL data."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input}")

    values_by_layer: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    observations: dict[int, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    line_count = 0

    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            line_count += 1
            record = json.loads(line)

            layers = record.get("layers")
            if not isinstance(layers, dict):
                raise ValueError(
                    f"Line {line_number}: expected an object at 'layers'."
                )

            for layer_text, kinds in layers.items():
                try:
                    layer = int(layer_text)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Line {line_number}: invalid layer key {layer_text!r}."
                    ) from exc

                if not isinstance(kinds, dict):
                    raise ValueError(
                        f"Line {line_number}: expected sublayer kinds for "
                        f"layer {layer}."
                    )

                for kind, metrics in kinds.items():
                    if not isinstance(metrics, dict):
                        raise ValueError(
                            f"Line {line_number}: expected metrics for "
                            f"layer {layer}, kind {kind!r}."
                        )

                    observations[layer][str(kind)] += 1

                    for key, value in metrics.items():
                        if isinstance(value, (int, float)) and not isinstance(
                            value, bool
                        ):
                            numeric = float(value)
                            if math.isfinite(numeric):
                                values_by_layer[layer][
                                    f"{kind}_{key}"
                                ].append(numeric)

    summary: dict[str, dict[str, object]] = {}
    for layer, metrics in sorted(values_by_layer.items()):
        summary[str(layer)] = {
            "layer": layer,
            "observations": dict(sorted(observations[layer].items())),
            **{
                f"{key}_mean": mean(values)
                for key, values in sorted(metrics.items())
            },
        }

    if line_count > 0 and not summary:
        raise RuntimeError(
            "The JSONL contained records, but no numeric sublayer metrics "
            "could be summarized."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Input records: {line_count}")
    print(f"Layers summarized: {len(summary)}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
