"""Rescore a saved CFI GSM8K result without regenerating model outputs.

This tool is intentionally separate from inference. It preserves the original
generation run and applies a flexible GSM8K extraction pass to the saved raw
outputs. It reports explicit-answer matches, flexible final-number matches,
truncated generations, and accuracy under each policy.
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


def canonical_number(value: str | None) -> str | None:
    if value is None:
        return None
    raw = value.strip().replace(",", "")
    try:
        number = Decimal(raw)
    except InvalidOperation:
        return raw
    if number == number.to_integral_value():
        return str(int(number))
    return format(number.normalize(), "f").rstrip("0").rstrip(".")


def extract_explicit(text: str) -> str | None:
    matches = re.findall(
        r"(?:The answer is|####)\s*:?\s*<?\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*>?",
        text,
        re.IGNORECASE,
    )
    return canonical_number(matches[-1]) if matches else None


def extract_flexible(text: str) -> str | None:
    # Fallback only: use the last numeric token when no explicit final-answer
    # marker is present. Commas and currency symbols are normalized.
    matches = re.findall(
        r"[-+]?\d[\d,]*(?:\.\d+)?",
        text,
    )
    return canonical_number(matches[-1]) if matches else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the rescored JSON.",
    )
    args = parser.parse_args()

    data = json.loads(args.results_json.read_text(encoding="utf-8"))
    rows = data.get("results", [])
    if not rows:
        raise ValueError("No per-question results found.")

    explicit_correct = 0
    flexible_correct = 0
    explicit_count = 0
    flexible_count = 0
    truncated_count = 0

    for row in rows:
        text = str(row.get("output", ""))
        expected = canonical_number(str(row.get("expected", "")))

        explicit = extract_explicit(text)
        flexible = extract_flexible(text)
        truncated = (
            int(row.get("generated_tokens", 0))
            >= int(data.get("max_new_tokens", 512))
        )

        row["original_predicted"] = row.get("predicted")
        row["rescore_explicit"] = explicit
        row["rescore_flexible"] = flexible
        row["truncated"] = truncated

        if explicit is not None:
            explicit_count += 1
            explicit_correct += int(explicit == expected)

        if flexible is not None:
            flexible_count += 1
            flexible_correct += int(flexible == expected)

        truncated_count += int(truncated)

    n = len(rows)

    report = {
        "source_results": str(args.results_json),
        "questions": n,
        "original_accuracy": data.get("accuracy"),
        "explicit_format": {
            "answers_extracted": explicit_count,
            "correct": explicit_correct,
            "accuracy": explicit_correct / n,
        },
        "flexible_final_number": {
            "answers_extracted": flexible_count,
            "correct": flexible_correct,
            "accuracy": flexible_correct / n,
        },
        "truncated_generations": truncated_count,
        "interpretation": (
            "Use flexible accuracy as a diagnostic, not as a replacement for a "
            "pre-registered metric unless the protocol explicitly adopts it. "
            "Truncated generations are flagged separately."
        ),
        "results": rows,
    }

    output = args.output or args.results_json.with_name(
        args.results_json.stem + "_rescored.json"
    )
    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=" * 64)
    print("CFI GSM8K RESCORE")
    print("=" * 64)
    print(f"Questions: {n}")
    print(f"Original recorded accuracy: {data.get('accuracy', 0):.1%}")
    print(
        f"Explicit-answer accuracy: "
        f"{explicit_correct}/{n} = {explicit_correct / n:.1%}"
    )
    print(
        f"Flexible-final-number accuracy: "
        f"{flexible_correct}/{n} = {flexible_correct / n:.1%}"
    )
    print(f"Responses truncated at token ceiling: {truncated_count}")
    print(f"Saved: {output}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
