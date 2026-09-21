"""Post-hoc analysis for the saved EXP-0002 depth sweep.

This performs no model inference. It rescales the saved JSON using robust
numeric extraction and reports per-depth speedup relative to full depth.
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

def extract_flexible(text: str) -> str | None:
    explicit = re.findall(
        r"(?:The answer is|####)\s*:?\s*<?\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*>?",
        text,
        re.IGNORECASE,
    )
    if explicit:
        return canonical_number(explicit[-1])
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return canonical_number(matches[-1]) if matches else None

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", type=Path)
    args = parser.parse_args()
    data = json.loads(args.results_json.read_text(encoding="utf-8"))
    variants = data["variants"]
    results = data["results"]
    full = next(v for v in variants if v["depth"] == data["total_model_layers"])
    print("=" * 70)
    print("EXP-0002 DEPTH SWEEP POST-HOC ANALYSIS")
    print("=" * 70)
    for variant in variants:
        depth = str(variant["depth"])
        rows = results[depth]
        flexible_correct = 0
        missing = 0
        for row in rows:
            predicted = extract_flexible(str(row.get("output", "")))
            expected = canonical_number(str(row.get("expected")))
            if predicted is None:
                missing += 1
            flexible_correct += int(predicted == expected)
        speedup = full["average_time_per_question_seconds"] / variant["average_time_per_question_seconds"]
        print(f"{variant['depth']:>2}/{data['total_model_layers']} layers | saved={variant['accuracy']:.1%} | flexible={flexible_correct}/{len(rows)}={flexible_correct/len(rows):.1%} | no numeric={missing}/{len(rows)} | speedup={speedup:.2f}x | tok/s={variant['overall_tokens_per_second']:.2f}")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
