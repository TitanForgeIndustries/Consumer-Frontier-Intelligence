"""Run the CFI EXP-0002 transformer-depth sweep on Qwen3-4B.

This is a controlled feasibility experiment for dynamic computation. It does
not train or modify model weights. It replaces the executed Qwen3 transformer
layer list with prefixes of the original 36-layer stack and evaluates each
depth on the same fixed GSM8K set.

Default depths: 12, 18, 24, 30, 36.

The 36-layer result is the full-depth reference. Reduced-depth variants are
true reduced-computation models because layers beyond the selected depth are
not executed during generation.

Example:
    python scripts/evaluate_depth_sweep.py --questions 5

Official run:
    python scripts/evaluate_depth_sweep.py --questions 100
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from cfi_experiment_logger.core import ExperimentLogger
    from cfi_experiment_logger.hardware import collect_hardware_snapshot
except ImportError:
    ExperimentLogger = None
    collect_hardware_snapshot = None


DEFAULT_MODEL = Path(
    r"E:\Titan Forge Industries\CFI-Data\HuggingFace\hub\models--Qwen--Qwen3-4B-Base\snapshots\906bfd4b4dc7f14ee4320094d8b41684abff8539"
)
DEFAULT_DATASET = Path(
    r"E:\Titan Forge Industries\CFI-Data\Datasets\CFI-Eval-0001-GSM8K\gsm8k_test_100.jsonl"
)
DEFAULT_OUTPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0002-DepthSweep"
)

DEFAULT_DEPTHS = (12, 18, 24, 30, 36)
MAX_NEW_TOKENS_DEFAULT = 512
SEED_BASE_DEFAULT = 42000
TEMPERATURE_DEFAULT = 0.6
TOP_P_DEFAULT = 0.95
TOP_K_DEFAULT = 20
HARDWARE_INTERVAL_DEFAULT = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CFI EXP-0002 Qwen3-4B depth sweep."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=list(DEFAULT_DEPTHS),
        help="Transformer depths to evaluate.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS_DEFAULT)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE_DEFAULT)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    parser.add_argument("--top-p", type=float, default=TOP_P_DEFAULT)
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    parser.add_argument("--experiment-id")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("experiments"),
    )
    parser.add_argument(
        "--hardware-interval",
        type=float,
        default=HARDWARE_INTERVAL_DEFAULT,
    )
    return parser.parse_args()


def load_rows(path: Path, count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("--questions must be positive")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Dataset row is not an object: {path}")
            rows.append(row)
            if len(rows) >= count:
                break

    if not rows:
        raise ValueError(f"No questions found in {path}")
    return rows


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


def extract_expected(text: str) -> str | None:
    match = re.search(
        r"####\s*(?:<\s*)?\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*>?",
        text,
        re.IGNORECASE,
    )
    return canonical_number(match.group(1)) if match else None


def extract_predicted(text: str) -> str | None:
    matches = re.findall(
        r"(?:The answer is|####)\s*:?\s*<?\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*>?",
        text,
        re.IGNORECASE,
    )
    if matches:
        return canonical_number(matches[-1])

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        match = re.fullmatch(
            r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?",
            line,
        )
        if match:
            return canonical_number(match.group(0).replace("$", "").strip())

    return None


class HardwareSampler:
    def __init__(self, output_path: Path, interval_seconds: float) -> None:
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.count = 0

    def start(self) -> None:
        if collect_hardware_snapshot is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds * 2.0))
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            snapshot = collect_hardware_snapshot()
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot) + "\n")
            self.count += 1
            self._stop.wait(self.interval_seconds)


def build_model(model_path: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.eval()
    return tokenizer, model


def get_layers(model):
    try:
        return model.model.layers
    except AttributeError as exc:
        raise RuntimeError(
            "Expected a Qwen3ForCausalLM model with model.layers."
        ) from exc


def run_variant(
    *,
    model,
    tokenizer,
    rows,
    depth: int,
    seed_base: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> list[dict[str, Any]]:
    original_layers = model.model.layers
    total_layers = len(original_layers)

    if depth < 1 or depth > total_layers:
        raise ValueError(
            f"Depth {depth} is invalid for a {total_layers}-layer model."
        )

    # Keep the original ModuleList alive and execute only its prefix.
    model.model.layers = torch.nn.ModuleList(
        list(original_layers[:depth])
    )

    try:
        results = []

        for index, row in enumerate(rows, 1):
            seed = seed_base + index
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            prompt = (
                "Question: "
                + str(row["question"])
                + "\n"
                + "Solve the problem step by step. "
                + "End with: The answer is <number>.\n"
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            ).to("cuda:0")

            prompt_length = int(inputs["input_ids"].shape[-1])

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            start = time.perf_counter()

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=tokenizer.eos_token_id,
                )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            generated = output[0, prompt_length:]
            generated_tokens = int(generated.shape[-1])
            tok_per_sec = generated_tokens / elapsed if elapsed > 0 else 0.0

            text = tokenizer.decode(
                generated,
                skip_special_tokens=True,
            ).strip()

            expected = extract_expected(str(row["answer"]))
            predicted = extract_predicted(text)
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

            result = {
                "index": index,
                "seed": seed,
                "depth": depth,
                "total_model_layers": total_layers,
                "expected": expected,
                "predicted": predicted,
                "correct": predicted == expected,
                "elapsed_seconds": round(elapsed, 6),
                "generated_tokens": generated_tokens,
                "tokens_per_second": round(tok_per_sec, 6),
                "peak_vram_gib": round(peak_vram, 6),
                "output": text,
            }
            results.append(result)

            print(
                f"  [{index}/{len(rows)}] "
                f"expected={expected} "
                f"predicted={predicted} "
                f"correct={result['correct']} "
                f"time={elapsed:.2f}s "
                f"tokens={generated_tokens} "
                f"tok/s={tok_per_sec:.2f} "
                f"VRAM={peak_vram:.2f} GiB"
            )

        return results
    finally:
        model.model.layers = original_layers


def summarize_variant(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    correct = sum(int(r["correct"]) for r in results)
    total_time = sum(r["elapsed_seconds"] for r in results)
    total_tokens = sum(r["generated_tokens"] for r in results)

    return {
        "questions": n,
        "correct": correct,
        "accuracy": correct / n,
        "total_generation_time_seconds": round(total_time, 6),
        "average_time_per_question_seconds": round(total_time / n, 6),
        "total_generated_tokens": total_tokens,
        "overall_tokens_per_second": round(
            total_tokens / total_time if total_time > 0 else 0.0,
            6,
        ),
        "mean_peak_vram_gib": round(
            sum(r["peak_vram_gib"] for r in results) / n,
            6,
        ),
        "max_peak_vram_gib": max(r["peak_vram_gib"] for r in results),
        "truncated_responses": sum(
            int(r["generated_tokens"] >= 512)
            for r in results
        ),
    }


def main() -> int:
    args = parse_args()

    if args.questions <= 0:
        raise ValueError("--questions must be positive")

    args.output.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.dataset, args.questions)

    tokenizer, model = build_model(args.model)
    full_layers = model.model.layers
    total_layers = len(full_layers)

    for depth in args.depths:
        if depth < 1 or depth > total_layers:
            raise ValueError(
                f"Requested depth {depth}, but model has {total_layers} layers."
            )

    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", args.model)
    print("Dataset:", args.dataset)
    print("Questions:", len(rows))
    print("Model layers:", total_layers)
    print("Depth sweep:", args.depths)

    hardware_output = args.output / "hardware_samples.jsonl"
    experiment_logger = None

    if args.experiment_id:
        if ExperimentLogger is None:
            raise RuntimeError("CFI experiment logger is unavailable.")
        experiment_logger = ExperimentLogger(
            args.experiment_id,
            args.experiment_root,
        )
        if not experiment_logger.config_path.exists():
            raise FileNotFoundError(
                f"Experiment is not initialized: {experiment_logger.path}"
            )
        hardware_output = experiment_logger.hardware_path

    sampler = HardwareSampler(
        hardware_output,
        args.hardware_interval,
    )

    variants = []
    all_results = {}

    sampler.start()

    try:
        for depth in args.depths:
            print("\n" + "=" * 64)
            print(f"DEPTH {depth}/{total_layers}")
            print("=" * 64)

            results = run_variant(
                model=model,
                tokenizer=tokenizer,
                rows=rows,
                depth=depth,
                seed_base=args.seed_base,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )

            summary = summarize_variant(results)
            variants.append({
                "depth": depth,
                **summary,
            })
            all_results[str(depth)] = results

            print(
                f"Depth {depth}: "
                f"{summary['correct']}/{summary['questions']} "
                f"= {summary['accuracy']:.1%}, "
                f"{summary['overall_tokens_per_second']:.2f} tok/s"
            )
    finally:
        sampler.stop()

    full_depth = next(
        (v for v in variants if v["depth"] == total_layers),
        None,
    )

    for variant in variants:
        if full_depth:
            variant["accuracy_delta_vs_full"] = round(
                variant["accuracy"] - full_depth["accuracy"],
                6,
            )
            variant["relative_depth"] = variant["depth"] / total_layers
            variant["time_ratio_vs_full"] = round(
                variant["average_time_per_question_seconds"]
                / full_depth["average_time_per_question_seconds"],
                6,
            )

    summary = {
        "schema_version": 1,
        "experiment": "EXP-0002",
        "benchmark": "CFI-Eval-0002-DepthSweep",
        "model": str(args.model),
        "dataset": str(args.dataset),
        "questions": len(rows),
        "total_model_layers": total_layers,
        "depths": args.depths,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed_base": args.seed_base,
        },
        "quantization": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
        },
        "hardware_samples": sampler.count,
        "variants": variants,
        "results": all_results,
    }

    result_path = args.output / "depth_sweep_results.json"
    result_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if experiment_logger is not None:
        experiment_logger.record_evaluation(
            benchmark="GSM8K-depth-sweep",
            samples=len(rows),
            variants=variants,
            results_file=str(result_path),
        )

    print("\n" + "=" * 64)
    print("EXP-0002 DEPTH SWEEP COMPLETE")
    print("=" * 64)

    for variant in variants:
        print(
            f"{variant['depth']:>2}/{total_layers} layers | "
            f"{variant['accuracy']:.1%} accuracy | "
            f"{variant['average_time_per_question_seconds']:.2f}s/q | "
            f"{variant['overall_tokens_per_second']:.2f} tok/s | "
            f"VRAM {variant['max_peak_vram_gib']:.2f} GiB"
        )

    print(f"Hardware samples: {sampler.count}")
    print(f"Results: {result_path}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
