"""Evaluate a causal LM on the fixed CFI GSM8K benchmark.

This runner keeps the evaluation protocol identical across a base model and a
PEFT/QLoRA adapter:
- fixed 100-question JSONL evaluation set
- per-question deterministic seed
- 4-bit NF4 base model with BF16 compute
- Qwen3-style sampling parameters
- identical prompt and generation limits
- per-question accuracy, latency, token count, throughput, and VRAM
- optional experiment-logger integration and background hardware sampling

Examples:

    python scripts/evaluate_gsm8k.py --model /path/to/model

    python scripts/evaluate_gsm8k.py \
        --model /path/to/model \
        --adapter /path/to/checkpoint-250 \
        --experiment-id EXP-0001

All paths can be supplied explicitly so results and source data can remain on
the user's chosen data drive.
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - only needed when --adapter is used.
    PeftModel = None

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
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0001-GSM8K"
)

MAX_NEW_TOKENS_DEFAULT = 512
SEED_BASE_DEFAULT = 42000
TEMPERATURE_DEFAULT = 0.6
TOP_P_DEFAULT = 0.95
TOP_K_DEFAULT = 20
HARDWARE_INTERVAL_DEFAULT = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed CFI GSM8K evaluation benchmark.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Local base-model snapshot directory.",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Optional PEFT/QLoRA adapter directory.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Fixed GSM8K JSONL evaluation set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for benchmark results.",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=100,
        help="Number of questions from the fixed set to evaluate.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS_DEFAULT,
        help="Generation ceiling.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=SEED_BASE_DEFAULT,
        help="Per-question seed is seed-base + question index.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE_DEFAULT,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=TOP_P_DEFAULT,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_DEFAULT,
    )
    parser.add_argument(
        "--experiment-id",
        help="Optional CFI experiment ID for evaluation/hardware logging.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("experiments"),
        help="Experiment root when --experiment-id is supplied.",
    )
    parser.add_argument(
        "--hardware-interval",
        type=float,
        default=HARDWARE_INTERVAL_DEFAULT,
        help="Seconds between nvidia-smi/system samples.",
    )
    return parser.parse_args()


def load_rows(dataset_path: Path, count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("--questions must be positive")

    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Dataset row is not an object: {dataset_path}")
            rows.append(value)
            if len(rows) >= count:
                break

    if not rows:
        raise ValueError(f"No questions found in {dataset_path}")

    return rows


CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CHINESE_UNITS = {
    "十": 10,
    "百": 100,
    "千": 1000,
    "万": 10_000,
    "亿": 100_000_000,
}


def parse_chinese_integer(text: str) -> int | None:
    text = text.strip()
    if not text or any(ch not in CHINESE_DIGITS and ch not in CHINESE_UNITS for ch in text):
        return None

    total = 0
    section = 0
    digit = 0
    saw_unit = False

    for ch in text:
        if ch in CHINESE_DIGITS:
            digit = CHINESE_DIGITS[ch]
            continue

        saw_unit = True
        unit = CHINESE_UNITS[ch]

        if unit < 10_000:
            if digit == 0 and ch == "十":
                digit = 1
            section += digit * unit
            digit = 0
        elif unit in (10_000, 100_000_000):
            section += digit
            total += section * unit
            section = 0
            digit = 0

    value = total + section + digit
    return value if saw_unit or len(text) == 1 else None


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


def extract_expected(answer_text: str) -> str | None:
    match = re.search(
        r"####\s*(?:<\s*)?\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*>?",
        answer_text,
        re.IGNORECASE,
    )
    return canonical_number(match.group(1)) if match else None


def extract_predicted(text: str) -> str | None:
    explicit = re.findall(
        r"(?:The answer is|####)\s*:?\s*<?\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*>?",
        text,
        re.IGNORECASE,
    )
    if explicit:
        return canonical_number(explicit[-1])

    explicit_chinese = re.findall(
        r"(?:The answer is|####)\s*:?\s*<?\s*([零〇一二两三四五六七八九十百千万亿]+)\s*>?",
        text,
        re.IGNORECASE,
    )
    if explicit_chinese:
        value = parse_chinese_integer(explicit_chinese[-1])
        if value is not None:
            return str(value)

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]
    for line in reversed(lines):
        match = re.fullmatch(
            r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?",
            line,
        )
        if match:
            return canonical_number(match.group(0).replace("$", "").strip())

    return None


class HardwareSampler:
    """Background sampler using the CFI logger's existing hardware collector."""

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
        self._thread = threading.Thread(
            target=self._run,
            name="cfi-hardware-sampler",
            daemon=True,
        )
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
                handle.flush()
            self.count += 1
            self._stop.wait(self.interval_seconds)


def build_model(model_path: Path, adapter_path: Path | None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

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

    if adapter_path is not None:
        if PeftModel is None:
            raise RuntimeError(
                "The --adapter option requires PEFT. "
                "Install it in the active CFI environment first."
            )
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )

    model.eval()
    return tokenizer, model


def run_one(
    *,
    tokenizer,
    model,
    row: dict[str, Any],
    index: int,
    seed_base: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[str, Any]:
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
            stop_strings=["\nQuestion:", "\nProblem:"],
            tokenizer=tokenizer,
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

    peak_vram = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available()
        else None
    )

    return {
        "index": index,
        "seed": seed,
        "expected": expected,
        "predicted": predicted,
        "correct": predicted == expected,
        "elapsed_seconds": round(elapsed, 6),
        "generated_tokens": generated_tokens,
        "tokens_per_second": round(tok_per_sec, 6),
        "peak_vram_gib": round(peak_vram, 6) if peak_vram is not None else None,
        "output": text,
    }


def main() -> int:
    args = parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.hardware_interval <= 0:
        raise ValueError("--hardware-interval must be positive")

    args.output.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.dataset, args.questions)

    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", args.model)
    print("Adapter:", args.adapter if args.adapter else "None")
    print("Dataset:", args.dataset)
    print("Questions:", len(rows))

    print("\nLoading 4-bit NF4 model...")
    tokenizer, model = build_model(args.model, args.adapter)

    experiment_logger = None
    hardware_output = args.output / "hardware_samples.jsonl"

    if args.experiment_id:
        if ExperimentLogger is None:
            raise RuntimeError(
                "CFI experiment logger is unavailable in the active environment."
            )
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

    results: list[dict[str, Any]] = []
    correct = 0
    total_time = 0.0
    total_tokens = 0

    print("\nRunning GSM8K benchmark...\n")
    sampler.start()

    try:
        for index, row in enumerate(rows, 1):
            result = run_one(
                tokenizer=tokenizer,
                model=model,
                row=row,
                index=index,
                seed_base=args.seed_base,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            results.append(result)

            correct += int(result["correct"])
            total_time += result["elapsed_seconds"]
            total_tokens += result["generated_tokens"]

            print(
                f"[{index}/{len(rows)}] "
                f"expected={result['expected']} "
                f"predicted={result['predicted']} "
                f"correct={result['correct']} "
                f"time={result['elapsed_seconds']:.2f}s "
                f"tokens={result['generated_tokens']} "
                f"tok/s={result['tokens_per_second']:.2f} "
                f"VRAM={result['peak_vram_gib']:.2f} GiB"
            )
    finally:
        sampler.stop()

    question_count = len(results)
    accuracy = correct / question_count
    average_time = total_time / question_count
    overall_tok_per_sec = total_tokens / total_time if total_time > 0 else 0.0

    summary = {
        "schema_version": 1,
        "benchmark": "CFI-Eval-0001-GSM8K",
        "model_path": str(args.model),
        "adapter_path": str(args.adapter) if args.adapter else None,
        "dataset_path": str(args.dataset),
        "questions": question_count,
        "correct": correct,
        "accuracy": accuracy,
        "total_generation_time_seconds": round(total_time, 6),
        "average_time_per_question_seconds": round(average_time, 6),
        "total_generated_tokens": total_tokens,
        "overall_tokens_per_second": round(overall_tok_per_sec, 6),
        "max_new_tokens": args.max_new_tokens,
        "sampling": {
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
        "results": results,
    }

    label = "adapter" if args.adapter else "base"
    result_path = args.output / f"gsm8k_{label}_results.json"
    result_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if experiment_logger is not None:
        experiment_logger.record_evaluation(
            benchmark="GSM8K",
            accuracy=accuracy,
            samples=question_count,
            correct=correct,
            total_generation_time_seconds=round(total_time, 6),
            average_time_per_question_seconds=round(average_time, 6),
            total_generated_tokens=total_tokens,
            overall_tokens_per_second=round(overall_tok_per_sec, 6),
            results_file=str(result_path),
        )

    print("\n" + "=" * 64)
    print("CFI GSM8K EVALUATION COMPLETE")
    print("=" * 64)
    print(f"Accuracy: {correct}/{question_count} = {accuracy:.1%}")
    print(f"Total generation time: {total_time:.2f}s")
    print(f"Average/question: {average_time:.2f}s")
    print(f"Generated tokens: {total_tokens}")
    print(f"Overall throughput: {overall_tok_per_sec:.2f} tok/s")
    print(f"Hardware samples: {sampler.count}")
    print(f"Results: {result_path}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
