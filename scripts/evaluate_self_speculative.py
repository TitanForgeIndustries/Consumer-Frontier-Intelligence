"""EXP-0006: benchmark native self-speculative decoding at layer 30.

This compares ordinary 36-layer generation against Transformers' native
self-speculative decoding, using the Qwen3-4B model's intermediate layer 30 as
the assistant/early-exit path.

The model weights are unchanged. The experiment uses the same 5-question
feasibility gate, sampling settings, and seeds as the CFI GSM8K evaluator.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = Path(
    r"E:\Titan Forge Industries\CFI-Data\HuggingFace\hub\models--Qwen--Qwen3-4B-Base\snapshots\906bfd4b4dc7f14ee4320094d8b41684abff8539"
)
DEFAULT_DATASET = Path(
    r"E:\Titan Forge Industries\CFI-Data\Datasets\CFI-Eval-0001-GSM8K\gsm8k_test_100.jsonl"
)
DEFAULT_OUTPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0006-SelfSpeculative"
)


def load_rows(path: Path, count: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= count:
                    break
    if not rows:
        raise ValueError(f"No examples in {path}")
    return rows


def prompt_for(question: str) -> str:
    return (
        "Question: "
        + question
        + "\n"
        + "Solve the problem step by step. "
        + "End with: The answer is <number>.\n"
    )


def canonical(value: str | None) -> str | None:
    if value is None:
        return None
    raw = value.strip().replace(",", "").replace("$", "")
    try:
        number = Decimal(raw)
    except InvalidOperation:
        return raw
    if number == number.to_integral_value():
        return str(int(number))
    return format(number.normalize(), "f").rstrip("0").rstrip(".")


def expected_answer(text: str) -> str | None:
    match = re.search(
        r"####\s*(?:<\s*)?\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*>?",
        text,
        re.IGNORECASE,
    )
    return canonical(match.group(1)) if match else None


def predicted_answer(text: str) -> str | None:
    matches = re.findall(
        r"(?:The answer is|####)\s*:?\s*<?\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*>?",
        text,
        re.IGNORECASE,
    )
    if matches:
        return canonical(matches[-1])

    for line in reversed([x.strip() for x in text.splitlines() if x.strip()]):
        if re.fullmatch(r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?", line):
            return canonical(line)
    return None


def load_model(model_path: Path):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=quant,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.eval()
    return tokenizer, model


def run_generation(
    model,
    tokenizer,
    prompt: str,
    *,
    seed: int,
    self_speculative: bool,
    max_new_tokens: int,
    num_assistant_tokens: int,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    prompt_len = int(inputs["input_ids"].shape[-1])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    if self_speculative:
        kwargs.update(
            {
                "assistant_early_exit": 30,
                "num_assistant_tokens": num_assistant_tokens,
                "num_assistant_tokens_schedule": "constant",
            }
        )

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    new_tokens = output[0, prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return {
        "text": text,
        "tokens": int(new_tokens.shape[-1]),
        "elapsed_seconds": elapsed,
        "tokens_per_second": (
            int(new_tokens.shape[-1]) / elapsed if elapsed > 0 else 0.0
        ),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-assistant-tokens", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=42000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    rows = load_rows(args.dataset, args.questions)
    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(args.model)

    baseline = []
    assisted = []

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model layers:", len(model.model.layers))
    print("Self-speculative exit:", 30)
    print("Assistant tokens:", args.num_assistant_tokens)
    print("Questions:", len(rows))

    for index, row in enumerate(rows, 1):
        seed = args.seed_base + index
        expected = expected_answer(str(row["answer"]))
        prompt = prompt_for(str(row["question"]))

        base = run_generation(
            model,
            tokenizer,
            prompt,
            seed=seed,
            self_speculative=False,
            max_new_tokens=args.max_new_tokens,
            num_assistant_tokens=args.num_assistant_tokens,
        )
        base["index"] = index
        base["seed"] = seed
        base["expected"] = expected
        base["predicted"] = predicted_answer(base["text"])
        base["correct"] = base["predicted"] == expected
        baseline.append(base)

        assisted_run = run_generation(
            model,
            tokenizer,
            prompt,
            seed=seed,
            self_speculative=True,
            max_new_tokens=args.max_new_tokens,
            num_assistant_tokens=args.num_assistant_tokens,
        )
        assisted_run["index"] = index
        assisted_run["seed"] = seed
        assisted_run["expected"] = expected
        assisted_run["predicted"] = predicted_answer(assisted_run["text"])
        assisted_run["correct"] = assisted_run["predicted"] == expected
        assisted.append(assisted_run)

        print(
            f"[{index}/{len(rows)}] "
            f"base={base['correct']} {base['elapsed_seconds']:.2f}s "
            f"spec={assisted_run['correct']} {assisted_run['elapsed_seconds']:.2f}s"
        )

    base_time = sum(x["elapsed_seconds"] for x in baseline)
    spec_time = sum(x["elapsed_seconds"] for x in assisted)
    base_tokens = sum(x["tokens"] for x in baseline)
    spec_tokens = sum(x["tokens"] for x in assisted)

    result = {
        "schema_version": 1,
        "benchmark": "CFI-Eval-0006-SelfSpeculative",
        "model_path": str(args.model),
        "questions": len(rows),
        "exit_layer": 30,
        "total_layers": len(model.model.layers),
        "assistant_tokens": args.num_assistant_tokens,
        "sampling": {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "seed_base": args.seed_base,
        },
        "baseline": {
            "correct": sum(int(x["correct"]) for x in baseline),
            "accuracy": (
                sum(int(x["correct"]) for x in baseline) / len(baseline)
            ),
            "total_time_seconds": base_time,
            "avg_time_per_question": base_time / len(baseline),
            "total_tokens": base_tokens,
            "overall_tok_s": base_tokens / base_time if base_time else 0.0,
            "runs": baseline,
        },
        "self_speculative": {
            "correct": sum(int(x["correct"]) for x in assisted),
            "accuracy": (
                sum(int(x["correct"]) for x in assisted) / len(assisted)
            ),
            "total_time_seconds": spec_time,
            "avg_time_per_question": spec_time / len(assisted),
            "total_tokens": spec_tokens,
            "overall_tok_s": spec_tokens / spec_time if spec_time else 0.0,
            "runs": assisted,
        },
        "speedup": (
            base_time / spec_time if spec_time > 0 else 0.0
        ),
    }

    path = args.output / "self_speculative_results.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print("EXP-0006 SELF-SPECULATIVE DECODING COMPLETE")
    print("=" * 64)
    print(
        f"Baseline: {result['baseline']['correct']}/{len(baseline)} "
        f"accuracy={result['baseline']['accuracy']:.1%} "
        f"time={base_time:.2f}s "
        f"tok/s={result['baseline']['overall_tok_s']:.2f}"
    )
    print(
        f"Self-spec: {result['self_speculative']['correct']}/{len(assisted)} "
        f"accuracy={result['self_speculative']['accuracy']:.1%} "
        f"time={spec_time:.2f}s "
        f"tok/s={result['self_speculative']['overall_tok_s']:.2f}"
    )
    print(f"Speedup: {result['speedup']:.3f}x")
    print(f"Results: {path}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
