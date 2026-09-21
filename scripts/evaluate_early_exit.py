"""Evaluate the EXP-0003 trained 30-layer exit adapter.

Uses the same fixed CFI GSM8K protocol as EXP-0001. The adapter is inserted
before the original final RMSNorm, while the Qwen3 transformer layer list is
reduced to the configured exit depth. This lets standard Transformers generate
using only the trained prefix plus the lightweight exit adapter.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = Path(
    r"E:\Titan Forge Industries\CFI-Data\HuggingFace\hub\models--Qwen--Qwen3-4B-Base\snapshots\906bfd4b4dc7f14ee4320094d8b41684abff8539"
)
DEFAULT_DATASET = Path(
    r"E:\Titan Forge Industries\CFI-Data\Datasets\CFI-Eval-0001-GSM8K\gsm8k_test_100.jsonl"
)
DEFAULT_ADAPTER = Path(
    r"E:\Titan Forge Industries\CFI-Data\Models\CFI-EXP-0003-EarlyExit-30\exit_adapter.pt"
)
DEFAULT_OUTPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0003-EarlyExit"
)


class ExitAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck, bias=True)
        self.up = nn.Linear(bottleneck, hidden_size, bias=True)
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.up(self.activation(self.down(hidden_states)))


class AdapterNorm(nn.Module):
    def __init__(self, adapter: ExitAdapter, norm: nn.Module) -> None:
        super().__init__()
        self.adapter = adapter
        self.norm = norm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.adapter(hidden_states))


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


def load_rows(path: Path, count: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= count:
                    break
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def build_model(model_path: Path, adapter_path: Path, depth: int):
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

    all_layers = model.model.layers
    total = len(all_layers)
    if depth < 1 or depth >= total:
        raise ValueError(f"Depth must be between 1 and {total - 1}.")

    checkpoint = torch.load(adapter_path, map_location="cuda:0", weights_only=True)
    adapter = ExitAdapter(
        int(checkpoint["hidden_size"]),
        int(checkpoint["bottleneck"]),
    ).to("cuda:0", dtype=torch.bfloat16)
    adapter.load_state_dict(checkpoint["state_dict"])
    adapter.eval()

    model.model.layers = nn.ModuleList(list(all_layers[:depth]))
    model.model.norm = AdapterNorm(adapter, model.model.norm)
    return tokenizer, model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--questions", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed-base", type=int, default=42000)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.dataset, args.questions)

    tokenizer, model = build_model(args.model, args.adapter, depth=30)

    results = []
    correct = 0
    total_time = 0.0
    total_tokens = 0

    print("GPU:", torch.cuda.get_device_name(0))
    print("Exit depth: 30/36")
    print("Questions:", len(rows))

    for index, row in enumerate(rows, 1):
        seed = args.seed_base + index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        prompt = (
            "Question: "
            + str(row["question"])
            + "\n"
            + "Solve the problem step by step. "
            + "End with: The answer is <number>.\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        prompt_len = int(inputs["input_ids"].shape[-1])

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        generated = output[0, prompt_len:]
        tokens = int(generated.shape[-1])
        text = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        expected = extract_expected(str(row["answer"]))
        predicted = extract_predicted(text)
        is_correct = predicted == expected
        correct += int(is_correct)
        total_time += elapsed
        total_tokens += tokens

        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

        results.append(
            {
                "index": index,
                "seed": seed,
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "elapsed_seconds": elapsed,
                "generated_tokens": tokens,
                "tokens_per_second": tokens / elapsed if elapsed > 0 else 0.0,
                "peak_vram_gib": peak_vram,
                "output": text,
            }
        )

        print(
            f"[{index}/{len(rows)}] "
            f"expected={expected} "
            f"predicted={predicted} "
            f"correct={is_correct} "
            f"time={elapsed:.2f}s "
            f"tokens={tokens} "
            f"tok/s={tokens / elapsed:.2f} "
            f"VRAM={peak_vram:.2f} GiB"
        )

    summary = {
        "schema_version": 1,
        "benchmark": "CFI-Eval-0003-EarlyExit",
        "model_path": str(args.model),
        "adapter_path": str(args.adapter),
        "depth": 30,
        "total_model_layers": 36,
        "questions": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "total_generation_time_seconds": total_time,
        "average_time_per_question_seconds": total_time / len(rows),
        "total_generated_tokens": total_tokens,
        "overall_tokens_per_second": (
            total_tokens / total_time if total_time > 0 else 0.0
        ),
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
        "results": results,
    }

    path = args.output / "early_exit_30_results.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print("EXP-0003 EARLY-EXIT EVALUATION COMPLETE")
    print("=" * 64)
    print(f"Accuracy: {correct}/{len(rows)} = {correct / len(rows):.1%}")
    print(f"Average/question: {total_time / len(rows):.2f}s")
    print(f"Overall throughput: {total_tokens / total_time:.2f} tok/s")
    print(f"Results: {path}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
