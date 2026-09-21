"""EXP-0006: benchmark shared-weight truncated-assistant speculative decoding.

Qwen3-4B-Base is not an early-exit-trained checkpoint. Therefore this
experiment does NOT use Transformers' assistant_early_exit mechanism.

Instead, it creates a lightweight 30-layer assistant object that SHARES the
target model's loaded modules and weights. The ordinary assistant_model
speculative-decoding path then lets the 30-layer assistant draft candidate
tokens and the 36-layer target verify them.

No model weights are changed or duplicated.
"""

from __future__ import annotations

import argparse
import copy
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
DEFAULT_OUTPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0006-SelfSpeculative"
)


def load_rows(path: Path, count: int) -> list[dict]:
    rows: list[dict] = []
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


def make_shared_weight_assistant(target_model, depth: int):
    """Create a truncated assistant without duplicating loaded weights."""
    total_layers = len(target_model.model.layers)
    if depth <= 0 or depth >= total_layers:
        raise ValueError(
            f"Assistant depth must be between 1 and {total_layers - 1}."
        )

    assistant = copy.copy(target_model)
    assistant.model = copy.copy(target_model.model)

    # The assistant needs its own config because its cache must contain only
    # the truncated number of layers. The target config remains at 36 layers.
    assistant.config = copy.deepcopy(target_model.config)
    assistant.model.config = assistant.config
    assistant.config.num_hidden_layers = depth

    # Share all actual module weights. Only the ModuleList container is new.
    assistant.model.layers = nn.ModuleList(
        list(target_model.model.layers[:depth])
    )
    assistant.model.embed_tokens = target_model.model.embed_tokens
    assistant.model.norm = target_model.model.norm
    assistant.lm_head = target_model.lm_head

    assistant.generation_config = copy.deepcopy(target_model.generation_config)
    assistant.generation_config.assistant_early_exit = None
    assistant.generation_config.num_assistant_tokens = 4
    # Qwen3 speculative rollback can encounter uninitialized DynamicCache
    # entries. StaticCache allocates every layer up front, making candidate
    # rollback deterministic for this feasibility experiment.
    assistant.generation_config.cache_implementation = "static"
    assistant.eval()

    return assistant


def run_baseline(
    model,
    tokenizer,
    prompt: str,
    *,
    seed: int,
    max_new_tokens: int,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    prompt_len = int(inputs["input_ids"].shape[-1])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            cache_implementation="static",
        )
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


def run_speculative(
    model,
    tokenizer,
    assistant_model,
    prompt: str,
    *,
    seed: int,
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

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            assistant_model=assistant_model,
            max_new_tokens=max_new_tokens,
            num_assistant_tokens=num_assistant_tokens,
            num_assistant_tokens_schedule="constant",
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            cache_implementation="static",
        )
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

    total_layers = len(model.model.layers)
    if total_layers != 36:
        raise ValueError(
            f"Expected Qwen3-4B-Base with 36 layers, found {total_layers}."
        )

    assistant_model = make_shared_weight_assistant(model, depth=30)

    shared_param_check = (
        assistant_model.model.layers[0].self_attn.q_proj.weight.data_ptr()
        == model.model.layers[0].self_attn.q_proj.weight.data_ptr()
    )

    baseline = []
    assisted = []

    print("GPU:", torch.cuda.get_device_name(0))
    print("Target layers:", total_layers)
    print("Assistant layers:", len(assistant_model.model.layers))
    print("Shared weights:", shared_param_check)
    print("Assistant tokens:", args.num_assistant_tokens)
    print("Questions:", len(rows))

    for index, row in enumerate(rows, 1):
        seed = args.seed_base + index
        expected = expected_answer(str(row["answer"]))
        prompt = prompt_for(str(row["question"]))

        base = run_baseline(
            model,
            tokenizer,
            prompt,
            seed=seed,
            max_new_tokens=args.max_new_tokens,
        )
        base.update(
            index=index,
            seed=seed,
            expected=expected,
            predicted=predicted_answer(base["text"]),
        )
        base["correct"] = base["predicted"] == expected
        baseline.append(base)

        assisted_run = run_speculative(
            model,
            tokenizer,
            assistant_model,
            prompt,
            seed=seed,
            max_new_tokens=args.max_new_tokens,
            num_assistant_tokens=args.num_assistant_tokens,
        )
        assisted_run.update(
            index=index,
            seed=seed,
            expected=expected,
            predicted=predicted_answer(assisted_run["text"]),
        )
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
        "schema_version": 2,
        "benchmark": "CFI-Eval-0006-SelfSpeculative",
        "method": (
            "Shared-weight truncated assistant speculative decoding. "
            "Qwen3-4B-Base remains the 36-layer target; a separate assistant "
            "object executes the first 30 shared layers and the ordinary "
            "assistant_model verification path validates candidates."
        ),
        "model_path": str(args.model),
        "questions": len(rows),
        "target_layers": total_layers,
        "assistant_layers": len(assistant_model.model.layers),
        "weights_shared": shared_param_check,
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
            "accuracy": sum(int(x["correct"]) for x in baseline) / len(baseline),
            "total_time_seconds": base_time,
            "avg_time_per_question": base_time / len(baseline),
            "total_tokens": base_tokens,
            "overall_tok_s": base_tokens / base_time if base_time else 0.0,
            "runs": baseline,
        },
        "assisted": {
            "correct": sum(int(x["correct"]) for x in assisted),
            "accuracy": sum(int(x["correct"]) for x in assisted) / len(assisted),
            "total_time_seconds": spec_time,
            "avg_time_per_question": spec_time / len(assisted),
            "total_tokens": spec_tokens,
            "overall_tok_s": spec_tokens / spec_time if spec_time else 0.0,
            "runs": assisted,
        },
        "speedup": base_time / spec_time if spec_time else 0.0,
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
        f"Speculative: {result['assisted']['correct']}/{len(assisted)} "
        f"accuracy={result['assisted']['accuracy']:.1%} "
        f"time={spec_time:.2f}s "
        f"tok/s={result['assisted']['overall_tok_s']:.2f}"
    )
    print(f"Speedup: {result['speedup']:.3f}x")
    print(f"Results: {path}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
