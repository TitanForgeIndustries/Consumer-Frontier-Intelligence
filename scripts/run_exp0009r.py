"""EXP-0009R: scorable, cache-safe evaluation of the frozen Q predictor."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import nn

from run_exp0008 import extract_expected, extract_predicted, format_prompt, load_rows
from run_exp0009n import build_model, seed_all
from run_exp0009q import (
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    MLPReplacement,
    MLPTransition,
    compare_tokens,
)


DEFAULT_CHECKPOINT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009Q-Cache-Safe-MLP-Reconstruction-rerun-20260922\mlp_transition_state_dict_fp32.pt"
)
DEFAULT_OUTPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009R-Scorable-Cache-Safe-Evaluation"
)
MODES = ("full", "mlp_zero", "mlp_predicted")
HELD_OUT_INDICES = (8, 9, 10)
PREFIX_CAPS = (128, 256, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--runtime-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=42000)
    return parser.parse_args()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prefix_answers(
    tokens: list[int], tokenizer: Any, caps: tuple[int, ...]
) -> dict[str, dict[str, Any]]:
    scores = {}
    for cap in caps:
        prefix = tokens[:cap]
        text = tokenizer.decode(prefix, skip_special_tokens=True).strip()
        scores[str(cap)] = {
            "generated_tokens": len(prefix),
            "predicted": extract_predicted(text),
        }
    return scores


def scorable_baselines(results: list[dict[str, Any]]) -> bool:
    return len(results) == len(HELD_OUT_INDICES) and all(
        result["predicted"] is not None for result in results
    )


def measurement_order(question_index: int, repeat: int) -> tuple[str, ...]:
    offset = (question_index - HELD_OUT_INDICES[0] + repeat - 1) % len(MODES)
    return MODES[offset:] + MODES[:offset]


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    seed: int,
    max_new_tokens: int,
    fixed_length: bool = False,
) -> dict[str, Any]:
    seed_all(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    prompt_length = int(inputs["input_ids"].shape[-1])
    options = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if fixed_length:
        options["min_new_tokens"] = max_new_tokens

    torch.cuda.reset_peak_memory_stats(device="cuda:0")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, **options)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    tokens = generated[0, prompt_length:].detach().cpu().tolist()
    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    return {
        "tokens": tokens,
        "generated_tokens": len(tokens),
        "text": text,
        "predicted": extract_predicted(text),
        "seconds": elapsed,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device="cuda:0"),
        "ended_with_eos": bool(tokens and tokens[-1] == tokenizer.eos_token_id),
    }


class ZeroMLP(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(hidden)


def generate_mode(
    model: Any,
    tokenizer: Any,
    prompt: str,
    seed: int,
    max_new_tokens: int,
    mode: str,
    predictor: MLPTransition | None,
    fixed_length: bool = False,
) -> dict[str, Any]:
    if mode == "full":
        return generate(model, tokenizer, prompt, seed, max_new_tokens, fixed_length)
    if mode == "mlp_zero":
        replacement = ZeroMLP().to("cuda:0")
    elif mode == "mlp_predicted" and predictor is not None:
        replacement = predictor
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    swap = MLPReplacement(model, replacement)
    try:
        return generate(model, tokenizer, prompt, seed, max_new_tokens, fixed_length)
    finally:
        swap.remove()


def load_predictor(path: Path, model: Any) -> MLPTransition:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["experiment"] != "EXP-0009Q" or not checkpoint["base_model_frozen"]:
        raise ValueError("Expected a frozen-base EXP-0009Q predictor checkpoint")
    hidden_size = model.config.hidden_size
    if checkpoint["hidden_size"] != hidden_size or checkpoint["bottleneck"] != 128:
        raise ValueError("EXP-0009Q predictor architecture does not match")
    predictor = MLPTransition(hidden_size, 128)
    predictor.load_state_dict(checkpoint["state_dict"], strict=True)
    return predictor.to(device="cuda:0", dtype=next(model.model.norm.parameters()).dtype).eval()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_new_tokens != PREFIX_CAPS[-1]:
        raise ValueError("The predeclared natural-stop cap is 512 tokens")
    if args.runtime_tokens != 128 or args.warmup_tokens != 16 or args.repeats != 2:
        raise ValueError("R timing is predeclared at 128 tokens, 16 warmup, 2 repeats")
    rows = load_rows(args.dataset, max(HELD_OUT_INDICES))
    tokenizer, model = build_model(args.model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    summary: dict[str, Any] = {
        "experiment": "EXP-0009R",
        "status": "in_progress",
        "base_model_frozen": True,
        "checkpoint_trained_in": "EXP-0009Q",
        "model": str(args.model),
        "dataset": str(args.dataset),
        "dataset_sha256": digest(args.dataset),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest(args.checkpoint),
        "config": {
            "held_out_indices": list(HELD_OUT_INDICES),
            "max_new_tokens": args.max_new_tokens,
            "prefix_caps": list(PREFIX_CAPS),
            "natural_stop": True,
            "decoding": "greedy deterministic",
            "runtime_tokens": args.runtime_tokens,
            "warmup_tokens": args.warmup_tokens,
            "repeats": args.repeats,
            "seed_base": args.seed_base,
            "timing_order": "rotated across questions and repeats",
        },
        "baseline": [],
    }

    for index in HELD_OUT_INDICES:
        row = rows[index - 1]
        prompt = format_prompt(str(row["question"]))
        expected = extract_expected(str(row["answer"]))
        full = generate(model, tokenizer, prompt, args.seed_base + index, args.max_new_tokens)
        full["prefix_answers"] = prefix_answers(full["tokens"], tokenizer, PREFIX_CAPS)
        full["question_index"] = index
        full["expected"] = expected
        full["correct"] = full["predicted"] == expected
        summary["baseline"].append(full)
        save_json(args.output / "summary.json", summary)
        print(
            f"Q{index} full: tokens={full['generated_tokens']} "
            f"answer={full['predicted']} expected={expected} "
            f"eos={full['ended_with_eos']} time={full['seconds']:.2f}s",
            flush=True,
        )

    if not scorable_baselines(summary["baseline"]):
        summary["status"] = "baseline_unscorable"
        save_json(args.output / "summary.json", summary)
        return summary
    if args.baseline_only:
        summary["status"] = "baseline_ready"
        save_json(args.output / "summary.json", summary)
        return summary

    predictor = load_predictor(args.checkpoint, model)
    summary["comparison"] = []
    for index in HELD_OUT_INDICES:
        row = rows[index - 1]
        prompt = format_prompt(str(row["question"]))
        expected = extract_expected(str(row["answer"]))
        full = summary["baseline"][index - HELD_OUT_INDICES[0]]
        candidates = {}
        for mode in MODES[1:]:
            result = generate_mode(
                model, tokenizer, prompt, args.seed_base + index,
                args.max_new_tokens, mode, predictor,
            )
            result["correct"] = result["predicted"] == expected
            result["vs_full"] = compare_tokens(full["tokens"], result["tokens"])
            candidates[mode] = result
            print(
                f"Q{index} {mode}: tokens={result['generated_tokens']} "
                f"answer={result['predicted']} correct={result['correct']} "
                f"first_divergence={result['vs_full']['first_divergence_generated_token']}",
                flush=True,
            )

        timings = {mode: [] for mode in MODES}
        for repeat in range(1, args.repeats + 1):
            for mode in measurement_order(index, repeat):
                generate_mode(
                    model, tokenizer, prompt, args.seed_base + index,
                    args.warmup_tokens, mode, predictor, fixed_length=True,
                )
                timed = generate_mode(
                    model, tokenizer, prompt, args.seed_base + index,
                    args.runtime_tokens, mode, predictor, fixed_length=True,
                )
                timings[mode].append({
                    "repeat": repeat,
                    "seconds": timed["seconds"],
                    "generated_tokens": timed["generated_tokens"],
                    "peak_memory_bytes": timed["peak_memory_bytes"],
                })
        summary["comparison"].append({
            "question_index": index,
            "expected": expected,
            "candidates": candidates,
            "fixed_length_timings": timings,
            "fixed_length_speedup": {
                mode: (
                    sum(item["seconds"] for item in timings["full"])
                    / sum(item["seconds"] for item in timings[mode])
                )
                for mode in MODES[1:]
            },
        })
        save_json(args.output / "summary.json", summary)

    summary["status"] = "completed"
    save_json(args.output / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {args.output}")
    args.output.mkdir(parents=True)
    try:
        summary = run(args)
    except Exception as failure:
        save_json(args.output / "implementation_failure.json", {
            "experiment": "EXP-0009R",
            "error": repr(failure),
            "traceback": traceback.format_exc(),
        })
        raise
    print(f"R status: {summary['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
