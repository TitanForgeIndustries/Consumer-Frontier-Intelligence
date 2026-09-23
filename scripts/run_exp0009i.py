"""CFI EXP-0009I: decomposition stability and whole-layer-36 skip diagnostic.

This experiment asks two questions:
1. Is the internal L36 attention/MLP decomposition stable across fresh forwards?
2. What happens if the entire L36 transformation is functionally skipped?

This is diagnostic only. The layer-skip hook still computes L36 internally, so it
does not measure a speedup.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_exp0008 import extract_expected, extract_predicted, format_prompt, load_rows


DEFAULT_MODEL = Path(
    r"E:\Titan Forge Industries\CFI-Data\HuggingFace\hub\models--Qwen--Qwen3-4B-Base\snapshots\906bfd4b4dc7f14ee4320094d8b41684abff8539"
)
DEFAULT_DATASET = Path(
    r"E:\Titan Forge Industries\CFI-Data\Datasets\CFI-Eval-0001-GSM8K\gsm8k_test_100.jsonl"
)
DEFAULT_OUTPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009I-Decomposition-Stability-Layer-Skip"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CFI EXP-0009I.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--questions", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed-base", type=int, default=42000)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def component_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Expected tensor-like output, got {type(output).__name__}")


def build_model(model_path: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-0009I.")

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
        attn_implementation="eager",
    )
    model.eval()
    return tokenizer, model


def capture_forward(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    h35: list[torch.Tensor] = []
    attn: list[torch.Tensor] = []
    mlp: list[torch.Tensor] = []
    h36: list[torch.Tensor] = []

    layer = model.model.layers[35]

    def capture_h35(_module: Any, _inputs: Any, output: Any) -> None:
        h35.append(component_tensor(output).detach().float().cpu())

    def capture_attn(_module: Any, _inputs: Any, output: Any) -> None:
        attn.append(component_tensor(output).detach().float().cpu())

    def capture_mlp(_module: Any, _inputs: Any, output: Any) -> None:
        mlp.append(component_tensor(output).detach().float().cpu())

    def capture_h36(_module: Any, _inputs: Any, output: Any) -> None:
        h36.append(component_tensor(output).detach().float().cpu())

    hooks = [
        model.model.layers[34].register_forward_hook(capture_h35),
        layer.self_attn.register_forward_hook(capture_attn),
        layer.mlp.register_forward_hook(capture_mlp),
        layer.register_forward_hook(capture_h36),
    ]
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
    finally:
        for handle in hooks:
            handle.remove()

    for name, values in (
        ("H35", h35),
        ("L36 attention", attn),
        ("L36 MLP", mlp),
        ("H36", h36),
    ):
        if len(values) != 1:
            raise RuntimeError(f"Expected one {name} capture, got {len(values)}.")

    return {
        "logits": outputs.logits.detach().float().cpu()[0],
        "h35": h35[0][0],
        "attention": attn[0][0],
        "mlp": mlp[0][0],
        "h36": h36[0][0],
    }


def abs_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = b.float() - a.float()
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "mean_l2": float(torch.linalg.vector_norm(diff, dim=-1).mean().item()),
        "cosine_mean": float(
            F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()
        ),
    }


def logits_metrics(
    oracle_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> dict[str, float]:
    oracle = oracle_logits.float()
    candidate = candidate_logits.float()
    target = target_ids.long().unsqueeze(-1)

    oracle_logp = F.log_softmax(oracle, dim=-1)
    candidate_logp = F.log_softmax(candidate, dim=-1)
    oracle_probs = oracle_logp.exp()

    kl = (oracle_probs * (oracle_logp - candidate_logp)).sum(dim=-1)
    oracle_top = oracle.argmax(dim=-1)
    candidate_top = candidate.argmax(dim=-1)

    oracle_target = oracle_probs.gather(-1, target).squeeze(-1)
    candidate_target = candidate_logp.exp().gather(-1, target).squeeze(-1)

    return {
        "token_count": float(target_ids.numel()),
        "top1_agreement": float((oracle_top == candidate_top).float().mean().item()),
        "target_probability_ratio_mean": float(
            (candidate_target / oracle_target.clamp_min(1e-20)).mean().item()
        ),
        "kl_oracle_to_candidate_mean": float(kl.mean().item()),
        "logit_l2_mean": float(
            torch.linalg.vector_norm(oracle - candidate, dim=-1).mean().item()
        ),
    }


class WholeLayerSkip:
    """Functionally replace L36's output with its input hidden state."""

    def __init__(self, layer: Any) -> None:
        self.handle = layer.register_forward_hook(self.hook)

    def hook(self, _module: Any, inputs: Any, output: Any) -> Any:
        hidden = inputs[0]
        if torch.is_tensor(output):
            return hidden
        if isinstance(output, tuple):
            values = list(output)
            values[0] = hidden
            return tuple(values)
        if isinstance(output, list):
            values = list(output)
            values[0] = hidden
            return values
        raise TypeError(f"Unexpected L36 output: {type(output).__name__}")

    def remove(self) -> None:
        self.handle.remove()


def fresh_pair(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    first = capture_forward(model, input_ids, attention_mask)
    second = capture_forward(model, input_ids, attention_mask)
    return first, second


def evaluate_row(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = format_prompt(str(row["question"]))
    expected = extract_expected(str(row["answer"]))

    seed = args.seed_base + index
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    prompt_length = int(inputs["input_ids"].shape[-1])

    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            pad_token_id=tokenizer.eos_token_id,
            stop_strings=["\nQuestion:", "\nProblem:"],
            tokenizer=tokenizer,
        )
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - started

    generated_ids = generated[0, prompt_length:].detach().cpu()
    baseline_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    baseline_predicted = extract_predicted(baseline_text)

    full_ids = generated.detach()
    attention_mask = torch.ones_like(full_ids)

    first, second = fresh_pair(model, full_ids, attention_mask)

    positions = torch.arange(
        prompt_length - 1,
        full_ids.shape[-1] - 1,
        device="cuda:0",
    )
    positions_cpu = positions.cpu()
    next_ids = full_ids[0, positions + 1].detach().cpu()

    fresh_fresh = {
        "logits": logits_metrics(
            first["logits"][positions_cpu],
            second["logits"][positions_cpu],
            next_ids,
        ),
        "h35": abs_metrics(first["h35"][positions_cpu], second["h35"][positions_cpu]),
        "attention": abs_metrics(
            first["attention"][positions_cpu],
            second["attention"][positions_cpu],
        ),
        "mlp": abs_metrics(first["mlp"][positions_cpu], second["mlp"][positions_cpu]),
        "h36": abs_metrics(first["h36"][positions_cpu], second["h36"][positions_cpu]),
    }

    skip_control = WholeLayerSkip(model.model.layers[35])
    try:
        with torch.inference_mode():
            skipped = model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[0, positions].float().cpu()
    finally:
        skip_control.remove()

    oracle = first["logits"][positions_cpu]
    skip_metrics = logits_metrics(oracle, skipped, next_ids)

    del inputs, generated, full_ids, first, second
    torch.cuda.empty_cache()

    return {
        "question_index": index,
        "expected": expected,
        "baseline_predicted": baseline_predicted,
        "baseline_correct": baseline_predicted == expected,
        "generated_tokens": int(generated_ids.numel()),
        "generation_seconds": generation_seconds,
        "evaluated_positions": int(positions.numel()),
        "fresh_vs_fresh": fresh_fresh,
        "whole_layer36_skip": skip_metrics,
    }


def main() -> int:
    args = parse_args()
    if args.questions <= 0:
        raise ValueError("--questions must be > 0.")

    seed_all(args.seed_base)
    rows = load_rows(args.dataset, args.questions)
    args.output.mkdir(parents=True, exist_ok=True)

    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", args.model)
    print("Dataset:", args.dataset)
    print("Questions:", args.questions)
    print("Max new tokens:", args.max_new_tokens)
    print("\nLoading established 4-bit NF4 model...")
    tokenizer, model = build_model(args.model)

    if len(model.model.layers) < 36:
        raise RuntimeError("EXP-0009I requires at least 36 decoder layers.")

    results = []
    for index, row in enumerate(rows, 1):
        print(f"\n[{index}/{args.questions}] generate + fresh/fresh + L36 skip...")
        result = evaluate_row(model, tokenizer, row, index, args)
        results.append(result)

        ff = result["fresh_vs_fresh"]
        skip = result["whole_layer36_skip"]

        print(
            f"  expected={result['expected']} "
            f"predicted={result['baseline_predicted']} "
            f"correct={result['baseline_correct']} "
            f"tokens={result['generated_tokens']} "
            f"time={result['generation_seconds']:.2f}s"
        )
        print(
            f"  fresh/fresh logits: KL={ff['logits']['kl_oracle_to_candidate_mean']:.8f} "
            f"top1={ff['logits']['top1_agreement']:.6f}"
        )
        for name in ("h35", "attention", "mlp", "h36"):
            m = ff[name]
            print(
                f"  fresh/fresh {name}: "
                f"cos={m['cosine_mean']:.4f} "
                f"mean_abs={m['mean_abs']:.6f} "
                f"l2={m['mean_l2']:.6f}"
            )
        print(
            f"  whole L36 skip: KL={skip['kl_oracle_to_candidate_mean']:.4f} "
            f"top1={skip['top1_agreement']:.4f} "
            f"target_ratio={skip['target_probability_ratio_mean']:.4f}"
        )

    summary = {
        "schema_version": 1,
        "experiment": "EXP-0009I",
        "title": "Decomposition Stability and Whole-Layer-36 Skip Diagnostic",
        "status": "completed",
        "model_path": str(args.model),
        "dataset_path": str(args.dataset),
        "questions": len(results),
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "seed_base": args.seed_base,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        "baseline": {
            "accuracy": round(
                sum(r["baseline_correct"] for r in results) / len(results), 8
            ),
            "mean_generation_seconds": round(
                sum(r["generation_seconds"] for r in results) / len(results), 8
            ),
            "mean_generated_tokens": round(
                sum(r["generated_tokens"] for r in results) / len(results), 8
            ),
        },
        "results": results,
    }

    fresh_keys = ("h35", "attention", "mlp", "h36")
    fresh_summary = {}
    for key in fresh_keys:
        values = [r["fresh_vs_fresh"][key] for r in results]
        fresh_summary[key] = {
            metric: round(sum(v[metric] for v in values) / len(values), 8)
            for metric in ("max_abs", "mean_abs", "mean_l2", "cosine_mean")
        }
    logit_values = [r["fresh_vs_fresh"]["logits"] for r in results]
    skip_values = [r["whole_layer36_skip"] for r in results]
    summary["fresh_vs_fresh_mean"] = {
        "logits": {
            metric: round(sum(v[metric] for v in logit_values) / len(logit_values), 8)
            for metric in (
                "top1_agreement",
                "target_probability_ratio_mean",
                "kl_oracle_to_candidate_mean",
                "logit_l2_mean",
            )
        },
        **fresh_summary,
    }
    summary["whole_layer36_skip_mean"] = {
        metric: round(sum(v[metric] for v in skip_values) / len(skip_values), 8)
        for metric in (
            "top1_agreement",
            "target_probability_ratio_mean",
            "kl_oracle_to_candidate_mean",
            "logit_l2_mean",
        )
    }

    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (args.output / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result) + "\n")

    print("\n=== EXP-0009I summary ===")
    print(json.dumps({
        "fresh_vs_fresh_mean": summary["fresh_vs_fresh_mean"],
        "whole_layer36_skip_mean": summary["whole_layer36_skip_mean"],
    }, indent=2))
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
