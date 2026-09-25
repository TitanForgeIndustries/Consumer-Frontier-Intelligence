"""CFI EXP-0009J: capture-path control.

Isolate the EXP-0009H stored-vs-fresh discrepancy by comparing identical inputs
under the two Transformers forward configurations:
- output_hidden_states=False
- output_hidden_states=True

Also compare the two configurations against each other.

This is an implementation/control experiment only.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from cfi_paths import data_path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_exp0008 import format_prompt, load_rows


DEFAULT_MODEL = data_path("HuggingFace", "hub", "models--Qwen--Qwen3-4B-Base", "snapshots", "906bfd4b4dc7f14ee4320094d8b41684abff8539")
DEFAULT_DATASET = data_path("Datasets", "CFI-Eval-0001-GSM8K", "gsm8k_test_100.jsonl")
DEFAULT_OUTPUT = data_path("Results", "CFI-Eval-0009J-Capture-Path-Control")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CFI EXP-0009J.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--questions", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed-base", type=int, default=42000)
    return p.parse_args()


def component_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Expected tensor-like output, got {type(output).__name__}")


def build_model(model_path: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-0009J.")
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


def capture(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    output_hidden_states: bool,
) -> dict[str, torch.Tensor]:
    h35: list[torch.Tensor] = []
    attn: list[torch.Tensor] = []
    mlp: list[torch.Tensor] = []
    h36: list[torch.Tensor] = []

    layer = model.model.layers[35]

    def cap_h35(_m: Any, _i: Any, o: Any) -> None:
        h35.append(component_tensor(o).detach().float().cpu())

    def cap_attn(_m: Any, _i: Any, o: Any) -> None:
        attn.append(component_tensor(o).detach().float().cpu())

    def cap_mlp(_m: Any, _i: Any, o: Any) -> None:
        mlp.append(component_tensor(o).detach().float().cpu())

    def cap_h36(_m: Any, _i: Any, o: Any) -> None:
        h36.append(component_tensor(o).detach().float().cpu())

    hooks = [
        model.model.layers[34].register_forward_hook(cap_h35),
        layer.self_attn.register_forward_hook(cap_attn),
        layer.mlp.register_forward_hook(cap_mlp),
        layer.register_forward_hook(cap_h36),
    ]
    try:
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=output_hidden_states,
            )
    finally:
        for h in hooks:
            h.remove()

    return {
        "logits": out.logits.detach().float().cpu()[0],
        "h35": h35[0][0],
        "attention": attn[0][0],
        "mlp": mlp[0][0],
        "h36": h36[0][0],
    }


def tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    d = b.float() - a.float()
    return {
        "max_abs": float(d.abs().max().item()),
        "mean_abs": float(d.abs().mean().item()),
        "mean_l2": float(torch.linalg.vector_norm(d, dim=-1).mean().item()),
        "cosine_mean": float(F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()),
    }


def logits_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    la = F.log_softmax(a.float(), dim=-1)
    lb = F.log_softmax(b.float(), dim=-1)
    pa = la.exp()
    return {
        "kl": float((pa * (la - lb)).sum(dim=-1).mean().item()),
        "top1": float((a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean().item()),
        "logit_l2": float(torch.linalg.vector_norm(a.float() - b.float(), dim=-1).mean().item()),
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed_base)
    torch.manual_seed(args.seed_base)
    torch.cuda.manual_seed_all(args.seed_base)

    rows = load_rows(args.dataset, args.questions)
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading established 4-bit NF4 model...")
    tokenizer, model = build_model(args.model)

    results = []
    for index, row in enumerate(rows, 1):
        prompt = format_prompt(str(row["question"]))
        seed = args.seed_base + index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
            stop_strings=["\nQuestion:", "\nProblem:"],
            tokenizer=tokenizer,
        )
        full_ids = generated.detach()
        mask = torch.ones_like(full_ids)

        false_a = capture(model, full_ids, mask, False)
        false_b = capture(model, full_ids, mask, False)
        true_a = capture(model, full_ids, mask, True)
        true_b = capture(model, full_ids, mask, True)

        prompt_len = int(inputs["input_ids"].shape[-1])
        positions = torch.arange(prompt_len - 1, full_ids.shape[-1] - 1)
        p = positions

        comparisons = {}
        for name in ("h35", "attention", "mlp", "h36"):
            comparisons[f"{name}_false_false"] = tensor_metrics(
                false_a[name][p], false_b[name][p]
            )
            comparisons[f"{name}_true_true"] = tensor_metrics(
                true_a[name][p], true_b[name][p]
            )
            comparisons[f"{name}_true_vs_false"] = tensor_metrics(
                true_a[name][p], false_a[name][p]
            )
        comparisons["logits_false_false"] = logits_metrics(false_a["logits"][p], false_b["logits"][p])
        comparisons["logits_true_true"] = logits_metrics(true_a["logits"][p], true_b["logits"][p])
        comparisons["logits_true_vs_false"] = logits_metrics(true_a["logits"][p], false_a["logits"][p])

        result = {
            "question_index": index,
            "generated_tokens": int(full_ids.shape[-1] - prompt_len),
            "evaluated_positions": int(p.numel()),
            "comparisons": comparisons,
        }
        results.append(result)

        print(f"\n[{index}/{len(rows)}]")
        for name in ("h35", "attention", "mlp", "h36"):
            ff = comparisons[f"{name}_false_false"]
            tt = comparisons[f"{name}_true_true"]
            tf = comparisons[f"{name}_true_vs_false"]
            print(
                f"  {name}: false/false mean_abs={ff['mean_abs']:.6f} "
                f"true/true={tt['mean_abs']:.6f} true-vs-false={tf['mean_abs']:.6f}"
            )
        print(
            f"  logits: false/false KL={comparisons['logits_false_false']['kl']:.8f} "
            f"true/true KL={comparisons['logits_true_true']['kl']:.8f} "
            f"true-vs-false KL={comparisons['logits_true_vs_false']['kl']:.8f}"
        )

        del inputs, generated, full_ids, mask, false_a, false_b, true_a, true_b
        torch.cuda.empty_cache()

    summary = {
        "schema_version": 1,
        "experiment": "EXP-0009J",
        "title": "Capture-Path Control",
        "status": "completed",
        "questions": len(results),
        "results": results,
    }

    with (args.output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
