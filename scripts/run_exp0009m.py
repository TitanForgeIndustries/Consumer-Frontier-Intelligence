"""CFI EXP-0009M: L36 boundary equivalence control.

EXP-0009I showed a mild behavioral effect when L36 output was replaced by its
input. EXP-0009L showed catastrophic behavior when stored H35 was passed through
the same final RMSNorm + LM head directly.

Those should agree if the captured H35 is exactly the L36 input. M tests that
equivalence on the same forward, explicitly capturing the L36 input and comparing:
1. captured L36 input vs H35 capture from layer 35
2. direct final output stack on the captured L36 input
3. same-forward functional L36 skip
4. oracle full-model output

This is a control experiment only.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_exp0008 import format_prompt, load_rows


DEFAULT_MODEL = Path(
    r"E:\Titan Forge Industries\CFI-Data\HuggingFace\hub\models--Qwen--Qwen3-4B-Base\snapshots\906bfd4b4dc7f14ee4320094d8b41684abff8539"
)
DEFAULT_DATASET = Path(
    r"E:\Titan Forge Industries\CFI-Data\Datasets\CFI-Eval-0001-GSM8K\gsm8k_test_100.jsonl"
)
DEFAULT_OUTPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009M-Layer-Boundary-Equivalence"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CFI EXP-0009M.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--questions", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=64)
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
        raise RuntimeError("CUDA is required for EXP-0009M.")
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


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = b.float() - a.float()
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "mean_l2": float(torch.linalg.vector_norm(diff, dim=-1).mean().item()),
        "cosine_mean": float(F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()),
    }


def logits_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    # Keep both operands on one device. The explicit CUDA->CPU conversion is
    # useful here because most captured reference tensors are stored on CPU.
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    la = F.log_softmax(a, dim=-1)
    lb = F.log_softmax(b, dim=-1)
    pa = la.exp()
    return {
        "kl": float((pa * (la - lb)).sum(dim=-1).mean().item()),
        "top1": float((a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean().item()),
        "l2": float(torch.linalg.vector_norm(a.float() - b.float(), dim=-1).mean().item()),
    }


def direct_final(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    norm = model.model.norm
    lm_head = model.lm_head
    x = hidden.to(device=norm.weight.device, dtype=norm.weight.dtype)
    return lm_head(norm(x)).float()


class SameForwardSkip:
    def __init__(self, layer: Any) -> None:
        self.handle = layer.register_forward_hook(self.hook)
        self.captured_input: list[torch.Tensor] = []
        self.captured_output: list[torch.Tensor] = []

    def hook(self, _module: Any, inputs: Any, output: Any) -> Any:
        hidden = inputs[0]
        self.captured_input.append(hidden.detach().float().cpu())
        self.captured_output.append(component_tensor(output).detach().float().cpu())

        if torch.is_tensor(output):
            return hidden.clone()
        if isinstance(output, tuple):
            values = list(output)
            values[0] = hidden.clone()
            return tuple(values)
        if isinstance(output, list):
            values = list(output)
            values[0] = hidden.clone()
            return values
        raise TypeError(f"Unexpected layer output: {type(output).__name__}")

    def remove(self) -> None:
        self.handle.remove()


def capture_once(
    model: Any,
    full_ids: torch.Tensor,
    capture_h35: bool,
) -> dict[str, torch.Tensor]:
    layer = model.model.layers[35]
    h35_values: list[torch.Tensor] = []

    def h35_hook(_m: Any, _i: Any, o: Any) -> None:
        h35_values.append(component_tensor(o).detach().float().cpu())

    handle = None
    if capture_h35:
        handle = model.model.layers[34].register_forward_hook(h35_hook)

    try:
        with torch.inference_mode():
            out = model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                use_cache=False,
            )
    finally:
        if handle is not None:
            handle.remove()

    result = {
        "logits": out.logits.detach().float().cpu()[0],
    }
    if capture_h35:
        result["h35"] = h35_values[0][0]
    del out
    return result


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
        torch.manual_seed(args.seed_base + index)
        torch.cuda.manual_seed_all(args.seed_base + index)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

        with torch.inference_mode():
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

        # Oracle + H35 capture.
        oracle = capture_once(model, full_ids, True)

        # Same-forward L36 skip. The hook captures the exact tensor entering
        # L36 before replacing the layer's output with that same tensor.
        skipper = SameForwardSkip(model.model.layers[35])
        try:
            with torch.inference_mode():
                skipped_out = model(
                    input_ids=full_ids,
                    attention_mask=torch.ones_like(full_ids),
                    use_cache=False,
                )
        finally:
            skipper.remove()

        if len(skipper.captured_input) != 1 or len(skipper.captured_output) != 1:
            raise RuntimeError("Expected one L36 input/output capture.")

        l36_input = skipper.captured_input[0][0]
        l36_output = skipper.captured_output[0][0]

        h35 = oracle["h35"]

        prompt_len = int(inputs["input_ids"].shape[-1])
        seq_len = int(full_ids.shape[-1])
        if seq_len <= prompt_len:
            raise RuntimeError(
                f"Generated sequence has no predicted positions: "
                f"prompt_len={prompt_len}, seq_len={seq_len}"
            )
        positions = torch.arange(
            prompt_len - 1,
            seq_len - 1,
            dtype=torch.long,
        )

        pos = positions.cpu()

        captured_h35 = h35[pos]
        captured_l36_input = l36_input[pos]

        direct = direct_final(model, captured_l36_input)
        direct_h35 = direct_final(model, captured_h35)
        skipped_logits = skipped_out.logits.detach().float().cpu()[0]

        result = {
            "question_index": index,
            "generated_tokens": int(full_ids.shape[-1] - prompt_len),
            "evaluated_positions": int(pos.numel()),
            "h35_vs_l36_input": metrics(captured_h35, captured_l36_input),
            "direct_l36_input_vs_skip": logits_metrics(
                direct[pos],
                skipped_logits[pos],
            ),
            "direct_h35_vs_skip": logits_metrics(
                direct_h35[pos],
                skipped_logits[pos],
            ),
            "direct_l36_input_vs_oracle": logits_metrics(
                direct[pos],
                oracle["logits"][pos],
            ),
            "direct_h35_vs_oracle": logits_metrics(
                direct_h35[pos],
                oracle["logits"][pos],
            ),
            "h36_change_from_l36": metrics(
                captured_l36_input,
                l36_output[pos],
            ),
        }
        results.append(result)

        print(f"\n[{index}/{len(rows)}]")
        print(
            f"  H35 vs actual L36 input: "
            f"mean_abs={result['h35_vs_l36_input']['mean_abs']:.8f} "
            f"l2={result['h35_vs_l36_input']['mean_l2']:.8f} "
            f"cos={result['h35_vs_l36_input']['cosine_mean']:.8f}"
        )
        for name in (
            "direct_l36_input_vs_skip",
            "direct_h35_vs_skip",
            "direct_l36_input_vs_oracle",
            "direct_h35_vs_oracle",
        ):
            m = result[name]
            print(
                f"  {name}: KL={m['kl']:.8f} top1={m['top1']:.6f} l2={m['l2']:.4f}"
            )
        print(
            f"  L36 transformation: mean_abs={result['h36_change_from_l36']['mean_abs']:.4f} "
            f"l2={result['h36_change_from_l36']['mean_l2']:.4f} "
            f"cos={result['h36_change_from_l36']['cosine_mean']:.4f}"
        )

        del inputs, generated, full_ids, oracle, skipper, skipped_out
        torch.cuda.empty_cache()

    summary = {
        "schema_version": 1,
        "experiment": "EXP-0009M",
        "title": "L36 Boundary Equivalence Control",
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
