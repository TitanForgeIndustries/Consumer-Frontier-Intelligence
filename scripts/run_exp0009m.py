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
    """Compare already-materialized CPU logits only."""
    a = a.detach().float().cpu().contiguous()
    b = b.detach().float().cpu().contiguous()
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise RuntimeError(
            f"Logit comparison shape mismatch: a={tuple(a.shape)} b={tuple(b.shape)}"
        )

    la = F.log_softmax(a, dim=-1)
    lb = F.log_softmax(b, dim=-1)
    pa = la.exp()

    return {
        "kl": float((pa * (la - lb)).sum(dim=-1).mean().item()),
        "top1": float(
            (a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean().item()
        ),
        "l2": float(
            torch.linalg.vector_norm(a - b, dim=-1).mean().item()
        ),
    }


def direct_final_selected(model: Any, hidden_cpu: torch.Tensor) -> torch.Tensor:
    """Run only selected hidden states through the existing output stack."""
    if hidden_cpu.ndim != 2:
        raise RuntimeError(
            f"Expected [positions, hidden] CPU tensor, got {tuple(hidden_cpu.shape)}"
        )

    norm = model.model.norm
    lm_head = model.lm_head

    x = hidden_cpu.to(
        device=norm.weight.device,
        dtype=norm.weight.dtype,
        non_blocking=False,
    )
    with torch.inference_mode():
        logits = lm_head(norm(x))
    torch.cuda.synchronize()
    return logits.detach().float().cpu().contiguous()
def direct_final(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    norm = model.model.norm
    lm_head = model.lm_head
    x = hidden.to(device=norm.weight.device, dtype=norm.weight.dtype)
    return lm_head(norm(x)).float()


class SameForwardBoundary:
    """Capture H35 and the exact L36 input/output in one forward."""

    def __init__(self, model: Any) -> None:
        self.layer35 = model.model.layers[34]
        self.layer36 = model.model.layers[35]
        self.handle35 = self.layer35.register_forward_hook(self.capture_h35)
        self.handle36 = self.layer36.register_forward_hook(self.capture_l36)

        self.h35: list[torch.Tensor] = []
        self.l36_input: list[torch.Tensor] = []
        self.l36_output: list[torch.Tensor] = []

    def capture_h35(self, _module: Any, _inputs: Any, output: Any) -> None:
        self.h35.append(component_tensor(output).detach().float().cpu())

    def capture_l36(
        self,
        _module: Any,
        inputs: Any,
        output: Any,
    ) -> Any:
        hidden = inputs[0]
        self.l36_input.append(hidden.detach().float().cpu())
        self.l36_output.append(component_tensor(output).detach().float().cpu())

        # Functional skip: preserve all auxiliary return values but replace
        # the primary hidden-state output with the exact L36 input.
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
        self.handle35.remove()
        self.handle36.remove()


def capture_oracle(
    model: Any,
    full_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        out = model(
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            use_cache=False,
        )
    torch.cuda.synchronize()
    return {
        "logits": out.logits.detach().float().cpu().contiguous()[0],
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
        torch.cuda.synchronize()

        full_ids = generated.detach()
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
        if positions.min().item() < 0 or positions.max().item() >= seq_len:
            raise RuntimeError(
                f"Invalid position range: min={positions.min().item()} "
                f"max={positions.max().item()} seq_len={seq_len}"
            )

        # Untouched oracle forward.
        oracle = capture_oracle(model, full_ids)

        # One separate forward captures H35 and the exact tensor entering L36,
        # then replaces the L36 output with that exact input.
        boundary = SameForwardBoundary(model)
        try:
            with torch.inference_mode():
                skipped = model(
                    input_ids=full_ids,
                    attention_mask=torch.ones_like(full_ids),
                    use_cache=False,
                )
            torch.cuda.synchronize()
        finally:
            boundary.remove()

        for name, values in (
            ("H35", boundary.h35),
            ("L36 input", boundary.l36_input),
            ("L36 output", boundary.l36_output),
        ):
            if len(values) != 1:
                raise RuntimeError(f"Expected one {name} capture, got {len(values)}.")

        h35 = boundary.h35[0][0].contiguous()
        l36_input = boundary.l36_input[0][0].contiguous()
        l36_output = boundary.l36_output[0][0].contiguous()
        skipped_logits = skipped.logits.detach().float().cpu().contiguous()[0]

        h35_sel = h35[positions]
        l36_input_sel = l36_input[positions]
        l36_output_sel = l36_output[positions]
        oracle_sel = oracle["logits"][positions]
        skipped_sel = skipped_logits[positions]

        if (
            h35.shape[0] != seq_len
            or l36_input.shape[0] != seq_len
            or l36_output.shape[0] != seq_len
            or skipped_logits.shape[0] != seq_len
            or oracle["logits"].shape[0] != seq_len
        ):
            raise RuntimeError(
                "Sequence-length mismatch: "
                f"full={seq_len}, h35={h35.shape[0]}, "
                f"l36_input={l36_input.shape[0]}, l36_output={l36_output.shape[0]}, "
                f"skip_logits={skipped_logits.shape[0]}, oracle={oracle['logits'].shape[0]}"
            )

        torch.cuda.synchronize()
        direct_l36_input = direct_final_selected(model, l36_input_sel)
        direct_h35 = direct_final_selected(model, h35_sel)

        result = {
            "question_index": index,
            "generated_tokens": int(seq_len - prompt_len),
            "evaluated_positions": int(positions.numel()),
            "h35_vs_actual_l36_input": metrics(h35_sel, l36_input_sel),
            "direct_l36_input_vs_skip": logits_metrics(
                direct_l36_input,
                skipped_sel,
            ),
            "direct_h35_vs_skip": logits_metrics(
                direct_h35,
                skipped_sel,
            ),
            "direct_l36_input_vs_oracle": logits_metrics(
                direct_l36_input,
                oracle_sel,
            ),
            "direct_h35_vs_oracle": logits_metrics(
                direct_h35,
                oracle_sel,
            ),
            "l36_transformation": metrics(
                l36_input_sel,
                l36_output_sel,
            ),
        }
        results.append(result)

        print(f"\n[{index}/{len(rows)}]")
        h = result["h35_vs_actual_l36_input"]
        print(
            f"  H35 vs actual L36 input: "
            f"mean_abs={h['mean_abs']:.8f} "
            f"l2={h['mean_l2']:.8f} "
            f"cos={h['cosine_mean']:.8f}"
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
        m = result["l36_transformation"]
        print(
            f"  L36 transformation: mean_abs={m['mean_abs']:.4f} "
            f"l2={m['mean_l2']:.4f} cos={m['cosine_mean']:.4f}"
        )

        del inputs, generated, full_ids, oracle, skipped
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
