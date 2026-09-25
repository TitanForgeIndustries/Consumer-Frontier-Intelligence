"""CFI EXP-0009K: direct behavioral exit head.

Tests a different replacement target from EXP-0009C-G:
rather than reconstructing H36, learn a compact low-rank head that maps H35
directly to the final token distribution.

The intended future use is to bypass the full L36 transition. Current evaluation
is behavioral only and does not claim a runtime speedup.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from cfi_paths import data_path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_exp0008 import extract_expected, extract_predicted, format_prompt, load_rows


DEFAULT_MODEL = data_path("HuggingFace", "hub", "models--Qwen--Qwen3-4B-Base", "snapshots", "906bfd4b4dc7f14ee4320094d8b41684abff8539")
DEFAULT_DATASET = data_path("Datasets", "CFI-Eval-0001-GSM8K", "gsm8k_test_100.jsonl")
DEFAULT_OUTPUT = data_path("Results", "CFI-Eval-0009K-Direct-Behavioral-Exit-Head")


@dataclass
class Trace:
    index: int
    expected: str | None
    baseline_predicted: str | None
    baseline_correct: bool
    prompt_length: int
    input_ids: torch.Tensor
    h35: torch.Tensor
    positions: torch.Tensor
    train_positions: torch.Tensor
    train_teacher_log_probs: torch.Tensor
    generated_tokens: int
    baseline_seconds: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CFI EXP-0009K.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--questions", type=int, default=4)
    p.add_argument("--train-questions", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed-base", type=int, default=42000)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-train-positions", type=int, default=32)
    p.add_argument("--bottleneck", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument(
        "--max-eval-tokens",
        type=int,
        default=0,
        help="Maximum held-out positions evaluated; 0 = all.",
    )
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
        raise RuntimeError("CUDA is required for EXP-0009K.")
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


def capture_teacher_trace(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    index: int,
    args: argparse.Namespace,
) -> Trace:
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
    elapsed = time.perf_counter() - started

    full_ids = generated.detach()
    generated_ids = generated[0, prompt_length:].detach().cpu()
    baseline_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    h35_capture: list[torch.Tensor] = []
    layer = model.model.layers[34]

    def capture_h35(_module: Any, _inputs: Any, output: Any) -> None:
        h35_capture.append(component_tensor(output).detach().float().cpu())

    hook = layer.register_forward_hook(capture_h35)
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                use_cache=False,
            )
    finally:
        hook.remove()

    if len(h35_capture) != 1:
        raise RuntimeError(f"Expected one H35 capture, got {len(h35_capture)}.")

    logits = outputs.logits[0].detach().float().cpu()
    positions = torch.arange(
        prompt_length - 1,
        full_ids.shape[-1] - 1,
        dtype=torch.long,
    )
    train_positions = positions
    if args.max_train_positions > 0 and train_positions.numel() > args.max_train_positions:
        keep = torch.linspace(
            0,
            train_positions.numel() - 1,
            steps=args.max_train_positions,
        ).round().long()
        train_positions = train_positions[keep]

    teacher_log_probs = F.log_softmax(
        logits[train_positions],
        dim=-1,
    ).half()

    input_ids = full_ids[0].detach().cpu()
    predicted = extract_predicted(baseline_text)

    trace = Trace(
        index=index,
        expected=expected,
        baseline_predicted=predicted,
        baseline_correct=predicted == expected,
        prompt_length=prompt_length,
        input_ids=input_ids,
        h35=h35_capture[0][0],
        positions=positions,
        train_positions=train_positions,
        train_teacher_log_probs=teacher_log_probs,
        generated_tokens=int(generated_ids.numel()),
        baseline_seconds=elapsed,
    )

    del inputs, generated, full_ids, outputs, logits
    torch.cuda.empty_cache()
    return trace


class DirectExitHead(nn.Module):
    """Compact H35 -> final-vocabulary distribution predictor."""

    def __init__(self, hidden_size: int, vocab_size: int, bottleneck: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(self.norm(hidden))))


def behavior_kl(
    candidate_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
) -> torch.Tensor:
    candidate_log_probs = F.log_softmax(candidate_logits.float(), dim=-1)
    teacher_log_probs = teacher_log_probs.to(
        device=candidate_log_probs.device,
        dtype=candidate_log_probs.dtype,
    )
    teacher_probs = teacher_log_probs.exp()
    return (
        teacher_probs * (teacher_log_probs - candidate_log_probs)
    ).sum(dim=-1).mean()


def train_exit_head(
    model: Any,
    traces: list[Trace],
    args: argparse.Namespace,
    device: torch.device,
    vocab_size: int,
) -> tuple[DirectExitHead, dict[str, Any]]:
    hidden_size = traces[0].h35.shape[-1]
    head = DirectExitHead(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        bottleneck=args.bottleneck,
    ).to(device=device, dtype=torch.float32)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0

        for trace in traces:
            positions = trace.train_positions
            source = trace.h35[positions].to(device)
            teacher_log_probs = trace.train_teacher_log_probs.to(device)

            predicted_logits = head(source)
            loss = behavior_kl(predicted_logits, teacher_log_probs)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())

        mean_loss = total / max(len(traces), 1)
        history.append({"epoch": epoch, "mean_loss": mean_loss})
        print(f"    epoch {epoch:02d}/{args.epochs}: mean_loss={mean_loss:.6f}")

    return head, {
        "training_seconds": time.perf_counter() - started,
        "epochs": args.epochs,
        "max_train_positions": args.max_train_positions,
        "bottleneck": args.bottleneck,
        "parameter_count": sum(p.numel() for p in head.parameters()),
        "history": history,
    }


def distribution_metrics(
    oracle: torch.Tensor,
    candidate: torch.Tensor,
    target_ids: torch.Tensor,
) -> dict[str, float]:
    oracle = oracle.float()
    candidate = candidate.float()
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
        "target_log_probability_delta_mean": float(
            (
                candidate_logp.gather(-1, target).squeeze(-1)
                - oracle_logp.gather(-1, target).squeeze(-1)
            ).mean().item()
        ),
        "kl_oracle_to_candidate_mean": float(kl.mean().item()),
        "logit_l2_mean": float(
            torch.linalg.vector_norm(oracle - candidate, dim=-1).mean().item()
        ),
    }


class WholeLayerSkip:
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
        raise TypeError(f"Unexpected layer output: {type(output).__name__}")

    def remove(self) -> None:
        self.handle.remove()


def evaluate(
    model: Any,
    head: DirectExitHead,
    trace: Trace,
    args: argparse.Namespace,
) -> dict[str, Any]:
    positions_idx = torch.arange(trace.positions.numel())
    if args.max_eval_tokens > 0 and positions_idx.numel() > args.max_eval_tokens:
        keep = torch.linspace(
            0,
            positions_idx.numel() - 1,
            steps=args.max_eval_tokens,
        ).round().long()
        positions_idx = positions_idx[keep]

    positions = trace.positions[positions_idx]
    next_ids = trace.input_ids[positions + 1]

    input_ids = trace.input_ids.unsqueeze(0).to("cuda:0")
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        oracle = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits[0, positions.to("cuda:0")].float().cpu()

    skip = WholeLayerSkip(model.model.layers[35])
    try:
        with torch.inference_mode():
            skipped = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[0, positions.to("cuda:0")].float().cpu()
    finally:
        skip.remove()

    head.eval()
    with torch.inference_mode():
        predicted = head(trace.h35[positions_idx].to("cuda:0")).float().cpu()

    oracle_m = distribution_metrics(oracle, predicted, next_ids)
    skip_m = distribution_metrics(oracle, skipped, next_ids)

    return {
        "question_index": trace.index,
        "evaluated_positions": int(positions.numel()),
        "direct_exit_behavior": oracle_m,
        "whole_layer36_skip_behavior": skip_m,
    }


def main() -> int:
    args = parse_args()
    if not 1 <= args.train_questions < args.questions:
        raise ValueError("--train-questions must be >=1 and < --questions.")

    seed_all(args.seed_base)
    rows = load_rows(args.dataset, args.questions)
    args.output.mkdir(parents=True, exist_ok=True)

    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", args.model)
    print("Dataset:", args.dataset)
    print("Questions:", args.questions)
    print("Train questions:", args.train_questions)
    print("Bottleneck:", args.bottleneck)

    print("\nLoading established 4-bit NF4 model...")
    tokenizer, model = build_model(args.model)
    vocab_size = int(model.config.vocab_size)

    traces = []
    for index, row in enumerate(rows, 1):
        print(f"\n[{index}/{args.questions}] baseline + H35/teacher capture...")
        trace = capture_teacher_trace(model, tokenizer, row, index, args)
        traces.append(trace)
        print(
            f"  expected={trace.expected} "
            f"predicted={trace.baseline_predicted} "
            f"correct={trace.baseline_correct} "
            f"tokens={trace.generated_tokens} "
            f"time={trace.baseline_seconds:.2f}s"
        )

    train_traces = traces[: args.train_questions]
    eval_traces = traces[args.train_questions :]

    print("\n=== train direct behavioral exit head ===")
    head, training = train_exit_head(
        model,
        train_traces,
        args,
        torch.device("cuda:0"),
        vocab_size,
    )

    results = []
    for trace in eval_traces:
        print(f"  evaluating held-out question {trace.index}...")
        result = evaluate(model, head, trace, args)
        results.append(result)

        d = result["direct_exit_behavior"]
        s = result["whole_layer36_skip_behavior"]
        print(
            f"    direct_exit: KL={d['kl_oracle_to_candidate_mean']:.4f} "
            f"top1={d['top1_agreement']:.4f} "
            f"target_ratio={d['target_probability_ratio_mean']:.4f}"
        )
        print(
            f"    whole_L36_skip: KL={s['kl_oracle_to_candidate_mean']:.4f} "
            f"top1={s['top1_agreement']:.4f} "
            f"target_ratio={s['target_probability_ratio_mean']:.4f}"
        )

    def mean_metrics(key: str) -> dict[str, float]:
        subset = [r[key] for r in results]
        return {
            metric: round(sum(r[metric] for r in subset) / len(subset), 8)
            for metric in (
                "top1_agreement",
                "target_probability_ratio_mean",
                "target_log_probability_delta_mean",
                "kl_oracle_to_candidate_mean",
                "logit_l2_mean",
            )
        }

    summary = {
        "schema_version": 1,
        "experiment": "EXP-0009K",
        "title": "Direct Behavioral Exit Head",
        "status": "completed",
        "model_path": str(args.model),
        "dataset_path": str(args.dataset),
        "questions": len(traces),
        "train_questions": len(train_traces),
        "eval_questions": len(eval_traces),
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "seed_base": args.seed_base,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_train_positions": args.max_train_positions,
            "bottleneck": args.bottleneck,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
        },
        "baseline": {
            "accuracy": round(
                sum(t.baseline_correct for t in traces) / len(traces),
                8,
            ),
            "mean_generation_seconds": round(
                sum(t.baseline_seconds for t in traces) / len(traces),
                8,
            ),
            "mean_generated_tokens": round(
                sum(t.generated_tokens for t in traces) / len(traces),
                8,
            ),
        },
        "training": training,
        "direct_exit_mean": mean_metrics("direct_exit_behavior") if results else {},
        "whole_layer36_skip_mean": mean_metrics("whole_layer36_skip_behavior") if results else {},
        "results": results,
    }

    with (args.output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (args.output / "results.jsonl").open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print("\n=== EXP-0009K summary ===")
    print(json.dumps({
        "direct_exit_mean": summary["direct_exit_mean"],
        "whole_layer36_skip_mean": summary["whole_layer36_skip_mean"],
        "training": training,
    }, indent=2))
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
