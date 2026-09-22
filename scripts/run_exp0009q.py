"""CFI EXP-0009Q: cache-safe L36 MLP reconstruction.

EXP-0009P showed that replacing the entire final decoder layer can improve
teacher-forced behavior while failing under free-running generation.

Q isolates the MLP contribution while leaving L36 self-attention untouched.
This preserves the real L36 KV cache semantics on every step.

The base Qwen3-4B-Base model remains frozen. A compact predictor is trained to
replace only the L36 MLP output.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from run_exp0008 import (
    extract_expected,
    extract_predicted,
    format_prompt,
    load_rows,
)
from run_exp0009n import base_exit_logits, build_model, seed_all


DEFAULT_MODEL = Path(
    r"E:\Titan Forge Industries\CFI-Data\HuggingFace\hub\models--Qwen--Qwen3-4B-Base\snapshots\906bfd4b4dc7f14ee4320094d8b41684abff8539"
)
DEFAULT_DATASET = Path(
    r"E:\Titan Forge Industries\CFI-Data\Datasets\CFI-Eval-0001-GSM8K\gsm8k_test_100.jsonl"
)
DEFAULT_OUTPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009Q-Cache-Safe-MLP-Reconstruction"
)


@dataclass
class MLPTrace:
    index: int
    expected: str | None
    baseline_predicted: str | None
    baseline_correct: bool
    prompt_length: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    train_positions: torch.Tensor
    mlp_residual: torch.Tensor
    mlp_input: torch.Tensor
    mlp_output: torch.Tensor
    teacher_log_probs: torch.Tensor
    generated_tokens: int
    baseline_seconds: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CFI EXP-0009Q.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--questions", type=int, default=10)
    p.add_argument("--train-questions", type=int, default=7)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed-base", type=int, default=42000)
    p.add_argument("--max-train-positions", type=int, default=64)
    p.add_argument("--bottleneck", type=int, default=128)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--mse-weight", type=float, default=0.05)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--warmup-tokens", type=int, default=16)
    return p.parse_args()


def capture_trace(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    index: int,
    args: argparse.Namespace,
) -> MLPTrace:
    prompt = format_prompt(str(row["question"]))
    expected = extract_expected(str(row["answer"]))

    seed = args.seed_base + index
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    prompt_length = int(inputs["input_ids"].shape[-1])

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    generated_ids = generated[0, prompt_length:].detach().cpu()
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    full_ids = generated.detach()

    residual_capture: list[torch.Tensor] = []
    mlp_input_capture: list[torch.Tensor] = []
    mlp_output_capture: list[torch.Tensor] = []

    layer = model.model.layers[35]

    def capture_residual(_module: Any, args_tuple: tuple[Any, ...]) -> None:
        if not args_tuple or not torch.is_tensor(args_tuple[0]):
            raise RuntimeError("L36 post-attention layernorm input was not a tensor.")
        residual_capture.append(args_tuple[0].detach().float().cpu())

    def capture_mlp_input(_module: Any, args_tuple: tuple[Any, ...]) -> None:
        if not args_tuple or not torch.is_tensor(args_tuple[0]):
            raise RuntimeError("L36 MLP input was not a tensor.")
        mlp_input_capture.append(args_tuple[0].detach().float().cpu())

    def capture_mlp_output(
        _module: Any,
        _inputs: Any,
        output: Any,
    ) -> None:
        if not torch.is_tensor(output):
            raise RuntimeError(
                f"Expected tensor L36 MLP output, got {type(output).__name__}."
            )
        mlp_output_capture.append(output.detach().float().cpu())

    h_residual = layer.post_attention_layernorm.register_forward_pre_hook(
        capture_residual
    )
    h_mlp_input = layer.mlp.register_forward_pre_hook(capture_mlp_input)
    h_mlp_output = layer.mlp.register_forward_hook(capture_mlp_output)

    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                use_cache=False,
            )
    finally:
        h_residual.remove()
        h_mlp_input.remove()
        h_mlp_output.remove()

    if not residual_capture or not mlp_input_capture or not mlp_output_capture:
        raise RuntimeError("Missing one or more L36 MLP captures.")

    if not (
        len(residual_capture) == 1
        and len(mlp_input_capture) == 1
        and len(mlp_output_capture) == 1
    ):
        raise RuntimeError(
            "Expected one full-sequence MLP capture for each component."
        )

    logits = outputs.logits[0].detach().float().cpu()
    positions = torch.arange(
        prompt_length - 1,
        full_ids.shape[-1] - 1,
        dtype=torch.long,
    )
    train_positions = positions
    if (
        args.max_train_positions > 0
        and train_positions.numel() > args.max_train_positions
    ):
        keep = torch.linspace(
            0,
            train_positions.numel() - 1,
            steps=args.max_train_positions,
        ).round().long()
        train_positions = train_positions[keep]

    teacher_log_probs = F.log_softmax(
        logits[train_positions],
        dim=-1,
    ).float()

    return MLPTrace(
        index=index,
        expected=expected,
        baseline_predicted=extract_predicted(generated_text),
        baseline_correct=extract_predicted(generated_text) == expected,
        prompt_length=prompt_length,
        input_ids=full_ids[0].detach().cpu(),
        positions=positions,
        train_positions=train_positions,
        mlp_residual=residual_capture[0][0],
        mlp_input=mlp_input_capture[0][0],
        mlp_output=mlp_output_capture[0][0],
        teacher_log_probs=teacher_log_probs,
        generated_tokens=int(generated_ids.numel()),
        baseline_seconds=elapsed,
    )


class MLPTransition(nn.Module):
    """Predict the L36 MLP residual from its normalized input."""

    def __init__(
        self,
        hidden_size: int,
        bottleneck: int,
    ) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, mlp_input: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(mlp_input)))


def train_mlp_transition(
    model: Any,
    traces: list[MLPTrace],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[MLPTransition, dict[str, Any]]:
    hidden_size = traces[0].mlp_input.shape[-1]
    transition = MLPTransition(
        hidden_size=hidden_size,
        bottleneck=args.bottleneck,
    ).to(device=device, dtype=torch.float32)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        transition.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        transition.train()
        total = 0.0

        for trace in traces:
            positions = trace.train_positions

            source = trace.mlp_input[positions].to(
                device=device,
                dtype=torch.float32,
            )
            residual = trace.mlp_residual[positions].to(
                device=device,
                dtype=torch.float32,
            )
            teacher_output = trace.teacher_log_probs.to(device)

            predicted_mlp = transition(source)
            predicted_state = residual + predicted_mlp
            predicted_logits = base_exit_logits(
                model,
                predicted_state,
            )

            selected_teacher = teacher_output
            student_logp = F.log_softmax(predicted_logits, dim=-1)
            teacher_probs = selected_teacher.exp()

            kl_loss = (
                teacher_probs * (selected_teacher - student_logp)
            ).sum(dim=-1).mean()

            target_mlp = trace.mlp_output[positions].to(
                device=device,
                dtype=torch.float32,
            )
            target_rms = (
                target_mlp.square().mean(dim=-1).sqrt().clamp_min(1e-6)
            )
            mse_loss = (
                (predicted_mlp - target_mlp).square().mean(dim=-1)
                / target_rms.square()
            ).mean()

            loss = kl_loss + args.mse_weight * mse_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transition.parameters(), 1.0)
            optimizer.step()

            total += float(loss.item())

        mean_loss = total / max(len(traces), 1)
        history.append(
            {
                "epoch": epoch,
                "mean_loss": mean_loss,
            }
        )
        print(
            f"    epoch {epoch:02d}/{args.epochs}: "
            f"mean_loss={mean_loss:.6f}"
        )

    return transition, {
        "training_seconds": time.perf_counter() - started,
        "epochs": args.epochs,
        "bottleneck": args.bottleneck,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "mse_weight": args.mse_weight,
        "parameter_count": sum(p.numel() for p in transition.parameters()),
        "history": history,
    }


def compare_tokens(
    full_tokens: list[int],
    candidate_tokens: list[int],
) -> dict[str, Any]:
    shared = min(len(full_tokens), len(candidate_tokens))
    matching = sum(
        1
        for idx in range(shared)
        if full_tokens[idx] == candidate_tokens[idx]
    )
    first_divergence = None
    for idx in range(shared):
        if full_tokens[idx] != candidate_tokens[idx]:
            first_divergence = idx + 1
            break
    return {
        "full_token_count": len(full_tokens),
        "candidate_token_count": len(candidate_tokens),
        "shared_token_count": shared,
        "exact_token_agreement": matching / max(shared, 1),
        "matching_tokens": matching,
        "first_divergence_generated_token": first_divergence,
    }


class MLPReplacement:
    """Temporarily replace only L36 MLP; L36 attention and KV cache remain real."""

    def __init__(
        self,
        model: Any,
        replacement: Any,
    ) -> None:
        self.layer = model.model.layers[35]
        self.mlp = self.layer.mlp
        self.original_forward = self.mlp.forward
        self.replacement = replacement

        def replacement_forward(
            _module: Any,
            hidden: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            del args, kwargs
            runtime_dtype = next(
                self.replacement.parameters()
            ).dtype
            output = self.replacement(
                hidden.to(dtype=runtime_dtype)
            )
            return output.to(dtype=hidden.dtype)

        self.mlp.forward = replacement_forward.__get__(
            self.mlp,
            type(self.mlp),
        )

    def remove(self) -> None:
        self.mlp.forward = self.original_forward


def benchmark_mode(
    model: Any,
    tokenizer: Any,
    trace: MLPTrace,
    args: argparse.Namespace,
    mode: str,
    transition: MLPTransition | None,
) -> dict[str, Any]:
    replacement: MLPReplacement | None = None

    if mode == "mlp_skip":
        class ZeroMLP(nn.Module):
            def forward(self, hidden: torch.Tensor) -> torch.Tensor:
                return torch.zeros_like(hidden)
        replacement = MLPReplacement(
            model,
            ZeroMLP().to("cuda:0"),
        )
    elif mode == "mlp_predicted":
        if transition is None:
            raise RuntimeError("mlp_predicted requires transition.")
        replacement = MLPReplacement(model, transition)
    elif mode != "full":
        raise ValueError(f"Unsupported mode: {mode}")

    try:
        input_ids = trace.input_ids[: trace.prompt_length].unsqueeze(0).to(
            "cuda:0"
        )
        attention_mask = torch.ones_like(input_ids)

        warmup_kwargs = {
            "max_new_tokens": args.warmup_tokens,
            "min_new_tokens": args.warmup_tokens,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": tokenizer.eos_token_id,
        }

        with torch.inference_mode():
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **warmup_kwargs,
            )
        torch.cuda.synchronize()

        runs: list[dict[str, Any]] = []
        reference_tokens: torch.Tensor | None = None

        for repeat in range(1, args.repeats + 1):
            torch.cuda.reset_peak_memory_stats(device="cuda:0")
            torch.cuda.synchronize()
            started = time.perf_counter()

            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

            generated_tokens = generated[
                0,
                trace.prompt_length:,
            ].detach().cpu()

            if reference_tokens is None:
                reference_tokens = generated_tokens.clone()
            elif not torch.equal(reference_tokens, generated_tokens):
                raise RuntimeError(
                    f"{mode} was not deterministic across repeats for "
                    f"question {trace.index}."
                )

            runs.append(
                {
                    "repeat": repeat,
                    "seconds": elapsed,
                    "generated_tokens": int(generated_tokens.numel()),
                    "tokens_per_second": int(generated_tokens.numel())
                    / max(elapsed, 1e-12),
                    "peak_memory_bytes": int(
                        torch.cuda.max_memory_allocated(device="cuda:0")
                    ),
                }
            )

        assert reference_tokens is not None
        text_output = tokenizer.decode(
            reference_tokens,
            skip_special_tokens=True,
        ).strip()

        return {
            "mode": mode,
            "generated_tokens": int(reference_tokens.numel()),
            "text": text_output,
            "predicted": extract_predicted(text_output),
            "runs": runs,
            "mean_seconds": sum(
                item["seconds"] for item in runs
            ) / len(runs),
            "mean_tokens_per_second": sum(
                item["tokens_per_second"] for item in runs
            ) / len(runs),
            "mean_peak_memory_bytes": sum(
                item["peak_memory_bytes"] for item in runs
            ) / len(runs),
            "tokens": reference_tokens.tolist(),
        }
    finally:
        if replacement is not None:
            replacement.remove()


def teacher_forced_eval(
    model: Any,
    transition: MLPTransition,
    trace: MLPTrace,
) -> dict[str, Any]:
    positions = trace.positions
    target_ids = trace.input_ids[positions + 1]

    source = trace.mlp_input[positions]
    residual = trace.mlp_residual[positions]

    with torch.inference_mode():
        oracle = model(
            input_ids=trace.input_ids.unsqueeze(0).to("cuda:0"),
            attention_mask=torch.ones(
                (1, trace.input_ids.shape[0]),
                dtype=torch.long,
                device="cuda:0",
            ),
            use_cache=False,
        ).logits[0, positions.to("cuda:0")].float()

        base_hidden = residual.to("cuda:0", dtype=torch.float32)
        base_mlp = torch.zeros_like(base_hidden)

        base_logits = base_exit_logits(
            model,
            base_hidden + base_mlp,
        )

        predicted_mlp = transition(
            source.to("cuda:0", dtype=torch.float32)
        )
        predicted_logits = base_exit_logits(
            model,
            base_hidden + predicted_mlp,
        )

    return {
        "question_index": trace.index,
        "mlp_zero": distribution_metrics(
            oracle,
            base_logits,
            target_ids.to("cuda:0"),
        ),
        "mlp_predicted": distribution_metrics(
            oracle,
            predicted_logits,
            target_ids.to("cuda:0"),
        ),
    }


def mean_metric(
    results: list[dict[str, Any]],
    mode: str,
    metric: str,
) -> float:
    return sum(
        row[mode][metric]
        for row in results
    ) / len(results)


def main() -> int:
    args = parse_args()

    if not 1 <= args.train_questions < args.questions:
        raise ValueError("--train-questions must be >=1 and < --questions.")
    if args.repeats < 1:
        raise ValueError("--repeats must be >=1.")

    seed_all(args.seed_base)
    rows = load_rows(args.dataset, args.questions)
    args.output.mkdir(parents=True, exist_ok=True)

    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", args.model)
    print("Dataset:", args.dataset)
    print("Questions:", len(rows))
    print("Train questions:", args.train_questions)
    print("Decoding: greedy deterministic")

    print("\nLoading established 4-bit NF4 model...")
    tokenizer, model = build_model(args.model)

    traces: list[MLPTrace] = []
    for index, row in enumerate(rows, 1):
        print(f"\n[{index}/{len(rows)}] greedy MLP trace...")
        trace = capture_trace(model, tokenizer, row, index, args)
        traces.append(trace)
        print(
            f"  expected={trace.expected} "
            f"predicted={trace.baseline_predicted} "
            f"correct={trace.baseline_correct} "
            f"tokens={trace.generated_tokens} "
            f"time={trace.baseline_seconds:.2f}s"
        )

    train_traces = traces[: args.train_questions]
    eval_traces = traces[args.train_questions:]

    print("\n=== train L36 MLP transition ===")
    transition, training = train_mlp_transition(
        model,
        train_traces,
        args,
        torch.device("cuda:0"),
    )

    transition_runtime = copy.deepcopy(transition).to(
        device="cuda:0",
        dtype=next(model.model.norm.parameters()).dtype,
    )
    transition_runtime.eval()

    adapter_path = args.output / "mlp_transition_state_dict_fp32.pt"
    torch.save(
        {
            "experiment": "EXP-0009Q",
            "base_model_frozen": True,
            "state_dict": {
                name: value.detach().cpu()
                for name, value in transition.state_dict().items()
            },
            "parameter_count": sum(
                p.numel() for p in transition.parameters()
            ),
            "hidden_size": train_traces[0].mlp_input.shape[-1],
            "bottleneck": args.bottleneck,
        },
        adapter_path,
    )

    teacher_results: list[dict[str, Any]] = []
    runtime_results: list[dict[str, Any]] = []

    for trace in eval_traces:
        print(f"\n=== Q held-out question {trace.index} ===")

        teacher = teacher_forced_eval(
            model,
            transition_runtime,
            trace,
        )
        teacher_results.append(teacher)

        z = teacher["mlp_zero"]
        p = teacher["mlp_predicted"]
        print(
            f"  teacher-forced MLP zero: "
            f"KL={z['kl_oracle_to_candidate_mean']:.4f} "
            f"top1={z['top1_agreement']:.4f}"
        )
        print(
            f"  teacher-forced MLP predicted: "
            f"KL={p['kl_oracle_to_candidate_mean']:.4f} "
            f"top1={p['top1_agreement']:.4f}"
        )

        full = benchmark_mode(
            model,
            tokenizer,
            trace,
            args,
            "full",
            transition=None,
        )
        skipped = benchmark_mode(
            model,
            tokenizer,
            trace,
            args,
            "mlp_skip",
            transition=None,
        )
        predicted = benchmark_mode(
            model,
            tokenizer,
            trace,
            args,
            "mlp_predicted",
            transition=transition_runtime,
        )

        runtime = {
            "question_index": trace.index,
            "expected": trace.expected,
            "baseline_predicted": trace.baseline_predicted,
            "baseline_correct": trace.baseline_correct,
            "full": full,
            "mlp_skip": skipped,
            "mlp_predicted": predicted,
            "mlp_skip_vs_full": compare_tokens(
                full["tokens"],
                skipped["tokens"],
            ),
            "mlp_predicted_vs_full": compare_tokens(
                full["tokens"],
                predicted["tokens"],
            ),
            "mlp_skip_speedup": full["mean_seconds"]
            / max(skipped["mean_seconds"], 1e-12),
            "mlp_predicted_speedup": full["mean_seconds"]
            / max(predicted["mean_seconds"], 1e-12),
        }
        runtime_results.append(runtime)

        print(
            f"  full:         {full['mean_seconds']:.3f}s "
            f"({full['mean_tokens_per_second']:.2f} tok/s)"
        )
        print(
            f"  MLP skip:     {skipped['mean_seconds']:.3f}s "
            f"speedup={runtime['mlp_skip_speedup']:.3f}x "
            f"agreement={runtime['mlp_skip_vs_full']['exact_token_agreement']:.3f}"
        )
        print(
            f"  MLP predicted:{predicted['mean_seconds']:.3f}s "
            f"speedup={runtime['mlp_predicted_speedup']:.3f}x "
            f"agreement={runtime['mlp_predicted_vs_full']['exact_token_agreement']:.3f}"
        )

    summary = {
        "schema_version": 1,
        "experiment": "EXP-0009Q",
        "title": "Cache-Safe L36 MLP Reconstruction",
        "status": "completed",
        "hypothesis": (
            "Replacing only L36 MLP while retaining exact L36 attention "
            "and KV-cache updates can preserve autoregressive behavior more "
            "reliably than replacing the whole layer."
        ),
        "model": {
            "path": str(args.model),
            "base_model_frozen": True,
            "num_layers": 36,
            "target_layer": 36,
            "replaced_component": "mlp",
            "transition_parameters": sum(
                p.numel() for p in transition_runtime.parameters()
            ),
        },
        "config": {
            "questions": len(traces),
            "train_questions": len(train_traces),
            "eval_questions": len(eval_traces),
            "max_new_tokens": args.max_new_tokens,
            "max_train_positions": args.max_train_positions,
            "bottleneck": args.bottleneck,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "mse_weight": args.mse_weight,
            "repeats": args.repeats,
            "warmup_tokens": args.warmup_tokens,
            "decoding": "greedy deterministic",
        },
        "training": training,
        "checkpoint": str(adapter_path),
        "teacher_forced_heldout": teacher_results,
        "runtime": runtime_results,
        "aggregate": {
            "teacher_zero_kl": mean_metric(
                teacher_results,
                "mlp_zero",
                "kl_oracle_to_candidate_mean",
            ),
            "teacher_predicted_kl": mean_metric(
                teacher_results,
                "mlp_predicted",
                "kl_oracle_to_candidate_mean",
            ),
            "teacher_zero_top1": mean_metric(
                teacher_results,
                "mlp_zero",
                "top1_agreement",
            ),
            "teacher_predicted_top1": mean_metric(
                teacher_results,
                "mlp_predicted",
                "top1_agreement",
            ),
            "runtime_mlp_skip_speedup": sum(
                row["mlp_skip_speedup"]
                for row in runtime_results
            ) / len(runtime_results),
            "runtime_mlp_predicted_speedup": sum(
                row["mlp_predicted_speedup"]
                for row in runtime_results
            ) / len(runtime_results),
            "runtime_mlp_skip_token_agreement": sum(
                row["mlp_skip_vs_full"]["exact_token_agreement"]
                for row in runtime_results
            ) / len(runtime_results),
            "runtime_mlp_predicted_token_agreement": sum(
                row["mlp_predicted_vs_full"]["exact_token_agreement"]
                for row in runtime_results
            ) / len(runtime_results),
        },
    }

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== EXP-0009Q summary ===")
    print(
        json.dumps(
            summary["aggregate"],
            indent=2,
        )
    )
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
