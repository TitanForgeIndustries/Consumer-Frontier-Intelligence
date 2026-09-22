"""CFI EXP-0009Q: adaptive early-exit gating.

P showed that a one-shot learned H35 replacement improves teacher-forced
behavior but diverges during free-running generation. Q introduces a
conservative gate that decides whether to bypass L36 on each decode step.

The base Qwen3-4B-Base model remains frozen. The existing 663,168-parameter
transition is unchanged. A small gate is trained to predict whether the
adapted H35 path will select the same next token as the full model.

For predicted-safe steps, L36 is skipped. For predicted-risk steps, the real
L36 forward executes, preserving the full-model trajectory when the gate is
correct.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from run_exp0009n import (
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    ResidualTransition,
    Trace,
    base_exit_logits,
    build_model,
    distribution_metrics,
    extract_expected,
    extract_predicted,
    format_prompt,
    load_rows,
    seed_all,
    train_transition,
)
from run_exp0009o import generation_kwargs, prompt_inputs
from run_exp0009p import capture_greedy_trace


DEFAULT_OUTPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009Q-Adaptive-Early-Exit-Gate"
)
DEFAULT_THRESHOLDS = (0.90, 0.95, 0.99)


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
    p.add_argument("--max-update-ratio", type=float, default=0.5)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--gate-bottleneck", type=int, default=64)
    p.add_argument("--gate-epochs", type=int, default=32)
    p.add_argument("--gate-learning-rate", type=float, default=2e-3)
    p.add_argument("--gate-weight-decay", type=float, default=1e-5)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--warmup-tokens", type=int, default=16)
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )
    return p.parse_args()


class SafetyGate(nn.Module):
    """Predicts whether the adapted H35 path will match full-model top-1."""

    def __init__(self, hidden_size: int, bottleneck: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden)
        x = self.act(self.down(x))
        return self.up(x).squeeze(-1)


def make_gate_dataset(
    model: Any,
    transition: ResidualTransition,
    traces: list[Trace],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    transition.eval()
    with torch.inference_mode():
        for trace in traces:
            positions = trace.train_positions
            source = trace.h35[positions].to(
                device=device,
                dtype=next(transition.parameters()).dtype,
            )

            adapted_state = transition(source)
            adapted_logits = base_exit_logits(model, adapted_state)

            teacher_top1 = trace.teacher_log_probs.argmax(dim=-1).to(device)
            adapted_top1 = adapted_logits.argmax(dim=-1)
            safe = (adapted_top1 == teacher_top1).float()

            features.append(
                source.to(dtype=torch.float32).cpu()
            )
            labels.append(safe.to(dtype=torch.float32).cpu())

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def train_gate(
    hidden_size: int,
    features: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[SafetyGate, dict[str, Any]]:
    gate = SafetyGate(
        hidden_size=hidden_size,
        bottleneck=args.gate_bottleneck,
    ).to(device=device, dtype=torch.float32)

    pos = float(labels.sum().item())
    neg = float(labels.numel() - pos)
    # Bias the classifier toward conservative risk detection. A false safe
    # prediction can change the autoregressive trajectory; a false risk
    # prediction only spends L36 unnecessarily.
    positive_weight = min(max(0.5 * neg / max(pos, 1.0), 0.1), 1.0)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            positive_weight,
            dtype=torch.float32,
            device=device,
        )
    )
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=args.gate_learning_rate,
        weight_decay=args.gate_weight_decay,
    )

    x = features.to(device)
    y = labels.to(device)

    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, args.gate_epochs + 1):
        gate.train()
        logits = gate(x)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
        optimizer.step()

        with torch.inference_mode():
            probs = torch.sigmoid(logits)
            pred = probs >= 0.5
            accuracy = float((pred == (y >= 0.5)).float().mean().item())

        history.append(
            {
                "epoch": epoch,
                "mean_loss": float(loss.item()),
                "train_accuracy_at_0_5": accuracy,
            }
        )
        print(
            f"    gate epoch {epoch:02d}/{args.gate_epochs}: "
            f"loss={loss.item():.6f} accuracy={accuracy:.4f}"
        )

    safe_fraction = float(labels.mean().item())
    return gate, {
        "training_seconds": time.perf_counter() - started,
        "epochs": args.gate_epochs,
        "bottleneck": args.gate_bottleneck,
        "parameter_count": sum(p.numel() for p in gate.parameters()),
        "positive_weight": positive_weight,
        "training_safe_label_fraction": safe_fraction,
        "history": history,
    }


class AdaptiveLayerBypass:
    """Conditionally skip L36 during actual generation."""

    def __init__(
        self,
        model: Any,
        transition: ResidualTransition,
        gate: SafetyGate,
        threshold: float,
    ) -> None:
        self.model = model
        self.layer = model.model.layers[35]
        self.original_forward = self.layer.forward
        self.transition = transition
        self.gate = gate
        self.threshold = threshold

        self.decode_steps = 0
        self.skipped_steps = 0
        self.full_steps = 0
        self.forced_prefill_steps = 0
        self.safe_prob_sum = 0.0

        def bypass_forward(
            _layer: Any,
            hidden_states: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            if hidden_states.shape[1] != 1:
                self.forced_prefill_steps += int(hidden_states.shape[1])
                self.full_steps += int(hidden_states.shape[1])
                return self.original_forward(
                    hidden_states,
                    *args,
                    **kwargs,
                )

            past = kwargs.get("past_key_values")
            if past is None:
                self.forced_prefill_steps += 1
                self.full_steps += 1
                return self.original_forward(
                    hidden_states,
                    *args,
                    **kwargs,
                )

            self.decode_steps += 1
            gate_dtype = next(self.gate.parameters()).dtype
            gate_input = hidden_states.to(dtype=gate_dtype)
            with torch.no_grad():
                probability = torch.sigmoid(self.gate(gate_input)).reshape(-1)[0]

            safe_probability = float(probability.item())
            self.safe_prob_sum += safe_probability

            if safe_probability >= self.threshold:
                self.skipped_steps += 1
                source_dtype = hidden_states.dtype
                transition_dtype = next(
                    self.transition.parameters()
                ).dtype
                adapted = self.transition(
                    hidden_states.to(dtype=transition_dtype)
                )
                return adapted.to(dtype=source_dtype)

            self.full_steps += 1
            return self.original_forward(
                hidden_states,
                *args,
                **kwargs,
            )

        self.layer.forward = bypass_forward.__get__(self.layer, type(self.layer))

    def remove(self) -> None:
        self.layer.forward = self.original_forward

    @property
    def skip_rate(self) -> float:
        return self.skipped_steps / max(self.decode_steps, 1)

    @property
    def full_decode_rate(self) -> float:
        return self.full_steps / max(self.decode_steps, 1)

    @property
    def mean_safe_probability(self) -> float:
        return self.safe_prob_sum / max(self.decode_steps, 1)


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


def benchmark_gated(
    model: Any,
    tokenizer: Any,
    trace: Trace,
    args: argparse.Namespace,
    transition: ResidualTransition,
    gate: SafetyGate,
    threshold: float,
) -> dict[str, Any]:
    bypass = AdaptiveLayerBypass(
        model=model,
        transition=transition,
        gate=gate,
        threshold=threshold,
    )

    try:
        # Warmup uses the exact same gate and threshold but is excluded from
        # measured results.
        input_ids, attention_mask = prompt_inputs(model, trace)
        kwargs = generation_kwargs(tokenizer, args)
        kwargs["max_new_tokens"] = args.warmup_tokens
        kwargs["min_new_tokens"] = args.warmup_tokens

        with torch.inference_mode():
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        torch.cuda.synchronize()

        runs: list[dict[str, Any]] = []
        reference_tokens: torch.Tensor | None = None
        decision_snapshots: list[dict[str, Any]] = []

        for repeat in range(1, args.repeats + 1):
            input_ids, attention_mask = prompt_inputs(model, trace)
            torch.cuda.reset_peak_memory_stats(device="cuda:0")

            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs(tokenizer, args),
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
                    f"Gated threshold {threshold} was not deterministic across "
                    f"repeats for question {trace.index}."
                )

            decision_snapshots.append(
                {
                    "repeat": repeat,
                    "decode_steps": bypass.decode_steps,
                    "skipped_steps": bypass.skipped_steps,
                    "full_decode_steps": bypass.full_steps,
                    "skip_rate": bypass.skip_rate,
                    "mean_safe_probability": bypass.mean_safe_probability,
                    "forced_prefill_steps": bypass.forced_prefill_steps,
                }
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

        # Use the last measured run's decision counts for a representative
        # runtime trace. Repeats are expected to make identical decisions.
        decisions = decision_snapshots[-1]

        return {
            "threshold": threshold,
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
            "decision_metrics": decisions,
            "tokens": reference_tokens.tolist(),
        }
    finally:
        bypass.remove()


def train_transition_from_traces(
    model: Any,
    traces: list[Trace],
    args: argparse.Namespace,
) -> ResidualTransition:
    transition, _training = train_transition(
        model,
        traces,
        args,
        torch.device("cuda:0"),
    )
    return transition


def main() -> int:
    args = parse_args()

    if not 1 <= args.train_questions < args.questions:
        raise ValueError("--train-questions must be >=1 and < --questions.")
    if args.repeats < 1:
        raise ValueError("--repeats must be >=1.")
    if not args.thresholds:
        raise ValueError("--thresholds must not be empty.")
    for threshold in args.thresholds:
        if not 0.0 < threshold < 1.0:
            raise ValueError("Every threshold must be in (0, 1).")

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
    print("Gate thresholds:", args.thresholds)

    print("\nLoading established 4-bit NF4 model...")
    tokenizer, model = build_model(args.model)

    traces: list[Trace] = []
    for index, row in enumerate(rows, 1):
        print(f"\n[{index}/{len(rows)}] greedy training trace...")
        trace = capture_greedy_trace(model, tokenizer, row, index, args)
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

    print("\n=== train anchored residual transition ===")
    transition, transition_training = train_transition(
        model,
        train_traces,
        args,
        torch.device("cuda:0"),
    )

    print("\n=== build gate labels from matched full-model traces ===")
    gate_features, gate_labels = make_gate_dataset(
        model,
        transition,
        train_traces,
        torch.device("cuda:0"),
    )

    safe_fraction = float(gate_labels.mean().item())
    print(
        f"  positions={gate_labels.numel()} "
        f"safe_label_fraction={safe_fraction:.4f}"
    )

    print("\n=== train safety gate ===")
    gate, gate_training = train_gate(
        hidden_size=train_traces[0].h35.shape[-1],
        features=gate_features,
        labels=gate_labels,
        args=args,
        device=torch.device("cuda:0"),
    )

    adapter_path = args.output / "transition_state_dict_fp32.pt"
    gate_path = args.output / "gate_state_dict_fp32.pt"

    torch.save(
        {
            "experiment": "EXP-0009Q",
            "source_experiments": ["EXP-0009L", "EXP-0009N", "EXP-0009P"],
            "state_dict": {
                name: value.detach().cpu()
                for name, value in transition.state_dict().items()
            },
            "parameter_count": sum(
                p.numel() for p in transition.parameters()
            ),
            "hidden_size": train_traces[0].h35.shape[-1],
            "bottleneck": args.bottleneck,
        },
        adapter_path,
    )
    torch.save(
        {
            "experiment": "EXP-0009Q",
            "state_dict": {
                name: value.detach().cpu()
                for name, value in gate.state_dict().items()
            },
            "parameter_count": sum(p.numel() for p in gate.parameters()),
            "hidden_size": train_traces[0].h35.shape[-1],
            "bottleneck": args.gate_bottleneck,
        },
        gate_path,
    )

    transition_runtime = copy.deepcopy(transition).to(
        device="cuda:0",
        dtype=next(model.model.norm.parameters()).dtype,
    )
    transition_runtime.eval()

    gate_runtime = copy.deepcopy(gate).to(
        device="cuda:0",
        dtype=next(model.model.norm.parameters()).dtype,
    )
    gate_runtime.eval()

    threshold_results: dict[str, list[dict[str, Any]]] = {
        str(threshold): [] for threshold in args.thresholds
    }

    for trace in eval_traces:
        print(f"\n=== Q held-out question {trace.index} ===")

        for threshold in args.thresholds:
            result = benchmark_gated(
                model=model,
                tokenizer=tokenizer,
                trace=trace,
                args=args,
                transition=transition_runtime,
                gate=gate_runtime,
                threshold=threshold,
            )
            threshold_results[str(threshold)].append(result)

            print(
                f"  threshold={threshold:.2f}: "
                f"time={result['mean_seconds']:.3f}s "
                f"tok/s={result['mean_tokens_per_second']:.2f} "
                f"skip_rate={result['decision_metrics']['skip_rate']:.3f} "
                f"agreement_pending"
            )

    # Establish the full-model and always-adapted reference runs separately.
    # This avoids treating a gated run as the speed reference.
    reference_results: list[dict[str, Any]] = []

    from run_exp0009o import benchmark_mode

    for trace in eval_traces:
        full = benchmark_mode(
            model,
            tokenizer,
            trace,
            args,
            "full",
            transition=None,
        )
        adapted = benchmark_mode(
            model,
            tokenizer,
            trace,
            args,
            "adapted",
            transition=transition_runtime,
        )
        reference_results.append(
            {
                "question_index": trace.index,
                "full": full,
                "adapted": adapted,
                "adapted_vs_full": compare_tokens(
                    full["tokens"],
                    adapted["tokens"],
                ),
            }
        )

        print(
            f"  reference Q{trace.index}: "
            f"full={full['mean_seconds']:.3f}s "
            f"adapted={adapted['mean_seconds']:.3f}s "
            f"adapted_agreement="
            f"{reference_results[-1]['adapted_vs_full']['exact_token_agreement']:.3f}"
        )

    aggregate: dict[str, Any] = {}
    for threshold in args.thresholds:
        rows_for_threshold = threshold_results[str(threshold)]
        per_question: list[dict[str, Any]] = []

        for idx, gated in enumerate(rows_for_threshold):
            full = reference_results[idx]["full"]
            comparison = compare_tokens(
                full["tokens"],
                gated["tokens"],
            )
            speedup = full["mean_seconds"] / max(
                gated["mean_seconds"],
                1e-12,
            )
            per_question.append(
                {
                    "question_index": gated.get(
                        "question_index",
                        eval_traces[idx].index,
                    ),
                    "speedup": speedup,
                    "token_comparison": comparison,
                    "gated": gated,
                }
            )

        aggregate[str(threshold)] = {
            "mean_speedup": sum(
                item["speedup"] for item in per_question
            ) / len(per_question),
            "mean_token_agreement": sum(
                item["token_comparison"]["exact_token_agreement"]
                for item in per_question
            ) / len(per_question),
            "mean_skip_rate": sum(
                item["gated"]["decision_metrics"]["skip_rate"]
                for item in per_question
            ) / len(per_question),
            "mean_generation_seconds": sum(
                item["gated"]["mean_seconds"] for item in per_question
            ) / len(per_question),
            "mean_tokens_per_second": sum(
                item["gated"]["mean_tokens_per_second"]
                for item in per_question
            ) / len(per_question),
            "per_question": per_question,
        }

    summary = {
        "schema_version": 1,
        "experiment": "EXP-0009Q",
        "title": "Adaptive Early-Exit Gate",
        "status": "completed",
        "hypothesis": (
            "A conservative H35-only gate can skip L36 on low-risk decode "
            "steps while falling back to the true L36 on predicted-risk steps."
        ),
        "model": {
            "path": str(args.model),
            "base_model_frozen": True,
            "num_layers": 36,
            "exit_boundary": "H35",
            "bypassed_layer": 36,
            "transition_parameters": sum(
                p.numel() for p in transition_runtime.parameters()
            ),
            "gate_parameters": sum(
                p.numel() for p in gate_runtime.parameters()
            ),
        },
        "config": {
            "questions": len(traces),
            "train_questions": len(train_traces),
            "eval_questions": len(eval_traces),
            "max_new_tokens": args.max_new_tokens,
            "max_train_positions": args.max_train_positions,
            "transition_bottleneck": args.bottleneck,
            "transition_epochs": args.epochs,
            "gate_bottleneck": args.gate_bottleneck,
            "gate_epochs": args.gate_epochs,
            "repeats": args.repeats,
            "warmup_tokens": args.warmup_tokens,
            "decoding": "greedy deterministic",
            "thresholds": args.thresholds,
        },
        "transition_training": transition_training,
        "gate_training": gate_training,
        "gate_labels": {
            "positions": int(gate_labels.numel()),
            "safe_fraction": safe_fraction,
        },
        "checkpoints": {
            "transition": str(adapter_path),
            "gate": str(gate_path),
        },
        "always_adapted_reference": reference_results,
        "threshold_results": aggregate,
    }

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== EXP-0009Q summary ===")
    print(
        json.dumps(
            {
                threshold: {
                    "mean_speedup": values["mean_speedup"],
                    "mean_token_agreement": values["mean_token_agreement"],
                    "mean_skip_rate": values["mean_skip_rate"],
                }
                for threshold, values in aggregate.items()
            },
            indent=2,
        )
    )
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
