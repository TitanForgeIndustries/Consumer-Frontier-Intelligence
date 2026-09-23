"""CFI EXP-0009P: matched-greedy trajectory early-exit validation.

EXP-0009O trained the transition on sampled trajectories but benchmarked it with
greedy decoding. P removes that confound by generating the training traces with
the same greedy decoding used by the runtime benchmark.

The base Qwen3-4B-Base model remains frozen. Only the 663,168-parameter
anchored residual transition is trained.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from run_exp0009n import (
    ResidualTransition,
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    build_model,
    extract_expected,
    extract_predicted,
    format_prompt,
    load_rows,
    seed_all,
    train_transition,
    base_exit_logits,
    distribution_metrics,
    Trace,
)
from run_exp0009o import (
    IntegratedLayerBypass,
    benchmark_mode,
    generation_kwargs,
    prompt_inputs,
)


DEFAULT_OUTPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009P-Matched-Greedy-Trajectory"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CFI EXP-0009P.")
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
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--warmup-tokens", type=int, default=16)
    return p.parse_args()


def capture_greedy_trace(
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

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            **generation_kwargs(tokenizer, args),
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    generated_ids = generated[0, prompt_length:].detach().cpu()
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    full_ids = generated.detach()
    h35_capture: list[torch.Tensor] = []

    def capture_h35(_module: Any, _inputs: Any, output: Any) -> None:
        if torch.is_tensor(output):
            h35_capture.append(output.detach().float().cpu())
        else:
            raise RuntimeError(
                f"Unexpected L35 output type: {type(output).__name__}"
            )

    handle = model.model.layers[34].register_forward_hook(capture_h35)
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                use_cache=False,
            )
    finally:
        handle.remove()

    if len(h35_capture) != 1:
        raise RuntimeError(
            f"Expected one H35 capture, got {len(h35_capture)}."
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

    input_ids = full_ids[0].detach().cpu()
    predicted = extract_predicted(generated_text)

    return Trace(
        index=index,
        expected=expected,
        baseline_predicted=predicted,
        baseline_correct=predicted == expected,
        prompt_length=prompt_length,
        input_ids=input_ids,
        h35=h35_capture[0][0],
        positions=positions,
        train_positions=train_positions,
        teacher_log_probs=teacher_log_probs,
        generated_tokens=int(generated_ids.numel()),
        baseline_seconds=elapsed,
    )


def heldout_teacher_forced(
    model: Any,
    transition: ResidualTransition,
    trace: Trace,
) -> dict[str, Any]:
    positions = trace.positions
    target_ids = trace.input_ids[positions + 1]

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

        source = trace.h35[positions]
        base = base_exit_logits(
            model,
            source.to("cuda:0"),
        ).float()

        transition.eval()
        transition_dtype = next(transition.parameters()).dtype
        adapted_state = transition(
            source.to(
                device="cuda:0",
                dtype=transition_dtype,
            )
        )
        adapted = base_exit_logits(
            model,
            adapted_state,
        ).float()

    return {
        "question_index": trace.index,
        "base_h35": distribution_metrics(
            oracle,
            base,
            target_ids.to("cuda:0"),
        ),
        "adapted": distribution_metrics(
            oracle,
            adapted,
            target_ids.to("cuda:0"),
        ),
    }


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

    traces: list[Trace] = []
    for index, row in enumerate(rows, 1):
        print(f"\n[{index}/{len(rows)}] greedy training trace...")
        trace = capture_greedy_trace(
            model,
            tokenizer,
            row,
            index,
            args,
        )
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

    print("\n=== train anchored residual transition on matched greedy traces ===")
    transition_fp32, training = train_transition(
        model,
        train_traces,
        args,
        torch.device("cuda:0"),
    )

    adapter_path = args.output / "transition_state_dict_fp32.pt"
    state_dict_cpu = {
        name: value.detach().cpu()
        for name, value in transition_fp32.state_dict().items()
    }
    torch.save(
        {
            "experiment": "EXP-0009P",
            "source_experiment": "EXP-0009L/N",
            "state_dict": state_dict_cpu,
            "parameter_count": sum(
                p.numel() for p in transition_fp32.parameters()
            ),
            "hidden_size": train_traces[0].h35.shape[-1],
            "bottleneck": args.bottleneck,
            "training_decode_mode": "greedy",
        },
        adapter_path,
    )

    runtime_dtype = next(model.model.norm.parameters()).dtype
    transition_runtime = copy.deepcopy(transition_fp32).to(
        device="cuda:0",
        dtype=runtime_dtype,
    )
    transition_runtime.eval()

    results: list[dict[str, Any]] = []
    heldout_teacher: list[dict[str, Any]] = []

    for trace in eval_traces:
        print(f"\n=== P held-out question {trace.index} ===")

        teacher = heldout_teacher_forced(
            model=model,
            transition=transition_runtime,
            trace=trace,
        )
        heldout_teacher.append(teacher)

        b = teacher["base_h35"]
        a = teacher["adapted"]
        print(
            f"  teacher-forced base H35: "
            f"KL={b['kl_oracle_to_candidate_mean']:.4f} "
            f"top1={b['top1_agreement']:.4f}"
        )
        print(
            f"  teacher-forced adapted: "
            f"KL={a['kl_oracle_to_candidate_mean']:.4f} "
            f"top1={a['top1_agreement']:.4f}"
        )

        full = benchmark_mode(
            model,
            tokenizer,
            trace,
            args,
            "full",
            transition=None,
        )
        raw = benchmark_mode(
            model,
            tokenizer,
            trace,
            args,
            "raw_h35",
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

        results.append(
            {
                "question_index": trace.index,
                "expected": trace.expected,
                "baseline_predicted": trace.baseline_predicted,
                "baseline_correct": trace.baseline_correct,
                "teacher_forced": teacher,
                "full": full,
                "raw_h35": raw,
                "adapted": adapted,
                "raw_vs_full": compare_tokens(
                    full["tokens"],
                    raw["tokens"],
                ),
                "adapted_vs_full": compare_tokens(
                    full["tokens"],
                    adapted["tokens"],
                ),
            }
        )

        full_time = full["mean_seconds"]
        raw_time = raw["mean_seconds"]
        adapted_time = adapted["mean_seconds"]
        print(
            f"  full:    {full_time:.3f}s "
            f"({full['mean_tokens_per_second']:.2f} tok/s)"
        )
        print(
            f"  raw H35: {raw_time:.3f}s "
            f"({raw['mean_tokens_per_second']:.2f} tok/s), "
            f"speedup={full_time / raw_time:.3f}x, "
            f"agreement={results[-1]['raw_vs_full']['exact_token_agreement']:.3f}"
        )
        print(
            f"  adapted: {adapted_time:.3f}s "
            f"({adapted['mean_tokens_per_second']:.2f} tok/s), "
            f"speedup={full_time / adapted_time:.3f}x, "
            f"agreement={results[-1]['adapted_vs_full']['exact_token_agreement']:.3f}"
        )

    def mean_teacher(mode: str, metric: str) -> float:
        return sum(
            row[mode][metric]
            for row in heldout_teacher
        ) / len(heldout_teacher)

    full_times = [row["full"]["mean_seconds"] for row in results]
    raw_times = [row["raw_h35"]["mean_seconds"] for row in results]
    adapted_times = [row["adapted"]["mean_seconds"] for row in results]

    full_tps = [
        row["full"]["mean_tokens_per_second"] for row in results
    ]
    raw_tps = [
        row["raw_h35"]["mean_tokens_per_second"] for row in results
    ]
    adapted_tps = [
        row["adapted"]["mean_tokens_per_second"] for row in results
    ]

    summary = {
        "schema_version": 1,
        "experiment": "EXP-0009P",
        "title": "Matched-Greedy Trajectory Early-Exit Validation",
        "status": "completed",
        "model": {
            "path": str(args.model),
            "base_model_frozen": True,
            "num_layers": 36,
            "exit_boundary": "H35",
            "last_layer_bypassed": 36,
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
            "repeats": args.repeats,
            "warmup_tokens": args.warmup_tokens,
            "decoding": "greedy deterministic",
            "seed_base": args.seed_base,
        },
        "training": training,
        "adapter_checkpoint": str(adapter_path),
        "teacher_forced_heldout_mean": {
            "base_h35_kl": mean_teacher(
                "base_h35",
                "kl_oracle_to_candidate_mean",
            ),
            "adapted_kl": mean_teacher(
                "adapted",
                "kl_oracle_to_candidate_mean",
            ),
            "base_h35_top1": mean_teacher(
                "base_h35",
                "top1_agreement",
            ),
            "adapted_top1": mean_teacher(
                "adapted",
                "top1_agreement",
            ),
        },
        "runtime": {
            "full_mean_seconds": sum(full_times) / len(full_times),
            "raw_h35_mean_seconds": sum(raw_times) / len(raw_times),
            "adapted_mean_seconds": sum(adapted_times) / len(adapted_times),
            "full_mean_tokens_per_second": sum(full_tps) / len(full_tps),
            "raw_h35_mean_tokens_per_second": sum(raw_tps) / len(raw_tps),
            "adapted_mean_tokens_per_second": sum(adapted_tps) / len(adapted_tps),
            "raw_h35_speedup_mean": sum(
                full_times[i] / raw_times[i]
                for i in range(len(results))
            ) / len(results),
            "adapted_speedup_mean": sum(
                full_times[i] / adapted_times[i]
                for i in range(len(results))
            ) / len(results),
            "raw_h35_token_agreement_mean": sum(
                r["raw_vs_full"]["exact_token_agreement"]
                for r in results
            ) / len(results),
            "adapted_token_agreement_mean": sum(
                r["adapted_vs_full"]["exact_token_agreement"]
                for r in results
            ) / len(results),
        },
        "results": results,
    }

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.output / "results.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in results:
            handle.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    print("\n=== EXP-0009P summary ===")
    print(
        json.dumps(
            {
                "teacher_forced_heldout_mean": summary[
                    "teacher_forced_heldout_mean"
                ],
                "runtime": summary["runtime"],
            },
            indent=2,
        )
    )
    print(f"Results: {args.output}")
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
