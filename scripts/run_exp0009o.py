"""CFI EXP-0009O: integrated early-exit runtime benchmark.

This experiment turns the EXP-0009N behavioral result into an actual runtime
prototype. The last Qwen3 decoder layer (L36) is bypassed inside the model
forward path, rather than executed and overwritten afterward.

Modes:
1. full: execute all 36 decoder layers
2. raw_h35: execute layers 1-35, then bypass L36 and use H35 directly
3. adapted: execute layers 1-35, bypass L36, run the trained 663k-parameter
   residual transition, then use the existing final RMSNorm + LM head

The base Qwen3-4B-Base weights remain frozen.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from types import MethodType
from typing import Any, Callable

import torch
from transformers import AutoTokenizer

from run_exp0009n import (
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    ResidualTransition,
    build_model,
    capture_trace,
    extract_predicted,
    load_rows,
    seed_all,
    train_transition,
)


DEFAULT_OUTPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009O-Integrated-Early-Exit-Runtime"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CFI EXP-0009O.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--questions", type=int, default=10)
    p.add_argument("--train-questions", type=int, default=7)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed-base", type=int, default=42000)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-train-positions", type=int, default=64)
    p.add_argument("--bottleneck", type=int, default=128)
    p.add_argument("--max-update-ratio", type=float, default=0.5)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--warmup-tokens", type=int, default=16)
    return p.parse_args()


def generation_kwargs(
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    # Greedy decoding makes the A/B comparison deterministic. min_new_tokens
    # prevents an early EOS from shortening one mode relative to another.
    return {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }


def prompt_inputs(model: Any, trace: Any) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = trace.input_ids[: trace.prompt_length].unsqueeze(0).to("cuda:0")
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def inspect_last_layer_api(model: Any, trace: Any) -> dict[str, Any]:
    """Verify that the installed Qwen3 layer returns a tensor directly."""
    seen: list[Any] = []

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        seen.append(output)

    handle = model.model.layers[35].register_forward_hook(capture)
    input_ids, attention_mask = prompt_inputs(model, trace)
    try:
        with torch.inference_mode():
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
    finally:
        handle.remove()

    if len(seen) != 1:
        raise RuntimeError(
            f"Expected one L36 output during API probe, got {len(seen)}."
        )

    output = seen[0]
    if not torch.is_tensor(output):
        raise RuntimeError(
            "EXP-0009O requires a tensor-returning Qwen3 decoder layer. "
            f"Installed L36 returned {type(output).__name__}. "
            "Refusing to monkey-patch an unknown cache/output ABI."
        )

    return {
        "layer_index": 36,
        "return_type": type(output).__name__,
        "shape": list(output.shape),
        "dtype": str(output.dtype),
    }


def make_bypass_forward(
    mode: str,
    transition: torch.nn.Module | None,
) -> Callable[..., torch.Tensor]:
    if mode not in {"raw_h35", "adapted"}:
        raise ValueError(f"Unsupported bypass mode: {mode}")

    def bypass_forward(
        _layer: Any,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del args, kwargs

        if mode == "raw_h35":
            return hidden_states

        if transition is None:
            raise RuntimeError("Adapted bypass requires a trained transition.")

        source_dtype = hidden_states.dtype
        runtime_dtype = next(transition.parameters()).dtype
        adapted = transition(
            hidden_states.to(dtype=runtime_dtype)
        )
        return adapted.to(dtype=source_dtype)

    return bypass_forward


class IntegratedLayerBypass:
    """Replace the actual L36 forward implementation while benchmarking."""

    def __init__(
        self,
        model: Any,
        mode: str,
        transition: torch.nn.Module | None = None,
    ) -> None:
        self.model = model
        self.layer = model.model.layers[35]
        self.original_forward = self.layer.forward
        self.mode = mode
        self.transition = transition

        patched = make_bypass_forward(mode, transition)
        self.layer.forward = MethodType(patched, self.layer)

    def remove(self) -> None:
        self.layer.forward = self.original_forward


def generate_once(
    model: Any,
    tokenizer: Any,
    trace: Any,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, float, int]:
    input_ids, attention_mask = prompt_inputs(model, trace)
    kwargs = generation_kwargs(tokenizer, args)

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    generated_new = generated[0, trace.prompt_length :].detach().cpu()
    return generated_new, elapsed, int(generated_new.numel())


def warmup_mode(
    model: Any,
    tokenizer: Any,
    trace: Any,
    args: argparse.Namespace,
) -> None:
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


def benchmark_mode(
    model: Any,
    tokenizer: Any,
    trace: Any,
    args: argparse.Namespace,
    mode: str,
    transition: torch.nn.Module | None,
) -> dict[str, Any]:
    if mode == "full":
        bypass = None
    else:
        bypass = IntegratedLayerBypass(model, mode, transition)

    try:
        # One warmup per mode removes first-use CUDA/kernel initialization from
        # the measured runs.
        warmup_mode(model, tokenizer, trace, args)

        runs: list[dict[str, Any]] = []
        reference_tokens: torch.Tensor | None = None

        for repeat in range(1, args.repeats + 1):
            torch.cuda.reset_peak_memory_stats(device="cuda:0")
            generated, elapsed, token_count = generate_once(
                model,
                tokenizer,
                trace,
                args,
            )

            if reference_tokens is None:
                reference_tokens = generated.clone()
            elif not torch.equal(reference_tokens, generated):
                raise RuntimeError(
                    f"{mode} was not deterministic across repeats for "
                    f"question {trace.index}."
                )

            peak_memory = int(torch.cuda.max_memory_allocated(device="cuda:0"))
            runs.append(
                {
                    "repeat": repeat,
                    "seconds": elapsed,
                    "generated_tokens": token_count,
                    "tokens_per_second": token_count / max(elapsed, 1e-12),
                    "peak_memory_bytes": peak_memory,
                }
            )

        assert reference_tokens is not None
        mean_seconds = sum(r["seconds"] for r in runs) / len(runs)
        mean_tps = sum(r["tokens_per_second"] for r in runs) / len(runs)
        mean_peak_memory = sum(r["peak_memory_bytes"] for r in runs) / len(runs)

        return {
            "mode": mode,
            "generated_tokens": int(reference_tokens.numel()),
            "text": tokenizer.decode(
                reference_tokens,
                skip_special_tokens=True,
            ).strip(),
            "predicted": extract_predicted(
                tokenizer.decode(
                    reference_tokens,
                    skip_special_tokens=True,
                ).strip()
            ),
            "runs": runs,
            "mean_seconds": mean_seconds,
            "mean_tokens_per_second": mean_tps,
            "mean_peak_memory_bytes": mean_peak_memory,
            "tokens": reference_tokens.tolist(),
        }
    finally:
        if bypass is not None:
            bypass.remove()


def compare_sequences(
    full_tokens: list[int],
    candidate_tokens: list[int],
) -> dict[str, Any]:
    shared = min(len(full_tokens), len(candidate_tokens))
    equal = sum(
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
        "exact_token_agreement": equal / max(shared, 1),
        "matching_tokens": equal,
        "first_divergence_generated_token": first_divergence,
    }


def main() -> int:
    args = parse_args()

    if not 1 <= args.train_questions < args.questions:
        raise ValueError("--train-questions must be >=1 and < --questions.")
    if args.repeats < 1:
        raise ValueError("--repeats must be >=1.")
    if args.warmup_tokens < 1:
        raise ValueError("--warmup-tokens must be >=1.")

    seed_all(args.seed_base)
    rows = load_rows(args.dataset, args.questions)
    args.output.mkdir(parents=True, exist_ok=True)

    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", args.model)
    print("Dataset:", args.dataset)
    print("Questions:", args.questions)
    print("Train questions:", args.train_questions)
    print("Runtime repeats:", args.repeats)

    print("\nLoading established 4-bit NF4 model...")
    tokenizer, model = build_model(args.model)

    traces = []
    for index, row in enumerate(rows, 1):
        print(f"\n[{index}/{args.questions}] capture training trace...")
        trace = capture_trace(model, tokenizer, row, index, args)
        traces.append(trace)
        print(
            f"  expected={trace.expected} "
            f"predicted={trace.baseline_predicted} "
            f"correct={trace.baseline_correct} "
            f"tokens={trace.generated_tokens} "
            f"time={trace.baseline_seconds:.2f}s"
        )

    api_info = inspect_last_layer_api(model, traces[0])
    print("\nL36 API probe:", json.dumps(api_info))

    train_traces = traces[: args.train_questions]
    eval_traces = traces[args.train_questions :]

    print("\n=== train anchored residual transition ===")
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
            "experiment": "EXP-0009O",
            "source_experiment": "EXP-0009N",
            "state_dict": state_dict_cpu,
            "parameter_count": sum(
                p.numel() for p in transition_fp32.parameters()
            ),
            "hidden_size": train_traces[0].h35.shape[-1],
            "bottleneck": args.bottleneck,
        },
        adapter_path,
    )

    runtime_dtype = next(model.model.norm.parameters()).dtype
    transition_runtime = copy.deepcopy(transition_fp32).to(
        device="cuda:0",
        dtype=runtime_dtype,
    )
    transition_runtime.eval()

    del transition_fp32
    torch.cuda.empty_cache()

    results: list[dict[str, Any]] = []

    for trace in eval_traces:
        print(f"\n=== Runtime benchmark question {trace.index} ===")

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

        raw_agreement = compare_sequences(
            full["tokens"],
            raw["tokens"],
        )
        adapted_agreement = compare_sequences(
            full["tokens"],
            adapted["tokens"],
        )

        full_seconds = full["mean_seconds"]
        raw_speedup = full_seconds / max(raw["mean_seconds"], 1e-12)
        adapted_speedup = full_seconds / max(adapted["mean_seconds"], 1e-12)

        print(
            f"  full:    {full['mean_seconds']:.3f}s "
            f"({full['mean_tokens_per_second']:.2f} tok/s)"
        )
        print(
            f"  raw H35: {raw['mean_seconds']:.3f}s "
            f"({raw['mean_tokens_per_second']:.2f} tok/s), "
            f"speedup={raw_speedup:.3f}x, "
            f"token_agreement={raw_agreement['exact_token_agreement']:.3f}"
        )
        print(
            f"  adapted: {adapted['mean_seconds']:.3f}s "
            f"({adapted['mean_tokens_per_second']:.2f} tok/s), "
            f"speedup={adapted_speedup:.3f}x, "
            f"token_agreement={adapted_agreement['exact_token_agreement']:.3f}"
        )

        results.append(
            {
                "question_index": trace.index,
                "expected": trace.expected,
                "full": full,
                "raw_h35": raw,
                "adapted": adapted,
                "raw_vs_full": raw_agreement,
                "adapted_vs_full": adapted_agreement,
                "raw_speedup_x": raw_speedup,
                "adapted_speedup_x": adapted_speedup,
            }
        )

    def avg(key: str) -> float:
        return sum(r[key] for r in results) / len(results)

    raw_speedups = [r["raw_speedup_x"] for r in results]
    adapted_speedups = [r["adapted_speedup_x"] for r in results]
    raw_agreements = [
        r["raw_vs_full"]["exact_token_agreement"]
        for r in results
    ]
    adapted_agreements = [
        r["adapted_vs_full"]["exact_token_agreement"]
        for r in results
    ]

    full_times = [
        r["full"]["mean_seconds"]
        for r in results
    ]
    raw_times = [
        r["raw_h35"]["mean_seconds"]
        for r in results
    ]
    adapted_times = [
        r["adapted"]["mean_seconds"]
        for r in results
    ]

    summary = {
        "schema_version": 1,
        "experiment": "EXP-0009O",
        "title": "Integrated Early-Exit Runtime Benchmark",
        "status": "completed",
        "model": {
            "path": str(args.model),
            "base_model_frozen": True,
            "num_layers": 36,
            "exit_boundary": "H35",
            "last_layer_bypassed": 36,
            "runtime_transition_parameters": sum(
                p.numel() for p in transition_runtime.parameters()
            ),
            "runtime_transition_dtype": str(runtime_dtype),
        },
        "config": {
            "questions": len(traces),
            "train_questions": len(train_traces),
            "eval_questions": len(eval_traces),
            "max_new_tokens": args.max_new_tokens,
            "warmup_tokens": args.warmup_tokens,
            "repeats": args.repeats,
            "bottleneck": args.bottleneck,
            "epochs": args.epochs,
            "seed_base": args.seed_base,
            "decoding": "greedy deterministic",
        },
        "l36_api_probe": api_info,
        "training": training,
        "adapter_checkpoint": str(adapter_path),
        "runtime": {
            "full_mean_seconds": avg("full_mean_seconds")
            if "full_mean_seconds" in results[0]
            else sum(full_times) / len(full_times),
            "raw_h35_mean_seconds": sum(raw_times) / len(raw_times),
            "adapted_mean_seconds": sum(adapted_times) / len(adapted_times),
            "raw_h35_mean_tokens_per_second": sum(
                r["raw_h35"]["mean_tokens_per_second"] for r in results
            ) / len(results),
            "adapted_mean_tokens_per_second": sum(
                r["adapted"]["mean_tokens_per_second"] for r in results
            ) / len(results),
            "raw_h35_speedup_mean": sum(raw_speedups) / len(raw_speedups),
            "adapted_speedup_mean": sum(adapted_speedups) / len(adapted_speedups),
            "raw_h35_token_agreement_mean": sum(raw_agreements) / len(raw_agreements),
            "adapted_token_agreement_mean": sum(adapted_agreements) / len(adapted_agreements),
        },
        "results": results,
    }

    # Remove a dead helper key if present. The explicit means above are the
    # canonical aggregate fields.
    summary["runtime"]["full_mean_seconds"] = sum(full_times) / len(full_times)

    with (args.output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (args.output / "results.jsonl").open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print("\n=== EXP-0009O summary ===")
    print(
        json.dumps(
            {
                "full_mean_seconds": summary["runtime"]["full_mean_seconds"],
                "raw_h35_mean_seconds": summary["runtime"]["raw_h35_mean_seconds"],
                "adapted_mean_seconds": summary["runtime"]["adapted_mean_seconds"],
                "raw_h35_speedup_mean": summary["runtime"]["raw_h35_speedup_mean"],
                "adapted_speedup_mean": summary["runtime"]["adapted_speedup_mean"],
                "raw_h35_token_agreement_mean": summary["runtime"][
                    "raw_h35_token_agreement_mean"
                ],
                "adapted_token_agreement_mean": summary["runtime"][
                    "adapted_token_agreement_mean"
                ],
            },
            indent=2,
        )
    )
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
