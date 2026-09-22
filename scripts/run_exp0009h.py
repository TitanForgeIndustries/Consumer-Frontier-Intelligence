"""CFI EXP-0009H: layer-36 sublayer predictability diagnostic.

This experiment stops changing the generic latent-state predictor and instead
measures whether the actual layer-36 attention and MLP outputs are predictable
from H35.

It is diagnostic, not a speedup experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = Path(
    r"E:\Titan Forge Industries\CFI-Data\HuggingFace\hub\models--Qwen--Qwen3-4B-Base\snapshots\906bfd4b4dc7f14ee4320094d8b41684abff8539"
)
DEFAULT_DATASET = Path(
    r"E:\Titan Forge Industries\CFI-Data\Datasets\CFI-Eval-0001-GSM8K\gsm8k_test_100.jsonl"
)
DEFAULT_OUTPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009H-Sublayer-Predictability-Diagnostic"
)


@dataclass
class Trace:
    index: int
    expected: str | None
    baseline_predicted: str | None
    baseline_correct: bool
    prompt_length: int
    input_ids: torch.Tensor
    source_h35: torch.Tensor
    attention_output_h36: torch.Tensor
    mlp_output_h36: torch.Tensor
    final_h36: torch.Tensor
    positions: torch.Tensor
    next_token_ids: torch.Tensor
    generated_tokens: int
    baseline_seconds: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CFI EXP-0009H.")
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
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--max-train-positions", type=int, default=32)
    p.add_argument("--bottleneck", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument(
        "--max-eval-tokens",
        type=int,
        default=0,
        help="Maximum evaluated positions per held-out question; 0 = all.",
    )
    return p.parse_args()


def build_h_model(model_path: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-0009H.")

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


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def component_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        if torch.is_tensor(output[0]):
            return output[0]
    raise TypeError(f"Expected tensor-like decoder output, got {type(output).__name__}")


def capture_trace(
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
        output = model.generate(
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

    full_ids = output.detach()
    generated_ids = output[0, prompt_length:].detach().cpu()
    baseline_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    h35_capture: list[torch.Tensor] = []
    attn_capture: list[torch.Tensor] = []
    mlp_capture: list[torch.Tensor] = []
    h36_capture: list[torch.Tensor] = []

    layer = model.model.layers[35]

    def capture_h35(_module: Any, _inputs: Any, output_value: Any) -> None:
        hidden = component_tensor(output_value)
        h35_capture.append(hidden.detach().float().cpu())

    def capture_attn(_module: Any, _inputs: Any, output_value: Any) -> None:
        attn_capture.append(component_tensor(output_value).detach().float().cpu())

    def capture_mlp(_module: Any, _inputs: Any, output_value: Any) -> None:
        mlp_capture.append(component_tensor(output_value).detach().float().cpu())

    def capture_h36(_module: Any, _inputs: Any, output_value: Any) -> None:
        h36_capture.append(component_tensor(output_value).detach().float().cpu())

    hooks = [
        model.model.layers[34].register_forward_hook(capture_h35),
        layer.self_attn.register_forward_hook(capture_attn),
        layer.mlp.register_forward_hook(capture_mlp),
        layer.register_forward_hook(capture_h36),
    ]

    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                use_cache=False,
                output_hidden_states=True,
            )
    finally:
        for handle in hooks:
            handle.remove()

    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) <= 35:
        raise RuntimeError("Model did not expose hidden states through layer 35.")

    for name, values in (
        ("H35", h35_capture),
        ("H36 attention", attn_capture),
        ("H36 MLP", mlp_capture),
        ("H36", h36_capture),
    ):
        if len(values) != 1:
            raise RuntimeError(f"Expected one {name} capture, got {len(values)}.")

    positions = torch.arange(
        prompt_length - 1,
        full_ids.shape[-1] - 1,
    )
    input_ids = full_ids[0].detach().cpu()

    trace = Trace(
        index=index,
        expected=expected,
        baseline_predicted=extract_predicted(baseline_text),
        baseline_correct=extract_predicted(baseline_text) == expected,
        prompt_length=prompt_length,
        input_ids=input_ids,
        source_h35=h35_capture[0][0],
        attention_output_h36=attn_capture[0][0],
        mlp_output_h36=mlp_capture[0][0],
        final_h36=h36_capture[0][0],
        positions=positions.cpu(),
        next_token_ids=input_ids[positions + 1],
        generated_tokens=int(generated_ids.numel()),
        baseline_seconds=elapsed,
    )

    del outputs, hidden_states, full_ids, output, inputs
    torch.cuda.empty_cache()
    return trace


class Probe(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(self.norm(source))))


class ComponentInjector:
    def __init__(
        self,
        module: Any,
        positions: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self.positions = positions.to("cuda:0")
        self.values = values.to("cuda:0")
        self.handle = module.register_forward_hook(self.hook)

    def hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        def replace(tensor: torch.Tensor) -> torch.Tensor:
            cloned = tensor.clone()
            values = self.values.to(
                device=cloned.device,
                dtype=cloned.dtype,
            )
            cloned[:, self.positions, :] = values
            return cloned

        if torch.is_tensor(output):
            return replace(output)

        if isinstance(output, tuple):
            values = list(output)
            values[0] = replace(values[0])
            return tuple(values)

        if isinstance(output, list):
            values = list(output)
            values[0] = replace(values[0])
            return values

        raise TypeError(
            f"Unexpected component output: {type(output).__name__}"
        )

    def remove(self) -> None:
        self.handle.remove()


def choose_positions(trace: Trace, max_positions: int) -> torch.Tensor:
    count = int(trace.positions.numel())
    if max_positions <= 0 or max_positions >= count:
        return torch.arange(count)
    selected = torch.linspace(
        0,
        count - 1,
        steps=max_positions,
    ).round().long()
    return torch.unique(selected, sorted=True)


def representation_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    pred = prediction.float()
    tgt = target.float()
    return {
        "cosine_mean": float(
            F.cosine_similarity(pred, tgt, dim=-1).mean().item()
        ),
        "relative_error_mean": float(
            (
                torch.linalg.vector_norm(pred - tgt, dim=-1)
                / torch.linalg.vector_norm(tgt, dim=-1).clamp_min(1e-8)
            ).mean().item()
        ),
        "mse_mean": float(F.mse_loss(pred, tgt).item()),
    }


def distribution_metrics(
    oracle: torch.Tensor,
    candidate: torch.Tensor,
    target_ids: torch.Tensor,
) -> dict[str, float]:
    oracle = oracle.float()
    candidate = candidate.float()
    target = target_ids.to(
        device=oracle.device,
        dtype=torch.long,
    ).unsqueeze(-1)

    oracle_logp = F.log_softmax(oracle, dim=-1)
    candidate_logp = F.log_softmax(candidate, dim=-1)
    oracle_probs = oracle_logp.exp()

    oracle_top = oracle.argmax(dim=-1)
    candidate_top = candidate.argmax(dim=-1)
    oracle_target = oracle_probs.gather(-1, target).squeeze(-1)
    candidate_target = candidate_logp.exp().gather(-1, target).squeeze(-1)
    kl = (
        oracle_probs * (oracle_logp - candidate_logp)
    ).sum(dim=-1)

    return {
        "token_count": float(target_ids.numel()),
        "top1_agreement": float(
            (oracle_top == candidate_top).float().mean().item()
        ),
        "target_probability_ratio_mean": float(
            (
                candidate_target
                / oracle_target.clamp_min(1e-20)
            ).mean().item()
        ),
        "target_log_probability_delta_mean": float(
            (
                candidate_logp.gather(-1, target).squeeze(-1)
                - oracle_logp.gather(-1, target).squeeze(-1)
            ).mean().item()
        ),
        "kl_oracle_to_candidate_mean": float(
            kl.mean().item()
        ),
        "logit_l2_mean": float(
            torch.linalg.vector_norm(
                oracle - candidate,
                dim=-1,
            ).mean().item()
        ),
    }


def evaluate_probe(
    model: Any,
    trace: Trace,
    probe: Probe,
    component: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    positions_idx = choose_positions(trace, args.max_eval_tokens or 0)
    positions = trace.positions[positions_idx]
    source = trace.source_h35[positions_idx]
    if component == "attention":
        target = trace.attention_output_h36[positions_idx]
        module = model.model.layers[35].self_attn
    elif component == "mlp":
        target = trace.mlp_output_h36[positions_idx]
        module = model.model.layers[35].mlp
    else:
        raise ValueError(f"Unknown component {component}")

    input_ids = trace.input_ids.unsqueeze(0).to("cuda:0")
    attention_mask = torch.ones_like(input_ids)
    next_ids = trace.next_token_ids[positions_idx]

    with torch.inference_mode():
        oracle = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits[0, positions.to("cuda:0")].float()

    # Re-run the untouched model once before component replay. This detects
    # CUDA attention nondeterminism separately from hook-boundary errors.
    with torch.inference_mode():
        repeat_oracle = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits[0, positions.to("cuda:0")].float()

    repeat_metrics = distribution_metrics(
        oracle,
        repeat_oracle,
        next_ids,
    )
    print(
        f"    repeatability_control: "
        f"KL={repeat_metrics['kl_oracle_to_candidate_mean']:.8f} "
        f"top1={repeat_metrics['top1_agreement']:.6f}"
    )
    if (
        repeat_metrics["top1_agreement"] < 0.999999
        or repeat_metrics["kl_oracle_to_candidate_mean"] > 1e-5
    ):
        raise RuntimeError(
            f"Baseline forward repeatability failed for question {trace.index}: "
            f"top1={repeat_metrics['top1_agreement']:.6f}, "
            f"KL={repeat_metrics['kl_oracle_to_candidate_mean']:.8f}"
        )

    # Re-capture the untouched component immediately before replay. This
    # distinguishes stored-capture mismatch from injector semantics.
    fresh_capture: list[torch.Tensor] = []

    def capture_fresh(_module: Any, _inputs: Any, output_value: Any) -> None:
        fresh_capture.append(
            component_tensor(output_value).detach().float().cpu()
        )

    fresh_handle = module.register_forward_hook(capture_fresh)
    try:
        with torch.inference_mode():
            fresh_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
    finally:
        fresh_handle.remove()

    if len(fresh_capture) != 1:
        raise RuntimeError(
            f"Expected one fresh {component} capture, got {len(fresh_capture)}."
        )

    fresh_component = fresh_capture[0][0][positions_idx]
    capture_diff = fresh_component - target
    capture_l2 = torch.linalg.vector_norm(
        capture_diff,
        dim=-1,
    )
    capture_metrics = {
        "max_abs": float(capture_diff.abs().max().item()),
        "mean_abs": float(capture_diff.abs().mean().item()),
        "l2_mean": float(capture_l2.mean().item()),
    }
    print(
        f"    stored_capture_vs_fresh: "
        f"max_abs={capture_metrics['max_abs']:.6f} "
        f"mean_abs={capture_metrics['mean_abs']:.6f} "
        f"l2_mean={capture_metrics['l2_mean']:.6f}"
    )

    del fresh_outputs, fresh_capture

    exact_injector = ComponentInjector(module, positions, target)
    try:
        with torch.inference_mode():
            exact = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[0, positions.to("cuda:0")].float()
    finally:
        exact_injector.remove()

    exact_metrics = distribution_metrics(oracle, exact, next_ids)
    if (
        exact_metrics["top1_agreement"] < 0.999999
        or exact_metrics["kl_oracle_to_candidate_mean"] > 1e-5
    ):
        raise RuntimeError(
            f"Exact {component} injection control failed for question {trace.index}: "
            f"top1={exact_metrics['top1_agreement']:.6f}, "
            f"KL={exact_metrics['kl_oracle_to_candidate_mean']:.8f}"
        )

    probe.eval()
    with torch.inference_mode():
        predicted_all = probe(trace.source_h35.unsqueeze(0).to("cuda:0"))[0]
        predicted = predicted_all[positions_idx].float()

    prediction_metrics = representation_metrics(predicted.cpu(), target)

    learned_injector = ComponentInjector(module, positions, predicted)
    try:
        with torch.inference_mode():
            candidate = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[0, positions.to("cuda:0")].float()
    finally:
        learned_injector.remove()

    learned_metrics = distribution_metrics(oracle, candidate, next_ids)

    zero = torch.zeros_like(target)
    zero_injector = ComponentInjector(module, positions, zero)
    try:
        with torch.inference_mode():
            zero_candidate = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[0, positions.to("cuda:0")].float()
    finally:
        zero_injector.remove()

    zero_metrics = distribution_metrics(oracle, zero_candidate, next_ids)

    return {
        "question_index": trace.index,
        "component": component,
        "evaluated_pairs": len(positions_idx),
        "representation": prediction_metrics,
        "repeatability_control": repeat_metrics,
        "stored_capture_vs_fresh": capture_metrics,
        "exact_injection_control": exact_metrics,
        "predicted_component_behavior": learned_metrics,
        "zero_component_behavior": zero_metrics,
    }


def train_probe(
    component: str,
    model: Any,
    traces: list[Trace],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Probe, dict[str, Any]]:
    hidden_size = traces[0].source_h35.shape[-1]
    if component == "attention":
        targets = [trace.attention_output_h36 for trace in traces]
    elif component == "mlp":
        targets = [trace.mlp_output_h36 for trace in traces]
    else:
        raise ValueError(f"Unknown component {component}")

    probe = Probe(hidden_size, args.bottleneck).to(device=device, dtype=torch.float32)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        probe.train()
        total = 0.0

        for trace in traces:
            positions_idx = choose_positions(
                trace,
                args.max_train_positions,
            )
            source = trace.source_h35[positions_idx].to(device)
            target = targets[traces.index(trace)][positions_idx].to(device)

            predicted = probe(source)
            normalized_mse = (
                F.mse_loss(predicted, target)
                / target.square().mean().clamp_min(1e-8)
            )
            cosine = 1.0 - F.cosine_similarity(
                predicted,
                target,
                dim=-1,
            ).mean()
            loss = normalized_mse + 0.05 * cosine

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())

        mean_loss = total / max(len(traces), 1)
        history.append({"epoch": float(epoch), "mean_loss": mean_loss})
        print(
            f"    epoch {epoch:02d}/{args.epochs}: mean_loss={mean_loss:.6f}"
        )

    return probe, {
        "training_seconds": time.perf_counter() - started,
        "epochs": args.epochs,
        "max_train_positions": args.max_train_positions,
        "parameter_count": float(sum(p.numel() for p in probe.parameters())),
        "history": history,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-0009H.")

    if args.questions <= 1:
        raise ValueError("--questions must be greater than 1.")
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
    print("Probe bottleneck:", args.bottleneck)

    print("\nLoading established 4-bit NF4 model...")
    tokenizer, model = build_h_model(args.model)

    if len(model.model.layers) < 36:
        raise RuntimeError("EXP-0009H requires at least 36 decoder layers.")

    traces: list[Trace] = []
    for index, row in enumerate(rows, 1):
        print(f"\n[{index}/{args.questions}] baseline + sublayer capture...")
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
    eval_traces = traces[args.train_questions :]

    results: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []

    for component in ("attention", "mlp"):
        print(f"\n=== layer36_{component} ===")
        probe, meta = train_probe(
            component,
            model,
            train_traces,
            args,
            torch.device("cuda:0"),
        )
        training.append({"component": component, **meta})

        for trace in eval_traces:
            print(f"  evaluating held-out question {trace.index}...")
            result = evaluate_probe(
                model,
                trace,
                probe,
                component,
                args,
            )
            results.append(result)

            r = result["representation"]
            b = result["predicted_component_behavior"]
            z = result["zero_component_behavior"]
            e = result["exact_injection_control"]
            print(
                f"    repr_cos={r['cosine_mean']:.4f} "
                f"repr_rel={r['relative_error_mean']:.4f} "
                f"KL={b['kl_oracle_to_candidate_mean']:.4f} "
                f"top1={b['top1_agreement']:.4f} "
                f"target_ratio={b['target_probability_ratio_mean']:.4f}"
            )
            print(
                f"    exact_control: KL={e['kl_oracle_to_candidate_mean']:.6f} "
                f"top1={e['top1_agreement']:.6f}"
            )
            print(
                f"    zero_component: KL={z['kl_oracle_to_candidate_mean']:.4f} "
                f"top1={z['top1_agreement']:.4f}"
            )

        del probe
        torch.cuda.empty_cache()

    by_component: dict[str, dict[str, float]] = {}
    for component in ("attention", "mlp"):
        subset = [r for r in results if r["component"] == component]
        if subset:
            by_component[component] = {
                "repr_cosine_mean": round(
                    sum(r["representation"]["cosine_mean"] for r in subset)
                    / len(subset),
                    8,
                ),
                "repr_relative_error_mean": round(
                    sum(r["representation"]["relative_error_mean"] for r in subset)
                    / len(subset),
                    8,
                ),
                "predicted_kl_mean": round(
                    sum(r["predicted_component_behavior"]["kl_oracle_to_candidate_mean"] for r in subset)
                    / len(subset),
                    8,
                ),
                "predicted_top1_mean": round(
                    sum(r["predicted_component_behavior"]["top1_agreement"] for r in subset)
                    / len(subset),
                    8,
                ),
                "predicted_target_ratio_mean": round(
                    sum(r["predicted_component_behavior"]["target_probability_ratio_mean"] for r in subset)
                    / len(subset),
                    8,
                ),
                "zero_component_kl_mean": round(
                    sum(r["zero_component_behavior"]["kl_oracle_to_candidate_mean"] for r in subset)
                    / len(subset),
                    8,
                ),
                "zero_component_top1_mean": round(
                    sum(r["zero_component_behavior"]["top1_agreement"] for r in subset)
                    / len(subset),
                    8,
                ),
            }

    summary = {
        "schema_version": 1,
        "experiment": "EXP-0009H",
        "title": "Layer-36 Sublayer Predictability Diagnostic",
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
            "epochs": args.epochs,
            "max_train_positions": args.max_train_positions,
            "bottleneck": args.bottleneck,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "baseline": {
            "accuracy": round(
                sum(trace.baseline_correct for trace in traces) / len(traces),
                8,
            ),
            "mean_generation_seconds": round(
                sum(trace.baseline_seconds for trace in traces) / len(traces),
                8,
            ),
            "mean_generated_tokens": round(
                sum(trace.generated_tokens for trace in traces) / len(traces),
                8,
            ),
        },
        "training": training,
        "by_component": by_component,
        "results": results,
    }

    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (args.output / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")

    print("\n=== EXP-0009H summary ===")
    print(json.dumps(by_component, indent=2))
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
