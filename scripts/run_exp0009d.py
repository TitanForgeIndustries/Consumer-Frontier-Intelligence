"""CFI EXP-0009D: behaviorally trained latent-state replacement.

EXP-0009C showed that MSE/cosine hidden-state reconstruction can improve
geometric state similarity while making downstream behavior worse than simple
source-state copying.

EXP-0009D changes the training objective. The predictor starts as identity-plus-
residual and is trained primarily against a frozen downstream teacher using
its top-k next-token distribution. Hidden-state losses remain small auxiliary
terms.

This is still a teacher-forced substitution experiment. It does not claim a
hardware speedup.
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
    build_model,
    extract_expected,
    extract_predicted,
    format_prompt,
    load_rows,
)


DEFAULT_OUTPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009D-Behavioral-Latent-Predictor"
)

RELATIONS = {
    "h30_to_h35": (30, 35),
    "h35_to_h36": (35, 36),
    "h30_to_h36": (30, 36),
}


@dataclass
class Trace:
    index: int
    expected: str | None
    baseline_predicted: str | None
    baseline_correct: bool
    prompt_length: int
    input_ids: torch.Tensor
    states: dict[int, torch.Tensor]
    teacher_positions: torch.Tensor
    teacher_top_ids: torch.Tensor
    teacher_top_probs: torch.Tensor
    generated_tokens: int
    baseline_seconds: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CFI EXP-0009D.")
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--questions", type=int, default=6)
    p.add_argument("--train-questions", type=int, default=4)
    p.add_argument(
        "--relations",
        nargs="+",
        choices=tuple(RELATIONS),
        default=list(RELATIONS),
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed-base", type=int, default=42000)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--teacher-top-k", type=int, default=16)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--max-train-positions", type=int, default=32)
    p.add_argument("--bottleneck", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--state-cosine-weight", type=float, default=0.05)
    p.add_argument("--state-mse-weight", type=float, default=0.01)
    p.add_argument(
        "--max-eval-tokens",
        type=int,
        default=0,
        help="Maximum target positions evaluated per held-out question; 0 = all.",
    )
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    if args.model is None:
        from run_exp0008 import DEFAULT_MODEL

        args.model = DEFAULT_MODEL
    if args.dataset is None:
        from run_exp0008 import DEFAULT_DATASET

        args.dataset = DEFAULT_DATASET


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_layer_states(
    model: Any,
    full_ids: torch.Tensor,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    captured: dict[int, list[torch.Tensor]] = {30: [], 35: [], 36: []}
    handles = []

    for layer in (30, 35):
        def make_hook(target_layer: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                if torch.is_tensor(output):
                    tensor = output
                elif isinstance(output, (tuple, list)) and output:
                    tensor = output[0]
                else:
                    raise TypeError(
                        f"Unexpected layer-{target_layer} output: "
                        f"{type(output).__name__}"
                    )
                captured[target_layer].append(
                    tensor.detach().float().cpu()
                )
            return hook

        handles.append(
            model.model.layers[layer - 1].register_forward_hook(
                make_hook(layer)
            )
        )

    def hook36(_module: Any, _inputs: Any, output: Any) -> None:
        if torch.is_tensor(output):
            tensor = output
        elif isinstance(output, (tuple, list)) and output:
            tensor = output[0]
        else:
            raise TypeError(
                f"Unexpected layer-36 output: {type(output).__name__}"
            )
        captured[36].append(tensor.detach().float().cpu())

    handles.append(model.model.layers[35].register_forward_hook(hook36))

    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                use_cache=False,
                output_hidden_states=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) <= 35:
        raise RuntimeError("Model did not expose hidden states through layer 35.")

    states = {
        30: hidden_states[30][0].float().cpu(),
        35: hidden_states[35][0].float().cpu(),
        36: captured[36][0][0],
    }
    return states, outputs.logits.detach()


def collect_trace(
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

    states, logits = capture_layer_states(model, full_ids)

    # Measure the same next-token positions used by the downstream evaluator.
    positions = torch.arange(
        prompt_length - 1,
        full_ids.shape[-1] - 1,
    )
    teacher_logits = logits[0, positions].float()
    top_k = min(args.teacher_top_k, teacher_logits.shape[-1])
    top_values, top_ids = torch.topk(
        teacher_logits,
        k=top_k,
        dim=-1,
    )
    top_probs = torch.softmax(top_values, dim=-1).cpu()

    input_ids = full_ids[0].cpu()
    del logits, states[36], full_ids, output, inputs
    torch.cuda.empty_cache()

    predicted = extract_predicted(baseline_text)
    return Trace(
        index=index,
        expected=expected,
        baseline_predicted=predicted,
        baseline_correct=predicted == expected,
        prompt_length=prompt_length,
        input_ids=input_ids,
        states=states,
        teacher_positions=positions.cpu(),
        teacher_top_ids=top_ids.cpu(),
        teacher_top_probs=top_probs,
        generated_tokens=int(generated_ids.numel()),
        baseline_seconds=elapsed,
    )


class Predictor(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size)

        # Zero residual at initialization means the first predictor state is
        # exactly the source-state copy baseline.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        update = self.up(self.act(self.down(self.norm(source))))
        return source + update


class Injector:
    def __init__(
        self,
        model: Any,
        layer: int,
        positions: torch.Tensor,
        states: torch.Tensor,
    ) -> None:
        self.positions = positions.to("cuda:0")
        self.states = states.to("cuda:0")
        self.handle = model.model.layers[layer - 1].register_forward_hook(
            self.hook
        )

    def hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        def replace(tensor: torch.Tensor) -> torch.Tensor:
            cloned = tensor.clone()
            states = self.states.to(
                device=cloned.device,
                dtype=cloned.dtype,
            )
            cloned[:, self.positions, :] = states
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
            f"Unexpected decoder layer output: {type(output).__name__}"
        )

    def remove(self) -> None:
        self.handle.remove()


def relation_states(
    trace: Trace,
    relation: str,
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    source_layer, target_layer = RELATIONS[relation]
    source = trace.states[source_layer]
    target = trace.states[target_layer]
    positions = trace.teacher_positions
    return (
        source_layer,
        target_layer,
        source[positions],
        target[positions],
    )


def behavioral_loss(
    candidate_logits: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_probs: torch.Tensor,
) -> torch.Tensor:
    logp = F.log_softmax(candidate_logits.float(), dim=-1)
    ids = teacher_ids.to(device=logp.device, dtype=torch.long)
    probs = teacher_probs.to(device=logp.device, dtype=logp.dtype)
    selected = logp.gather(-1, ids)
    return -(probs * selected).sum(dim=-1).mean()


def choose_positions(trace: Trace, max_positions: int) -> torch.Tensor:
    count = int(trace.teacher_positions.numel())
    if max_positions <= 0 or max_positions >= count:
        return torch.arange(count)
    keep = torch.linspace(
        0,
        count - 1,
        steps=max_positions,
    ).round().long()
    return torch.unique(keep, sorted=True)


def train_predictor(
    relation: str,
    model: Any,
    traces: list[Trace],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Predictor, dict[str, Any]]:
    source_layer, target_layer, _, _ = relation_states(
        traces[0],
        relation,
    )
    hidden_size = traces[0].states[source_layer].shape[-1]

    predictor = Predictor(
        hidden_size=hidden_size,
        bottleneck=args.bottleneck,
    ).to(device=device, dtype=torch.float32)

    # The base model is a frozen teacher. We still allow gradients with
    # respect to the predictor-generated hidden states.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        predictor.train()
        epoch_loss = 0.0

        for trace in traces:
            positions_idx = choose_positions(
                trace,
                args.max_train_positions,
            )
            positions = trace.teacher_positions[positions_idx]
            source = trace.states[source_layer][positions_idx]
            target = trace.states[target_layer][positions_idx]

            teacher_ids = trace.teacher_top_ids[positions_idx]
            teacher_probs = trace.teacher_top_probs[positions_idx]

            source_gpu = source.to(device)
            target_gpu = target.to(device)

            predicted = predictor(source_gpu)
            injector = Injector(
                model,
                target_layer,
                positions,
                predicted,
            )
            try:
                # Gradient must remain enabled through the frozen downstream
                # network so the behavioral loss can shape the predictor.
                candidate = model(
                    input_ids=trace.input_ids.unsqueeze(0).to(device),
                    attention_mask=torch.ones(
                        (1, trace.input_ids.shape[0]),
                        dtype=torch.long,
                        device=device,
                    ),
                    use_cache=False,
                ).logits[0, positions.to(device)].float()

                behavior = behavioral_loss(
                    candidate,
                    teacher_ids,
                    teacher_probs,
                )
                cosine_loss = 1.0 - F.cosine_similarity(
                    predicted,
                    target_gpu,
                    dim=-1,
                ).mean()
                denom = target_gpu.square().mean().clamp_min(1e-8)
                mse_loss = F.mse_loss(
                    predicted,
                    target_gpu,
                ) / denom

                loss = (
                    behavior
                    + args.state_cosine_weight * cosine_loss
                    + args.state_mse_weight * mse_loss
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    predictor.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()
                epoch_loss += float(loss.item())
            finally:
                injector.remove()
                del candidate, predicted
                torch.cuda.empty_cache()

        mean_loss = epoch_loss / max(len(traces), 1)
        history.append(
            {
                "epoch": float(epoch),
                "mean_training_loss": mean_loss,
            }
        )
        print(
            f"    epoch {epoch:02d}/{args.epochs}: "
            f"mean_loss={mean_loss:.6f}"
        )

    elapsed = time.perf_counter() - started
    return predictor, {
        "training_seconds": elapsed,
        "epochs": args.epochs,
        "max_train_positions": args.max_train_positions,
        "final_mean_training_loss": history[-1]["mean_training_loss"],
        "parameter_count": float(
            sum(parameter.numel() for parameter in predictor.parameters())
        ),
        "history": history,
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
        oracle_probs
        * (oracle_logp - candidate_logp)
    ).sum(dim=-1)

    return {
        "token_count": float(target_ids.numel()),
        "top1_agreement": float(
            (oracle_top == candidate_top).float().mean().item()
        ),
        "oracle_target_probability_mean": float(
            oracle_target.mean().item()
        ),
        "candidate_target_probability_mean": float(
            candidate_target.mean().item()
        ),
        "target_probability_ratio_mean": float(
            (candidate_target / oracle_target.clamp_min(1e-20))
            .mean()
            .item()
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


def evaluate(
    model: Any,
    trace: Trace,
    relation: str,
    predictor: Predictor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_layer, target_layer, source, target = relation_states(
        trace,
        relation,
    )

    count = source.shape[0]
    if count < 1:
        raise RuntimeError(
            f"No evaluation positions for {relation} / "
            f"question {trace.index}."
        )

    max_tokens = args.max_eval_tokens or count
    if max_tokens >= count:
        keep = torch.arange(count)
    else:
        keep = torch.linspace(
            0,
            count - 1,
            steps=max_tokens,
        ).round().long()
        keep = torch.unique(keep, sorted=True)

    positions = trace.teacher_positions[keep]
    source_eval = source[keep]
    target_eval = target[keep]

    input_ids = trace.input_ids.unsqueeze(0).to("cuda:0")
    attention = torch.ones_like(input_ids)

    with torch.inference_mode():
        oracle = model(
            input_ids=input_ids,
            attention_mask=attention,
            use_cache=False,
        ).logits[0, positions.to("cuda:0")].float()

    # Exact-state injection is the semantic control.
    exact_injector = Injector(
        model,
        target_layer,
        positions,
        target_eval,
    )
    try:
        with torch.inference_mode():
            exact = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
            ).logits[0, positions.to("cuda:0")].float()
    finally:
        exact_injector.remove()

    next_ids = trace.input_ids[positions + 1]
    exact_metrics = distribution_metrics(
        oracle,
        exact,
        next_ids,
    )
    if (
        exact_metrics["top1_agreement"] < 0.999999
        or exact_metrics["kl_oracle_to_candidate_mean"] > 1e-5
    ):
        raise RuntimeError(
            "Exact-state injection control failed for "
            f"{relation} / question {trace.index}: "
            f"top1={exact_metrics['top1_agreement']:.6f}, "
            f"KL={exact_metrics['kl_oracle_to_candidate_mean']:.8f}"
        )

    predictor.eval()
    with torch.inference_mode():
        predicted = predictor(
            source_eval.to("cuda:0")
        ).float()

    predictor_injector = Injector(
        model,
        target_layer,
        positions,
        predicted,
    )
    try:
        with torch.inference_mode():
            candidate = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
            ).logits[0, positions.to("cuda:0")].float()
    finally:
        predictor_injector.remove()

    metrics = distribution_metrics(
        oracle,
        candidate,
        next_ids,
    )

    copy_injector = Injector(
        model,
        target_layer,
        positions,
        source_eval,
    )
    try:
        with torch.inference_mode():
            copy_candidate = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
            ).logits[0, positions.to("cuda:0")].float()
    finally:
        copy_injector.remove()

    copy_metrics = distribution_metrics(
        oracle,
        copy_candidate,
        next_ids,
    )

    metrics.update(
        {
            "state_cosine_mean": float(
                F.cosine_similarity(
                    predicted.cpu(),
                    target_eval,
                    dim=-1,
                ).mean().item()
            ),
            "state_relative_error_mean": float(
                (
                    torch.linalg.vector_norm(
                        predicted.cpu() - target_eval,
                        dim=-1,
                    )
                    / torch.linalg.vector_norm(
                        target_eval,
                        dim=-1,
                    ).clamp_min(1e-8)
                ).mean().item()
            ),
            "state_mse_mean": float(
                F.mse_loss(
                    predicted.cpu(),
                    target_eval,
                ).item()
            ),
        }
    )

    return {
        "relation": relation,
        "source_layer": source_layer,
        "target_layer": target_layer,
        "depth_gap": target_layer - source_layer,
        "question_index": trace.index,
        "expected": trace.expected,
        "baseline_predicted": trace.baseline_predicted,
        "baseline_correct": trace.baseline_correct,
        "generated_tokens": trace.generated_tokens,
        "evaluated_pairs": len(keep),
        "metrics": metrics,
        "exact_state_injection_control": exact_metrics,
        "copy_baseline": copy_metrics,
    }


def round_float(value: float) -> float | None:
    return round(value, 8) if math.isfinite(value) else None


def main() -> int:
    args = parse_args()
    resolve_paths(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-0009D.")
    if args.questions <= 1:
        raise ValueError("--questions must be greater than 1.")
    if not 1 <= args.train_questions < args.questions:
        raise ValueError("--train-questions must be >=1 and < --questions.")
    if args.epochs <= 0 or args.bottleneck <= 0:
        raise ValueError("epochs and bottleneck must be positive.")
    if args.max_train_positions <= 0:
        raise ValueError("--max-train-positions must be positive.")
    if args.teacher_top_k <= 0:
        raise ValueError("--teacher-top-k must be positive.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer configuration.")

    seed_all(args.seed_base)
    rows = load_rows(args.dataset, args.questions)
    args.output.mkdir(parents=True, exist_ok=True)

    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", args.model)
    print("Dataset:", args.dataset)
    print("Questions:", args.questions)
    print("Train questions:", args.train_questions)
    print("Relations:", args.relations)

    print("\nLoading established 4-bit NF4 model...")
    tokenizer, model = build_model(args.model)
    total_layers = len(model.model.layers)
    if total_layers < 36:
        raise RuntimeError(
            f"EXP-0009D requires at least 36 layers, found {total_layers}."
        )

    traces: list[Trace] = []
    for index, row in enumerate(rows, 1):
        print(
            f"\n[{index}/{args.questions}] "
            "baseline + state capture..."
        )
        trace = collect_trace(
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

    device = torch.device("cuda:0")
    all_results: list[dict[str, Any]] = []
    training_summary: list[dict[str, Any]] = []
    predictor_paths: list[str] = []

    for relation in args.relations:
        print(f"\n=== {relation} ===")
        predictor, training_meta = train_predictor(
            relation,
            model,
            train_traces,
            args,
            device,
        )

        path = args.output / f"{relation}_behavior_predictor.pt"
        torch.save(
            {
                "schema_version": 1,
                "experiment": "EXP-0009D",
                "relation": relation,
                "source_layer": RELATIONS[relation][0],
                "target_layer": RELATIONS[relation][1],
                "bottleneck": args.bottleneck,
                "behavioral_objective": {
                    "teacher_top_k": args.teacher_top_k,
                    "state_cosine_weight": args.state_cosine_weight,
                    "state_mse_weight": args.state_mse_weight,
                },
                "state_dict": predictor.state_dict(),
            },
            path,
        )
        predictor_paths.append(str(path))
        training_summary.append(
            {
                "relation": relation,
                **{
                    key: (
                        value
                        if key == "history"
                        else round_float(value)
                    )
                    for key, value in training_meta.items()
                },
                "predictor_path": str(path),
            }
        )

        for trace in eval_traces:
            print(
                f"  evaluating held-out question {trace.index}..."
            )
            result = evaluate(
                model,
                trace,
                relation,
                predictor,
                args,
            )
            all_results.append(result)

            m = result["metrics"]
            c = result["copy_baseline"]
            exact = result["exact_state_injection_control"]
            print(
                f"    state_cos={m['state_cosine_mean']:.4f} "
                f"state_rel={m['state_relative_error_mean']:.4f} "
                f"KL={m['kl_oracle_to_candidate_mean']:.4f} "
                f"top1={m['top1_agreement']:.4f} "
                f"target_ratio={m['target_probability_ratio_mean']:.4f}"
            )
            print(
                f"    exact_injection_control: "
                f"KL={exact['kl_oracle_to_candidate_mean']:.6f} "
                f"top1={exact['top1_agreement']:.6f}"
            )
            print(
                f"    copy_baseline: "
                f"KL={c['kl_oracle_to_candidate_mean']:.4f} "
                f"top1={c['top1_agreement']:.4f}"
            )

        del predictor
        torch.cuda.empty_cache()

    metric_names = (
        "state_cosine_mean",
        "state_relative_error_mean",
        "state_mse_mean",
        "top1_agreement",
        "target_probability_ratio_mean",
        "target_log_probability_delta_mean",
        "kl_oracle_to_candidate_mean",
        "logit_l2_mean",
    )
    by_relation: dict[str, dict[str, float]] = {}

    for relation in args.relations:
        subset = [
            result
            for result in all_results
            if result["relation"] == relation
        ]
        if subset:
            by_relation[relation] = {
                metric: round(
                    sum(
                        float(result["metrics"][metric])
                        for result in subset
                    )
                    / len(subset),
                    8,
                )
                for metric in metric_names
            }
            by_relation[relation]["depth_gap"] = (
                RELATIONS[relation][1] - RELATIONS[relation][0]
            )

    summary = {
        "schema_version": 1,
        "experiment": "EXP-0009D",
        "title": "Behaviorally Trained Predictive Latent-State Replacement",
        "status": "completed",
        "method": "downstream_teacher_distillation_with_state_regularization",
        "model_path": str(args.model),
        "dataset_path": str(args.dataset),
        "model_layers": total_layers,
        "questions": len(traces),
        "train_questions": len(train_traces),
        "eval_questions": len(eval_traces),
        "relations": args.relations,
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "seed_base": args.seed_base,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "teacher_top_k": args.teacher_top_k,
            "epochs": args.epochs,
            "max_train_positions": args.max_train_positions,
            "bottleneck": args.bottleneck,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "state_cosine_weight": args.state_cosine_weight,
            "state_mse_weight": args.state_mse_weight,
            "max_eval_tokens": args.max_eval_tokens,
        },
        "baseline": {
            "accuracy": round(
                sum(trace.baseline_correct for trace in traces)
                / len(traces),
                8,
            ),
            "mean_generation_seconds": round(
                sum(trace.baseline_seconds for trace in traces)
                / len(traces),
                8,
            ),
            "mean_generated_tokens": round(
                sum(trace.generated_tokens for trace in traces)
                / len(traces),
                8,
            ),
        },
        "training": training_summary,
        "held_out_by_relation": by_relation,
        "predictor_paths": predictor_paths,
        "question_baselines": [
            {
                "question_index": trace.index,
                "expected": trace.expected,
                "baseline_predicted": trace.baseline_predicted,
                "baseline_correct": trace.baseline_correct,
                "generated_tokens": trace.generated_tokens,
                "baseline_generation_seconds": round(
                    trace.baseline_seconds,
                    8,
                ),
            }
            for trace in traces
        ],
        "interpretation_guidance": [
            "The predictor is initialized as source-state persistence, making copy a direct zero-learning baseline.",
            "Behavioral distillation is the primary training objective.",
            "Hidden-state cosine and MSE are auxiliary regularizers rather than the main target.",
            "Exact-state injection must reproduce the oracle before predictor metrics are interpreted.",
            "Improvement over the copy baseline is the key evidence that behavioral training adds useful information.",
            "The experiment remains teacher-forced and does not establish a hardware speedup.",
        ],
    }

    with (args.output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (args.output / "downstream_results.jsonl").open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in all_results:
            f.write(json.dumps(row) + "\n")

    with (args.output / "training_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(training_summary, f, indent=2)

    with (args.output / "question_baselines.jsonl").open(
        "w",
        encoding="utf-8",
    ) as f:
        for trace in traces:
            f.write(
                json.dumps(
                    {
                        "question_index": trace.index,
                        "expected": trace.expected,
                        "baseline_predicted": trace.baseline_predicted,
                        "baseline_correct": trace.baseline_correct,
                        "prompt_length": trace.prompt_length,
                        "generated_tokens": trace.generated_tokens,
                        "baseline_generation_seconds": round(
                            trace.baseline_seconds,
                            8,
                        ),
                    }
                )
                + "\n"
            )

    print("\n=== EXP-0009D summary ===")
    print(json.dumps(by_relation, indent=2))
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
