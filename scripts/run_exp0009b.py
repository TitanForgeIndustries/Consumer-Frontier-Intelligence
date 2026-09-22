"""CFI EXP-0009B: conditioned predictive latent-state replacement.

This follow-up tests two better-conditioned prediction problems:

1. Same-position depth prediction:
       H30(t) -> H35(t)
       H35(t) -> H36(t)

2. Token-conditioned temporal prediction:
       [H30(t), Emb(token[t+1])] -> H30(t+1)
       [H35(t), Emb(token[t+1])] -> H35(t+1)

The experiment keeps exact-state injection controls and source-state copy
baselines. It is teacher-forced and does not claim a hardware speedup.
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
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009B-Conditioned-Latent-State"
)

RELATIONS = {
    "h30_to_h35_same_position": {
        "source_layer": 30,
        "target_layer": 35,
        "temporal": False,
    },
    "h35_to_h36_same_position": {
        "source_layer": 35,
        "target_layer": 36,
        "temporal": False,
    },
    "h30_plus_token_to_h30_next": {
        "source_layer": 30,
        "target_layer": 30,
        "temporal": True,
    },
    "h35_plus_token_to_h35_next": {
        "source_layer": 35,
        "target_layer": 35,
        "temporal": True,
    },
}


@dataclass
class Trace:
    index: int
    expected: str | None
    baseline_predicted: str | None
    baseline_correct: bool
    prompt_length: int
    input_ids: torch.Tensor
    h30: torch.Tensor
    h35: torch.Tensor
    h36: torch.Tensor
    token_embeddings: torch.Tensor
    generated_tokens: int
    baseline_seconds: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CFI EXP-0009B.")
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
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed-base", type=int, default=42000)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--bottleneck", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hidden-loss-weight", type=float, default=0.1)
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

    start = time.perf_counter()
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
    elapsed = time.perf_counter() - start

    full_ids = output.detach()
    generated_ids = output[0, prompt_length:].detach().cpu()
    baseline_text = tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()

    with torch.inference_mode():
        outputs = model(
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            use_cache=False,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) <= 36:
            raise RuntimeError(
                "Model did not expose hidden states through layer 36."
            )

        h30 = hidden_states[30][0].float().cpu()
        h35 = hidden_states[35][0].float().cpu()
        h36 = hidden_states[36][0].float().cpu()
        token_embeddings = (
            model.get_input_embeddings()(full_ids)[0].float().cpu()
        )

    input_ids = full_ids[0].cpu()

    del outputs, hidden_states, full_ids, output, inputs
    torch.cuda.empty_cache()

    predicted = extract_predicted(baseline_text)
    return Trace(
        index=index,
        expected=expected,
        baseline_predicted=predicted,
        baseline_correct=predicted == expected,
        prompt_length=prompt_length,
        input_ids=input_ids,
        h30=h30,
        h35=h35,
        h36=h36,
        token_embeddings=token_embeddings,
        generated_tokens=int(generated_ids.numel()),
        baseline_seconds=elapsed,
    )


def relation_spec(name: str) -> tuple[int, int, bool]:
    item = RELATIONS[name]
    return item["source_layer"], item["target_layer"], bool(item["temporal"])


def build_pairs(
    trace: Trace,
    relation: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_layer, target_layer, temporal = relation_spec(relation)

    source_by_layer = {30: trace.h30, 35: trace.h35}
    target_by_layer = {30: trace.h30, 35: trace.h35, 36: trace.h36}
    source = source_by_layer[source_layer]
    target = target_by_layer[target_layer]

    if temporal:
        # source at p predicts the target state at p+1.
        start = trace.prompt_length - 1
        stop = target.shape[0] - 2
        if stop <= start:
            empty = source.new_empty((0, source.shape[-1]))
            return empty, target.new_empty((0, target.shape[-1])), empty
        source_positions = torch.arange(start, stop)
        target_positions = source_positions + 1
        token_features = trace.token_embeddings[target_positions]
    else:
        # Same token position: source depth predicts later depth.
        start = trace.prompt_length - 1
        stop = target.shape[0] - 1
        if stop <= start:
            empty = source.new_empty((0, source.shape[-1]))
            return empty, target.new_empty((0, target.shape[-1])), empty
        source_positions = torch.arange(start, stop)
        target_positions = source_positions
        token_features = source.new_empty((source_positions.numel(), 0))

    return (
        source[source_positions],
        target[target_positions],
        token_features,
    )


class Predictor(nn.Module):
    def __init__(
        self,
        source_size: int,
        feature_size: int,
        hidden_size: int,
        bottleneck: int,
    ) -> None:
        super().__init__()
        input_size = source_size + feature_size
        self.norm = nn.LayerNorm(input_size)
        self.down = nn.Linear(input_size, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        source: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([source, features], dim=-1)
        update = self.up(self.act(self.down(self.norm(x))))
        return source + update


def train_predictor(
    relation: str,
    traces: list[Trace],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Predictor, dict[str, float]]:
    source_layer, target_layer, temporal = relation_spec(relation)

    pairs = [build_pairs(trace, relation) for trace in traces]
    xs = [x for x, y, f in pairs if x.shape[0] > 0]
    ys = [y for x, y, f in pairs if y.shape[0] > 0]
    fs = [f for x, y, f in pairs if x.shape[0] > 0]
    if not xs:
        raise RuntimeError(f"No training state pairs for {relation}.")

    x = torch.cat(xs)
    y = torch.cat(ys)
    features = torch.cat(fs)

    model = Predictor(
        source_size=x.shape[-1],
        feature_size=features.shape[-1],
        hidden_size=y.shape[-1],
        bottleneck=args.bottleneck,
    ).to(device=device, dtype=torch.float32)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1000 + source_layer * 100 + target_layer + int(temporal))

    model.train()
    start = time.perf_counter()
    last_loss = math.nan
    for epoch in range(args.epochs):
        order = torch.randperm(x.shape[0], generator=generator)
        total = 0.0
        for offset in range(0, x.shape[0], args.batch_size):
            idx = order[offset : offset + args.batch_size]
            xb = x[idx].to(device)
            fb = features[idx].to(device)
            yb = y[idx].to(device)
            pred = model(xb, fb)
            mse = F.mse_loss(pred, yb)
            cos = 1.0 - F.cosine_similarity(pred, yb, dim=-1).mean()
            loss = mse + args.hidden_loss_weight * cos
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.shape[0]
        last_loss = total / x.shape[0]
        print(
            f"    epoch {epoch + 1:02d}/{args.epochs}: "
            f"loss={last_loss:.6f}"
        )

    elapsed = time.perf_counter() - start
    with torch.inference_mode():
        sample = min(4096, x.shape[0])
        pred = model(
            x[:sample].to(device),
            features[:sample].to(device),
        ).cpu()
        truth = y[:sample]
        train_cos = float(
            F.cosine_similarity(pred, truth, dim=-1).mean().item()
        )
        train_rel = float(
            (
                torch.linalg.vector_norm(pred - truth, dim=-1)
                / torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-8)
            ).mean().item()
        )

    meta = {
        "train_pairs": float(x.shape[0]),
        "final_train_loss": float(last_loss),
        "sample_train_cosine": train_cos,
        "sample_train_relative_error": train_rel,
        "training_seconds": elapsed,
        "parameter_count": float(sum(p.numel() for p in model.parameters())),
    }
    return model, meta


def distribution_metrics(
    oracle: torch.Tensor,
    candidate: torch.Tensor,
    target_ids: torch.Tensor,
) -> dict[str, float]:
    oracle_device = oracle.device
    candidate_device = candidate.device
    if oracle_device != candidate_device:
        raise RuntimeError("Oracle and candidate logits are on different devices.")

    oracle = oracle.float()
    candidate = candidate.float()
    oracle_logp = F.log_softmax(oracle, dim=-1)
    candidate_logp = F.log_softmax(candidate, dim=-1)
    oracle_p = oracle_logp.exp()
    target = target_ids.to(device=oracle.device, dtype=torch.long).unsqueeze(-1)

    oracle_top = oracle.argmax(-1)
    candidate_top = candidate.argmax(-1)
    oracle_target = oracle_p.gather(-1, target).squeeze(-1)
    candidate_target = candidate_logp.exp().gather(-1, target).squeeze(-1)
    kl = (oracle_p * (oracle_logp - candidate_logp)).sum(-1)

    return {
        "token_count": float(target_ids.numel()),
        "top1_agreement": float(
            (oracle_top == candidate_top).float().mean().item()
        ),
        "oracle_target_probability_mean": float(oracle_target.mean().item()),
        "candidate_target_probability_mean": float(candidate_target.mean().item()),
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


class Injector:
    def __init__(
        self,
        model: Any,
        layer: int,
        positions: torch.Tensor,
        states: torch.Tensor,
    ) -> None:
        self.layer = layer
        self.positions = positions.to("cuda:0")
        self.states = states.to("cuda:0")
        self.handle = model.model.layers[layer - 1].register_forward_hook(self.hook)

    def hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        states = self.states
        if torch.is_tensor(output):
            tensor = output.clone()
            states = states.to(device=tensor.device, dtype=tensor.dtype)
            tensor[:, self.positions, :] = states
            return tensor

        if isinstance(output, tuple):
            values = list(output)
            values[0] = values[0].clone()
            states = states.to(
                device=values[0].device,
                dtype=values[0].dtype,
            )
            values[0][:, self.positions, :] = states
            return tuple(values)

        if isinstance(output, list):
            values = list(output)
            values[0] = values[0].clone()
            states = states.to(
                device=values[0].device,
                dtype=values[0].dtype,
            )
            values[0][:, self.positions, :] = states
            return values

        raise TypeError(
            f"Unexpected decoder layer output: {type(output).__name__}"
        )

    def remove(self) -> None:
        self.handle.remove()


def evaluate(
    model: Any,
    trace: Trace,
    relation: str,
    predictor: Predictor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_layer, target_layer, temporal = relation_spec(relation)
    source, target, features = build_pairs(trace, relation)
    if source.shape[0] < 1:
        raise RuntimeError(
            f"No evaluation pairs for {relation} / question {trace.index}."
        )

    count = source.shape[0]
    max_tokens = args.max_eval_tokens or count
    if max_tokens >= count:
        keep = torch.arange(count)
    else:
        keep = torch.linspace(0, count - 1, steps=max_tokens).round().long()
        keep = torch.unique(keep, sorted=True)

    target_positions_all = (
        torch.arange(
            trace.prompt_length - 1,
            trace.input_ids.shape[0] - 1,
        )
        if not temporal
        else torch.arange(
            trace.prompt_length,
            trace.input_ids.shape[0] - 1,
        )
    )
    target_positions = target_positions_all[keep]

    source_eval = source[keep]
    target_eval = target[keep]
    feature_eval = features[keep]

    input_ids = trace.input_ids.unsqueeze(0).to("cuda:0")
    attention = torch.ones_like(input_ids)

    with torch.inference_mode():
        oracle = model(
            input_ids=input_ids,
            attention_mask=attention,
            use_cache=False,
        ).logits[0, target_positions.to("cuda:0")].float()

    predictor.eval()
    with torch.inference_mode():
        pred_states = predictor(
            source_eval.to("cuda:0"),
            feature_eval.to("cuda:0"),
        ).float()

    exact_injector = Injector(
        model,
        target_layer,
        target_positions,
        target_eval,
    )
    try:
        with torch.inference_mode():
            exact_candidate = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
            ).logits[0, target_positions.to("cuda:0")].float()
    finally:
        exact_injector.remove()

    exact_metrics = distribution_metrics(
        oracle,
        exact_candidate,
        trace.input_ids[target_positions + 1],
    )
    if (
        exact_metrics["top1_agreement"] < 0.999999
        or exact_metrics["kl_oracle_to_candidate_mean"] > 1e-5
    ):
        raise RuntimeError(
            "Exact-state injection control failed: "
            f"top1={exact_metrics['top1_agreement']:.6f}, "
            f"KL={exact_metrics['kl_oracle_to_candidate_mean']:.8f}. "
            "Do not interpret predictor metrics until injection semantics are fixed."
        )

    predictor_injector = Injector(
        model,
        target_layer,
        target_positions,
        pred_states,
    )
    try:
        with torch.inference_mode():
            candidate = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
            ).logits[0, target_positions.to("cuda:0")].float()
    finally:
        predictor_injector.remove()

    metrics = distribution_metrics(
        oracle,
        candidate,
        trace.input_ids[target_positions + 1],
    )
    state_cos = F.cosine_similarity(pred_states.cpu(), target_eval, dim=-1)
    state_rel = (
        torch.linalg.vector_norm(pred_states.cpu() - target_eval, dim=-1)
        / torch.linalg.vector_norm(target_eval, dim=-1).clamp_min(1e-8)
    )
    metrics.update(
        {
            "state_cosine_mean": float(state_cos.mean().item()),
            "state_relative_error_mean": float(state_rel.mean().item()),
            "state_mse_mean": float(F.mse_loss(pred_states.cpu(), target_eval).item()),
            "evaluated_pairs": float(len(keep)),
        }
    )

    # The source-state copy is a zero-extra-model baseline for every relation.
    copy_state = source_eval
    copy_injector = Injector(
        model,
        target_layer,
        target_positions,
        copy_state,
    )
    try:
        with torch.inference_mode():
            copy_candidate = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
            ).logits[0, target_positions.to("cuda:0")].float()
    finally:
        copy_injector.remove()

    copy_logits = distribution_metrics(
        oracle,
        copy_candidate,
        trace.input_ids[target_positions + 1],
    )
    copy_state_metrics = {
        "state_cosine_mean": float(
            F.cosine_similarity(copy_state, target_eval, dim=-1).mean().item()
        ),
        "state_relative_error_mean": float(
            (
                torch.linalg.vector_norm(copy_state - target_eval, dim=-1)
                / torch.linalg.vector_norm(target_eval, dim=-1).clamp_min(1e-8)
            ).mean().item()
        ),
        "state_mse_mean": float(F.mse_loss(copy_state, target_eval).item()),
    }

    return {
        "relation": relation,
        "source_layer": source_layer,
        "target_layer": target_layer,
        "temporal": temporal,
        "injection_layer": target_layer,
        "question_index": trace.index,
        "expected": trace.expected,
        "baseline_predicted": trace.baseline_predicted,
        "baseline_correct": trace.baseline_correct,
        "generated_tokens": trace.generated_tokens,
        "evaluated_pairs": len(keep),
        "metrics": metrics,
        "exact_state_injection_control": exact_metrics,
        "copy_baseline": {
            **copy_state_metrics,
            **{
                f"downstream_{key}": value
                for key, value in copy_logits.items()
                if key != "token_count"
            },
        },
    }


def finite_round(value: float) -> float | None:
    return round(value, 8) if math.isfinite(value) else None


def main() -> int:
    args = parse_args()
    resolve_paths(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-0009B.")
    if args.questions <= 1:
        raise ValueError("--questions must be greater than 1.")
    if not 1 <= args.train_questions < args.questions:
        raise ValueError("--train-questions must be >=1 and < --questions.")
    if args.epochs <= 0 or args.batch_size <= 0 or args.bottleneck <= 0:
        raise ValueError("epochs, batch-size, and bottleneck must be positive.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer configuration.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")

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
            f"EXP-0009B requires at least 36 layers, found {total_layers}."
        )

    traces: list[Trace] = []
    for index, row in enumerate(rows, 1):
        print(f"\n[{index}/{args.questions}] baseline + hidden-state capture...")
        trace = collect_trace(model, tokenizer, row, index, args)
        traces.append(trace)
        print(
            f"  expected={trace.expected} predicted={trace.baseline_predicted} "
            f"correct={trace.baseline_correct} tokens={trace.generated_tokens} "
            f"time={trace.baseline_seconds:.2f}s"
        )

    train_traces = traces[: args.train_questions]
    eval_traces = traces[args.train_questions :]
    device = torch.device("cuda:0")
    all_results: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    predictor_paths: list[str] = []

    for relation in args.relations:
        print(f"\n=== {relation} ===")
        predictor, train_meta = train_predictor(
            relation,
            train_traces,
            args,
            device,
        )

        path = args.output / f"{relation}_predictor.pt"
        torch.save(
            {
                "schema_version": 1,
                "experiment": "EXP-0009B",
                "relation": relation,
                "source_layer": RELATIONS[relation]["source_layer"],
                "target_layer": RELATIONS[relation]["target_layer"],
                "temporal": RELATIONS[relation]["temporal"],
                "bottleneck": args.bottleneck,
                "state_dict": predictor.state_dict(),
            },
            path,
        )
        predictor_paths.append(str(path))
        training.append(
            {
                "relation": relation,
                **{k: finite_round(v) for k, v in train_meta.items()},
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
            metrics = result["metrics"]
            print(
                f"    state_cos={metrics['state_cosine_mean']:.4f} "
                f"state_rel={metrics['state_relative_error_mean']:.4f} "
                f"KL={metrics['kl_oracle_to_candidate_mean']:.4f} "
                f"top1={metrics['top1_agreement']:.4f} "
                f"target_ratio={metrics['target_probability_ratio_mean']:.4f}"
            )
            copy = result["copy_baseline"]
            print(
                f"    exact_injection_control: "
                f"KL={result['exact_state_injection_control']['kl_oracle_to_candidate_mean']:.6f} "
                f"top1={result['exact_state_injection_control']['top1_agreement']:.6f}"
            )
            print(
                f"    copy_baseline: "
                f"state_cos={copy['state_cosine_mean']:.4f} "
                f"state_rel={copy['state_relative_error_mean']:.4f} "
                f"KL={copy['downstream_kl_oracle_to_candidate_mean']:.4f} "
                f"top1={copy['downstream_top1_agreement']:.4f}"
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
        subset = [r for r in all_results if r["relation"] == relation]
        if subset:
            by_relation[relation] = {
                name: round(
                    sum(float(r["metrics"][name]) for r in subset) / len(subset),
                    8,
                )
                for name in metric_names
            }

    summary = {
        "schema_version": 1,
        "experiment": "EXP-0009B",
        "title": "Conditioned Predictive Latent-State Replacement",
        "status": "completed",
        "method": "teacher_forced_conditioned_hidden_state_injection",
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
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "bottleneck": args.bottleneck,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_loss_weight": args.hidden_loss_weight,
            "max_eval_tokens": args.max_eval_tokens,
        },
        "baseline": {
            "accuracy": round(
                sum(t.baseline_correct for t in traces) / len(traces), 8
            ),
            "mean_generation_seconds": round(
                sum(t.baseline_seconds for t in traces) / len(traces), 8
            ),
            "mean_generated_tokens": round(
                sum(t.generated_tokens for t in traces) / len(traces), 8
            ),
        },
        "training": training,
        "held_out_by_relation": by_relation,
        "predictor_paths": predictor_paths,
        "question_baselines": [
            {
                "question_index": t.index,
                "expected": t.expected,
                "baseline_predicted": t.baseline_predicted,
                "baseline_correct": t.baseline_correct,
                "generated_tokens": t.generated_tokens,
                "baseline_generation_seconds": round(t.baseline_seconds, 8),
            }
            for t in traces
        ],
        "interpretation_guidance": [
            "Exact-state injection must reproduce the oracle before predictor metrics are trusted.",
            "Same-position depth prediction tests whether downstream transformer computation can be forecast from an earlier layer.",
            "Token-conditioned temporal prediction restores the identity of the next token that drives the state transition.",
            "Source-state copying is a baseline for whether learning adds useful predictive information.",
            "The experiment is teacher-forced and does not establish a hardware speedup.",
            "High state similarity with poor downstream agreement indicates amplification of prediction error by later computation.",
            "A strong downstream result is evidence for a later autoregressive rollout and hardware-timing experiment.",
        ],
    }

    with (args.output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (args.output / "downstream_results.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for row in all_results:
            f.write(json.dumps(row) + "\n")

    with (args.output / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(training, f, indent=2)

    with (args.output / "question_baselines.jsonl").open(
        "w", encoding="utf-8"
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
                            trace.baseline_seconds, 8
                        ),
                    }
                )
                + "\n"
            )

    print("\n=== EXP-0009B summary ===")
    print(json.dumps(by_relation, indent=2))
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
