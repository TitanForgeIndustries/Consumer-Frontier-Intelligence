"""CFI EXP-0009N: early-exit generalization with matched baselines.

Validates the EXP-0009L anchored early-exit idea on a larger split while keeping
all evaluation conditions matched on the same generated traces.

Compares:
1. full model oracle
2. raw H35 direct exit through existing final norm + LM head
3. learned bounded residual H35 exit
4. functional whole-L36 skip

Behavioral diagnostic only. No runtime speedup claim.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_exp0008 import extract_expected, extract_predicted, format_prompt, load_rows


DEFAULT_MODEL = Path(
    r"E:\Titan Forge Industries\CFI-Data\HuggingFace\hub\models--Qwen--Qwen3-4B-Base\snapshots\906bfd4b4dc7f14ee4320094d8b41684abff8539"
)
DEFAULT_DATASET = Path(
    r"E:\Titan Forge Industries\CFI-Data\Datasets\CFI-Eval-0001-GSM8K\gsm8k_test_100.jsonl"
)
DEFAULT_OUTPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009N-Early-Exit-Generalization"
)


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
    teacher_log_probs: torch.Tensor
    generated_tokens: int
    baseline_seconds: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CFI EXP-0009N.")
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
    p.add_argument("--max-eval-tokens", type=int, default=0)
    p.add_argument("--bottleneck", type=int, default=128)
    p.add_argument("--max-update-ratio", type=float, default=0.5)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=16)
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
        raise RuntimeError("CUDA is required for EXP-0009N.")

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

    generated_ids = generated[0, prompt_length:].detach().cpu()
    baseline_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    full_ids = generated.detach()
    h35_capture: list[torch.Tensor] = []

    def capture_h35(_module: Any, _inputs: Any, output: Any) -> None:
        h35_capture.append(component_tensor(output).detach().float().cpu())

    handle = model.model.layers[34].register_forward_hook(capture_h35)
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                use_cache=False,
            )
        torch.cuda.synchronize()
    finally:
        handle.remove()

    if len(h35_capture) != 1:
        raise RuntimeError(f"Expected one H35 capture, got {len(h35_capture)}.")

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
        teacher_log_probs=teacher_log_probs,
        generated_tokens=int(generated_ids.numel()),
        baseline_seconds=elapsed,
    )

    del inputs, generated, full_ids, outputs, logits
    torch.cuda.empty_cache()
    return trace


class ResidualTransition(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bottleneck: int,
        max_update_ratio: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size)
        self.max_update_ratio = max_update_ratio

        # Identity at initialization: exact raw-H35 exit baseline.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        update = self.up(self.act(self.down(self.norm(source))))
        source_rms = source.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        update = torch.tanh(update / source_rms) * (
            self.max_update_ratio * source_rms
        )
        return source + update


def final_stack(model: Any, hidden_cpu: torch.Tensor) -> torch.Tensor:
    if hidden_cpu.ndim != 2:
        raise RuntimeError(
            f"Expected [positions, hidden] CPU tensor, got {tuple(hidden_cpu.shape)}"
        )

    norm = model.model.norm
    lm_head = model.lm_head
    hidden = hidden_cpu.to(
        device=norm.weight.device,
        dtype=norm.weight.dtype,
    )
    with torch.inference_mode():
        logits = lm_head(norm(hidden))
    torch.cuda.synchronize()
    return logits.detach().float().cpu().contiguous()


def distribution_metrics(
    oracle: torch.Tensor,
    candidate: torch.Tensor,
    target_ids: torch.Tensor,
) -> dict[str, float]:
    oracle = oracle.detach().float().cpu().contiguous()
    candidate = candidate.detach().float().cpu().contiguous()
    target = target_ids.detach().long().cpu().unsqueeze(-1)

    if oracle.shape != candidate.shape:
        raise RuntimeError(
            f"Distribution shape mismatch: oracle={tuple(oracle.shape)} "
            f"candidate={tuple(candidate.shape)}"
        )

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


def train_transition(
    model: Any,
    traces: list[Trace],
    args: argparse.Namespace,
) -> tuple[ResidualTransition, dict[str, Any]]:
    hidden_size = traces[0].h35.shape[-1]
    transition = ResidualTransition(
        hidden_size,
        args.bottleneck,
        args.max_update_ratio,
    ).to(device="cuda:0", dtype=torch.float32)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        transition.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        transition.train()
        total = 0.0

        for trace in traces:
            pos = trace.train_positions
            source = trace.h35[pos].to("cuda:0")
            teacher_log_probs = trace.teacher_log_probs.to("cuda:0")

            predicted_state = transition(source)
            predicted_logits = final_stack(model, predicted_state.detach().float().cpu())

            # Do not build a second full model graph. This initial N implementation
            # trains the compact transition through the frozen output stack only.
            # Gradient through the quantized output stack is intentionally omitted.
            # Therefore the loss below trains the transition indirectly only through
            # a differentiable local surrogate and is not used as a causal claim.
            #
            # For this validation experiment, use representation-free regression
            # to teacher logits as a proxy target instead.
            with torch.inference_mode():
                teacher_logits = torch.stack(
                    [trace.teacher_log_probs[i].float().to("cuda:0")
                     for i in range(teacher_log_probs.shape[0])],
                    dim=0,
                ).cpu()

            student_logp = F.log_softmax(predicted_logits, dim=-1)
            target_logp = teacher_logits.to(student_logp.device)
            loss = F.mse_loss(student_logp, target_logp)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transition.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())

        mean_loss = total / max(len(traces), 1)
        history.append({"epoch": epoch, "mean_loss": mean_loss})
        print(f"    epoch {epoch:02d}/{args.epochs}: mean_loss={mean_loss:.6f}")

    return transition, {
        "training_seconds": time.perf_counter() - started,
        "epochs": args.epochs,
        "max_train_positions": args.max_train_positions,
        "bottleneck": args.bottleneck,
        "max_update_ratio": args.max_update_ratio,
        "parameter_count": sum(p.numel() for p in transition.parameters()),
        "history": history,
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

    traces = []
    for index, row in enumerate(rows, 1):
        print(f"\n[{index}/{args.questions}] baseline + H35/teacher capture...")
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

    # IMPORTANT: this implementation trains through a detached final stack,
    # which cannot propagate gradient into the transition. Refuse rather than
    # silently produce a non-learning experiment.
    raise RuntimeError(
        "EXP-0009N implementation scaffold requires a differentiable frozen "
        "output stack before training. Do not run this scaffold yet."
    )


if __name__ == "__main__":
    raise SystemExit(main())
