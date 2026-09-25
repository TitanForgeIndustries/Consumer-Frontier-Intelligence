"""CFI EXP-0008: causal attention/MLP ablation on baseline trajectories."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from cfi_paths import data_path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = data_path("HuggingFace", "hub", "models--Qwen--Qwen3-4B-Base", "snapshots", "906bfd4b4dc7f14ee4320094d8b41684abff8539")
DEFAULT_DATASET = data_path("Datasets", "CFI-Eval-0001-GSM8K", "gsm8k_test_100.jsonl")
DEFAULT_OUTPUT = data_path("Results", "CFI-Eval-0008-Causal-Sublayer-Ablation")
DEFAULT_LAYERS = (30, 35, 36)
DEFAULT_COMPONENTS = ("attention", "mlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CFI EXP-0008 causal sublayer ablations."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAYERS),
        help="1-based decoder layers to ablate.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=("attention", "mlp"),
        default=list(DEFAULT_COMPONENTS),
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed-base", type=int, default=42000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--free-run-questions",
        type=int,
        default=0,
        help="Number of questions for optional free-running ablation generation.",
    )
    return parser.parse_args()


def load_rows(path: Path, count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("--questions must be positive")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Dataset row is not an object: {path}")
            rows.append(row)
            if len(rows) >= count:
                break

    if not rows:
        raise ValueError(f"No questions found in {path}")
    return rows


def format_prompt(question: str) -> str:
    return (
        "Question: "
        + question
        + "\n"
        + "Solve the problem step by step. "
        + "End with: The answer is <number>.\n"
    )


def canonical_number(value: str | None) -> str | None:
    if value is None:
        return None
    raw = value.strip().replace(",", "")
    try:
        number = float(raw)
    except ValueError:
        return raw
    return str(int(number)) if number.is_integer() else str(number)


def extract_expected(text: str) -> str | None:
    import re

    match = re.search(
        r"####\s*(?:<\s*)?\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*>?",
        text,
        re.IGNORECASE,
    )
    return canonical_number(match.group(1)) if match else None


def extract_predicted(text: str) -> str | None:
    import re

    matches = re.findall(
        r"(?:The answer is|####)\s*:?\s*<?\s*\$?\s*"
        r"([-+]?\d[\d,]*(?:\.\d+)?)\s*>?",
        text,
        re.IGNORECASE,
    )
    if matches:
        return canonical_number(matches[-1])

    for line in reversed(
        [line.strip() for line in text.splitlines() if line.strip()]
    ):
        if re.fullmatch(r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?", line):
            return canonical_number(line.replace("$", "").strip())
    return None


def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value):
                return value
    raise TypeError(f"Expected tensor-like output, got {type(output).__name__}")


def zero_sublayer_output(output: Any) -> Any:
    if torch.is_tensor(output):
        return torch.zeros_like(output)

    if isinstance(output, tuple):
        values = list(output)
        if not values or not torch.is_tensor(values[0]):
            raise TypeError("Sublayer tuple has no tensor at position 0.")
        values[0] = torch.zeros_like(values[0])
        return tuple(values)

    if isinstance(output, list):
        values = list(output)
        if not values or not torch.is_tensor(values[0]):
            raise TypeError("Sublayer list has no tensor at position 0.")
        values[0] = torch.zeros_like(values[0])
        return values

    raise TypeError(
        f"Cannot ablate sublayer output of type {type(output).__name__}"
    )


@dataclass(frozen=True)
class Ablation:
    layer: int
    component: str

    @property
    def name(self) -> str:
        return f"layer_{self.layer}_{self.component}"


class Ablator:
    def __init__(self, model, ablation: Ablation | None) -> None:
        self.model = model
        self.ablation = ablation
        self.handle = None

    def _hook(self, _module, _inputs, output):
        return zero_sublayer_output(output)

    def install(self) -> None:
        if self.ablation is None:
            return

        layers = self.model.model.layers
        if not 1 <= self.ablation.layer <= len(layers):
            raise ValueError(
                f"Invalid layer {self.ablation.layer}; model has {len(layers)} layers."
            )

        layer = layers[self.ablation.layer - 1]
        if self.ablation.component == "attention":
            module = layer.self_attn
        elif self.ablation.component == "mlp":
            module = layer.mlp
        else:
            raise ValueError(f"Unknown component: {self.ablation.component}")

        self.handle = module.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def build_model(model_path: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-0008.")

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
    )
    model.eval()
    return tokenizer, model


def next_token_metrics(
    baseline_logits: torch.Tensor,
    counterfactual_logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> dict[str, float]:
    baseline = baseline_logits.float()
    counterfactual = counterfactual_logits.float()

    baseline_log_probs = F.log_softmax(baseline, dim=-1)
    counter_log_probs = F.log_softmax(counterfactual, dim=-1)
    baseline_probs = baseline_log_probs.exp()
    counter_probs = counter_log_probs.exp()

    target = target_ids.long().unsqueeze(-1)

    baseline_top1 = baseline.argmax(dim=-1)
    counter_top1 = counterfactual.argmax(dim=-1)

    baseline_target_prob = baseline_probs.gather(-1, target).squeeze(-1)
    counter_target_prob = counter_probs.gather(-1, target).squeeze(-1)

    kl = (
        baseline_probs
        * (baseline_log_probs - counter_log_probs)
    ).sum(dim=-1)

    target_log_prob_delta = (
        counter_log_probs.gather(-1, target).squeeze(-1)
        - baseline_log_probs.gather(-1, target).squeeze(-1)
    )

    return {
        "token_count": float(target_ids.numel()),
        "top1_agreement": float(
            (baseline_top1 == counter_top1).float().mean().item()
        ),
        "baseline_target_probability_mean": float(
            baseline_target_prob.mean().item()
        ),
        "counterfactual_target_probability_mean": float(
            counter_target_prob.mean().item()
        ),
        "target_probability_ratio_mean": float(
            (
                counter_target_prob
                / baseline_target_prob.clamp_min(1e-20)
            )
            .mean()
            .item()
        ),
        "target_log_probability_delta_mean": float(
            target_log_prob_delta.mean().item()
        ),
        "kl_baseline_to_counterfactual_mean": float(
            kl.mean().item()
        ),
        "logit_l2_mean": float(
            torch.linalg.vector_norm(
                baseline - counterfactual,
                dim=-1,
            )
            .mean()
            .item()
        ),
    }


def run_teacher_forced(
    model,
    tokenizer,
    full_ids: torch.Tensor,
    prompt_length: int,
    ablation: Ablation | None,
) -> tuple[dict[str, float], float]:
    inputs = {
        "input_ids": full_ids,
        "attention_mask": torch.ones_like(full_ids),
    }

    ablator = Ablator(model, ablation)
    ablator.install()
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        ablator.remove()

    return outputs.logits.float(), elapsed


def run_generation(
    tokenizer,
    model,
    prompt: str,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    ablation: Ablation | None,
) -> tuple[torch.Tensor, str, float]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    prompt_length = int(inputs["input_ids"].shape[-1])

    ablator = Ablator(model, ablation)
    ablator.install()
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=["\nQuestion:", "\nProblem:"],
                tokenizer=tokenizer,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        ablator.remove()

    generated = output[0, prompt_length:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return generated.detach().cpu(), text, elapsed


def summarize_question(
    tokenizer,
    model,
    row: dict[str, Any],
    question_index: int,
    args: argparse.Namespace,
    ablations: list[Ablation],
    free_run: bool,
) -> dict[str, Any]:
    prompt = format_prompt(str(row["question"]))
    expected = extract_expected(str(row["answer"]))

    baseline_tokens, baseline_text, baseline_time = run_generation(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        seed=args.seed_base + question_index,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        ablation=None,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda:0")
    prompt_length = int(inputs["input_ids"].shape[-1])
    full_ids = torch.cat(
        [
            inputs["input_ids"],
            baseline_tokens.to("cuda:0").unsqueeze(0),
        ],
        dim=-1,
    )

    baseline_logits, baseline_teacher_time = run_teacher_forced(
        model=model,
        tokenizer=tokenizer,
        full_ids=full_ids,
        prompt_length=prompt_length,
        ablation=None,
    )

    target_positions = torch.arange(
        prompt_length - 1,
        full_ids.shape[-1] - 1,
        device=full_ids.device,
    )
    target_ids = full_ids[0, prompt_length:]
    baseline_slice = baseline_logits[0, target_positions]

    ablation_results: list[dict[str, Any]] = []

    for ablation in ablations:
        counter_logits, teacher_time = run_teacher_forced(
            model=model,
            tokenizer=tokenizer,
            full_ids=full_ids,
            prompt_length=prompt_length,
            ablation=ablation,
        )
        counter_slice = counter_logits[0, target_positions]
        metrics = next_token_metrics(
            baseline_logits=baseline_slice,
            counterfactual_logits=counter_slice,
            target_ids=target_ids,
        )

        record: dict[str, Any] = {
            "ablation": ablation.name,
            "layer": ablation.layer,
            "component": ablation.component,
            "question_index": question_index,
            "expected": expected,
            "baseline_predicted": extract_predicted(baseline_text),
            "baseline_correct": extract_predicted(baseline_text) == expected,
            "baseline_generated_tokens": int(baseline_tokens.numel()),
            "baseline_generation_seconds": round(baseline_time, 6),
            "baseline_teacher_seconds": round(baseline_teacher_time, 6),
            "counterfactual_teacher_seconds": round(teacher_time, 6),
            **metrics,
        }

        if free_run:
            (
                cf_tokens,
                cf_text,
                cf_time,
            ) = run_generation(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                seed=args.seed_base + question_index,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                ablation=ablation,
            )
            cf_predicted = extract_predicted(cf_text)
            shared = min(
                int(baseline_tokens.numel()),
                int(cf_tokens.numel()),
            )
            if shared:
                same_prefix = float(
                    (
                        baseline_tokens[:shared] == cf_tokens[:shared]
                    ).float().mean().item()
                )
            else:
                same_prefix = 0.0

            record.update(
                {
                    "free_run_predicted": cf_predicted,
                    "free_run_correct": cf_predicted == expected,
                    "free_run_generated_tokens": int(cf_tokens.numel()),
                    "free_run_generation_seconds": round(cf_time, 6),
                    "free_run_token_agreement_with_baseline": same_prefix,
                    "free_run_output": cf_text,
                }
            )

        ablation_results.append(record)

    return {
        "question_index": question_index,
        "expected": expected,
        "baseline_predicted": extract_predicted(baseline_text),
        "baseline_correct": extract_predicted(baseline_text) == expected,
        "baseline_generated_tokens": int(baseline_tokens.numel()),
        "baseline_generation_seconds": round(baseline_time, 6),
        "baseline_output": baseline_text,
        "baseline_teacher_seconds": round(baseline_teacher_time, 6),
        "ablations": ablation_results,
    }


def mean_or_none(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def main() -> int:
    args = parse_args()

    if args.questions <= 0:
        raise ValueError("--questions must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.free_run_questions < 0:
        raise ValueError("--free-run-questions cannot be negative")
    if args.free_run_questions > args.questions:
        raise ValueError("--free-run-questions cannot exceed --questions")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    rows = load_rows(args.dataset, args.questions)
    args.output.mkdir(parents=True, exist_ok=True)

    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", args.model)
    print("Dataset:", args.dataset)
    print("Questions:", len(rows))

    print("\nLoading 4-bit NF4 model...")
    tokenizer, model = build_model(args.model)

    total_layers = len(model.model.layers)
    layers = sorted(set(args.layers))
    for layer in layers:
        if not 1 <= layer <= total_layers:
            raise ValueError(
                f"Invalid layer {layer}; model has {total_layers} layers."
            )

    ablations = [
        Ablation(layer=layer, component=component)
        for layer in layers
        for component in args.components
    ]

    print("Model layers:", total_layers)
    print("Ablation layers:", layers)
    print("Ablations:", [item.name for item in ablations])
    print(
        "Mode: teacher-forced counterfactuals"
        + (
            f" + free-running on first {args.free_run_questions} questions"
            if args.free_run_questions
            else ""
        )
    )

    question_results: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows, 1):
        print(f"\n[{index}/{len(rows)}] baseline + ablations...")
        result = summarize_question(
            tokenizer=tokenizer,
            model=model,
            row=row,
            question_index=index,
            args=args,
            ablations=ablations,
            free_run=index <= args.free_run_questions,
        )
        question_results.append(result)
        ablation_rows.extend(result["ablations"])

        print(
            f"baseline expected={result['expected']} "
            f"predicted={result['baseline_predicted']} "
            f"correct={result['baseline_correct']} "
            f"tokens={result['baseline_generated_tokens']} "
            f"time={result['baseline_generation_seconds']:.2f}s"
        )
        for item in result["ablations"]:
            print(
                f"  {item['ablation']}: "
                f"agreement={item['top1_agreement']:.3f} "
                f"KL={item['kl_baseline_to_counterfactual_mean']:.4f} "
                f"target_prob_ratio={item['target_probability_ratio_mean']:.3f}"
            )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "EXP-0008",
        "title": "Causal Sublayer Ablation on Late Qwen3 Layers",
        "status": "causal_ablation_run",
        "model_path": str(args.model),
        "dataset_path": str(args.dataset),
        "questions": len(question_results),
        "model_layers": total_layers,
        "ablation_layers": layers,
        "components": list(args.components),
        "hardware": {
            "cuda": True,
            "gpu": torch.cuda.get_device_name(0),
        },
        "quantization": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
        },
        "sampling": {
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed_base": args.seed_base,
            "max_new_tokens": args.max_new_tokens,
            "stop_strings": ["\nQuestion:", "\nProblem:"],
        },
        "counterfactual_method": {
            "mode": "teacher_forced",
            "baseline_sequence_replayed": True,
            "use_cache": False,
            "comparison": [
                "baseline_vs_counterfactual_top1_agreement",
                "baseline_target_probability",
                "counterfactual_target_probability",
                "target_probability_ratio",
                "target_log_probability_delta",
                "KL baseline to counterfactual",
                "logit L2 difference",
            ],
        },
        "free_run_questions": args.free_run_questions,
        "what_is_measured": [
            "Causal effect of zeroing one attention or MLP sublayer on the baseline trajectory",
            "Depth dependence of causal disruption",
            "Attention-versus-MLP causal asymmetry",
            "Optional free-running output changes under identical seeds",
        ],
        "what_is_not_measured": [
            "Whether an ablated component can be skipped while preserving quality",
            "Whether a cheaper replacement can reproduce an ablated component",
            "Hardware speedup from a real conditional implementation",
            "Global causal importance of a component outside the tested layers and tasks",
        ],
        "questions": question_results,
    }

    aggregate: dict[str, dict[str, Any]] = {}
    metric_keys = (
        "top1_agreement",
        "baseline_target_probability_mean",
        "counterfactual_target_probability_mean",
        "target_probability_ratio_mean",
        "target_log_probability_delta_mean",
        "kl_baseline_to_counterfactual_mean",
        "logit_l2_mean",
    )

    for ablation in ablations:
        subset = [
            row
            for row in ablation_rows
            if row["ablation"] == ablation.name
        ]
        aggregate[ablation.name] = {
            "layer": ablation.layer,
            "component": ablation.component,
            "questions": len(subset),
            **{
                metric: mean_or_none(
                    [float(row[metric]) for row in subset]
                )
                for metric in metric_keys
            },
        }

    summary["aggregate"] = aggregate

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.output / "ablations.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in ablation_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    with (args.output / "questions.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in question_results:
            compact = {
                key: value
                for key, value in row.items()
                if key != "ablations"
            }
            handle.write(
                json.dumps(compact, ensure_ascii=False) + "\n"
            )

    print("\n" + "=" * 72)
    print("CFI EXP-0008 COMPLETE")
    print("=" * 72)
    for name, values in aggregate.items():
        print(
            f"{name}: "
            f"agreement={values['top1_agreement']:.3f} "
            f"KL={values['kl_baseline_to_counterfactual_mean']:.4f} "
            f"target_prob_ratio="
            f"{values['target_probability_ratio_mean']:.3f}"
        )
    print(f"Results: {args.output}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
