
"""CFI EXP-0007: measure layer, sublayer, temporal, and token utility."""

from __future__ import annotations

import argparse
import json
import math
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = Path(
    r"E:\Titan Forge Industries\CFI-Data\HuggingFace\hub\models--Qwen--Qwen3-4B-Base\snapshots\906bfd4b4dc7f14ee4320094d8b41684abff8539"
)
DEFAULT_DATASET = Path(
    r"E:\Titan Forge Industries\CFI-Data\Datasets\CFI-Eval-0001-GSM8K\gsm8k_test_100.jsonl"
)
DEFAULT_OUTPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0007-Instrumentation"
)

DEFAULT_PROBE_LAYERS = (6, 12, 18, 24, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CFI EXP-0007 Qwen3 instrumentation."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument(
        "--probe-layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_PROBE_LAYERS),
        help="1-based layers for intermediate logit probes.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed-base", type=int, default=42000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--hardware-interval", type=float, default=2.0)
    return parser.parse_args()


def load_rows(path: Path, count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("--questions must be positive")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Dataset row is not an object: {path}")
            rows.append(value)
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


def parse_chinese_integer(text: str) -> int | None:
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    units = {
        "十": 10,
        "百": 100,
        "千": 1000,
        "万": 10000,
        "亿": 100000000,
    }

    text = text.strip()
    if not text or any(
        ch not in digits and ch not in units for ch in text
    ):
        return None

    total = 0
    section = 0
    digit = 0
    saw_unit = False

    for ch in text:
        if ch in digits:
            digit = digits[ch]
            continue

        saw_unit = True
        unit = units[ch]

        if unit < 10000:
            if digit == 0 and ch == "十":
                digit = 1
            section += digit * unit
            digit = 0
        else:
            section += digit
            total += section * unit
            section = 0
            digit = 0

    value = total + section + digit
    return value if saw_unit or len(text) == 1 else None


def extract_expected(text: str) -> str | None:
    match = re.search(
        r"####\s*(?:<\s*)?\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*>?",
        text,
        re.IGNORECASE,
    )
    return canonical_number(match.group(1)) if match else None


def extract_predicted(text: str) -> str | None:
    matches = re.findall(
        r"(?:The answer is|####)\s*:?\s*<?\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*>?",
        text,
        re.IGNORECASE,
    )
    if matches:
        return canonical_number(matches[-1])

    chinese = re.findall(
        r"(?:The answer is|####)\s*:?\s*([零〇一二两三四五六七八九十百千万亿]+)",
        text,
        re.IGNORECASE,
    )
    if chinese:
        value = parse_chinese_integer(chinese[-1])
        if value is not None:
            return str(value)

    for line in reversed(
        [line.strip() for line in text.splitlines() if line.strip()]
    ):
        if re.fullmatch(r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?", line):
            return canonical_number(line.replace("$", "").strip())

    return None


def _first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value):
                return value

    raise TypeError(
        f"Expected tensor output, got {type(output).__name__}"
    )


def _last_vector(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor[0, -1]
    if tensor.ndim == 2:
        return tensor[0]
    if tensor.ndim == 1:
        return tensor
    raise ValueError(f"Unexpected tensor shape: {tuple(tensor.shape)}")


def _first_input_tensor(
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    for value in inputs:
        if torch.is_tensor(value):
            return value

    for key in ("hidden_states", "hidden_state", "inputs_embeds"):
        value = kwargs.get(key)
        if torch.is_tensor(value):
            return value

    for value in kwargs.values():
        if torch.is_tensor(value):
            return value

    raise TypeError(
        "Could not find a tensor input in positional or keyword arguments."
    )


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            a.float().reshape(1, -1),
            b.float().reshape(1, -1),
        ).item()
    )


def _relative_delta(
    current: torch.Tensor,
    previous: torch.Tensor,
) -> float | None:
    numerator = torch.linalg.vector_norm(
        current.float() - previous.float()
    )
    denominator = torch.linalg.vector_norm(previous.float())
    denom = float(denominator.item())
    return None if denom == 0.0 else float(
        (numerator / denominator).item()
    )


def _activation_stats(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().float()
    total = max(values.numel(), 1)
    return {
        "output_norm": float(torch.linalg.vector_norm(values).item()),
        "mean_abs": float(values.abs().mean().item()),
        "zero_fraction": float((values == 0).sum().item() / total),
        "small_fraction": float(
            (values.abs() < 1e-3).sum().item() / total
        ),
    }


def _sublayer_stats(
    output: torch.Tensor,
    input_tensor: torch.Tensor,
) -> dict[str, float]:
    result = _activation_stats(output)
    input_norm = float(
        torch.linalg.vector_norm(input_tensor.detach().float()).item()
    )
    result["input_norm"] = input_norm
    result["output_to_input_norm_ratio"] = (
        result["output_norm"] / input_norm
        if input_norm > 0.0
        else float("nan")
    )
    return result


@dataclass
class StepState:
    hidden: dict[int, torch.Tensor] = field(default_factory=dict)
    layers: dict[int, dict[str, Any]] = field(default_factory=dict)
    sublayers: dict[int, dict[str, dict[str, float]]] = field(
        default_factory=dict
    )
    probe_top1: dict[int, int] = field(default_factory=dict)


class Recorder:
    def __init__(self, model, probe_layers: list[int]) -> None:
        self.model = model
        self.probe_layers = set(probe_layers)
        self.forward_calls = 0
        self.decode_step = 0
        self.is_decode = False
        self.previous_hidden: dict[int, torch.Tensor] = {}
        self.current = StepState()
        self.in_probe = False
        self.token_records: list[dict[str, Any]] = []
        self.layer_records: list[dict[str, Any]] = []
        self.sublayer_records: list[dict[str, Any]] = []
        self.handles: list[Any] = []

    def reset(self) -> None:
        self.forward_calls = 0
        self.decode_step = 0
        self.is_decode = False
        self.previous_hidden.clear()
        self.current = StepState()
        self.token_records.clear()
        self.layer_records.clear()
        self.sublayer_records.clear()

    def first_layer_pre_hook(self, _module, inputs) -> None:
        self.forward_calls += 1
        self.current = StepState()

        if self.forward_calls == 1:
            self.is_decode = False
            return

        hidden = inputs[0]
        self.is_decode = int(hidden.shape[-2]) == 1
        if self.is_decode:
            self.decode_step += 1

    def layer_hook(self, index: int, _module, _inputs, output) -> None:
        if not self.is_decode:
            return

        native_hidden = _last_vector(_first_tensor(output)).detach()
        hidden = native_hidden.float()

        previous_layer = self.current.hidden.get(index - 1)
        previous_token = self.previous_hidden.get(index)

        metrics: dict[str, Any] = {
            "layer": index,
            "hidden_norm": float(
                torch.linalg.vector_norm(hidden).item()
            ),
            "within_token_delta_norm": None,
            "within_token_relative_delta": None,
            "within_token_cosine_prev_layer": None,
            "temporal_relative_delta": None,
            "temporal_cosine_previous_token": None,
            "probe_top1_token_id": None,
            "probe_top1_confidence": None,
            "probe_top1_margin": None,
            "final_top1_token_id": None,
            "intermediate_vs_final_top1_agreement": None,
        }

        if previous_layer is not None:
            metrics["within_token_delta_norm"] = float(
                torch.linalg.vector_norm(
                    hidden - previous_layer
                ).item()
            )
            metrics["within_token_relative_delta"] = _relative_delta(
                hidden,
                previous_layer,
            )
            metrics["within_token_cosine_prev_layer"] = _cosine(
                hidden,
                previous_layer,
            )

        if previous_token is not None:
            metrics["temporal_relative_delta"] = _relative_delta(
                hidden,
                previous_token,
            )
            metrics["temporal_cosine_previous_token"] = _cosine(
                hidden,
                previous_token,
            )

        self.current.hidden[index] = hidden
        self.current.layers[index] = metrics

        if index not in self.probe_layers:
            return

        device = next(self.model.parameters()).device
        projected = self.model.model.norm(
            native_hidden.to(device).unsqueeze(0).unsqueeze(0)
        )
        self.in_probe = True
        try:
            logits = self.model.lm_head(projected).float()[0, -1]
        finally:
            self.in_probe = False
        top_values, top_indices = torch.topk(logits, k=2)

        top1_id = int(top_indices[0].item())
        self.current.probe_top1[index] = top1_id
        metrics["probe_top1_token_id"] = top1_id
        metrics["probe_top1_confidence"] = float(
            torch.softmax(logits, dim=-1).max().item()
        )
        metrics["probe_top1_margin"] = float(
            (top_values[0] - top_values[1]).item()
        )

    def sublayer_hook(
        self,
        index: int,
        kind: str,
        _module,
        inputs,
        kwargs,
        output,
    ) -> None:
        if not self.is_decode:
            return

        input_tensor = _last_vector(
            _first_input_tensor(inputs, kwargs).detach()
        )
        output_tensor = _last_vector(_first_tensor(output))

        self.current.sublayers.setdefault(index, {})[kind] = (
            _sublayer_stats(output_tensor, input_tensor)
        )

    def lm_head_hook(self, _module, _inputs, output) -> None:
        if not self.is_decode or self.in_probe:
            return

        logits = _first_tensor(output).float()[0, -1]
        probabilities = torch.softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(logits, k=2)
        final_top1 = int(top_indices[0].item())

        for index, metrics in self.current.layers.items():
            metrics["final_top1_token_id"] = final_top1
            probe = self.current.probe_top1.get(index)
            if probe is not None:
                metrics["intermediate_vs_final_top1_agreement"] = bool(
                    probe == final_top1
                )

        self.token_records.append(
            {
                "step": self.decode_step,
                "final_top1_token_id": final_top1,
                "final_top2_token_id": int(top_indices[1].item()),
                "final_top1_probability": float(
                    probabilities[top_indices[0]].item()
                ),
                "final_top1_margin": float(
                    (top_values[0] - top_values[1]).item()
                ),
                "final_entropy": float(
                    -(
                        probabilities
                        * probabilities.clamp_min(1e-20).log()
                    )
                    .sum()
                    .item()
                ),
            }
        )

        self.layer_records.append(
            {
                "step": self.decode_step,
                "layers": {
                    str(index): metrics
                    for index, metrics in self.current.layers.items()
                },
            }
        )
        self.sublayer_records.append(
            {
                "step": self.decode_step,
                "layers": {
                    str(index): metrics
                    for index, metrics in self.current.sublayers.items()
                },
            }
        )

        self.previous_hidden = {
            index: value.clone()
            for index, value in self.current.hidden.items()
        }

    def install(self) -> None:
        layers = self.model.model.layers
        if not layers:
            raise RuntimeError("Model has no decoder layers.")

        self.handles.append(
            layers[0].register_forward_pre_hook(
                self.first_layer_pre_hook
            )
        )

        for index, layer in enumerate(layers, start=1):
            if not hasattr(layer, "self_attn") or not hasattr(
                layer, "mlp"
            ):
                raise RuntimeError(
                    f"Layer {index} lacks self_attn/mlp; "
                    "EXP-0007 expects Qwen3-style decoder layers."
                )

            self.handles.append(
                layer.register_forward_hook(
                    lambda module, inputs, output, i=index:
                    self.layer_hook(i, module, inputs, output)
                )
            )
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    lambda module, inputs, kwargs, output, i=index:
                    self.sublayer_hook(
                        i, "attention", module, inputs, kwargs, output
                    ),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                layer.mlp.register_forward_hook(
                    lambda module, inputs, kwargs, output, i=index:
                    self.sublayer_hook(
                        i, "mlp", module, inputs, kwargs, output
                    ),
                    with_kwargs=True,
                )
            )

        self.handles.append(
            self.model.lm_head.register_forward_hook(
                self.lm_head_hook
            )
        )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class HardwareSampler:
    def __init__(self, path: Path, interval: float) -> None:
        self.path = path
        self.interval = interval
        self.count = 0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

        try:
            from cfi_experiment_logger.hardware import (
                collect_hardware_snapshot,
            )
        except ImportError:
            collect_hardware_snapshot = None

        self.collect = collect_hardware_snapshot

    def start(self) -> None:
        if self.collect is None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name="cfi-exp0007-hardware",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        if self.collect is None:
            return

        with self.path.open("a", encoding="utf-8") as handle:
            while not self.stop_event.is_set():
                snapshot = self.collect()
                handle.write(
                    json.dumps(
                        snapshot,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
                self.count += 1
                self.stop_event.wait(self.interval)

    def stop(self) -> None:
        if self.thread is None:
            return

        self.stop_event.set()
        self.thread.join(
            timeout=max(5.0, self.interval * 2.0)
        )
        self.thread = None


def build_model(model_path: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-0007.")

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


def run_question(
    tokenizer,
    model,
    row: dict[str, Any],
    index: int,
    args: argparse.Namespace,
    recorder: Recorder,
) -> dict[str, Any]:
    seed = args.seed_base + index
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    prompt = format_prompt(str(row["question"]))
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    prompt_length = int(inputs["input_ids"].shape[-1])

    recorder.reset()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()
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
    elapsed = time.perf_counter() - start

    new_tokens = generated[0, prompt_length:]
    output_text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    generated_tokens = int(new_tokens.shape[-1])
    observed_steps = len(recorder.token_records)

    # The initial forward pass processes the full prompt and directly
    # produces the first generated token. Our decode hooks intentionally
    # measure only subsequent one-token decode forwards, so one generated
    # token is expected to be outside the instrumentation stream.
    expected_decode_steps = max(generated_tokens - 1, 0)
    alignment_steps = min(expected_decode_steps, observed_steps)

    token_records = []
    for step in range(alignment_steps):
        record = dict(recorder.token_records[step])
        token_id = int(new_tokens[step + 1].item())
        record["sampled_token_id"] = token_id
        record["sampled_token"] = tokenizer.decode([token_id])
        token_records.append(record)

    expected = extract_expected(str(row["answer"]))
    predicted = extract_predicted(output_text)

    return {
        "index": index,
        "seed": seed,
        "expected": expected,
        "predicted": predicted,
        "correct": predicted == expected,
        "elapsed_seconds": round(elapsed, 6),
        "generated_tokens": generated_tokens,
        "prefill_generated_tokens": min(generated_tokens, 1),
        "expected_decode_steps": expected_decode_steps,
        "observed_instrumentation_steps": observed_steps,
        "alignment_steps": alignment_steps,
        "forward_calls": recorder.forward_calls,
        "peak_vram_gib": round(
            torch.cuda.max_memory_allocated()
            / (1024 ** 3),
            6,
        ),
        "output": output_text,
        "token_records": token_records,
        "layer_records": recorder.layer_records[:alignment_steps],
        "sublayer_records": recorder.sublayer_records[:alignment_steps],
    }


def mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def summarize(
    question_results: list[dict[str, Any]],
    group: str,
) -> dict[str, dict[str, Any]]:
    values_by_layer: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    observations: dict[int, int] = defaultdict(int)

    for question in question_results:
        for step_record in question[group]:
            for layer_text, metrics in step_record["layers"].items():
                layer = int(layer_text)
                observations[layer] += 1
                for key, value in metrics.items():
                    if key in {
                        "layer",
                        "probe_top1_token_id",
                        "final_top1_token_id",
                    }:
                        continue
                    if isinstance(value, (int, float)) and not isinstance(
                        value, bool
                    ):
                        values_by_layer[layer][key].append(float(value))

    result: dict[str, dict[str, Any]] = {}
    for layer, metrics in sorted(values_by_layer.items()):
        result[str(layer)] = {
            "layer": layer,
            "observations": observations[layer],
            **{
                f"{key}_mean": mean(values)
                for key, values in metrics.items()
            },
        }

    return result


def summarize_sublayers(
    question_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    values_by_layer: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    observations: dict[int, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for question in question_results:
        for step_record in question["sublayer_records"]:
            for layer_text, kinds in step_record["layers"].items():
                layer = int(layer_text)
                for kind, metrics in kinds.items():
                    observations[layer][kind] += 1
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)) and not isinstance(
                            value, bool
                        ):
                            values_by_layer[layer][
                                f"{kind}_{key}"
                            ].append(float(value))

    result: dict[str, dict[str, Any]] = {}
    for layer, metrics in sorted(values_by_layer.items()):
        result[str(layer)] = {
            "layer": layer,
            "observations": dict(
                sorted(observations[layer].items())
            ),
            **{
                f"{key}_mean": mean(values)
                for key, values in sorted(metrics.items())
            },
        }

    return result


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> int:
    args = parse_args()

    if args.questions <= 0:
        raise ValueError("--questions must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.hardware_interval <= 0:
        raise ValueError("--hardware-interval must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.dataset, args.questions)

    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", args.model)
    print("Dataset:", args.dataset)
    print("Questions:", len(rows))

    print("\nLoading 4-bit NF4 model...")
    tokenizer, model = build_model(args.model)

    total_layers = len(model.model.layers)
    probe_layers = sorted(
        {
            layer
            for layer in args.probe_layers
            if 1 <= layer <= total_layers
        }
    )
    if not probe_layers:
        raise ValueError("No valid probe layers were provided.")

    print("Model layers:", total_layers)
    print("Probe layers:", probe_layers)

    recorder = Recorder(model, probe_layers)
    recorder.install()

    hardware = HardwareSampler(
        args.output / "hardware_samples.jsonl",
        args.hardware_interval,
    )

    results: list[dict[str, Any]] = []
    all_tokens: list[dict[str, Any]] = []
    all_layers: list[dict[str, Any]] = []
    all_sublayers: list[dict[str, Any]] = []

    try:
        hardware.start()

        for index, row in enumerate(rows, 1):
            print(f"\n[{index}/{len(rows)}] running...")
            result = run_question(
                tokenizer,
                model,
                row,
                index,
                args,
                recorder,
            )
            results.append(result)

            for item in result["token_records"]:
                record = dict(item)
                record["question_index"] = index
                all_tokens.append(record)

            for item in result["layer_records"]:
                record = dict(item)
                record["question_index"] = index
                all_layers.append(record)

            for item in result["sublayer_records"]:
                record = dict(item)
                record["question_index"] = index
                all_sublayers.append(record)

            print(
                f"expected={result['expected']} "
                f"predicted={result['predicted']} "
                f"correct={result['correct']} "
                f"time={result['elapsed_seconds']:.2f}s "
                f"tokens={result['generated_tokens']} "
                f"instrumented={result['observed_instrumentation_steps']}"
            )
    finally:
        hardware.stop()
        recorder.remove()

    correct = sum(int(item["correct"]) for item in results)
    total_tokens = sum(
        int(item["generated_tokens"]) for item in results
    )
    total_time = sum(
        float(item["elapsed_seconds"]) for item in results
    )

    mismatches = [
        {
            "index": item["index"],
            "generated_tokens": item["generated_tokens"],
            "expected_decode_steps": max(
                item["generated_tokens"] - 1,
                0,
            ),
            "observed_instrumentation_steps": item[
                "observed_instrumentation_steps"
            ],
        }
        for item in results
        if max(item["generated_tokens"] - 1, 0)
        != item["observed_instrumentation_steps"]
    ]

    token_values: dict[str, list[float]] = defaultdict(list)
    for item in all_tokens:
        for key in (
            "final_top1_probability",
            "final_top1_margin",
            "final_entropy",
        ):
            value = item.get(key)
            if isinstance(value, (int, float)):
                token_values[key].append(float(value))

    summary = {
        "schema_version": 1,
        "experiment": "EXP-0007",
        "title": "Where Does a Pretrained LLM Actually Spend Its Computation and Memory?",
        "status": "instrumentation_run",
        "model_path": str(args.model),
        "dataset_path": str(args.dataset),
        "questions": len(results),
        "model_layers": total_layers,
        "probe_layers": probe_layers,
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
            "stop_strings": [
                "\nQuestion:",
                "\nProblem:",
            ],
        },
        "accuracy": correct / len(results),
        "correct": correct,
        "total_generated_tokens": total_tokens,
        "total_generation_time_seconds": round(total_time, 6),
        "generation_tok_per_sec": (
            total_tokens / total_time
            if total_time > 0
            else 0.0
        ),
        "hardware_samples": hardware.count,
        "token_summary": {
            "tokens_observed": len(all_tokens),
            "final_top1_probability_mean": mean(
                token_values["final_top1_probability"]
            ),
            "final_top1_margin_mean": mean(
                token_values["final_top1_margin"]
            ),
            "final_entropy_mean": mean(
                token_values["final_entropy"]
            ),
        },
        "instrumentation_alignment": {
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
        },
        "what_is_measured": [
            "Per-layer hidden-state magnitude and change",
            "Cross-token hidden-state temporal similarity",
            "Intermediate top-1 prediction and confidence",
            "Intermediate-vs-final top-1 agreement",
            "Attention output magnitude and activation sparsity",
            "MLP output magnitude and activation sparsity",
            "Final entropy, confidence, and margin",
            "Generation-level hardware telemetry",
        ],
        "what_is_not_measured": [
            "Exact per-layer VRAM/RAM/NVMe bytes moved",
            "Exact parameter bytes accessed",
            "Exact KV-cache reuse ratio",
            "Causal proof that a low-magnitude operation can be skipped",
        ],
    }

    (args.output / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (args.output / "layer_summary.json").write_text(
        json.dumps(
            summarize(results, "layer_records"),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (args.output / "sublayer_summary.json").write_text(
        json.dumps(
            summarize_sublayers(results),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_jsonl(
        args.output / "token_metrics.jsonl",
        all_tokens,
    )
    write_jsonl(
        args.output / "layer_metrics.jsonl",
        all_layers,
    )
    sublayer_summary = summarize_sublayers(results)
    if all_sublayers and not sublayer_summary:
        raise RuntimeError(
            "Sublayer records were collected but the sublayer summary is empty."
        )

    write_jsonl(
        args.output / "sublayer_metrics.jsonl",
        all_sublayers,
    )
    write_jsonl(
        args.output / "questions.jsonl",
        [
            {
                key: value
                for key, value in result.items()
                if key
                not in {
                    "token_records",
                    "layer_records",
                    "sublayer_records",
                }
            }
            for result in results
        ],
    )

    print("\n" + "=" * 72)
    print("CFI EXP-0007 COMPLETE")
    print("=" * 72)
    print(
        f"Accuracy: {correct}/{len(results)} = "
        f"{correct / len(results):.1%}"
    )
    print(f"Generated tokens: {total_tokens}")
    print(f"Generation time: {total_time:.2f}s")
    if total_time > 0:
        print(
            "Generation throughput: "
            f"{total_tokens / total_time:.2f} tok/s"
        )
    print(f"Instrumentation mismatches: {len(mismatches)}")
    print(f"Hardware samples: {hardware.count}")
    print(f"Results: {args.output}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
