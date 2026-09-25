"""Probe intermediate Qwen3-4B logits without generation.

EXP-0004 uses the fixed CFI GSM8K evaluation set and runs the full frozen model
once per example with output_hidden_states=True. For selected layers, it
applies the model's final RMSNorm and tied LM head to the intermediate hidden
states and evaluates the known answer sequence under teacher forcing.

This is a diagnostic, not a training experiment. It measures:
- token-level accuracy at each depth
- cross-entropy loss on answer tokens
- agreement with the final layer
- top-1 agreement with the final layer

It answers whether layer 30 contains useful next-token information even though
autoregressive early-exit generation failed in EXP-0002/0003.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from cfi_paths import data_path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = data_path("HuggingFace", "hub", "models--Qwen--Qwen3-4B-Base", "snapshots", "906bfd4b4dc7f14ee4320094d8b41684abff8539")
DEFAULT_DATASET = data_path("Datasets", "CFI-Eval-0001-GSM8K", "gsm8k_test_100.jsonl")
DEFAULT_OUTPUT = data_path("Results", "CFI-Eval-0004-LayerwiseProbe")

DEFAULT_LAYERS = (12, 18, 24, 30, 36)


def load_rows(path: Path, count: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= count:
                    break
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def format_prompt(question: str) -> str:
    return (
        "Question: "
        + question
        + "\n"
        + "Solve the problem step by step. "
        + "End with: The answer is <number>.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAYERS),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    rows = load_rows(args.dataset, args.questions)
    args.output.mkdir(parents=True, exist_ok=True)

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        quantization_config=quant,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.eval()

    total_layers = len(model.model.layers)
    valid_layers = [x for x in args.layers if 1 <= x <= total_layers]
    if total_layers not in valid_layers:
        valid_layers.append(total_layers)
    valid_layers = sorted(set(valid_layers))

    sums = {
        str(layer): {
            "tokens": 0,
            "correct": 0,
            "cross_entropy_sum": 0.0,
            "final_agreement": 0,
            "examples": 0,
        }
        for layer in valid_layers
    }

    examples: list[dict] = []

    for index, row in enumerate(rows, 1):
        prompt = format_prompt(row["question"])
        target = str(row["answer"])

        prompt_tokens = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        target_tokens = tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=max(
                1,
                2048 - int(prompt_tokens["input_ids"].shape[-1]),
            ),
            return_tensors="pt",
        )

        prompt_ids = prompt_tokens["input_ids"][0]
        target_ids = target_tokens["input_ids"][0]
        input_ids = torch.cat([prompt_ids, target_ids], dim=0).unsqueeze(0)
        labels = input_ids.clone()
        labels[:, : int(prompt_ids.shape[0])] = -100

        input_ids = input_ids.to("cuda:0")
        labels = labels.to("cuda:0")

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )

        example_layers: dict[str, dict] = {}
        final_token_predictions = None

        for layer in valid_layers:
            hidden = outputs.hidden_states[layer]
            normalized = model.model.norm(hidden)
            logits = model.lm_head(normalized).float()

            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            mask = shift_labels != -100

            token_logits = shift_logits[mask]
            token_labels = shift_labels[mask]

            if token_logits.numel() == 0:
                continue

            log_probs = F.log_softmax(token_logits, dim=-1)
            predictions = token_logits.argmax(dim=-1)
            correct = int((predictions == token_labels).sum().item())
            token_count = int(token_labels.numel())
            ce = float(
                F.nll_loss(
                    log_probs,
                    token_labels,
                    reduction="mean",
                ).item()
            )

            if layer == total_layers:
                final_token_predictions = predictions.detach()

            final_agreement = 0
            if final_token_predictions is not None:
                final_agreement = int(
                    (predictions == final_token_predictions).sum().item()
                )

            sums[str(layer)]["tokens"] += token_count
            sums[str(layer)]["correct"] += correct
            sums[str(layer)]["cross_entropy_sum"] += ce * token_count
            sums[str(layer)]["examples"] += 1
            sums[str(layer)]["final_agreement"] += final_agreement

            example_layers[str(layer)] = {
                "tokens": token_count,
                "correct": correct,
                "accuracy": correct / token_count,
                "cross_entropy": ce,
                "final_agreement": final_agreement,
                "final_agreement_rate": (
                    final_agreement / token_count
                    if layer == total_layers
                    else None
                ),
            }

        # Recompute agreement for all intermediate layers against final layer.
        if str(total_layers) in example_layers:
            final_logits = model.lm_head(
                model.model.norm(outputs.hidden_states[total_layers])
            ).float()
            final_preds = final_logits[:, :-1, :].argmax(dim=-1)
            mask = labels[:, 1:] != -100
            final_preds = final_preds[mask]

            for layer in valid_layers:
                if layer == total_layers or str(layer) not in example_layers:
                    continue
                hidden = outputs.hidden_states[layer]
                logits = model.lm_head(model.model.norm(hidden)).float()
                preds = logits[:, :-1, :].argmax(dim=-1)[mask]
                agree = int((preds == final_preds).sum().item())
                tokens = int(final_preds.numel())
                example_layers[str(layer)]["final_agreement"] = agree
                example_layers[str(layer)]["final_agreement_rate"] = (
                    agree / tokens if tokens else 0.0
                )
                sums[str(layer)]["final_agreement"] += agree

        examples.append(
            {
                "index": index,
                "layers": example_layers,
            }
        )

        if index == 1 or index % 10 == 0:
            print(f"[{index}/{len(rows)}] processed")

    summary = {}
    for layer in valid_layers:
        item = sums[str(layer)]
        token_count = item["tokens"]
        summary[str(layer)] = {
            "tokens": token_count,
            "token_accuracy": (
                item["correct"] / token_count if token_count else 0.0
            ),
            "cross_entropy": (
                item["cross_entropy_sum"] / token_count
                if token_count
                else math.nan
            ),
            "agreement_with_final_layer": (
                item["final_agreement"] / token_count
                if token_count
                else 0.0
            ),
            "examples": item["examples"],
        }

    result = {
        "schema_version": 1,
        "benchmark": "CFI-Eval-0004-LayerwiseProbe",
        "model_path": str(args.model),
        "dataset_path": str(args.dataset),
        "questions": len(rows),
        "model_layers": total_layers,
        "probed_layers": valid_layers,
        "quantization": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
        },
        "method": (
            "Teacher-forced next-token probe. Intermediate hidden states are "
            "passed through the original final RMSNorm and tied LM head. "
            "No generation and no weight updates."
        ),
        "summary": summary,
        "examples": examples,
    }

    output_path = args.output / "layerwise_probe_results.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print("EXP-0004 LAYERWISE LOGIT PROBE COMPLETE")
    print("=" * 64)
    for layer in valid_layers:
        item = summary[str(layer)]
        print(
            f"Layer {layer:>2}/{total_layers}: "
            f"token_acc={item['token_accuracy']:.2%} "
            f"CE={item['cross_entropy']:.4f} "
            f"final_agreement={item['agreement_with_final_layer']:.2%}"
        )
    print(f"Results: {output_path}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
