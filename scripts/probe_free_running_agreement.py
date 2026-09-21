"""EXP-0005: observe layer-30 predictions on the full model's free-running path.

The full 36-layer Qwen3-4B-Base remains the actual generator. Generation uses
the official CFI sampling settings. We request generation hidden states and
scores, then reconstruct the raw layer-30 and layer-36 next-token logits from
the hidden state at each generation step.

This is diagnostic only. No weights are changed and the generated sequence is
not altered.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

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
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0005-FreeRunningAgreement"
)


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


def canonical_number(value: str | None) -> str | None:
    if value is None:
        return None
    raw = value.strip().replace(",", "")
    try:
        number = float(raw)
    except ValueError:
        return raw
    if number.is_integer():
        return str(int(number))
    return str(number)


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
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    for line in reversed(lines):
        if re.fullmatch(r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?", line):
            return canonical_number(line.replace("$", "").strip())
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed-base", type=int, default=42000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
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
    exit_layer = 30
    if total_layers != 36:
        raise ValueError(f"Expected 36 layers, found {total_layers}.")

    summaries = []
    all_step_records = []

    for index, row in enumerate(rows, 1):
        seed = args.seed_base + index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        prompt = format_prompt(str(row["question"]))
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        prompt_len = int(inputs["input_ids"].shape[-1])

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                pad_token_id=tokenizer.eos_token_id,
                output_hidden_states=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

        sequence = generated.sequences[0]
        new_tokens = sequence[prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        scores = generated.scores
        hidden_steps = generated.hidden_states

        steps = min(len(scores), len(hidden_steps))
        if steps == 0:
            raise RuntimeError("Generation did not return hidden states/scores.")

        layer_agreements = 0
        sample_agreements = 0
        per_step = []

        for step_idx in range(steps):
            # For cached generation, hidden states for the new token are stored
            # as [batch, 1, hidden]. For the initial prefill, take the last
            # position because it predicts the first generated token.
            h30 = hidden_steps[step_idx][exit_layer]
            h36 = hidden_steps[step_idx][total_layers]
            h30 = h30[:, -1:, :]
            h36 = h36[:, -1:, :]

            logits30 = model.lm_head(model.model.norm(h30)).float()[:, -1, :]
            logits36 = scores[step_idx].float()

            # scores are processed generation scores. Raw final logits are
            # reconstructed separately so confidence is comparable.
            raw36 = model.lm_head(model.model.norm(h36)).float()[:, -1, :]

            pred30 = int(logits30.argmax(dim=-1).item())
            pred36 = int(raw36.argmax(dim=-1).item())
            sampled = int(new_tokens[step_idx].item())

            prob30 = float(
                F.softmax(logits30, dim=-1).max(dim=-1).values.item()
            )
            prob36 = float(
                F.softmax(raw36, dim=-1).max(dim=-1).values.item()
            )

            agrees = pred30 == pred36
            sample_agrees_30 = pred30 == sampled

            layer_agreements += int(agrees)
            sample_agreements += int(sample_agrees_30)

            per_step.append(
                {
                    "step": step_idx + 1,
                    "token_id": sampled,
                    "token": tokenizer.decode([sampled]),
                    "layer30_top1_id": pred30,
                    "layer30_top1": tokenizer.decode([pred30]),
                    "layer36_top1_id": pred36,
                    "layer36_top1": tokenizer.decode([pred36]),
                    "layer30_confidence": prob30,
                    "layer36_confidence": prob36,
                    "layer30_agrees_with_layer36": agrees,
                    "layer30_agrees_with_sample": sample_agrees_30,
                }
            )

        expected = extract_expected(str(row["answer"]))
        predicted = extract_predicted(text)
        summary = {
            "index": index,
            "seed": seed,
            "expected": expected,
            "predicted": predicted,
            "correct": predicted == expected,
            "generated_tokens": len(new_tokens),
            "layer30_vs_layer36_agreement": (
                layer_agreements / steps if steps else 0.0
            ),
            "layer30_vs_sample_agreement": (
                sample_agreements / steps if steps else 0.0
            ),
            "steps_observed": steps,
            "output": text,
        }
        summaries.append(summary)

        all_step_records.append(
            {
                "index": index,
                "seed": seed,
                "steps": per_step,
            }
        )

        print(
            f"[{index}/{len(rows)}] "
            f"correct={summary['correct']} "
            f"tokens={len(new_tokens)} "
            f"30v36={summary['layer30_vs_layer36_agreement']:.2%} "
            f"30vsampled={summary['layer30_vs_sample_agreement']:.2%}"
        )

    total_steps = sum(x["steps_observed"] for x in summaries)
    agreement = (
        sum(
            x["layer30_vs_layer36_agreement"] * x["steps_observed"]
            for x in summaries
        )
        / total_steps
        if total_steps
        else 0.0
    )
    sample_agreement = (
        sum(
            x["layer30_vs_sample_agreement"] * x["steps_observed"]
            for x in summaries
        )
        / total_steps
        if total_steps
        else 0.0
    )

    result = {
        "schema_version": 1,
        "benchmark": "CFI-Eval-0005-FreeRunningAgreement",
        "questions": len(rows),
        "model_layers": total_layers,
        "observed_layer": exit_layer,
        "sampling": {
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed_base": args.seed_base,
        },
        "method": (
            "Full 36-layer model remains the generator. Intermediate layer-30 "
            "and final layer-36 predictions are reconstructed from hidden "
            "states observed along the actual free-running generation path."
        ),
        "summary": {
            "questions_correct": sum(int(x["correct"]) for x in summaries),
            "total_steps_observed": total_steps,
            "layer30_vs_layer36_agreement": agreement,
            "layer30_vs_sample_agreement": sample_agreement,
        },
        "questions": summaries,
        "per_step": all_step_records,
    }

    path = args.output / "free_running_agreement_results.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print("EXP-0005 FREE-RUNNING LAYER AGREEMENT COMPLETE")
    print("=" * 64)
    print(
        "Layer-30 vs layer-36 agreement: "
        f"{agreement:.2%}"
    )
    print(
        "Layer-30 vs sampled-token agreement: "
        f"{sample_agreement:.2%}"
    )
    print(f"Results: {path}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
