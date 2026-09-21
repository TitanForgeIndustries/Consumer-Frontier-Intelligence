"""EXP-0006: manual shared-weight self-speculative decoding.

Qwen3-4B-Base is not an early-exit-trained checkpoint, so this experiment does
not use Transformers' assistant_early_exit or assistant_model generation path.

Instead, CFI implements the speculative draft/verify loop directly:

1. A 30-layer assistant object shares the target model's loaded modules/weights.
2. The assistant drafts up to K tokens with sequential cached forward passes.
3. The 36-layer target verifies the whole candidate block in one forward pass.
4. Draft tokens are accepted/rejected with the standard distribution-preserving
   speculative-sampling rule.
5. Rejected suffixes are rolled back explicitly with DynamicCache.crop().

This keeps cache ownership explicit and avoids the generic Transformers
_assisted_decoding cache path that failed for this shared-weight truncated
Qwen3 setup.

The benchmark keeps the CFI GSM8K sampling protocol:
- 4-bit NF4
- BF16 compute
- do_sample=True
- temperature=0.6
- top_p=0.95
- top_k=20
- per-question seed = 42000 + index
- max_new_tokens=512
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import threading
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DynamicCache,
)
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

try:
    from cfi_experiment_logger.hardware import collect_hardware_snapshot
except ImportError:
    collect_hardware_snapshot = None


DEFAULT_MODEL = Path(
    r"E:\Titan Forge Industries\CFI-Data\HuggingFace\hub\models--Qwen--Qwen3-4B-Base\snapshots\906bfd4b4dc7f14ee4320094d8b41684abff8539"
)
DEFAULT_DATASET = Path(
    r"E:\Titan Forge Industries\CFI-Data\Datasets\CFI-Eval-0001-GSM8K\gsm8k_test_100.jsonl"
)
DEFAULT_OUTPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0006-SelfSpeculative"
)

MAX_NEW_TOKENS_DEFAULT = 512
SEED_BASE_DEFAULT = 42000
TEMPERATURE_DEFAULT = 0.6
TOP_P_DEFAULT = 0.95
TOP_K_DEFAULT = 20
ASSISTANT_DEPTH_DEFAULT = 30
NUM_ASSISTANT_TOKENS_DEFAULT = 4
HARDWARE_INTERVAL_DEFAULT = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CFI EXP-0006 manual self-speculative decoding.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS_DEFAULT)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE_DEFAULT)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    parser.add_argument("--top-p", type=float, default=TOP_P_DEFAULT)
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    parser.add_argument("--assistant-depth", type=int, default=ASSISTANT_DEPTH_DEFAULT)
    parser.add_argument(
        "--num-assistant-tokens",
        type=int,
        default=NUM_ASSISTANT_TOKENS_DEFAULT,
    )
    parser.add_argument(
        "--hardware-interval",
        type=float,
        default=HARDWARE_INTERVAL_DEFAULT,
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


def canonical_number(value: str | None) -> str | None:
    if value is None:
        return None

    raw = value.strip().replace(",", "")
    try:
        number = Decimal(raw)
    except InvalidOperation:
        return raw

    if number == number.to_integral_value():
        return str(int(number))
    return format(number.normalize(), "f").rstrip("0").rstrip(".")


def extract_expected(text: str) -> str | None:
    match = re.search(
        r"####\s*(?:<\s*)?\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*>?",
        text,
        re.IGNORECASE,
    )
    return canonical_number(match.group(1)) if match else None


def extract_predicted(text: str) -> str | None:
    explicit = re.findall(
        r"(?:The answer is|####)\s*:?\s*<?\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*>?",
        text,
        re.IGNORECASE,
    )
    if explicit:
        return canonical_number(explicit[-1])

    for line in reversed([x.strip() for x in text.splitlines() if x.strip()]):
        match = re.fullmatch(r"[-+]?\$?\s*\d[\d,]*(?:\.\d+)?", line)
        if match:
            return canonical_number(match.group(0).replace("$", "").strip())

    return None


def prompt_for(question: str) -> str:
    return (
        "Question: "
        + question
        + "\n"
        + "Solve the problem step by step. "
        + "End with: The answer is <number>.\n"
    )


class HardwareSampler:
    """Background sampler using the existing CFI hardware collector."""

    def __init__(self, output_path: Path, interval_seconds: float) -> None:
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.count = 0

    def start(self) -> None:
        if collect_hardware_snapshot is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="cfi-exp0006-hardware",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds * 2.0))
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            snapshot = collect_hardware_snapshot()
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot) + "\n")
                handle.flush()
            self.count += 1
            self._stop.wait(self.interval_seconds)


def build_model(model_path: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.eval()
    return tokenizer, model


def make_shared_weight_assistant(target_model, depth: int):
    """Create a truncated assistant without duplicating loaded weights."""
    total_layers = len(target_model.model.layers)
    if depth <= 0 or depth >= total_layers:
        raise ValueError(
            f"Assistant depth must be between 1 and {total_layers - 1}."
        )

    assistant = copy.copy(target_model)
    assistant.model = copy.copy(target_model.model)
    assistant.config = copy.deepcopy(target_model.config)
    assistant.model.config = assistant.config
    assistant.config.num_hidden_layers = depth

    assistant.model.layers = nn.ModuleList(
        list(target_model.model.layers[:depth])
    )
    assistant.model.embed_tokens = target_model.model.embed_tokens
    assistant.model.norm = target_model.model.norm
    assistant.lm_head = target_model.lm_head
    assistant.eval()

    return assistant


def make_warpers(
    *,
    temperature: float,
    top_p: float,
    top_k: int,
):
    if temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if not 0 < top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if top_k < 0:
        raise ValueError("--top-k must be >= 0")

    return [
        TemperatureLogitsWarper(temperature),
        TopKLogitsWarper(top_k) if top_k > 0 else None,
        TopPLogitsWarper(top_p),
    ]


def distribution_from_logits(
    logits: torch.Tensor,
    warpers,
) -> torch.Tensor:
    """Apply the same temperature/top-k/top-p transforms used by generation."""
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    input_ids = torch.zeros(
        (logits.shape[0], 1),
        dtype=torch.long,
        device=logits.device,
    )
    scores = logits.float()

    for warper in warpers:
        if warper is not None:
            scores = warper(input_ids, scores)

    return torch.softmax(scores, dim=-1)


def sample_distribution(probs: torch.Tensor) -> int:
    if probs.dim() != 1:
        raise ValueError(f"Expected 1D probability vector, got {tuple(probs.shape)}")

    token = torch.multinomial(probs, num_samples=1)
    return int(token.item())


def prefill(
    model,
    input_ids: torch.Tensor,
    *,
    cache: DynamicCache,
):
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
        )
    return output.past_key_values, output.logits[:, -1, :]


def draft_candidates(
    assistant_model,
    *,
    cache: DynamicCache,
    next_logits: torch.Tensor,
    eos_token_id: int | None,
    num_tokens: int,
    warpers,
):
    tokens: list[int] = []
    probabilities: list[torch.Tensor] = []
    forward_calls = 0

    for _ in range(num_tokens):
        probs = distribution_from_logits(next_logits, warpers)[0]
        token_id = sample_distribution(probs)

        tokens.append(token_id)
        probabilities.append(probs)

        input_token = torch.tensor(
            [[token_id]],
            dtype=torch.long,
            device=next_logits.device,
        )

        with torch.inference_mode():
            output = assistant_model(
                input_ids=input_token,
                past_key_values=cache,
                use_cache=True,
            )

        cache = output.past_key_values
        next_logits = output.logits[:, -1, :]
        forward_calls += 1

        if eos_token_id is not None and token_id == eos_token_id:
            break

    return tokens, probabilities, cache, next_logits, forward_calls


def safe_crop_cache(cache: DynamicCache, tokens_to_remove: int) -> list[int]:
    """Crop only initialized KV layers.

    Transformers 5.17's DynamicCache can hold lazy, uninitialized layer slots.
    Its built-in DynamicLayer.crop() assumes keys/values are already tensors,
    so calling Cache.crop() can fail with NoneType on an untouched slot.
    Uninitialized slots contain no cached sequence to roll back and can be
    skipped safely.
    """
    if tokens_to_remove <= 0:
        return []

    skipped: list[int] = []
    for layer_idx, layer in enumerate(cache.layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)

        if keys is None or values is None:
            skipped.append(layer_idx)
            continue

        layer.crop(tokens_to_remove)

    return skipped


def speculative_decode(
    *,
    target_model,
    assistant_model,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    num_assistant_tokens: int,
    warpers,
):
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    device = prompt_ids.device
    target_cache = DynamicCache(config=target_model.config)
    assistant_cache = DynamicCache(config=assistant_model.config)

    target_cache, target_next_logits = prefill(
        target_model,
        prompt_ids,
        cache=target_cache,
    )
    assistant_cache, assistant_next_logits = prefill(
        assistant_model,
        prompt_ids,
        cache=assistant_cache,
    )

    generated: list[int] = []
    accepted_draft_tokens = 0
    proposed_draft_tokens = 0
    target_verification_calls = 0
    target_verified_candidate_tokens = 0
    assistant_draft_forward_calls = 0
    rejection_count = 0

    pending_target_token: int | None = None

    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        draft_count = min(num_assistant_tokens, remaining)

        (
            draft_tokens,
            draft_probs_list,
            assistant_cache,
            assistant_next_logits,
            draft_calls,
        ) = draft_candidates(
            assistant_model,
            cache=assistant_cache,
            next_logits=assistant_next_logits,
            eos_token_id=tokenizer.eos_token_id,
            num_tokens=draft_count,
            warpers=warpers,
        )
        assistant_draft_forward_calls += draft_calls
        proposed_draft_tokens += len(draft_tokens)

        if not draft_tokens:
            raise RuntimeError("Assistant produced no candidate tokens.")

        draft_tensor = torch.tensor(
            [draft_tokens],
            dtype=torch.long,
            device=device,
        )

        if pending_target_token is None:
            target_inputs = draft_tensor
            prefix_logits = target_next_logits.unsqueeze(1)
        else:
            pending_tensor = torch.tensor(
                [[pending_target_token]],
                dtype=torch.long,
                device=device,
            )
            target_inputs = torch.cat([pending_tensor, draft_tensor], dim=1)
            prefix_logits = None

        with torch.inference_mode():
            target_output = target_model(
                input_ids=target_inputs,
                past_key_values=target_cache,
                use_cache=True,
            )

        target_cache = target_output.past_key_values
        target_verification_calls += 1
        target_verified_candidate_tokens += len(draft_tokens)

        if pending_target_token is None:
            # Rows 0..K-2 are the target distributions for draft tokens 1..K-1.
            # The prefill logits provide the distribution for draft token 1.
            candidate_logits = torch.cat(
                [
                    prefix_logits,
                    target_output.logits[:, :-1, :],
                ],
                dim=1,
            )
        else:
            # Feeding [pending, draft_1, ..., draft_K] produces logits for
            # draft_1, ..., draft_K, bonus in rows 0..K.
            candidate_logits = target_output.logits[:, :-1, :]

        target_probs = distribution_from_logits(
            candidate_logits.reshape(-1, candidate_logits.shape[-1]),
            warpers,
        )

        rejected = False
        accepted_this_round = 0

        for position, token_id in enumerate(draft_tokens):
            p = target_probs[position, token_id]
            q = draft_probs_list[position][token_id]

            acceptance_probability = torch.clamp(
                p / torch.clamp(q, min=1e-12),
                max=1.0,
            )

            random_value = torch.rand(
                (),
                device=device,
            )

            if random_value < acceptance_probability:
                generated.append(token_id)
                accepted_draft_tokens += 1
                accepted_this_round += 1

                if token_id == tokenizer.eos_token_id:
                    rejected = False
                    pending_target_token = None
                    break
                continue

            rejection_count += 1
            rejected = True

            residual = torch.clamp(
                target_probs[position] - draft_probs_list[position],
                min=0.0,
            )
            residual_mass = residual.sum()

            if float(residual_mass) > 1e-12:
                replacement_probs = residual / residual_mass
            else:
                replacement_probs = target_probs[position]

            replacement = sample_distribution(replacement_probs)
            generated.append(replacement)
            pending_target_token = replacement

            remove_from_target = len(draft_tokens) - accepted_this_round
            remove_from_assistant = remove_from_target

            if remove_from_target > 0:
                safe_crop_cache(target_cache, remove_from_target)
                safe_crop_cache(assistant_cache, remove_from_assistant)

            replacement_tensor = torch.tensor(
                [[replacement]],
                dtype=torch.long,
                device=device,
            )

            with torch.inference_mode():
                assistant_output = assistant_model(
                    input_ids=replacement_tensor,
                    past_key_values=assistant_cache,
                    use_cache=True,
                )

            assistant_cache = assistant_output.past_key_values
            assistant_next_logits = assistant_output.logits[:, -1, :]
            assistant_draft_forward_calls += 1

            if replacement == tokenizer.eos_token_id:
                pending_target_token = None

            break

        if rejected:
            if generated[-1] == tokenizer.eos_token_id:
                break
            # The emitted replacement intentionally remains pending in the
            # target cache. It is consumed with the next verification block.
            continue

        if generated and generated[-1] == tokenizer.eos_token_id:
            break

        # All proposed tokens were accepted. The target's final verification
        # row gives the bonus-token distribution. Emit one target-sampled token.
        bonus_logits = target_output.logits[:, -1, :]
        bonus_probs = distribution_from_logits(bonus_logits, warpers)[0]
        bonus = sample_distribution(bonus_probs)

        generated.append(bonus)
        pending_target_token = bonus

        # The assistant cache must include the emitted bonus token so its next
        # draft distribution is conditioned on the complete generated text.
        bonus_tensor = torch.tensor(
            [[bonus]],
            dtype=torch.long,
            device=device,
        )

        with torch.inference_mode():
            assistant_output = assistant_model(
                input_ids=bonus_tensor,
                past_key_values=assistant_cache,
                use_cache=True,
            )

        assistant_cache = assistant_output.past_key_values
        assistant_next_logits = assistant_output.logits[:, -1, :]
        assistant_draft_forward_calls += 1

        if bonus == tokenizer.eos_token_id:
            pending_target_token = None
            break

    return {
        "token_ids": generated[:max_new_tokens],
        "accepted_draft_tokens": accepted_draft_tokens,
        "proposed_draft_tokens": proposed_draft_tokens,
        "acceptance_rate": (
            accepted_draft_tokens / proposed_draft_tokens
            if proposed_draft_tokens
            else 0.0
        ),
        "target_verification_calls": target_verification_calls,
        "target_verified_candidate_tokens": target_verified_candidate_tokens,
        "assistant_draft_forward_calls": assistant_draft_forward_calls,
        "rejections": rejection_count,
    }


def run_reference(
    *,
    model,
    tokenizer,
    prompt: str,
    seed: int,
    max_new_tokens: int,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    prompt_length = int(inputs["input_ids"].shape[-1])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
            stop_strings=["\nQuestion:", "\nProblem:"],
            tokenizer=tokenizer,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated = output[0, prompt_length:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    generated_tokens = int(generated.shape[-1])

    return {
        "text": text,
        "tokens": generated_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": (
            generated_tokens / elapsed if elapsed > 0 else 0.0
        ),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }


def run_speculative(
    *,
    target_model,
    assistant_model,
    tokenizer,
    prompt: str,
    seed: int,
    max_new_tokens: int,
    num_assistant_tokens: int,
    warpers,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    prompt_length = int(inputs["input_ids"].shape[-1])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    decoded = speculative_decode(
        target_model=target_model,
        assistant_model=assistant_model,
        tokenizer=tokenizer,
        prompt_ids=inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        num_assistant_tokens=num_assistant_tokens,
        warpers=warpers,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated = torch.tensor(
        decoded["token_ids"],
        dtype=torch.long,
        device=inputs["input_ids"].device,
    )
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    generated_tokens = int(generated.shape[-1])

    result = {
        "text": text,
        "tokens": generated_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": (
            generated_tokens / elapsed if elapsed > 0 else 0.0
        ),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }
    result.update(
        {key: value for key, value in decoded.items() if key != "token_ids"}
    )
    return result


def main() -> int:
    args = parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.num_assistant_tokens <= 0:
        raise ValueError("--num-assistant-tokens must be positive")
    if args.hardware_interval <= 0:
        raise ValueError("--hardware-interval must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.dataset, args.questions)

    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", args.model)
    print("Dataset:", args.dataset)
    print("Questions:", len(rows))

    tokenizer, model = build_model(args.model)

    total_layers = len(model.model.layers)
    if total_layers != 36:
        raise ValueError(
            f"Expected Qwen3-4B-Base with 36 layers, found {total_layers}."
        )
    if not 0 < args.assistant_depth < total_layers:
        raise ValueError(
            f"Assistant depth must be between 1 and {total_layers - 1}."
        )

    assistant_model = make_shared_weight_assistant(
        model,
        depth=args.assistant_depth,
    )

    shared_weights = (
        assistant_model.model.layers[0].self_attn.q_proj.weight.data_ptr()
        == model.model.layers[0].self_attn.q_proj.weight.data_ptr()
    )

    warpers = make_warpers(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    print("Target layers:", total_layers)
    print("Assistant layers:", len(assistant_model.model.layers))
    print("Shared weights:", shared_weights)
    print("Assistant tokens:", args.num_assistant_tokens)
    print(
        "Sampling:",
        f"temperature={args.temperature}",
        f"top_p={args.top_p}",
        f"top_k={args.top_k}",
    )

    sampler = HardwareSampler(
        args.output / "hardware_samples.jsonl",
        args.hardware_interval,
    )

    baseline_results: list[dict[str, Any]] = []
    assisted_results: list[dict[str, Any]] = []

    sampler.start()
    try:
        for index, row in enumerate(rows, 1):
            seed = args.seed_base + index
            expected = extract_expected(str(row["answer"]))
            prompt = prompt_for(str(row["question"]))

            reference = run_reference(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                seed=seed,
                max_new_tokens=args.max_new_tokens,
            )
            reference.update(
                index=index,
                seed=seed,
                expected=expected,
                predicted=extract_predicted(reference["text"]),
            )
            reference["correct"] = reference["predicted"] == expected
            baseline_results.append(reference)

            assisted = run_speculative(
                target_model=model,
                assistant_model=assistant_model,
                tokenizer=tokenizer,
                prompt=prompt,
                seed=seed,
                max_new_tokens=args.max_new_tokens,
                num_assistant_tokens=args.num_assistant_tokens,
                warpers=warpers,
            )
            assisted.update(
                index=index,
                seed=seed,
                expected=expected,
                predicted=extract_predicted(assisted["text"]),
            )
            assisted["correct"] = assisted["predicted"] == expected
            assisted_results.append(assisted)

            print(
                f"[{index}/{len(rows)}] "
                f"expected={expected} "
                f"base={reference['predicted']}:{reference['correct']} "
                f"base_time={reference['elapsed_seconds']:.2f}s "
                f"spec={assisted['predicted']}:{assisted['correct']} "
                f"spec_time={assisted['elapsed_seconds']:.2f}s "
                f"accept={assisted['acceptance_rate']:.1%} "
                f"draft={assisted['proposed_draft_tokens']} "
                f"verified={assisted['target_verification_calls']}"
            )
    finally:
        sampler.stop()

    base_time = sum(x["elapsed_seconds"] for x in baseline_results)
    spec_time = sum(x["elapsed_seconds"] for x in assisted_results)
    base_tokens = sum(x["tokens"] for x in baseline_results)
    spec_tokens = sum(x["tokens"] for x in assisted_results)

    base_correct = sum(int(x["correct"]) for x in baseline_results)
    spec_correct = sum(int(x["correct"]) for x in assisted_results)

    proposed = sum(x["proposed_draft_tokens"] for x in assisted_results)
    accepted = sum(x["accepted_draft_tokens"] for x in assisted_results)
    verification_calls = sum(
        x["target_verification_calls"] for x in assisted_results
    )
    assistant_calls = sum(
        x["assistant_draft_forward_calls"] for x in assisted_results
    )

    result = {
        "schema_version": 3,
        "benchmark": "CFI-Eval-0006-SelfSpeculative",
        "method": (
            "Manual distribution-preserving speculative sampling with a "
            f"{args.assistant_depth}-layer shared-weight draft and a "
            f"{total_layers}-layer target. Cache ownership and rollback are "
            "handled explicitly with DynamicCache."
        ),
        "model": "Qwen3-4B-Base",
        "questions": len(rows),
        "target_layers": total_layers,
        "assistant_layers": len(assistant_model.model.layers),
        "weights_shared": shared_weights,
        "assistant_tokens": args.num_assistant_tokens,
        "max_new_tokens": args.max_new_tokens,
        "sampling": {
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed_base": args.seed_base,
        },
        "baseline": {
            "correct": base_correct,
            "accuracy": base_correct / len(rows),
            "total_time_seconds": base_time,
            "average_time_per_question_seconds": base_time / len(rows),
            "total_generated_tokens": base_tokens,
            "overall_tokens_per_second": (
                base_tokens / base_time if base_time > 0 else 0.0
            ),
            "mean_peak_vram_gib": (
                sum(x["peak_vram_gib"] for x in baseline_results) / len(rows)
            ),
            "runs": baseline_results,
        },
        "assisted": {
            "correct": spec_correct,
            "accuracy": spec_correct / len(rows),
            "total_time_seconds": spec_time,
            "average_time_per_question_seconds": spec_time / len(rows),
            "total_generated_tokens": spec_tokens,
            "overall_tokens_per_second": (
                spec_tokens / spec_time if spec_time > 0 else 0.0
            ),
            "mean_peak_vram_gib": (
                sum(x["peak_vram_gib"] for x in assisted_results) / len(rows)
            ),
            "proposed_draft_tokens": proposed,
            "accepted_draft_tokens": accepted,
            "acceptance_rate": accepted / proposed if proposed else 0.0,
            "target_verification_calls": verification_calls,
            "assistant_forward_calls": assistant_calls,
            "runs": assisted_results,
        },
        "speedup": base_time / spec_time if spec_time > 0 else 0.0,
        "hardware_samples": sampler.count,
    }

    result_path = args.output / "self_speculative_results.json"
    result_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 64)
    print("EXP-0006 MANUAL SELF-SPECULATIVE DECODING COMPLETE")
    print("=" * 64)
    print(
        f"Baseline: {base_correct}/{len(rows)} "
        f"accuracy={base_correct / len(rows):.1%} "
        f"time={base_time:.2f}s "
        f"tok/s={base_tokens / base_time:.2f}"
    )
    print(
        f"Speculative: {spec_correct}/{len(rows)} "
        f"accuracy={spec_correct / len(rows):.1%} "
        f"time={spec_time:.2f}s "
        f"tok/s={spec_tokens / spec_time:.2f}"
    )
    print(
        f"Draft acceptance: {accepted}/{proposed} "
        f"({accepted / proposed:.1%})" if proposed else "Draft acceptance: 0/0"
    )
    print(f"Target verification calls: {verification_calls}")
    print(f"Assistant forward calls: {assistant_calls}")
    print(f"Speedup: {result['speedup']:.3f}x")
    print(f"Hardware samples: {sampler.count}")
    print(f"Results: {result_path}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
