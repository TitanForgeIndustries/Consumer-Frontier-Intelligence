"""Validate EXP-0006's manual decoder before stochastic benchmarking.

This diagnostic is deliberately greedy and short. It compares:
1. A full-recompute target oracle (no KV cache).
2. A cached 36-layer target greedy decoder.
3. The 30-layer shared-weight draft + 36-layer target speculative decoder.

The goal is token-level correctness, not benchmark accuracy. It isolates cache
alignment / acceptance-loop bugs without spending a full GSM8K run.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache


DEFAULT_MODEL = Path(
    r"E:\Titan Forge Industries\CFI-Data\HuggingFace\hub\models--Qwen--Qwen3-4B-Base\snapshots\906bfd4b4dc7f14ee4320094d8b41684abff8539"
)
DEFAULT_DATASET = Path(
    r"E:\Titan Forge Industries\CFI-Data\Datasets\CFI-Eval-0001-GSM8K\gsm8k_test_100.jsonl"
)


def load_question(path: Path, index: int) -> str:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(__import__("json").loads(line))
    if not 1 <= index <= len(rows):
        raise ValueError(f"Question index {index} outside dataset ({len(rows)} rows)")
    return str(rows[index - 1]["question"])


def load_model(model_path: Path):
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


def make_assistant(target_model, depth: int):
    assistant = copy.copy(target_model)
    assistant.__dict__ = target_model.__dict__.copy()
    assistant._modules = target_model._modules.copy()

    assistant_model = copy.copy(target_model.model)
    assistant_model.__dict__ = target_model.model.__dict__.copy()
    assistant_model._modules = target_model.model._modules.copy()

    assistant.model = assistant_model
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


def safe_crop(cache: DynamicCache, count: int) -> None:
    if count <= 0:
        return
    for layer in cache.layers:
        if getattr(layer, "keys", None) is None or getattr(layer, "values", None) is None:
            continue
        layer.crop(-count)


def full_recompute_greedy(model, prompt_ids: torch.Tensor, count: int) -> list[int]:
    ids = prompt_ids.clone()
    out: list[int] = []
    for _ in range(count):
        with torch.inference_mode():
            logits = model(input_ids=ids, use_cache=False).logits[:, -1, :]
        token = int(logits.argmax(dim=-1).item())
        out.append(token)
        ids = torch.cat(
            [ids, torch.tensor([[token]], device=ids.device, dtype=torch.long)],
            dim=1,
        )
    return out


def validate_cached_block(
    model,
    prompt_ids: torch.Tensor,
    block_tokens: torch.Tensor,
) -> bool:
    """Verify multi-token cached target logits against full recomputation."""
    cache = DynamicCache(config=model.config)
    with torch.inference_mode():
        prefill = model(
            input_ids=prompt_ids,
            past_key_values=cache,
            use_cache=True,
        )
        block_output = model(
            input_ids=block_tokens,
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )

        full_ids = torch.cat([prompt_ids, block_tokens], dim=1)
        full_output = model(input_ids=full_ids, use_cache=False)

    prompt_len = prompt_ids.shape[-1]
    block_len = block_tokens.shape[-1]
    cached_logits = block_output.logits[:, :, :]
    full_logits = full_output.logits[:, prompt_len : prompt_len + block_len, :]

    max_abs = (cached_logits.float() - full_logits.float()).abs().max().item()
    cached_tokens = cached_logits.argmax(dim=-1)[0].tolist()
    full_tokens = full_logits.argmax(dim=-1)[0].tolist()

    print("BLOCK CACHE MAX ABS LOGIT DIFF:", f"{max_abs:.6g}")
    print("BLOCK CACHE ARGMAX MATCH:", cached_tokens == full_tokens)
    if cached_tokens != full_tokens:
        first = next(
            (i for i, (a, b) in enumerate(zip(cached_tokens, full_tokens)) if a != b),
            None,
        )
        print("BLOCK CACHE FIRST MISMATCH:", first)
        print("Cached token:", cached_tokens[first] if first is not None else None)
        print("Full token:", full_tokens[first] if first is not None else None)
        return False
    return True


def cached_target_greedy(model, prompt_ids: torch.Tensor, count: int) -> list[int]:
    cache = DynamicCache(config=model.config)
    with torch.inference_mode():
        first = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    cache = first.past_key_values
    next_logits = first.logits[:, -1, :]
    out: list[int] = []

    for _ in range(count):
        token = int(next_logits.argmax(dim=-1).item())
        out.append(token)
        token_tensor = torch.tensor([[token]], device=prompt_ids.device, dtype=torch.long)
        with torch.inference_mode():
            step = model(input_ids=token_tensor, past_key_values=cache, use_cache=True)
        cache = step.past_key_values
        next_logits = step.logits[:, -1, :]

    return out


def speculative_greedy(
    target_model,
    assistant_model,
    tokenizer,
    prompt_ids: torch.Tensor,
    count: int,
    draft_k: int,
):
    """Greedy speculative decoding validated against a full-recompute target."""
    assistant_cache = DynamicCache(config=assistant_model.config)

    with torch.inference_mode():
        assistant_prefill = assistant_model(
            input_ids=prompt_ids,
            past_key_values=assistant_cache,
            use_cache=True,
        )

    assistant_cache = assistant_prefill.past_key_values
    assistant_next = assistant_prefill.logits[:, -1, :]
    generated: list[int] = []
    rounds = 0
    prompt_len = int(prompt_ids.shape[-1])

    while len(generated) < count:
        rounds += 1
        draft: list[int] = []

        for _ in range(min(draft_k, count - len(generated))):
            token = int(assistant_next.argmax(dim=-1).item())
            draft.append(token)
            token_tensor = torch.tensor(
                [[token]],
                device=prompt_ids.device,
                dtype=torch.long,
            )
            with torch.inference_mode():
                step = assistant_model(
                    input_ids=token_tensor,
                    past_key_values=assistant_cache,
                    use_cache=True,
                )
            assistant_cache = step.past_key_values
            assistant_next = step.logits[:, -1, :]

        context_len = prompt_len + len(generated)
        context_ids = torch.tensor(
            [generated],
            device=prompt_ids.device,
            dtype=torch.long,
        )
        draft_ids = torch.tensor(
            [draft],
            device=prompt_ids.device,
            dtype=torch.long,
        )
        full_ids = torch.cat([prompt_ids, context_ids, draft_ids], dim=1)

        with torch.inference_mode():
            target_output = target_model(
                input_ids=full_ids,
                use_cache=False,
            )

        # Logits at position context_len - 1 predict draft[0], and the next
        # len(draft) positions provide the target predictions for the full
        # candidate block plus its bonus token.
        target_logits = target_output.logits[
            :,
            context_len - 1 : context_len + len(draft),
            :,
        ]
        target_tokens = target_logits.argmax(dim=-1)[0].tolist()

        matches = 0
        while matches < len(draft) and draft[matches] == target_tokens[matches]:
            matches += 1

        generated.extend(draft[:matches])

        if len(generated) >= count:
            break

        committed = target_tokens[matches]
        generated.append(committed)

        rollback = len(draft) - matches
        if rollback:
            safe_crop(assistant_cache, rollback)

        committed_tensor = torch.tensor(
            [[committed]],
            device=prompt_ids.device,
            dtype=torch.long,
        )
        with torch.inference_mode():
            corrected = assistant_model(
                input_ids=committed_tensor,
                past_key_values=assistant_cache,
                use_cache=True,
            )

        assistant_cache = corrected.past_key_values
        assistant_next = corrected.logits[:, -1, :]

    return generated[:count], rounds



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--question", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--draft-tokens", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    tokenizer, model = load_model(args.model)
    question = load_question(args.dataset, args.question)
    prompt = (
        "Question: "
        + question
        + "\n"
        + "Solve the problem step by step. "
        + "End with: The answer is <number>.\n"
    )
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda:0")

    assistant = make_assistant(model, depth=30)

    print("Question:", args.question)
    print("Target layers:", len(model.model.layers))
    print("Assistant layers:", len(assistant.model.layers))
    print("Draft tokens:", args.draft_tokens)
    print("Validation tokens:", args.tokens)

    oracle = full_recompute_greedy(model, prompt_ids, args.tokens)
    cached = cached_target_greedy(model, prompt_ids, args.tokens)

    block_tokens = torch.tensor(
        [oracle[: min(args.draft_tokens, args.tokens)]],
        device=prompt_ids.device,
        dtype=torch.long,
    )
    if not validate_cached_block(model, prompt_ids, block_tokens):
        print("TARGET BLOCK CACHE CHECK: FAIL")
        return 4
    print("TARGET BLOCK CACHE CHECK: PASS")

    if oracle == cached:
        print("TARGET CACHE CHECK: PASS")
    else:
        first = next((i for i, (a, b) in enumerate(zip(oracle, cached)) if a != b), min(len(oracle), len(cached)))
        print("TARGET CACHE CHECK: FAIL")
        print("First mismatch:", first)
        return 2

    spec, rounds = speculative_greedy(
        model,
        assistant,
        tokenizer,
        prompt_ids,
        args.tokens,
        args.draft_tokens,
    )

    if oracle == spec:
        print("SPECULATIVE GREEDY CHECK: PASS")
        print("Rounds:", rounds)
        print("Token-identical to target oracle:", True)
        return 0

    first = next((i for i, (a, b) in enumerate(zip(oracle, spec)) if a != b), min(len(oracle), len(spec)))
    print("SPECULATIVE GREEDY CHECK: FAIL")
    print("First mismatch:", first)
    print("Oracle token:", oracle[first] if first < len(oracle) else None)
    print("Spec token:", spec[first] if first < len(spec) else None)
    print()
    print("Oracle:")
    print(tokenizer.decode(torch.tensor(oracle), skip_special_tokens=True))
    print()
    print("Speculative:")
    print(tokenizer.decode(torch.tensor(spec), skip_special_tokens=True))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
