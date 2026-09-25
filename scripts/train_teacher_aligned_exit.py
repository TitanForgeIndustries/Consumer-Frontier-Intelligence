"""Train a teacher-aligned EXP-0003 intermediate exit adapter.

This variant keeps the full Qwen3-4B-Base backbone frozen and learns only a
small residual adapter after layer 30. During training, one frozen full-depth
forward provides both layer-30 and final-layer hidden states. The adapter is
trained to map the layer-30 representation toward the layer-36 representation,
with an auxiliary language-model loss on GSM8K targets.

This directly tests whether representation alignment, rather than a larger
adapter or backbone fine-tune, is the missing ingredient in the first EXP-0003
configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path
from cfi_paths import data_path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from cfi_experiment_logger.hardware import collect_hardware_snapshot
except ImportError:
    collect_hardware_snapshot = None


DEFAULT_MODEL = data_path("HuggingFace", "hub", "models--Qwen--Qwen3-4B-Base", "snapshots", "906bfd4b4dc7f14ee4320094d8b41684abff8539")
DEFAULT_OUTPUT = data_path("Models", "CFI-EXP-0003B-TeacherExit-30")
DEFAULT_DATASET_CACHE = data_path("HuggingFace", "datasets")
DEFAULT_DEPTH = 30
DEFAULT_TRAIN_EXAMPLES = 512
DEFAULT_MAX_LENGTH = 512
DEFAULT_STEPS = 100
DEFAULT_LR = 2e-4
DEFAULT_GRAD_ACCUM = 8
DEFAULT_BOTTLENECK = 256
DEFAULT_TEACHER_WEIGHT = 1.0
DEFAULT_LM_WEIGHT = 0.25
SEED = 42003


class ExitAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.activation = nn.SiLU()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.up(self.activation(self.down(hidden_states)))


class HardwareSampler:
    def __init__(self, path: Path, interval: float) -> None:
        self.path = path
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.count = 0

    def start(self) -> None:
        if collect_hardware_snapshot is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.thread is None:
            return
        self.stop_event.set()
        self.thread.join(timeout=max(5.0, self.interval * 2.0))
        self.thread = None

    def _run(self) -> None:
        while not self.stop_event.is_set():
            snapshot = collect_hardware_snapshot()
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot) + "\n")
            self.count += 1
            self.stop_event.wait(self.interval)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--dataset-cache", type=Path, default=DEFAULT_DATASET_CACHE)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--train-examples", type=int, default=DEFAULT_TRAIN_EXAMPLES)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--bottleneck", type=int, default=DEFAULT_BOTTLENECK)
    p.add_argument("--teacher-weight", type=float, default=DEFAULT_TEACHER_WEIGHT)
    p.add_argument("--lm-weight", type=float, default=DEFAULT_LM_WEIGHT)
    p.add_argument("--hardware-interval", type=float, default=2.0)
    return p.parse_args()


def format_example(question: str) -> str:
    return (
        "Question: "
        + question
        + "\n"
        + "Solve the problem step by step. "
        + "End with: The answer is <number>.\n"
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-0003B.")
    if args.depth < 1:
        raise ValueError("--depth must be positive")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        cache_dir=str(args.dataset_cache),
    )
    dataset = dataset.shuffle(seed=SEED).select(
        range(min(args.train_examples, len(dataset)))
    )

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        quantization_config=quant,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.eval()

    layers = model.model.layers
    total_layers = len(layers)
    if total_layers != 36:
        raise ValueError(
            f"Expected 36-layer Qwen3-4B-Base, found {total_layers}."
        )
    if args.depth >= total_layers:
        raise ValueError("Exit depth must be below full depth.")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    adapter = ExitAdapter(
        int(model.config.hidden_size),
        args.bottleneck,
    ).to(device="cuda:0", dtype=torch.bfloat16)
    adapter.train()

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    hardware_path = args.output / "hardware_samples.jsonl"
    sampler = HardwareSampler(hardware_path, args.hardware_interval)
    sampler.start()

    examples = 0
    step = 0
    accum = 0
    total_lm = 0.0
    total_teacher = 0.0
    started = time.perf_counter()

    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model layers:", total_layers)
    print("Exit depth:", args.depth)
    print("Training examples:", len(dataset))
    print("Steps:", args.steps)
    print("Teacher representation weight:", args.teacher_weight)
    print("LM loss weight:", args.lm_weight)

    try:
        while step < args.steps:
            row = dataset[examples % len(dataset)]
            examples += 1

            prompt = format_example(row["question"])
            target = row["answer"]

            prompt_tokens = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            target_tokens = tokenizer(
                target,
                add_special_tokens=False,
                truncation=True,
                max_length=max(
                    1,
                    args.max_length - prompt_tokens["input_ids"].shape[-1],
                ),
                return_tensors="pt",
            )

            prompt_ids = prompt_tokens["input_ids"][0]
            target_ids = target_tokens["input_ids"][0]
            input_ids = torch.cat([prompt_ids, target_ids], dim=0)[
                : args.max_length
            ].unsqueeze(0).to("cuda:0")

            labels = input_ids.clone()
            prompt_len = min(int(prompt_ids.shape[0]), int(labels.shape[1]))
            labels[:, :prompt_len] = -100
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                teacher = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_30 = teacher.hidden_states[args.depth].detach()
                hidden_36 = teacher.hidden_states[total_layers].detach()

            if accum == 0:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                adapted = adapter(hidden_30)

                # Match normalized representations, which removes arbitrary
                # hidden-state scale differences before cosine/MSE comparison.
                adapted_norm = F.normalize(adapted.float(), dim=-1)
                teacher_norm = F.normalize(hidden_36.float(), dim=-1)
                teacher_loss = F.mse_loss(adapted_norm, teacher_norm)

                normed = model.model.norm(adapted)
                logits = model.lm_head(normed).float()

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                lm_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                loss = (
                    args.teacher_weight * teacher_loss
                    + args.lm_weight * lm_loss
                ) / args.grad_accum

            loss.backward()
            accum += 1

            total_teacher += float(teacher_loss.detach().cpu())
            total_lm += float(lm_loss.detach().cpu())

            if accum >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                step += 1

                if step == 1 or step % 10 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"step={step}/{args.steps} "
                        f"teacher_loss={total_teacher / (step * args.grad_accum):.6f} "
                        f"lm_loss={total_lm / (step * args.grad_accum):.6f} "
                        f"elapsed={elapsed:.1f}s"
                    )

                if step % 50 == 0 or step == args.steps:
                    torch.save(
                        {
                            "depth": args.depth,
                            "hidden_size": int(model.config.hidden_size),
                            "bottleneck": args.bottleneck,
                            "seed": SEED,
                            "state_dict": adapter.state_dict(),
                            "variant": "teacher-aligned",
                            "teacher_weight": args.teacher_weight,
                            "lm_weight": args.lm_weight,
                        },
                        args.output / f"exit_adapter_step_{step}.pt",
                    )
    finally:
        sampler.stop()

    elapsed = time.perf_counter() - started

    torch.save(
        {
            "depth": args.depth,
            "hidden_size": int(model.config.hidden_size),
            "bottleneck": args.bottleneck,
            "seed": SEED,
            "state_dict": adapter.state_dict(),
            "variant": "teacher-aligned",
            "teacher_weight": args.teacher_weight,
            "lm_weight": args.lm_weight,
        },
        args.output / "exit_adapter.pt",
    )

    metadata = {
        "experiment_id": "EXP-0003B",
        "parent_experiment": "EXP-0003",
        "variant": "teacher-aligned-intermediate-exit",
        "model": str(args.model),
        "depth": args.depth,
        "total_model_layers": total_layers,
        "hidden_size": int(model.config.hidden_size),
        "bottleneck": args.bottleneck,
        "train_examples": len(dataset),
        "steps": step,
        "learning_rate": args.lr,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "teacher_weight": args.teacher_weight,
        "lm_weight": args.lm_weight,
        "elapsed_seconds": elapsed,
        "seed": SEED,
        "hardware_samples": sampler.count,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 64)
    print("EXP-0003B TEACHER-ALIGNED EXIT TRAINING COMPLETE")
    print("=" * 64)
    print(f"Depth: {args.depth}/{total_layers}")
    print(f"Steps: {step}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Hardware samples: {sampler.count}")
    print(f"Adapter: {args.output / 'exit_adapter.pt'}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
