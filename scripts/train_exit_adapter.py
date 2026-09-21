"""Train a lightweight Qwen3-4B intermediate exit adapter for EXP-0003.

The 36-layer Qwen3-4B-Base backbone is frozen. Training executes only the first
30 layers and learns a residual bottleneck adapter immediately before the
original final RMSNorm and LM head.

The resulting adapter can be used with scripts/evaluate_early_exit.py.

Default training data is the GSM8K train split with a deterministic 512-example
subset. All model/data/output paths default to the user's E: CFI-Data layout.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from cfi_experiment_logger.core import ExperimentLogger
    from cfi_experiment_logger.hardware import collect_hardware_snapshot
except ImportError:
    ExperimentLogger = None
    collect_hardware_snapshot = None


DEFAULT_MODEL = Path(
    r"E:\Titan Forge Industries\CFI-Data\HuggingFace\hub\models--Qwen--Qwen3-4B-Base\snapshots\906bfd4b4dc7f14ee4320094d8b41684abff8539"
)
DEFAULT_OUTPUT = Path(
    r"E:\Titan Forge Industries\CFI-Data\Models\CFI-EXP-0003-EarlyExit-30"
)
DEFAULT_DATASET_CACHE = Path(
    r"E:\Titan Forge Industries\CFI-Data\HuggingFace\datasets"
)
DEFAULT_DEPTH = 30
DEFAULT_TRAIN_EXAMPLES = 512
DEFAULT_MAX_LENGTH = 512
DEFAULT_STEPS = 300
DEFAULT_LR = 2e-4
DEFAULT_GRAD_ACCUM = 8
DEFAULT_BOTTLENECK = 256
SEED = 42003


class ExitAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck, bias=True)
        self.up = nn.Linear(bottleneck, hidden_size, bias=True)
        self.activation = nn.SiLU()

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.up(self.activation(self.down(hidden_states)))


class AdapterNorm(nn.Module):
    def __init__(self, adapter: ExitAdapter, norm: nn.Module) -> None:
        super().__init__()
        self.adapter = adapter
        self.norm = norm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.adapter(hidden_states))


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
    p.add_argument("--experiment-id", default="EXP-0003")
    p.add_argument("--experiment-root", type=Path, default=Path("experiments"))
    p.add_argument("--hardware-interval", type=float, default=2.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.depth < 1:
        raise ValueError("--depth must be positive")

    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        cache_dir=str(args.dataset_cache),
    )
    dataset = dataset.shuffle(seed=SEED).select(range(min(args.train_examples, len(dataset))))

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

    if args.depth >= total_layers:
        raise ValueError("EXP-0003 requires an intermediate depth below full depth.")

    model.model.layers = nn.ModuleList(list(layers[:args.depth]))

    hidden_size = model.config.hidden_size
    adapter = ExitAdapter(hidden_size, args.bottleneck).to(
        device="cuda:0",
        dtype=torch.bfloat16,
    )

    original_norm = model.model.norm
    model.model.norm = AdapterNorm(adapter, original_norm)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for parameter in adapter.parameters():
        parameter.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    logger = None
    hardware_path = args.output / "hardware_samples.jsonl"

    if args.experiment_id and ExperimentLogger is not None:
        logger = ExperimentLogger(args.experiment_id, args.experiment_root)
        if not logger.config_path.exists():
            raise FileNotFoundError(
                f"Experiment is not initialized: {logger.path}"
            )
        hardware_path = logger.hardware_path

    sampler = HardwareSampler(hardware_path, args.hardware_interval)
    sampler.start()

    examples = 0
    step = 0
    accum = 0
    running_loss = 0.0
    started = time.perf_counter()

    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model layers:", total_layers)
    print("Exit depth:", args.depth)
    print("Training examples:", len(dataset))
    print("Steps:", args.steps)

    try:
        while step < args.steps:
            row = dataset[examples % len(dataset)]
            examples += 1

            prompt = (
                "Question: "
                + row["question"]
                + "\n"
                + "Solve the problem step by step. "
                + "End with: The answer is <number>.\n"
            )
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
                max_length=max(1, args.max_length - prompt_tokens["input_ids"].shape[-1]),
                return_tensors="pt",
            )

            prompt_ids = prompt_tokens["input_ids"][0]
            target_ids = target_tokens["input_ids"][0]

            input_ids = torch.cat([prompt_ids, target_ids], dim=0)
            input_ids = input_ids[:args.max_length].unsqueeze(0).to("cuda:0")

            labels = input_ids.clone()
            prompt_len = min(prompt_ids.shape[0], labels.shape[1])
            labels[:, :prompt_len] = -100

            attention_mask = torch.ones_like(input_ids)

            optimizer.zero_grad(set_to_none=True) if accum == 0 else None

            with torch.no_grad():
                pass

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                loss = outputs.loss / args.grad_accum

            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.grad_accum
            accum += 1

            if accum >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                accum = 0

                if step == 1 or step % 10 == 0:
                    elapsed = time.perf_counter() - started
                    avg = running_loss / step
                    print(
                        f"step={step}/{args.steps} "
                        f"loss={avg:.4f} "
                        f"elapsed={elapsed:.1f}s"
                    )
                    if logger is not None:
                        logger.record_training(
                            step=step,
                            loss=avg,
                            learning_rate=args.lr,
                            elapsed_seconds=elapsed,
                        )

    finally:
        sampler.stop()

    elapsed = time.perf_counter() - started

    torch.save(
        {
            "depth": args.depth,
            "hidden_size": hidden_size,
            "bottleneck": args.bottleneck,
            "seed": SEED,
            "state_dict": adapter.state_dict(),
        },
        args.output / "exit_adapter.pt",
    )

    metadata = {
        "experiment_id": args.experiment_id,
        "model": str(args.model),
        "depth": args.depth,
        "total_model_layers": total_layers,
        "hidden_size": hidden_size,
        "bottleneck": args.bottleneck,
        "train_examples": len(dataset),
        "steps": step,
        "learning_rate": args.lr,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "elapsed_seconds": elapsed,
        "seed": SEED,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 64)
    print("EXP-0003 EXIT ADAPTER TRAINING COMPLETE")
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
