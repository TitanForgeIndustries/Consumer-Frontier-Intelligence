"""EXP-0010 paired sparse/dense byte-model training and smoke evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfi_v0.checkpoint import save_checkpoint
from cfi_v0.data import sample_batch, split_bytes
from cfi_v0.model import CFIConfig, CFIModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--balance-weight", type=float, default=0.01)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--shared-ff", type=int, default=512)
    parser.add_argument("--expert-ff", type=int, default=512)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--experts-per-group", type=int, default=2)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--runtime-tokens", type=int, default=16)
    parser.add_argument("--warmup-tokens", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    return parser


def save_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def validate(
    model: CFIModel,
    data: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    model.eval()
    generator = torch.Generator().manual_seed(args.seed + 1)
    losses = []
    windows_hash = hashlib.sha256()
    loads = [0] * (args.groups * args.experts_per_group)
    repeats = 0
    transitions = 0
    for _ in range(args.val_batches):
        inputs, targets = sample_batch(
            data, args.batch_size, args.window, generator, device
        )
        windows_hash.update(inputs.cpu().numpy().tobytes())
        logits, _, routes = model(inputs)
        batch_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        if not torch.isfinite(batch_loss):
            raise FloatingPointError("validation loss became nonfinite")
        losses.append(batch_loss.item())
        if routes is not None:
            counts = torch.bincount(routes.flatten(), minlength=len(loads)).cpu().tolist()
            loads = [old + count for old, count in zip(loads, counts)]
            repeats += int((routes[:, 1:] == routes[:, :-1]).sum().item())
            transitions += routes.shape[0] * (routes.shape[1] - 1)
    return {
        "validation_nll": sum(losses) / len(losses),
        "validation_windows_sha256": windows_hash.hexdigest(),
        "expert_load": loads if model.mode == "sparse" else None,
        "adjacent_same_route_fraction": repeats / transitions if transitions else None,
    }


@torch.inference_mode()
def generate_and_time(
    model: CFIModel, args: argparse.Namespace, device: torch.device
) -> tuple[dict, dict]:
    model.eval()
    prompt_bytes = args.prompt.encode("utf-8")
    if not 1 <= len(prompt_bytes) <= args.context:
        raise ValueError("prompt must contain 1 through context UTF-8 bytes")
    prefix = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=device)
    full = model.generate(prefix, args.generation_tokens)
    new_bytes = bytes(full[0, len(prompt_bytes):].cpu().tolist())
    generation = {
        "prompt": args.prompt,
        "bytes": list(new_bytes),
        "text": new_bytes.decode("utf-8", errors="replace"),
    }
    times = []
    peaks = []
    for _ in range(args.repeats):
        warmed_prefix = model.generate(prefix, args.warmup_tokens)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        model.generate(warmed_prefix, args.runtime_tokens)
        synchronize(device)
        times.append(time.perf_counter() - start)
        peaks.append(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
    mean = sum(times) / len(times)
    return generation, {
        "seconds": times,
        "mean_seconds": mean,
        "generated_tokens": args.runtime_tokens,
        "tokens_per_second": args.runtime_tokens / mean,
        "peak_cuda_allocated_bytes": max(peaks) if device.type == "cuda" else None,
    }


def run(args: argparse.Namespace) -> dict:
    if (args.steps < 1 or args.batch_size < 1 or args.val_batches < 1
            or args.repeats < 1 or args.runtime_tokens < 1
            or args.warmup_tokens < 1 or args.generation_tokens < 1
            or args.learning_rate <= 0 or args.balance_weight < 0
            or args.weight_decay < 0):
        raise ValueError("training, validation, and timing budgets must be positive")
    config = CFIConfig(
        width=args.width, layers=args.layers, heads=args.heads,
        shared_ff=args.shared_ff, expert_ff=args.expert_ff, groups=args.groups,
        experts_per_group=args.experts_per_group, context=args.context,
    )
    if not 1 <= args.window <= config.context:
        raise ValueError("window must be positive and no larger than model context")
    corpus = args.corpus.read_bytes()
    train, validation = split_bytes(corpus, config.context, args.train_fraction)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    corpus_hash = hashlib.sha256(corpus).hexdigest()
    summary = {
        "experiment": "EXP-0010",
        "status": "in_progress",
        "corpus": str(args.corpus),
        "corpus_sha256": corpus_hash,
        "train_bytes": len(train),
        "validation_bytes": len(validation),
        "seed": args.seed,
        "config": config.to_dict(),
        "protocol": {
            "steps": args.steps, "batch_size": args.batch_size,
            "window": args.window,
            "val_batches": args.val_batches, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "balance_weight": args.balance_weight,
            "train_fraction": args.train_fraction,
            "runtime_tokens": args.runtime_tokens,
            "warmup_tokens": args.warmup_tokens,
            "repeats": args.repeats,
            "generation_tokens": args.generation_tokens,
        },
        "software": {"torch": torch.__version__, "python": sys.version.split()[0]},
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "modes": {},
    }
    save_json(args.output / "summary.json", summary)

    for mode in ("sparse", "dense"):
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        model = CFIModel(config, mode).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        generator = torch.Generator().manual_seed(args.seed + 2)
        windows_hash = hashlib.sha256()
        losses = []
        model.train()
        for _ in range(args.steps):
            inputs, targets = sample_batch(
                train, args.batch_size, args.window, generator, device
            )
            windows_hash.update(inputs.cpu().numpy().tobytes())
            optimizer.zero_grad(set_to_none=True)
            logits, auxiliary, _ = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            objective = loss + (args.balance_weight * auxiliary if mode == "sparse" else 0)
            if not torch.isfinite(objective):
                raise FloatingPointError(f"{mode} loss became nonfinite")
            objective.backward()
            optimizer.step()
            losses.append(loss.item())

        del optimizer, inputs, targets, logits, auxiliary, loss, objective
        if device.type == "cuda":
            torch.cuda.empty_cache()
        validation_metrics = validate(model, validation, args, device)
        generation, timing = generate_and_time(model, args, device)
        checkpoint_path = args.output / f"{mode}.pt"
        save_checkpoint(checkpoint_path, model, corpus_hash, args.seed, args.steps)
        summary["modes"][mode] = {
            "train_windows_sha256": windows_hash.hexdigest(),
            "last_train_nll": losses[-1],
            "mean_train_nll": sum(losses) / len(losses),
            **validation_metrics,
            "total_parameters": model.total_parameters,
            "active_parameters_per_token": model.active_parameters_per_token,
            "resident_parameters": model.total_parameters,
            "timing": timing,
            "generation": generation,
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        }
        save_json(args.output / "summary.json", summary)
        print(
            f"{mode}: val NLL={validation_metrics['validation_nll']:.4f} "
            f"tokens/s={timing['tokens_per_second']:.2f}", flush=True
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary["status"] = "completed"
    save_json(args.output / "summary.json", summary)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    output_preexisted = args.output.exists()
    try:
        summary = run(args)
    except Exception as failure:
        if args.output.is_dir() and not output_preexisted and not isinstance(failure, FileExistsError):
            save_json(args.output / "implementation_failure.json", {
                "experiment": "EXP-0010",
                "error": repr(failure),
                "traceback": traceback.format_exc(),
            })
        raise
    print(f"EXP-0010 status: {summary['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
