"""EXP-0011 fixed-condition route-collapse and capacity comparison."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfi_v0.checkpoint import save_checkpoint
from cfi_v0.data import sample_batch, split_bytes
from cfi_v0.model import CFIConfig, CFIModel
from train_cfi_v0 import generate_and_time, save_json, validate


CORPUS_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
CONDITIONS = {
    "sparse_baseline": ("sparse", 512, 0.01),
    "sparse_balanced": ("sparse", 512, 1.0),
    "dense_active": ("dense", 512, 0.0),
    "dense_parameters": ("dense", 4096, 0.0),
}


def condition_spec(condition: str) -> tuple[str, int, float]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    return CONDITIONS[condition]


def route_statistics(loads: list[int]) -> dict:
    total = sum(loads)
    if total < 1:
        raise ValueError("route loads must be nonempty")
    proportions = [load / total for load in loads]
    entropy = -sum(share * math.log(share) for share in proportions if share)
    return {
        "effective_experts": math.exp(entropy),
        "max_route_share": max(proportions),
        "experts_at_least_one_percent": sum(share >= 0.01 for share in proportions),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-corpus-sha256", default=CORPUS_SHA256)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--shared-ff", type=int, default=512)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--experts-per-group", type=int, default=2)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--runtime-tokens", type=int, default=32)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    return parser


def run(args: argparse.Namespace) -> dict:
    mode, expert_ff, balance_weight = condition_spec(args.condition)
    if (args.steps < 1 or args.batch_size < 1 or args.val_batches < 1
            or args.repeats < 1 or args.runtime_tokens < 1
            or args.warmup_tokens < 1 or args.generation_tokens < 1
            or args.learning_rate <= 0 or args.weight_decay < 0):
        raise ValueError("training, validation, and timing budgets must be positive")
    config = CFIConfig(
        width=args.width, layers=args.layers, heads=args.heads,
        shared_ff=args.shared_ff, expert_ff=expert_ff, groups=args.groups,
        experts_per_group=args.experts_per_group, context=args.context,
    )
    if not 1 <= args.window <= config.context:
        raise ValueError("window must be positive and no larger than context")
    corpus = args.corpus.read_bytes()
    corpus_hash = hashlib.sha256(corpus).hexdigest()
    if corpus_hash != args.expected_corpus_sha256:
        raise ValueError(f"corpus SHA-256 mismatch: {corpus_hash}")
    train, validation = split_bytes(corpus, config.context, args.train_fraction)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    summary = {
        "experiment": "EXP-0011", "status": "in_progress",
        "condition": args.condition, "mode": mode, "seed": args.seed,
        "corpus": str(args.corpus), "corpus_sha256": corpus_hash,
        "train_bytes": len(train), "validation_bytes": len(validation),
        "config": config.to_dict(),
        "protocol": {
            "steps": args.steps, "batch_size": args.batch_size,
            "window": args.window, "val_batches": args.val_batches,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "balance_weight": balance_weight, "train_fraction": args.train_fraction,
            "runtime_tokens": args.runtime_tokens, "warmup_tokens": args.warmup_tokens,
            "repeats": args.repeats, "generation_tokens": args.generation_tokens,
        },
        "software": {"torch": torch.__version__, "python": sys.version.split()[0]},
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    save_json(args.output / "summary.json", summary)

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
    route_intervals = []
    interval_load = torch.zeros(config.groups * config.experts_per_group,
                                dtype=torch.long, device=device) if mode == "sparse" else None
    model.train()
    for step in range(1, args.steps + 1):
        inputs, targets = sample_batch(
            train, args.batch_size, args.window, generator, device
        )
        windows_hash.update(inputs.cpu().numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        logits, auxiliary, routes = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        objective = loss + balance_weight * auxiliary if mode == "sparse" else loss
        if not torch.isfinite(objective):
            raise FloatingPointError(f"{args.condition} step {step} nonfinite loss")
        objective.backward()
        optimizer.step()
        losses.append(loss.item())
        if routes is not None:
            interval_load += torch.bincount(
                routes.detach().flatten(), minlength=interval_load.numel()
            )
            if step % 64 == 0 or step == args.steps:
                route_intervals.append({
                    "through_step": step, "loads": interval_load.cpu().tolist()
                })
                interval_load.zero_()

    del optimizer, inputs, targets, logits, auxiliary, routes, loss, objective
    if device.type == "cuda":
        torch.cuda.empty_cache()
    summary["training"] = {
        "losses": losses, "last_nll": losses[-1],
        "mean_nll": sum(losses) / len(losses),
        "windows_sha256": windows_hash.hexdigest(),
        "route_intervals": route_intervals,
    }
    save_json(args.output / "summary.json", summary)
    evaluation = validate(model, validation, args, device)
    if evaluation["expert_load"] is not None:
        evaluation.update(route_statistics(evaluation["expert_load"]))
    summary["evaluation"] = evaluation
    save_json(args.output / "summary.json", summary)
    generation, timing = generate_and_time(model, args, device)
    summary["generation"] = generation
    summary["timing"] = timing
    summary["total_parameters"] = model.total_parameters
    summary["resident_parameters"] = model.total_parameters
    summary["active_parameters_per_token"] = model.active_parameters_per_token
    checkpoint_path = args.output / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, corpus_hash, args.seed, args.steps)
    summary["checkpoint_sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    summary["status"] = "completed"
    save_json(args.output / "summary.json", summary)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    output_preexisted = args.output.exists()
    try:
        summary = run(args)
    except Exception as failure:
        if args.output.is_dir() and not output_preexisted:
            save_json(args.output / "implementation_failure.json", {
                "experiment": "EXP-0011", "condition": args.condition,
                "seed": args.seed, "error": repr(failure),
                "traceback": traceback.format_exc(),
            })
        raise
    print(f"{summary['condition']} seed {summary['seed']}: "
          f"val NLL={summary['evaluation']['validation_nll']:.4f}, "
          f"bytes/s={summary['timing']['tokens_per_second']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
