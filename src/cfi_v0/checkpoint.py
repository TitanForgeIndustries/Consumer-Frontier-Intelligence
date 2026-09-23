"""State-only checkpoints for the independent byte-language model."""

from pathlib import Path

import torch

from .model import CFIConfig, CFIModel


ARCHITECTURE_VERSION = 1


def save_checkpoint(
    path: Path,
    model: CFIModel,
    corpus_sha256: str,
    seed: int,
    step: int,
) -> None:
    payload = {
        "architecture_version": ARCHITECTURE_VERSION,
        "vocabulary": "utf8-bytes-256",
        "config": model.config.to_dict(),
        "mode": model.mode,
        "corpus_sha256": corpus_sha256,
        "seed": seed,
        "step": step,
        "state_dict": model.state_dict(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path, device: str | torch.device = "cpu"
) -> tuple[CFIModel, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("unsupported CFI architecture version")
    if payload.get("vocabulary") != "utf8-bytes-256":
        raise ValueError("unsupported CFI vocabulary")
    model = CFIModel(CFIConfig(**payload["config"]), payload["mode"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), {
        key: value for key, value in payload.items() if key != "state_dict"
    }
