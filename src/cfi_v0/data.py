"""Reproducible next-byte windows from an explicitly provided corpus."""

import torch


def split_bytes(
    data: bytes, context: int, fraction: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    if context < 1 or not 0 < fraction < 1:
        raise ValueError("context must be positive and train fraction between zero and one")
    cut = int(len(data) * fraction)
    if cut < context + 1 or len(data) - cut < context + 1:
        raise ValueError("both byte splits must hold at least context + 1 bytes")
    values = torch.tensor(list(data), dtype=torch.long)
    return values[:cut], values[cut:]


def sample_batch(
    data: torch.Tensor,
    batch_size: int,
    context: int,
    generator: torch.Generator,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size < 1 or context < 1 or data.numel() < context + 1:
        raise ValueError("invalid batch size, context, or corpus length")
    starts = torch.randint(
        0, data.numel() - context, (batch_size,), generator=generator
    )
    offsets = starts[:, None] + torch.arange(context)
    return data[offsets].to(device), data[offsets + 1].to(device)
