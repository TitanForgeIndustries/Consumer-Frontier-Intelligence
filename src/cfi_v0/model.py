"""A causal shared core with a physically selected hierarchical expert residual."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CFIConfig:
    width: int = 256
    layers: int = 4
    heads: int = 4
    shared_ff: int = 512
    expert_ff: int = 512
    groups: int = 4
    experts_per_group: int = 2
    context: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        sizes = (
            self.width, self.layers, self.heads, self.shared_ff,
            self.expert_ff, self.groups, self.experts_per_group, self.context,
        )
        if any(size < 1 for size in sizes) or self.width % self.heads:
            raise ValueError("positive sizes and a width divisible by heads are required")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)


def feed_forward(width: int, expanded: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(width, expanded), nn.GELU(),
        nn.Linear(expanded, width), nn.Dropout(dropout),
    )


class CausalBlock(nn.Module):
    def __init__(self, config: CFIConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = nn.MultiheadAttention(
            config.width, config.heads, dropout=config.dropout, batch_first=True
        )
        self.feed_forward_norm = nn.LayerNorm(config.width)
        self.feed_forward = feed_forward(config.width, config.shared_ff, config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        length = hidden.shape[1]
        future_mask = torch.ones(
            length, length, dtype=torch.bool, device=hidden.device
        ).triu_(diagonal=1)
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized, normalized, normalized, attn_mask=future_mask,
            need_weights=False,
        )
        hidden = hidden + attended
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))


class SparseResidual(nn.Module):
    def __init__(self, config: CFIConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.width)
        self.group_router = nn.Linear(config.width, config.groups)
        self.expert_router = nn.Linear(
            config.width, config.groups * config.experts_per_group
        )
        self.groups = config.groups
        self.experts_per_group = config.experts_per_group
        self.experts = nn.ModuleList([
            feed_forward(config.width, config.expert_ff, config.dropout)
            for _ in range(config.groups * config.experts_per_group)
        ])

    def forward(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, width = hidden.shape
        normalized = self.norm(hidden)
        group_probs = self.group_router(normalized).softmax(dim=-1)
        all_expert_probs = self.expert_router(normalized).reshape(
            batch, length, self.groups, self.experts_per_group
        ).softmax(dim=-1)
        selected_group = group_probs.argmax(dim=-1)
        chosen_group_probs = all_expert_probs.gather(
            2, selected_group[..., None, None].expand(
                -1, -1, 1, self.experts_per_group
            ),
        ).squeeze(2)
        selected_expert = chosen_group_probs.argmax(dim=-1)
        routes = selected_group * self.experts_per_group + selected_expert
        gate = (
            group_probs.gather(2, selected_group[..., None]).squeeze(-1)
            * chosen_group_probs.gather(2, selected_expert[..., None]).squeeze(-1)
        )

        flat_hidden = normalized.reshape(-1, width)
        flat_routes = routes.reshape(-1)
        outputs = torch.zeros_like(flat_hidden)
        for index, expert in enumerate(self.experts):
            positions = torch.nonzero(flat_routes == index, as_tuple=True)[0]
            if positions.numel():
                outputs.index_copy_(
                    0, positions, expert(flat_hidden.index_select(0, positions))
                )

        expected = (group_probs[..., :, None] * all_expert_probs).reshape(
            batch, length, -1
        ).mean(dim=(0, 1))
        observed = F.one_hot(flat_routes, len(self.experts)).float().mean(dim=0)
        balance = len(self.experts) * (observed.detach() * expected).sum()
        residual = outputs.reshape(batch, length, width) * gate[..., None]
        return hidden + residual, balance, routes


class CFIModel(nn.Module):
    def __init__(self, config: CFIConfig, mode: str = "sparse") -> None:
        super().__init__()
        if mode not in {"sparse", "dense"}:
            raise ValueError("mode must be sparse or dense")
        self.config = config
        self.mode = mode
        self.tokens = nn.Embedding(256, config.width)
        self.positions = nn.Embedding(config.context, config.width)
        self.blocks = nn.ModuleList([CausalBlock(config) for _ in range(config.layers)])
        self.sparse_layer = SparseResidual(config) if mode == "sparse" else None
        self.dense_layer = (
            nn.Sequential(
                nn.LayerNorm(config.width),
                feed_forward(config.width, config.expert_ff, config.dropout),
            ) if mode == "dense" else None
        )
        self.output_norm = nn.LayerNorm(config.width)

    @property
    def total_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def active_parameters_per_token(self) -> int:
        if self.sparse_layer is None:
            return self.total_parameters
        one_expert = sum(parameter.numel() for parameter in self.sparse_layer.experts[0].parameters())
        return self.total_parameters - (len(self.sparse_layer.experts) - 1) * one_expert

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= self.config.context:
            raise ValueError("input must be [batch, length] within the context window")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.tokens(tokens) + self.positions(positions)
        for block in self.blocks:
            hidden = block(hidden)
        routes = None
        auxiliary = hidden.new_zeros(())
        if self.sparse_layer is not None:
            hidden, auxiliary, routes = self.sparse_layer(hidden)
        else:
            hidden = hidden + self.dense_layer(hidden)
        logits = F.linear(self.output_norm(hidden), self.tokens.weight)
        return logits, auxiliary, routes

    @torch.inference_mode()
    def generate(self, prefix: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        if prefix.ndim != 2 or prefix.shape[1] < 1 or max_new_tokens < 0:
            raise ValueError("non-empty [batch, length] prefix and nonnegative length required")
        generated = prefix
        for _ in range(max_new_tokens):
            logits, _, _ = self(generated[:, -self.config.context:])
            next_byte = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_byte), dim=1)
        return generated
