import io
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfi_v0.data import sample_batch, split_bytes
from cfi_v0.model import CFIConfig, CFIModel


def small_config() -> CFIConfig:
    return CFIConfig(
        width=16,
        layers=1,
        heads=4,
        shared_ff=32,
        expert_ff=24,
        groups=2,
        experts_per_group=2,
        context=16,
    )


def test_invalid_config_and_context_fail_early():
    with pytest.raises(ValueError):
        CFIModel(CFIConfig(width=15, heads=4), "sparse")
    with pytest.raises(ValueError):
        CFIModel(small_config(), "unknown")
    with pytest.raises(ValueError):
        CFIModel(small_config(), "dense")(
            torch.zeros((1, 17), dtype=torch.long)
        )


@pytest.mark.parametrize("mode", ["sparse", "dense"])
def test_future_bytes_do_not_change_past_logits(mode):
    torch.manual_seed(11)
    model = CFIModel(small_config(), mode).eval()
    first = torch.tensor([[4, 5, 6, 7, 8]])
    second = torch.tensor([[4, 5, 6, 7, 99]])

    with torch.no_grad():
        left, _, _ = model(first)
        right, _, _ = model(second)

    torch.testing.assert_close(left[:, :-1], right[:, :-1], atol=1e-6, rtol=1e-6)


def test_only_selected_expert_executes():
    model = CFIModel(small_config(), "sparse")
    for router in (model.sparse_layer.group_router, model.sparse_layer.expert_router):
        torch.nn.init.zeros_(router.weight)
        torch.nn.init.zeros_(router.bias)
    called = [0 for _ in model.sparse_layer.experts]
    handles = [
        expert.register_forward_hook(lambda _module, _inputs, _output, index=index: called.__setitem__(index, called[index] + 1))
        for index, expert in enumerate(model.sparse_layer.experts)
    ]
    try:
        _, _, routes = model(torch.tensor([[1, 2, 3, 4]]))
    finally:
        for handle in handles:
            handle.remove()

    assert routes.tolist() == [[0, 0, 0, 0]]
    assert called == [1, 0, 0, 0]


def test_router_and_selected_expert_receive_training_gradients():
    torch.manual_seed(3)
    model = CFIModel(small_config(), "sparse")
    logits, auxiliary, routes = model(torch.randint(0, 256, (2, 9)))
    (logits.square().mean() + 0.01 * auxiliary).backward()

    assert routes.shape == (2, 9)
    assert torch.isfinite(auxiliary)
    assert model.sparse_layer.group_router.weight.grad.abs().sum() > 0
    assert model.sparse_layer.expert_router.weight.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for expert in model.sparse_layer.experts
        for parameter in expert.parameters()
    )


def test_state_round_trip_and_independent_generation():
    torch.manual_seed(5)
    config = small_config()
    model = CFIModel(config, "sparse").eval()
    tokens = torch.tensor([[72, 105]])
    checkpoint = io.BytesIO()
    torch.save({"config": config.to_dict(), "mode": "sparse", "state_dict": model.state_dict()}, checkpoint)
    checkpoint.seek(0)
    stored = torch.load(checkpoint, weights_only=True)
    loaded = CFIModel(CFIConfig(**stored["config"]), stored["mode"]).eval()
    loaded.load_state_dict(stored["state_dict"], strict=True)

    with torch.no_grad():
        torch.testing.assert_close(model(tokens)[0], loaded(tokens)[0])
        assert torch.equal(model.generate(tokens, 3), loaded.generate(tokens, 3))
    assert len(loaded.generate(tokens, 3)[0]) == 5
    assert model.total_parameters > model.active_parameters_per_token


def test_split_and_paired_batches_are_deterministic():
    with pytest.raises(ValueError):
        split_bytes(b"short", context=8, fraction=0.9)
    with pytest.raises(ValueError):
        split_bytes(b"x" * 64, context=8, fraction=1.0)
    train, validation = split_bytes(bytes(range(100)), context=8, fraction=0.8)
    assert len(train) == 80 and len(validation) == 20
    assert train[-1] == 79 and validation[0] == 80

    first = sample_batch(train, 3, 8, torch.Generator().manual_seed(42), "cpu")
    second = sample_batch(train, 3, 8, torch.Generator().manual_seed(42), "cpu")
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert first[0].shape == (3, 8) and first[1].shape == (3, 8)
