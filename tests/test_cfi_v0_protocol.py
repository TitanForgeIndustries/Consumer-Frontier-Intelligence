import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from cfi_v0.checkpoint import load_checkpoint, save_checkpoint
from cfi_v0.model import CFIConfig, CFIModel
import train_cfi_v0
import generate_cfi_v0
from train_cfi_v0 import build_parser, generate_and_time, run, validate


def test_checkpoint_rejects_wrong_version_and_restores_independent_model(tmp_path):
    model = CFIModel(CFIConfig(width=16, layers=1, heads=4, context=8), "dense")
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, corpus_sha256="abc", seed=4, step=2)
    restored, metadata = load_checkpoint(checkpoint, "cpu")

    assert metadata["corpus_sha256"] == "abc"
    assert metadata["step"] == 2
    with torch.no_grad():
        torch.testing.assert_close(
            model(torch.tensor([[1, 2, 3]]))[0],
            restored(torch.tensor([[1, 2, 3]]))[0],
        )
    payload = torch.load(checkpoint, weights_only=True)
    payload["architecture_version"] = 999
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="architecture version"):
        load_checkpoint(checkpoint, "cpu")


def test_paired_protocol_uses_same_windows_and_refuses_overwrite(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(bytes(range(256)) * 4)
    output = tmp_path / "run"
    args = build_parser().parse_args([
        "--corpus", str(corpus), "--output", str(output),
        "--device", "cpu", "--seed", "42", "--width", "16",
        "--layers", "1", "--heads", "4", "--shared-ff", "32",
        "--expert-ff", "24", "--groups", "2", "--experts-per-group", "2",
        "--context", "8", "--window", "8", "--batch-size", "2", "--steps", "2",
        "--val-batches", "1", "--runtime-tokens", "2",
        "--warmup-tokens", "1", "--repeats", "1",
        "--generation-tokens", "3", "--prompt", "A",
    ])

    summary = run(args)
    assert summary["status"] == "completed"
    assert summary["modes"]["sparse"]["train_windows_sha256"] == summary["modes"]["dense"]["train_windows_sha256"]
    assert summary["modes"]["sparse"]["validation_windows_sha256"] == summary["modes"]["dense"]["validation_windows_sha256"]
    assert summary["modes"]["sparse"]["total_parameters"] > summary["modes"]["sparse"]["active_parameters_per_token"]
    for mode in ("sparse", "dense"):
        assert summary["modes"][mode]["validation_nll"] > 0
        assert len(summary["modes"][mode]["generation"]["bytes"]) == 3
        restored, _ = load_checkpoint(output / f"{mode}.pt", "cpu")
        assert restored.mode == mode
    with pytest.raises(FileExistsError):
        run(args)


def test_runner_rejects_too_small_corpus_before_creating_output(tmp_path):
    corpus = tmp_path / "short.txt"
    corpus.write_bytes(b"abc")
    output = tmp_path / "run"
    args = build_parser().parse_args([
        "--corpus", str(corpus), "--output", str(output),
        "--device", "cpu", "--context", "16", "--window", "16",
    ])
    with pytest.raises(ValueError, match="byte splits"):
        run(args)
    assert not output.exists()


def test_runner_rejects_window_longer_than_context_before_output(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(bytes(range(256)) * 4)
    output = tmp_path / "run"
    args = build_parser().parse_args([
        "--corpus", str(corpus), "--output", str(output),
        "--device", "cpu", "--context", "8", "--window", "9",
    ])
    with pytest.raises(ValueError, match="window"):
        run(args)
    assert not output.exists()


def test_cli_refusal_does_not_write_into_existing_output(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(bytes(range(256)) * 4)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "summary.json"
    sentinel.write_text("original", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "train_cfi_v0.py", "--corpus", str(corpus), "--output", str(output),
        "--device", "cpu", "--window", "8", "--context", "8",
    ])
    with pytest.raises(FileExistsError):
        train_cfi_v0.main()
    assert sentinel.read_text(encoding="utf-8") == "original"
    assert not (output / "implementation_failure.json").exists()


def test_predeclared_128_byte_windows_with_256_byte_context(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(bytes(range(256)) * 32)
    output = tmp_path / "run"
    args = build_parser().parse_args([
        "--corpus", str(corpus), "--output", str(output), "--device", "cpu",
        "--width", "16", "--layers", "1", "--heads", "4",
        "--shared-ff", "32", "--expert-ff", "24", "--groups", "2",
        "--experts-per-group", "2", "--batch-size", "1", "--steps", "1",
        "--val-batches", "1", "--runtime-tokens", "1", "--warmup-tokens", "1",
        "--repeats", "1", "--generation-tokens", "1",
    ])
    sampled_lengths = []
    original_sample_batch = train_cfi_v0.sample_batch

    def record_batch(data, batch_size, context, generator, device):
        sampled_lengths.append(context)
        return original_sample_batch(data, batch_size, context, generator, device)

    monkeypatch.setattr(train_cfi_v0, "sample_batch", record_batch)
    summary = run(args)
    assert summary["config"]["context"] == 256
    assert summary["protocol"]["window"] == 128
    assert sampled_lengths == [128, 128, 128, 128]


def test_timing_uses_generated_warmup_as_prefix():
    class RecordingModel:
        def __init__(self):
            self.prefix_lengths = []

        def eval(self):
            return self

        def generate(self, prefix, max_new_tokens):
            self.prefix_lengths.append(prefix.shape[1])
            return torch.cat((prefix, torch.zeros(
                (1, max_new_tokens), dtype=torch.long
            )), dim=1)

    model = RecordingModel()
    args = SimpleNamespace(
        prompt="The ", context=256, generation_tokens=2,
        warmup_tokens=4, runtime_tokens=3, repeats=2,
    )
    _, timing = generate_and_time(model, args, torch.device("cpu"))
    assert model.prefix_lengths == [4, 4, 8, 4, 8]
    assert timing["generated_tokens"] == 3


def test_validation_nonfinite_loss_is_an_implementation_failure():
    class NonfiniteModel:
        mode = "dense"

        def eval(self):
            return self

        def __call__(self, inputs):
            return torch.full((*inputs.shape, 256), float("nan")), None, None

    args = SimpleNamespace(
        seed=42, batch_size=1, context=4, window=4, val_batches=1,
        groups=2, experts_per_group=2,
    )
    with pytest.raises(FloatingPointError, match="validation"):
        validate(NonfiniteModel(), torch.arange(20), args, torch.device("cpu"))


def test_standalone_cli_prints_invalid_utf8_on_cp1252_stdout(tmp_path, monkeypatch):
    class NonUtf8Model:
        mode = "dense"
        config = SimpleNamespace(context=8)

        def generate(self, prefix, max_new_tokens):
            return torch.cat((prefix, torch.full((1, max_new_tokens), 168)), dim=1)

    monkeypatch.setattr(
        generate_cfi_v0, "load_checkpoint",
        lambda _path, _device: (NonUtf8Model(), {"architecture_version": 1}),
    )
    monkeypatch.setattr(sys, "argv", [
        "generate_cfi_v0.py", "--checkpoint", str(tmp_path / "model.pt"),
        "--prompt", "A", "--max-new-tokens", "1",
    ])
    buffer = io.BytesIO()
    output = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", output)
    assert generate_cfi_v0.main() == 0
    output.flush()
    assert buffer.getvalue().decode("utf-8").replace("\r\n", "\n") == "A\ufffd\n"
