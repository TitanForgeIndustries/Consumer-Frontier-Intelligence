"""Generate bytes with an independent CFI v0 checkpoint; no Qwen weights."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfi_v0.checkpoint import load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    prompt = args.prompt.encode("utf-8")
    if not 1 <= len(prompt) <= model.config.context:
        parser.error("prompt must contain 1 through context UTF-8 bytes")
    tokens = torch.tensor([list(prompt)], dtype=torch.long, device=args.device)
    generated = model.generate(tokens, args.max_new_tokens)
    output = bytes(generated[0, len(prompt):].cpu().tolist())
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(args.prompt + output.decode("utf-8", errors="replace"))
    print(
        f"[CFI v{metadata['architecture_version']} {model.mode}; "
        f"{len(output)} new bytes; no Qwen weights loaded]", file=sys.stderr
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
