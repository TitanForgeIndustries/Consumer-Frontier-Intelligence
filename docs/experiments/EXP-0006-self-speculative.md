# EXP-0006: Shared-Weight Truncated-Assistant Speculative Decoding

## Correction to the initial implementation

The first EXP-0006 implementation attempted to use Transformers' assistant_early_exit on Qwen3-4B-Base.

Transformers documents assistant_early_exit for checkpoints trained so that intermediate-layer logits are interpretable as early-exit predictions. Qwen3-4B-Base is not an early-exit-trained checkpoint. The resulting run failed inside the speculative cache path before producing an assisted result.

That failure is classified as an implementation/model-support failure, not a scientific result.

## Corrected method

The corrected experiment uses the ordinary assistant_model speculative-decoding interface.

A second assistant model object is constructed by shallow-copying the loaded target and replacing its layer list with the first 30 layers. The underlying weights/modules are shared with the 36-layer target, so the experiment does not load a second copy of the 4B parameters.

The 30-layer assistant drafts candidate tokens. The full 36-layer target verifies the candidate block in one forward pass.

## Gate

Run five fixed CFI GSM8K questions with the existing sampling protocol.

Record:
- GSM8K correctness
- wall-clock generation time
- generated tokens
- tokens/second
- peak VRAM
- whether target and assistant weights are shared

Run the 100-question evaluation only after the five-question gate succeeds.

## Interpretation

A speedup with equivalent target quality would demonstrate that computation can be reorganized into shallow drafting plus selective full-depth verification.

No speedup would indicate that the 30-layer draft cost and verification overhead do not amortize on the RTX 3070 for this model.

A quality discrepancy should trigger validation of the speculative sampler and generation settings before being interpreted scientifically.
