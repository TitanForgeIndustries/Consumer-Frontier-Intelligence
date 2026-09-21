# EXP-0006: Manual Shared-Weight Self-Speculative Decoding

## Why the implementation changed

The first implementation attempted to use Transformers' `assistant_early_exit` path on Qwen3-4B-Base.

That checkpoint was not trained for intermediate-layer early exits. The run failed inside speculative cache rollback before producing an assisted result.

The second implementation used the ordinary `assistant_model` interface with a 30-layer assistant object sharing the target model's loaded weights. That path also failed during cache rollback. Switching the generation cache to `StaticCache` did not resolve the failure.

Those failures are classified as implementation/model-support failures, not scientific results.

## Current method

EXP-0006 now implements the speculative decoder directly in CFI instead of calling Transformers' assisted-generation machinery.

The target remains the full 36-layer Qwen3-4B-Base model. The draft assistant is a separate model object that executes only the first 30 layers while sharing the target's loaded modules and weights.

Each speculative iteration:

1. The 30-layer assistant drafts up to K candidate tokens with a normal KV cache.
2. The 36-layer target verifies the entire candidate block in one forward pass.
3. Candidate tokens are accepted with the standard distribution-preserving speculative-sampling rule.
4. On rejection, the residual distribution `max(0, target - draft)` is sampled and the rejected cache suffix is explicitly cropped.
5. The target cache may intentionally lag by one emitted token so the replacement/bonus token can be consumed together with the next verification block.

This makes cache ownership and rollback explicit rather than relying on the generic `_assisted_decoding()` implementation.

## Sampling protocol

The benchmark keeps the established CFI GSM8K generation settings:

- 4-bit NF4 quantization
- BF16 compute
- `do_sample=True`
- temperature 0.6
- top_p 0.95
- top_k 20
- per-question seed = 42000 + index
- maximum 512 new tokens
- fixed CFI 100-question GSM8K subset

The implementation uses the same Transformers temperature/top-k/top-p logits warpers for both draft and target distributions.

## Gate

Run five fixed CFI GSM8K questions first.

Record:

- GSM8K correctness
- wall-clock generation time
- generated tokens
- tokens/second
- peak VRAM
- draft tokens proposed
- draft tokens accepted
- acceptance rate
- target verification calls
- assistant forward calls
- whether target and assistant weights are shared

Run the 100-question evaluation only after the five-question gate succeeds.

## Interpretation

A successful result with measurable speedup and preserved target quality would support the CFI hypothesis that useful computation can be reorganized into shallow shared-weight drafting plus selective full-depth verification.

A quality discrepancy should first trigger validation of the manual sampler and cache transitions before being interpreted scientifically.

A negative speed result is still informative: it would indicate that, for this model and RTX 3070, the 30-layer draft cost does not amortize sufficiently against target verification overhead.
