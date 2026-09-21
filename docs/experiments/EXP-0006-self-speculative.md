# EXP-0006: Self-Speculative Decoding at Layer 30

## Research question

Can the useful predictive information already present at layer 30 be exploited
as a cheap draft path while the remaining layers verify and correct the draft?

## Motivation

EXP-0004 showed strong teacher-forced predictive signal at layer 30.
EXP-0005 showed that layer 30 agrees with the actual sampled token 64.78% of the
time during free-running generation.

EXP-0002 and EXP-0003 showed that using layer 30 as a standalone generator
fails. Self-speculative decoding tests a different computational organization:
layer 30 proposes several tokens, then layers 31-36 verify the block.

## Method

Use the native Transformers self-speculative decoding path with:

- Qwen3-4B-Base
- intermediate assistant/exit layer: 30
- same tokenizer
- 4-bit NF4 + double quant
- BF16 compute
- same fixed CFI GSM8K evaluation set
- same sampling parameters
- initial speculative block: 4 tokens
- constant speculative block size for the first gate

The target model remains the original 36-layer model. No weights are trained.

## Baseline

For the same questions and seeds, measure ordinary 36-layer generation.

## Metrics

- GSM8K accuracy
- total generation time
- average seconds/question
- generated tokens
- overall tokens/second

The primary systems result is wall-clock speed while preserving the target
model's output behavior.

## Gate

Run 5 questions first.

If self-speculative decoding produces correct outputs and a measurable wall-time
improvement, run the full 100-question CFI benchmark.

If it is correct but not faster, investigate dynamic speculation length and
implementation overhead before treating the approach as unsuccessful.

If it changes output quality unexpectedly, inspect the generation configuration
and model support before interpreting it as a scientific result.

## Scientific significance

This is the first CFI experiment that directly tests a computational
organization in which shallow computation is used for most token proposals and
deep computation is reserved for verification. This is closer to conditional
computation than simply truncating or adapting the model.
