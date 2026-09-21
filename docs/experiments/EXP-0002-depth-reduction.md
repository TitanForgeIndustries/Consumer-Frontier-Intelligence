# EXP-0002: Dynamic Compute Feasibility via Depth Reduction

## Research question

Can a Qwen3-4B model retain useful reasoning capability when fewer transformer
layers are executed, reducing inference computation?

## Hypothesis

Not every GSM8K problem requires the full 36-layer Qwen3-4B computation path.
A controlled depth sweep should reveal whether meaningful accuracy survives at
12, 18, 24, and 30 layers, compared with the full 36-layer reference.

This is a feasibility experiment, not yet a learned per-input early-exit
system. The purpose is to measure the capability/computation tradeoff before
adding a learned or confidence-based exit policy.

## Controlled variables

The following remain fixed across all depth variants:

- Qwen3-4B-Base weights
- 4-bit NF4 quantization
- BF16 compute
- RTX 3070 8 GB
- fixed CFI-Eval-0001-GSM8K 100-question set
- per-question seed = 42000 + question index
- temperature = 0.6
- top_p = 0.95
- top_k = 20
- max_new_tokens = 512
- same prompt and answer-scoring rules

## Independent variable

Executed transformer depth:

- 12 layers
- 18 layers
- 24 layers
- 30 layers
- 36 layers (full-depth reference)

Qwen3-4B has 36 transformer layers. Hugging Face also exposes hidden
states at the output of each layer, which makes layerwise analysis possible.

## Primary measurements

For every depth variant:

- GSM8K accuracy
- generation time per question
- generated tokens
- tokens per second
- peak VRAM
- hardware telemetry during the sweep

## Interpretation

The experiment is successful if it produces a reproducible curve showing how
capability changes as executed depth decreases.

A lower-depth variant is not considered an improvement merely because it is
faster. The result must be interpreted jointly as capability versus compute.

## Next experiment

If intermediate depths preserve substantial capability, EXP-0003 can test a
true adaptive policy that decides the execution depth per input/token.

If capability collapses sharply with depth, the evidence will redirect CFI toward
other computational mechanisms such as recurrence, sparsity, routing, or memory.
