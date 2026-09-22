# EXP-0009H: Layer-36 Sublayer Predictability Diagnostic

## Research question

Which part of the H35 -> H36 transition is actually predictable from H35?

After EXP-0009D through EXP-0009G, multiple state predictors remained close to source-state copying. This experiment stops changing the predictor objective and instead decomposes the actual layer-36 transformation.

## Measurements

For each token position the experiment captures:

- H35 source state
- layer-36 attention output
- layer-36 MLP output
- final H36 decoder-layer output

Separate compact probes are trained for:

1. attention output prediction from H35
2. MLP output prediction from H35

The probes use same-position H35 only. This deliberately gives a clean diagnostic of the information available in the source representation without adding another architectural mechanism.

## Downstream substitution

For each learned component predictor, evaluation performs:

- exact original component injection
- learned predicted component injection
- zero-component injection

The exact component injection must reproduce the oracle.

For attention prediction, the predicted attention output replaces the real layer-36 attention output while the real MLP executes normally.

For MLP prediction, the predicted MLP output replaces the real layer-36 MLP output while the real attention executes normally.

This separates representation predictability from downstream usefulness.

## Hypotheses

### A. One component is predictable

A component-level bottleneck may explain why whole-state prediction stalls.

### B. Both components are predictable but substitution is unstable

The information exists in H35, but the approximation is not sufficiently precise for downstream computation.

### C. Neither component is predictable

The missing information is not recoverable from H35 alone. A richer source representation is required.

## Controlled setup

- Qwen3-4B-Base
- 36 layers
- 4-bit NF4
- BF16 compute
- RTX 3070
- CFI-Eval-0001-GSM8K
- 4 questions
- 2 training questions
- 128 maximum generated tokens
- 4 probe-training epochs
- 256-dimensional bottleneck
- seed base 42000

Smoke command:

    python scripts/run_exp0009h.py --questions 4 --train-questions 2 --max-new-tokens 128 --epochs 4 --output "E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009H-Sublayer-Predictability-Diagnostic"

## Important limits

This experiment is diagnostic and teacher-forced.

It does not remove layer-36 computation.

It does not demonstrate a hardware speedup.

A positive component result does not by itself establish a usable replacement architecture.


## Control hardening after first run

The first smoke run failed the exact attention injection control at:

- top-1 = 0.937500
- KL = 0.07772846

Because the failure occurred in the mandatory exact control, no learned attention-predictor result was interpreted.

The implementation was hardened in commit `18c4b0e140cc30719057e2c1679b606d6b0445b7`:

- H now loads the established NF4 model with eager attention for this diagnostic
- a forward-to-forward repeatability control is run before component replay
- the repeatability control must itself reproduce the untouched model
- only then is exact component injection evaluated

This separates CUDA attention-kernel reproducibility from a hook-boundary or component-capture error.

The first H run therefore remains a control failure, not a scientific negative result.
