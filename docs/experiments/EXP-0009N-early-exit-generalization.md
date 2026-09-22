# EXP-0009N: Early-Exit Generalization with Matched Baselines

## Purpose

Validate the corrected EXP-0009L finding on a larger held-out split and compare the learned exit with a functional whole-L36 skip on the same traces.

Planned comparison:

1. full model oracle
2. raw H35 exit through the existing final RMSNorm + LM head
3. learned anchored H35 transition through the same existing final stack
4. functional whole-L36 skip

## Status

The branch contains a validation scaffold only. The training path is intentionally blocked until a differentiable frozen output-stack path is implemented correctly.

This prevents accidentally reporting a "trained" adapter whose gradient does not reach its parameters.

## Planned setup

- Qwen3-4B-Base
- 36 decoder layers
- 4-bit NF4
- BF16 compute
- RTX 3070
- eager attention
- CFI-Eval-0001-GSM8K
- 10 questions
- 7 train / 3 held out
- 128 max new tokens
- temperature 0.6
- top-p 0.95
- top-k 20
- 64 training positions/question
- bottleneck 128
- maximum update ratio 0.5
- 16 epochs
- learning rate 1e-3
- weight decay 1e-5

This experiment must not be run until the training implementation passes a gradient-flow unit check.
