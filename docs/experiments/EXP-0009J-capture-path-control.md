# EXP-0009J: Capture-Path Control

## Purpose

EXP-0009H measured a large stored-vs-fresh L36 attention difference even though final logits were reproducible. EXP-0009I then found exact fresh-to-fresh reproducibility when both forwards used the same configuration.

J isolates whether the discrepancy comes from the Transformers forward configuration.

## Conditions

For each identical generated sequence, capture the same objects under four forwards:

- False A: \`output_hidden_states=False\`
- False B: \`output_hidden_states=False\`
- True A: \`output_hidden_states=True\`
- True B: \`output_hidden_states=True\`

Compare:

1. False A vs False B
2. True A vs True B
3. True A vs False A

Captured objects:

- H35
- L36 attention
- L36 MLP
- H36
- final logits

## Decision

- False/false and true/true both stable, true-vs-false differs -> the configuration changes internal component tensors while preserving/recovering output behavior.
- Both same-path comparisons stable and true-vs-false also stable -> H's earlier mismatch had another cause.
- Same-path instability remains -> investigate hook ordering or tensor aliasing.

## Setup

- Qwen3-4B-Base
- 36 layers
- 4-bit NF4
- BF16 compute
- RTX 3070
- eager attention
- 4 questions
- 128 maximum generated tokens
- seed base 42000

Run:

    python scripts/run_exp0009j.py --questions 4 --max-new-tokens 128 --output "E:\\Titan Forge Industries\\CFI-Data\\Results\\CFI-Eval-0009J-Capture-Path-Control"

## Limits

J is a control experiment. It does not claim a scientific architecture result or speedup.
