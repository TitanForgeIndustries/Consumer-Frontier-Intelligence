# EXP-0009M: L36 Boundary Equivalence Control

## Purpose

EXP-0009I found a mild behavioral effect when L36 was functionally skipped, while EXP-0009L found that directly sending the stored H35 state through the final RMSNorm + LM head produced catastrophic mismatch.

These should be equivalent if the stored H35 is exactly the input to L36.

M isolates that boundary on the same forward.

## Same-forward measurements

For each generated sequence:

1. Capture H35 from layer 35 output.
2. Run another forward where a post-hook on L36 captures its exact input and exact output, then returns a clone of the input as the layer output.
3. Compare H35 with the actual L36 input.
4. Apply the existing final RMSNorm + LM head directly to the captured L36 input.
5. Compare those logits with the functional L36-skip logits.
6. Compare direct H35 logits with the functional L36-skip logits.
7. Measure the actual L36 transformation from input to output.

## Why this matters

If H35 == L36 input and direct-L36-input == skip, then the EXP-0009L discrepancy came from how the stored H35 trace was sourced or indexed.

If those are not equivalent, there is an unexpected boundary or tensor representation issue.

## Setup

- Qwen3-4B-Base
- 36 decoder layers
- 4-bit NF4
- BF16 compute
- RTX 3070
- eager attention
- default 2 questions
- 64 maximum generated tokens
- seed base 42000

Run:

    python scripts/run_exp0009m.py --questions 2 --max-new-tokens 64 --output "E:\\Titan Forge Industries\\CFI-Data\\Results\\CFI-Eval-0009M-Layer-Boundary-Equivalence"

## Limits

M is a control experiment. It does not claim a speedup or a new architecture.
