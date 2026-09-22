# EXP-0009L: Anchored Early-Exit Transition

## Motivation

EXP-0009K trained a new vocabulary head from scratch and failed badly on held-out questions. That result does not isolate whether H35 lacks the information needed for an early exit because K also discarded the model's learned final RMSNorm and vocabulary projection.

L keeps the pretrained output stack intact.

## Hypothesis

A compact residual transition can approximate the behavior of L36 sufficiently well when the model's own final RMSNorm and LM head are reused:

\`H35 -> cheap residual transition -> final RMSNorm -> existing LM head\`

The target remains the frozen teacher's full next-token distribution.

## Architecture

Learned component:

\`LayerNorm -> Linear(hidden, bottleneck) -> GELU -> Linear(bottleneck, hidden)\`

The output is a bounded residual added to H35.

Default bottleneck: 128.

The existing Qwen3 final normalization and LM head are frozen and reused.

This means the learned transition only approximates the missing transformation between H35 and the existing output stack.

## Controls

Evaluation compares:

1. **Base H35 exit:** final RMSNorm + existing LM head applied directly to H35
2. **Adapted exit:** learned residual transition applied to H35, followed by the same existing final RMSNorm + LM head
3. **Teacher oracle:** full untouched model

A whole-L36 skip remains available from EXP-0009I as the previous functional baseline.

## Limits

This is still a behavioral experiment.

The current implementation captures H35 using the full model before evaluating the exit, so it does not demonstrate generation speed.

An integrated decoder would need to stop after H35 and call the existing output stack or learned transition directly.

## Setup

- Qwen3-4B-Base
- 36 decoder layers
- 4-bit NF4
- BF16 compute
- RTX 3070
- eager attention
- CFI-Eval-0001-GSM8K
- default 6 questions
- 4 train / 2 held out
- 128 max new tokens
- temperature 0.6
- top-p 0.95
- top-k 20
- 64 training positions per training question
- bottleneck 128
- maximum update ratio 0.5
- 16 epochs
- learning rate 1e-3
- weight decay 1e-5

Run:

    python scripts/run_exp0009l.py --questions 6 --train-questions 4 --max-new-tokens 128 --epochs 16 --bottleneck 128 --max-train-positions 64 --output "E:\\Titan Forge Industries\\CFI-Data\\Results\\CFI-Eval-0009L-Anchored-Early-Exit"

## Decision rule

- Adapted exit materially approaches or improves the whole-L36 skip baseline -> pursue integrated early exit and runtime measurement.
- Base H35 exit is already close to skip -> the existing output stack is carrying most of the useful behavior; focus on thresholding and selective L36 execution.
- Adapted exit remains far below both -> H35 alone is insufficient for this compact transition; add a richer source representation such as H34-H35 features or a small recurrent/context state.

No speedup claim is made by L.


## Evaluation bug identified: 2026-09-21

The initial smoke evaluation used local evaluation indices to select H35:

\`source = trace.h35[indices]\`

while oracle logits were selected using absolute sequence positions:

\`oracle = logits[positions]\`

Because \`positions\` begins at \`prompt_length - 1\`, the learned exit was evaluated on the wrong hidden-state positions. This explains the catastrophic held-out KL values.

The corrected evaluation uses:

\`source = trace.h35[positions]\`

so the H35 source and oracle logits refer to the same sequence positions.

The initial EXP-0009L smoke result must therefore be treated as **invalid due to evaluation indexing**, not as a scientific negative result.

Commit containing the fix: \`748f001e601a0eba5ed000d6112ed58a4b18b60e\`.

A new smoke run should be performed before interpreting L.
