# EXP-0003B: Teacher-Aligned Intermediate Exit

EXP-0003A trained a residual layer-30 exit adapter directly against GSM8K
answer tokens and produced 0/5 on the feasibility evaluation.

EXP-0003B tests a more targeted hypothesis: the layer-30 representation may
still contain useful information, but its geometry is not aligned with the
final layer's representation expected by the original RMSNorm and tied LM
head.

The frozen 36-layer Qwen3-4B-Base is run once per example with hidden states
enabled. The adapter receives the layer-30 hidden state and is trained to
match the final layer-36 representation after normalization. A secondary
language-model loss encourages direct task usefulness on GSM8K.

The adapter remains the only trainable component.

## Initial gate

Run 100 training steps, then the existing 5-question EXP-0003 evaluator using
the saved \`exit_adapter.pt\`.

Do not run the 100-question benchmark unless the 5-question gate produces
usable outputs.

## Interpretation

A successful gate would justify a full fixed-set evaluation and potentially
an adaptive controller over multiple trained exits.

Failure would indicate that a simple representation-alignment adapter is
insufficient at layer 30, motivating a different exit head, deeper exit
location, or token-level conditional computation.

## Relation to EXP-0002

EXP-0002 showed that simply executing the first 30 layers was much faster but
produced no evaluator-scored answers. EXP-0003B directly addresses that
failure mode rather than assuming that raw truncation is a viable exit.
