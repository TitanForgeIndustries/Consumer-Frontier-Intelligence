# EXP-0005: Free-Running Layer Agreement

## Research question

Why does layer 30 show strong next-token prediction under teacher forcing but fail
when it is allowed to generate autoregressively?

## Hypothesis

The layer-30 representation contains useful predictive information, but small
errors compound when layer 30 generates its own previous tokens. If this is
true, agreement between layer 30 and layer 36 should deteriorate along the
model's own generated trajectory.

## Method

Run the unmodified full 36-layer Qwen3-4B-Base on the fixed CFI GSM8K
questions using the official generation settings.

During every generation step, retain the layer-30 hidden state and final
hidden state from the same forward pass. Apply the original final RMSNorm and
LM head to both representations and record:

- layer-30 top-1 prediction
- layer-36 top-1 prediction
- actually sampled token
- layer-30 confidence
- layer-36 confidence
- whether layer 30 agrees with layer 36
- whether layer 30 agrees with the sampled token

Because the full model remains the actual generator, the experiment does not
alter output quality. It only observes the intermediate path on the exact
free-running trajectory.

## Decision rule

If layer-30 agreement remains high during free generation but EXP-0003 still
fails, the primary problem is likely the learned exit readout.

If agreement collapses rapidly after generation begins, the result supports an
autoregressive exposure/feedback-loop explanation and motivates training the
intermediate path for self-consistent generation, such as layer-dropout or
early-exit-aware training.

## Relation to EXP-0004

EXP-0004 measured teacher-forced token predictions:

- layer 30: 62.12%
- layer 36: 44.78%

Those numbers do not directly measure generation quality because the next-token
prediction at every position was conditioned on the known reference sequence.
EXP-0005 measures the same intermediate representation on the model's own
generated sequence.
