# EXP-0009C Results: Predictive Depth Sweep

## Runs

Two runs were completed.

### Smoke test

- 4 questions
- 2 training questions
- 2 held-out questions
- 128 maximum generated tokens
- 4 predictor epochs
- bottleneck 256

The smoke-test exact-state controls passed for every relation.

### Full first gate

- 10 questions
- 7 training questions
- 3 held-out questions
- 256 maximum generated tokens
- 12 predictor epochs
- bottleneck 256
- RTX 3070
- Qwen3-4B-Base
- 4-bit NF4 with double quantization
- BF16
- seed base 42000

The baseline parser marked 6 of 10 generated answers correct. Several responses reached the 256-token limit without producing a parseable final answer; this does not affect the hidden-state or downstream substitution measurements.

Exact-state injection passed for every evaluated relation and question with KL 0.000000 and top-1 agreement 1.000000.

## Full-run results

### H24 -> H30, depth gap 6

Aggregate predictor:

- state cosine: 0.8897
- relative state error: 0.4561
- downstream top-1: 0.7200
- target-probability ratio: 0.6804
- KL: 0.8579

Source-state copy baseline:

- state cosine: 0.7998
- relative state error: 0.7181
- downstream top-1: 0.8041
- KL: 0.4895

The learned predictor substantially improves state similarity, but downstream behavior is worse than simply copying the source state.

### H30 -> H35, depth gap 5

Aggregate predictor:

- state cosine: 0.9330
- relative state error: 0.3615
- downstream top-1: 0.6327
- target-probability ratio: 0.6331
- KL: 1.6836

Source-state copy baseline:

- state cosine: 0.8404
- relative state error: 0.7114
- downstream top-1: 0.6972
- KL: 1.1109

Again, the predictor makes the hidden state substantially closer to the real target, but behavioral preservation is worse than source-state persistence.

### H35 -> H36, depth gap 1

Aggregate predictor:

- state cosine: 0.9488
- relative state error: 0.3279
- downstream top-1: 0.9264
- target-probability ratio: 0.9789
- KL: 0.1775

Source-state copy baseline:

- state cosine: approximately 0.5991
- relative state error: approximately 0.9300
- downstream top-1: approximately 0.9270
- KL: approximately 0.1496

The predictor dramatically improves hidden-state similarity but does not improve downstream behavior in this pilot. Copy remains marginally better by top-1 and KL.

### H30 -> H36, depth gap 6

Aggregate predictor:

- state cosine: 0.8673
- relative state error: 0.5027
- downstream top-1: 0.3712
- target-probability ratio: 0.3650
- KL: 3.2688

Source-state copy baseline:

- state cosine: approximately 0.4611
- relative state error: approximately 0.8936
- downstream top-1: approximately 0.6013
- KL: approximately 2.4754

The larger direct collapse is currently not behaviorally viable.

## Main finding

EXP-0009C produces the clearest evidence yet that **hidden-state reconstruction and behavioral reconstruction are different objectives**.

Across all four relations:

- learned predictors produce substantially higher state cosine than source-state copying
- this improvement does not translate into higher downstream top-1 agreement
- KL is generally worse than the copy baseline
- the problem becomes especially severe when the predicted state collapses a larger depth interval

This is not evidence that predictive depth replacement is impossible.

It is evidence that the current predictor objective is misaligned with the actual CFI objective.

## Why this matters

The CFI goal is not:

"produce a vector numerically close to the original hidden state."

The goal is:

"produce a replacement state that causes the downstream computation to behave like the original computation."

EXP-0009C demonstrates that a state can be geometrically close while landing in a behaviorally poor region of the downstream function.

The H35 -> H36 result makes this especially clear. A predictor with state cosine near 0.95 does not improve the downstream distribution over simply carrying H35 forward.

## Decision

Do not increase predictor capacity or training duration using the current MSE + cosine objective as the primary optimization target.

The next experiment should directly optimize downstream behavior.

## Next hypothesis

Train a behavioral or functional predictor using the downstream model as a frozen teacher.

Candidate objective:

predicted state
    |
real downstream transformer
    |
candidate logits

versus

real target state
    |
real downstream transformer
    |
oracle logits

The predictor loss should include a distributional distillation term, with hidden-state reconstruction retained only as an auxiliary regularizer.

The key question becomes:

**Can a predictor learn a state that is less geometrically accurate but more behaviorally equivalent?**

## Limits

The full gate has only 3 held-out questions, so these numbers are directional rather than final.

No hardware speedup has been demonstrated.

The substitution is teacher-forced and still computes the original path before replacing states.

A behaviorally trained predictor must still be tested against the source-state copy baseline and exact-state injection control.
