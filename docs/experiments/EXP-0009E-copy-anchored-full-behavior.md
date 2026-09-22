# EXP-0009E: Copy-Anchored Full-Behavior Latent-State Replacement

## Research question

Can a bounded residual predictor improve downstream behavior over source-state copying when the behavioral objective directly matches the full teacher distribution?

EXP-0009D used top-k teacher distillation plus hidden-state regularization. The smoke result was mixed at gap 1 and worse than copy at gaps 5 and 6.

EXP-0009E isolates two concrete issues:

1. top-k-only supervision does not explicitly penalize probability mass moving outside the teacher top-k set
2. an unconstrained learned residual can drift too far from the source-state copy baseline

## Hypothesis

If useful information about the target state is present in the source state, a small behaviorally trained correction should be able to improve on copy without requiring a large geometric reconstruction.

The predictor therefore remains identity-plus-residual, but:

- the behavioral objective is full-vocabulary teacher KL
- the residual is bounded relative to the source-state RMS
- a copy-anchor penalty discourages unnecessary movement
- hidden-state reconstruction losses are disabled by default

The test is deliberately conservative. It asks whether a small functional correction can beat persistence before adding architectural complexity.

## Relations

Primary smoke order:

- H35 -> H36, gap 1
- H30 -> H35, gap 5
- H30 -> H36, gap 6

H35 -> H36 is tested first because EXP-0009C and EXP-0009D show that this is the only relation currently showing even a weak behavioral signal beyond copy.

## Controlled setup

The base model and evaluation procedure remain aligned with EXP-0009D:

- Qwen3-4B-Base
- 36 decoder layers
- 4-bit NF4 with BF16 compute
- RTX 3070
- CFI-Eval-0001-GSM8K
- identity-plus-residual predictor
- 256-dimensional bottleneck by default
- 16 top-k values retained as a diagnostic/cache field
- full teacher log probabilities retained for the selected training positions
- seed base 42000
- temperature 0.6
- top-p 0.95
- top-k 20

Smoke configuration:

    python scripts/run_exp0009e.py --questions 4 --train-questions 2 --relations h35_to_h36 --max-new-tokens 128 --epochs 4

After the one-relation smoke gate passes, the larger smoke comparison is:

    python scripts/run_exp0009e.py --questions 4 --train-questions 2 --max-new-tokens 128 --epochs 4

## Predictor constraint

The predictor computes:

source
  |
bounded residual
  |
predicted target

The residual is normalized by the source-state RMS and bounded with a configurable maximum update ratio.

Default:

    max_update_ratio = 0.25
    copy_anchor_weight = 0.1

The normalized update ratio is also reported during evaluation.

This creates an explicit trust region around source-state persistence.

## Behavioral objective

EXP-0009D used the teacher's top-k distribution after renormalizing those selected probabilities.

EXP-0009E instead stores the teacher log probability for the full vocabulary at the selected training positions and computes:

    KL(teacher distribution || candidate distribution)

This penalizes both:

- incorrect probability among teacher-preferred tokens
- probability mass moving to tokens that were outside the teacher top-k set

The hidden-state cosine and normalized MSE losses are retained only as optional diagnostics/ablation terms and default to zero.

## Controls

Every held-out relation retains:

1. exact target-state injection
2. learned predictor
3. source-state copying

Exact-state injection must reproduce the oracle before predictor metrics are interpreted.

The primary comparison remains predictor versus copy.

## Success criterion

The predictor must improve downstream behavior over copy on held-out positions.

Useful evidence would be consistent improvement in KL and/or target probability without requiring a large source update ratio.

A relation that only improves hidden-state cosine is not considered successful.

A relation that only moves farther from copy without behavioral improvement is not considered successful.

## Possible outcomes

### A. Small correction consistently beats copy

This supports the idea that useful latent-state information can be extracted as a bounded functional correction.

### B. Full KL removes the D failure but only at gap 1

This would suggest local depth prediction may be possible, while larger gaps require additional source information or a hierarchical predictor.

### C. Predictor collapses back toward copy

This would suggest copy is already close to the useful optimum under the tested source representation and objective.

### D. Predictor remains worse than copy even with full KL and a trust region

This would strengthen the case that the missing ingredient is not predictor capacity or loss shaping. The source representation itself would likely need additional information, such as intermediate states, temporal context, attention context, or a structured multi-stage predictor.

## Important limits

This experiment is still teacher-forced.

It does not remove the original upstream transformer computation.

It does not demonstrate a hardware speedup.

The smoke test remains too small for generalization claims.

A positive teacher-forced result must still be followed by autoregressive rollout and timing.

## Research decision rule

Do not respond to a negative E result by immediately increasing bottleneck size or epochs.

First determine whether the source state contains enough information for the task under a bounded functional correction.

Only after that should the project consider adding richer conditioning or a multi-stage latent predictor.


## Smoke result: 2026-09-21

One-relation smoke command:

    python scripts/run_exp0009e.py --questions 4 --train-questions 2 --relations h35_to_h36 --max-new-tokens 128 --epochs 4

The exact-state injection control passed on both held-out questions:

- KL = 0.000000
- top-1 = 1.000000

### Held-out question 3

Predictor:

- state cosine: 0.5714
- state relative error: 0.9546
- KL: 0.1546
- top-1: 0.9531
- target probability ratio: 0.9943

Copy:

- KL: 0.1637
- top-1: 0.9531

### Held-out question 4

Predictor:

- state cosine: 0.6157
- state relative error: 0.9100
- KL: 0.1105
- top-1: 0.9375
- target probability ratio: 0.9802

Copy:

- KL: 0.1101
- top-1: 0.9375

Aggregate:

- predictor KL: 0.13257148
- copy KL: 0.1369
- predictor top-1: 0.9453125
- copy top-1: 0.9453125
- predictor target probability ratio: 0.98727277
- source update ratio: 0.00548158

The behavioral predictor therefore slightly reduces aggregate KL while leaving top-1 unchanged, but it does not produce a consistent held-out improvement. One held-out question improves and one is slightly worse.

The very small source update ratio is important. The learned state is staying extremely close to source-state copying. This suggests the copy anchor is successfully preventing large behavioral drift, but the experiment has not established that the predictor contains a materially useful replacement computation.

## Decision

EXP-0009E is not sufficient evidence for latent-state replacement.

It does, however, provide a useful constraint:

- full-vocabulary behavioral supervision is better behaved than the EXP-0009D top-k-only objective
- copy anchoring prevents the destructive drift seen in larger-gap D results
- the resulting predictor mostly learns to stay near copy
- there is no demonstrated consistent behavioral gain over copy

Do not increase predictor size or training duration yet.

The next experiment should change the architecture, not merely the optimizer.

## Next hypothesis

A one-layer H35 -> H36 transition may be too structured for a generic MLP residual predictor. EXP-0007 showed that late-layer computation is asymmetric, with attention and MLP contributions behaving differently and layer 36 being unusually active.

The next controlled architecture should therefore test a **factorized transition predictor**:

H35
  |
  +--> attention-like low-rank branch ----+
  |                                      |
  +--> MLP-like low-rank branch ---------+--> gated delta --> H36 prediction
  |
  +--> source persistence ----------------+

The branches remain lightweight and learned from H35, but the decomposition reflects the actual transform being approximated rather than treating the whole layer transition as one undifferentiated residual.

Behavioral full-KL remains primary, copy anchoring remains active, and exact-state injection remains mandatory.

This is a stronger architectural hypothesis than simply increasing the bottleneck.
