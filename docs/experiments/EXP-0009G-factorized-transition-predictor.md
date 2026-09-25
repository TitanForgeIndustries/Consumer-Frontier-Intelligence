# EXP-0009G: Factorized Global-Local Latent-State Transition Predictor

## Research question

Can a lightweight predictor approximate a one-layer transformer state transition more effectively when it explicitly separates global token interaction from token-wise nonlinear transformation?

EXP-0009D showed that behavioral distillation alone did not consistently beat source-state copying.

EXP-0009E showed that full-vocabulary KL and a copy anchor make the correction stable, but the learned correction is extremely small.

EXP-0009F showed that short-range causal convolution adds little measurable information for H35 -> H36.

The next hypothesis is that the predictor needs an explicit approximation of global attention-style mixing rather than local context alone.

## Architecture

Input: H_source sequence

Two compact branches operate on the same normalized source sequence:

1. Token-wise MLP branch
2. Compact causal global interaction branch

The global branch uses low-dimensional causal multi-head attention over the source sequence.

The outputs are combined by a learned per-token gate and passed through the same bounded residual mechanism used in EXP-0009E and EXP-0009F.

Conceptually:

H_source
  |
  +--> token-wise nonlinear branch --------+
  |                                        |
  +--> compact causal global interaction --+--> gated bounded residual --> H_pred
                                           |
                                           +--> source persistence

The final output projections are zero initialized, so the predictor starts exactly at copy.

## Why this architecture

EXP-0007 found that later transformer layers have distinct attention and MLP activity profiles, especially in the final layers.

EXP-0008 also showed that attention and MLP ablations do not have identical downstream effects.

EXP-0009F tested only a local causal context approximation and did not materially improve over copy.

This experiment therefore tests whether an explicitly global, factorized approximation is more informative.

## Behavioral objective

The full teacher distribution is used:

    KL(teacher || candidate)

A copy-anchor penalty remains active.

The residual is bounded relative to source-state RMS with the same default trust-region mechanism:

    max_update_ratio = 0.25
    copy_anchor_weight = 0.1

Hidden-state cosine and MSE losses remain disabled by default.

## Causality

The global branch uses a strict causal mask.

Position t can attend only to source positions <= t.

No future source states are visible.

## Relations

First gate:

- H35 -> H36, gap 1

Second gate, only if the first shows useful signal:

- H30 -> H35, gap 5
- H30 -> H36, gap 6

## Smoke command

    python scripts/run_exp0009g.py --questions 4 --train-questions 2 --relations h35_to_h36 --max-new-tokens 128 --epochs 4 --output ".cfi-data/Results/CFI-Eval-0009G-Factorized-Transition-Predictor"

## Controls

Every evaluation retains:

1. exact target-state injection
2. learned predictor
3. source-state copying

Exact-state injection must reproduce the oracle before predictor metrics are interpreted.

## Primary measurements

- top-1 agreement
- target probability ratio
- target log-probability delta
- KL divergence
- source update ratio
- state cosine and relative error as secondary diagnostics

The predictor is considered useful only if it provides behavioral improvement over copy.

## Interpretation

A positive result would support the idea that a compact approximation of the layer's global interaction structure contains information not captured by token-local or short-range predictors.

A negative result would suggest that even explicit global source-state interaction is insufficient, and that the missing information may come from features unavailable in H_source alone, such as richer intermediate representations or explicit attention statistics.

## Important limits

This remains teacher-forced.

It does not remove the original upstream transformer computation.

It does not demonstrate a hardware speedup.

The smoke test is too small for generalization claims.

A positive teacher-forced result still requires autoregressive rollout and timing.


## Smoke result: 2026-09-21

Command:

    python scripts/run_exp0009g.py --questions 4 --train-questions 2 --relations h35_to_h36 --max-new-tokens 128 --epochs 4 --output ".cfi-data/Results/CFI-Eval-0009G-Factorized-Transition-Predictor"

Exact-state injection passed:

- KL = 0.000000
- top-1 = 1.000000

### Held-out question 3

Predictor:

- state cosine: 0.5711
- state relative error: 0.9548
- KL: 0.1608
- top-1: 0.9531
- target probability ratio: 0.9946

Copy:

- KL: 0.1637
- top-1: 0.9531

### Held-out question 4

Predictor:

- state cosine: 0.6154
- state relative error: 0.9101
- KL: 0.1057
- top-1: 0.9375
- target probability ratio: 0.9815

Copy:

- KL: 0.1101
- top-1: 0.9375

Aggregate:

- predictor KL: 0.13322094
- copy KL: 0.1369
- predictor top-1: 0.9453125
- copy top-1: 0.9453125
- predictor target probability ratio: 0.9880363
- source update ratio: 0.00585843

The factorized global-local predictor remains near the copy solution. It gives a small KL reduction on both held-out questions, but top-1 behavior is unchanged and the state update remains very small.

## Decision

EXP-0009G does not demonstrate meaningful improvement over source-state copying.

Across D, E, F, and G, four different predictor formulations converge to approximately the same behavioral region for H35 -> H36:

- D: unconstrained behavioral predictor
- E: full-vocabulary KL plus copy anchor
- F: bounded local/context predictor
- G: bounded factorized global/local predictor

This repetition is important. Increasing architectural resemblance to the target layer has not produced a materially different result.

The research should therefore stop iterating on generic state predictors for this relation and test a more diagnostic question.

## Next experiment direction

The next experiment should decompose the actual H35 -> H36 transformation into its attention and MLP components and measure their predictability from H35.

The purpose is not to claim a usable predictor. It is to establish where the information bottleneck actually is.

For each token position, capture:

- H35 source state
- layer-36 attention output
- layer-36 MLP output
- final H36 state

Then train small source-only probes for:

1. attention output prediction
2. MLP output prediction
3. total layer delta prediction

Evaluate both representation error and downstream effect when each predicted component is substituted.

This will distinguish at least three possibilities:

- the layer components are individually predictable but the combined state is difficult to preserve
- one component is the dominant unpredictability bottleneck
- neither component is predictably recoverable from H35 alone

That diagnostic is more informative than another generic predictor architecture.

