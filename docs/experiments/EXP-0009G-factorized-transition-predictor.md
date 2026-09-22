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

    python scripts/run_exp0009g.py --questions 4 --train-questions 2 --relations h35_to_h36 --max-new-tokens 128 --epochs 4 --output "E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009G-Factorized-Transition-Predictor"

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
