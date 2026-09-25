# EXP-0009F: Causal Context-Aware Latent-State Transition Predictor

## Research question

Can a lightweight predictor improve latent-state replacement by using short-range causal context instead of predicting each target state independently from one token's source state?

EXP-0009D showed that behavioral distillation did not consistently beat source-state copying at gaps 5 and 6.

EXP-0009E showed that full-vocabulary teacher KL plus a strong copy anchor prevented large destructive corrections, but the learned correction stayed extremely small and did not produce a consistent held-out gain.

The next hypothesis is that the missing information is partly contextual.

## Hypothesis

A transformer layer contains attention-driven sequence interactions. A token-local predictor given only H35(t) cannot directly observe the neighboring source states that may contribute to the next-layer transformation.

EXP-0009F therefore adds a small causal context pathway while preserving the conservative E formulation.

Architecture:

H_source sequence
  |
  +--> token-local low-rank branch --------+
  |                                        |
  +--> causal depthwise-context branch ----+--> bounded residual --> H_pred
                                           |
                                           +--> source persistence

The context branch only sees source positions at or before the current position. It does not use future tokens.

## Design

The predictor has two equal-width branches within the existing bottleneck budget:

- local token transformation: LayerNorm -> Linear -> GELU -> Linear
- causal context transformation: LayerNorm -> Linear -> depthwise causal Conv1d -> GELU -> Linear

The branch outputs are summed into one residual.

The residual is bounded by source-state RMS using the same trust-region mechanism as EXP-0009E.

The final up projections are zero initialized, so the exact starting point remains:

predicted_state = source_state

This makes source-state copying the explicit zero-learning baseline.

## Behavioral objective

The training objective remains the full teacher distribution:

KL(teacher || candidate)

Hidden-state cosine/MSE terms remain disabled by default.

A copy-anchor penalty is retained to discourage unnecessary movement.

## Relations

Initial gate:

- H35 -> H36, gap 1
- H30 -> H35, gap 5
- H30 -> H36, gap 6

The H35 -> H36 relation is tested first because prior experiments show that it is the strongest case for functional preservation.

## Causality requirement

For position t, the context branch may depend only on source states <= t.

The implementation uses causal convolution followed by a left-to-right crop.

No future source state is available to the predictor.

## Controls

Every evaluation retains:

1. exact target-state injection
2. learned predictor
3. source-state copying

The exact-state injection control must reproduce the oracle before any predictor result is interpreted.

## Primary measurements

- top-1 agreement
- target probability ratio
- target log-probability delta
- KL divergence
- source update ratio
- state cosine and relative error as secondary diagnostics

The key comparison is predictor versus copy.

## Smoke command

First run only H35 -> H36:

    python scripts/run_exp0009f.py --questions 4 --train-questions 2 --relations h35_to_h36 --max-new-tokens 128 --epochs 4 --output ".cfi-data/Results/CFI-Eval-0009F-Causal-Context-Predictor"

After the one-relation gate completes:

    python scripts/run_exp0009f.py --questions 4 --train-questions 2 --max-new-tokens 128 --epochs 4 --output ".cfi-data/Results/CFI-Eval-0009F-Causal-Context-Predictor"

## Interpretation

A positive result would indicate that short-range causal source context contains useful information unavailable to a token-local predictor.

A negative result would strengthen the case that local context is insufficient and that the next source representation needs richer structure, such as multiple intermediate states or explicit attention summaries.

Do not immediately increase predictor width or epochs after a negative smoke result.

## Important limits

The experiment is teacher-forced.

It does not remove the original upstream transformer computation.

It does not demonstrate a hardware speedup.

The smoke test is too small for generalization claims.

A positive result still requires autoregressive rollout and timing before becoming a conditional-compute mechanism.


## Smoke result: 2026-09-21

Command:

    python scripts/run_exp0009f.py --questions 4 --train-questions 2 --relations h35_to_h36 --max-new-tokens 128 --epochs 4 --output ".cfi-data/Results/CFI-Eval-0009F-Causal-Context-Predictor"

Exact-state injection passed:

- KL = 0.000000
- top-1 = 1.000000

### Held-out question 3

Predictor:

- state cosine: 0.5712
- state relative error: 0.9547
- KL: 0.1606
- top-1: 0.9531
- target probability ratio: 0.9949

Copy:

- KL: 0.1637
- top-1: 0.9531

### Held-out question 4

Predictor:

- state cosine: 0.6155
- state relative error: 0.9100
- KL: 0.1062
- top-1: 0.9375
- target probability ratio: 0.9820

Copy:

- KL: 0.1101
- top-1: 0.9375

Aggregate:

- predictor KL: 0.13338269
- copy KL: 0.1369
- predictor top-1: 0.9453125
- copy top-1: 0.9453125
- predictor target probability ratio: 0.98845109
- source update ratio: 0.00403051

The causal context predictor therefore remains near the copy solution. KL is slightly lower than copy on both held-out questions, but top-1 is unchanged and the correction remains very small.

## Decision

EXP-0009F does not demonstrate a meaningful improvement over source-state copying.

The result specifically weakens the hypothesis that a short-range causal neighborhood is the missing information for H35 -> H36 prediction.

Because H35 is already a contextualized representation, local convolution adds little measurable information at this gap.

Do not respond by increasing the convolution kernel or width.

## Next hypothesis

The remaining transformer-layer transformation should be decomposed more explicitly.

A useful next architecture should approximate two qualitatively different operations from the same H35 source sequence:

1. a global interaction branch, approximating attention-style token-to-token mixing
2. a token-wise nonlinear branch, approximating the MLP-style transformation

The branches should be separately projected into a compact space and combined through a learned gate, while retaining the same bounded residual and full-vocabulary behavioral objective.

This tests whether the failure comes from using one generic predictor to approximate a layer whose computation has multiple distinct mechanisms.

