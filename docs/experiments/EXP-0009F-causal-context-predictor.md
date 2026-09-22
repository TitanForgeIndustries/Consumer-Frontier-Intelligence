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

    python scripts/run_exp0009f.py --questions 4 --train-questions 2 --relations h35_to_h36 --max-new-tokens 128 --epochs 4 --output "E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009F-Causal-Context-Predictor"

After the one-relation gate completes:

    python scripts/run_exp0009f.py --questions 4 --train-questions 2 --max-new-tokens 128 --epochs 4 --output "E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009F-Causal-Context-Predictor"

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
