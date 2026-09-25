# EXP-0009C: Predictive Depth Sweep

## Research question

How does predictive latent-state replacement behave as the depth gap between source and target layers increases?

EXP-0009B produced preliminary behavioral evidence for:

- H30(t) -> H35(t)
- H35(t) -> H36(t)

while token-conditioned temporal prediction remained unusable.

EXP-0009C therefore focuses only on same-position depth prediction.

## Relations

The sweep tests four depth gaps:

- H24(t) -> H30(t), gap 6
- H30(t) -> H35(t), gap 5
- H35(t) -> H36(t), gap 1
- H30(t) -> H36(t), gap 6

This separates two questions:

1. Can a target representation be predicted?
2. How far through the transformer can computation be replaced by that prediction before downstream behavior degrades?

## Controlled setup

Default configuration:

- Qwen3-4B-Base
- 36 decoder layers
- 4-bit NF4 with double quantization
- BF16 compute
- RTX 3070 target
- CFI-Eval-0001-GSM8K
- 10 questions
- 7 training questions
- 3 held-out questions
- 256 maximum generated tokens
- 12 predictor epochs
- 256-dimensional bottleneck
- seed base 42000
- temperature 0.6
- top-p 0.95
- top-k 20

Questions are split by question so held-out token states are not used to train the predictor.

## Predictor

Each relation uses the same small residual predictor:

LayerNorm
  |
Linear(hidden -> bottleneck)
  |
GELU
  |
Linear(bottleneck -> hidden)
  |
add source state

The predictor therefore learns an update to the source representation:

predicted = source + learned_update

A source-state copy remains the comparison baseline.

## State capture

The experiment captures:

- H24
- H30
- H35
- H36

Layer 36 is captured directly from decoder layer 36 output, before Qwen3's final model RMSNorm. This matches the state-injection boundary.

## Evaluation

For each held-out question:

1. Run the normal model on the fixed token sequence to obtain oracle logits.
2. Predict the target hidden state from the source hidden state.
3. Inject the predicted target state at the target layer.
4. Run the real downstream model.
5. Compare candidate logits with oracle logits.
6. Repeat with exact target-state injection as a control.
7. Repeat with source-state copying as a baseline.

### State metrics

- cosine similarity
- relative L2 error
- MSE

### Behavioral metrics

- top-1 agreement
- baseline-token probability ratio
- baseline-token log-probability change
- KL divergence
- logit L2 distance

The behavioral metrics are the primary evidence for whether computation is replaceable.

## First command

Smoke test:

    python scripts/run_exp0009c.py --questions 4 --train-questions 2 --max-new-tokens 128 --epochs 4

Full first gate:

    python scripts/run_exp0009c.py

A larger follow-up can use:

    python scripts/run_exp0009c.py --questions 20 --train-questions 14

## Outputs

Default output:

CFI_DATA_ROOT/Results/CFI-Eval-0009C-Predictive-Depth-Sweep

Files:

- summary.json
- downstream_results.jsonl
- training_summary.json
- question_baselines.jsonl
- predictor checkpoint for each relation

## Interpretation gates

### Gap dependence

If performance degrades smoothly as the depth gap increases, the result can be used to estimate how much downstream transformer computation can be predicted with a given error budget.

### Large-gap success

If H30 -> H36 retains strong downstream preservation, that would be substantially more important than the existing H30 -> H35 result because six layers would be represented by a single learned transformation.

### One-layer versus multi-layer behavior

H35 -> H36 provides a one-layer reference. Comparing it with the larger gaps can show whether prediction becomes qualitatively harder as more computation is collapsed.

### Predictor versus copy

A predictor should be compared against simply copying the source state. Improvement over copying is evidence that the learned transformation contains useful information about the missing computation.

## Connection to the CFI architecture

This experiment is the bridge between representation analysis and actual compute replacement.

A successful larger-gap relation would support the following future architecture:

current_state
    |
cheap predictor
    |
predicted_state
    |
verification / correction
    |
downstream computation

The current experiment still performs the original computation before substitution. It therefore does not demonstrate a speedup.

## Limits

The first gate is small and teacher-forced.

A relation that performs well on held-out token states still needs:

- broader question coverage
- autoregressive rollout testing
- actual compute measurement
- verification and fallback logic

No claim is made that the final CFI system should use predictive depth replacement until those stages succeed.
