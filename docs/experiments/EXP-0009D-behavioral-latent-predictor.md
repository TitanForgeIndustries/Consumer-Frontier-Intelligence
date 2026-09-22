# EXP-0009D: Behavioral Latent-State Predictor

## Research question

Can a predictor learn a replacement hidden state that preserves downstream model behavior better than a predictor trained only to minimize hidden-state reconstruction error?

EXP-0009C showed a consistent mismatch:

- hidden-state cosine improved substantially
- downstream top-1 and KL did not improve over source-state copying

That suggests geometric reconstruction is not the right primary objective for compute replacement.

## Hypothesis

A useful replacement state should be trained against the function that consumes it.

The predictor therefore starts as source-state persistence and learns a residual using a frozen downstream teacher:

source state
    |
predictor
    |
predicted target state
    |
real downstream transformer
    |
candidate next-token distribution

The teacher target is the distribution produced by the original model from the real target state.

## Relations

Initial behavioral training focuses on:

- H30 -> H35, gap 5
- H35 -> H36, gap 1
- H30 -> H36, gap 6

These relations cover the strongest existing depth signal and the larger direct collapse.

## Controlled setup

Default configuration:

- Qwen3-4B-Base
- 36 decoder layers
- 4-bit NF4 with double quantization
- BF16 compute
- RTX 3070
- CFI-Eval-0001-GSM8K
- 6 questions
- 4 training questions
- 2 held-out questions
- 256 maximum generated tokens
- 6 behavioral-training epochs
- 256-dimensional predictor bottleneck
- 16 teacher top-k logits
- 32 training positions per question
- seed base 42000
- temperature 0.6
- top-p 0.95
- top-k 20

For the smoke test, use 4 questions, 2 training questions, 128 generated tokens, and 4 epochs.

## Predictor

The predictor has the same residual structure used in EXP-0009C:

LayerNorm
  |
Linear(hidden -> bottleneck)
  |
GELU
  |
Linear(bottleneck -> hidden)
  |
add source state

The final two layers are initialized to zero.

Therefore, before learning:

predicted_state = source_state

This makes source-state copying an explicit zero-learning baseline.

## Behavioral objective

For each training question:

1. Capture source and target states from the original model.
2. Select a limited set of token positions.
3. Predict the target state for those positions.
4. Inject the predicted states into the frozen model.
5. Run the downstream model.
6. Compare candidate logits against the original teacher distribution.

The primary loss is top-k distributional distillation:

teacher top-k distribution
    versus
candidate distribution over those teacher-selected tokens

Small auxiliary terms retain:

- cosine state loss
- normalized MSE state loss

The behavioral objective remains dominant.

## Why top-k distillation

The full vocabulary distribution is expensive to store for every training position.

The experiment therefore stores the teacher's highest-probability tokens and their normalized probabilities.

This retains the model's most behaviorally important local distribution while keeping the training cache small.

## Evaluation

Held-out questions are evaluated using the same downstream substitution procedure established in EXP-0009C.

Three comparisons are produced:

1. Exact target-state injection
2. Learned behavioral predictor
3. Source-state copying

The exact-state injection control must reproduce the oracle.

The main behavioral metrics are:

- top-1 agreement
- baseline-token probability ratio
- baseline-token log-probability change
- KL divergence
- logit L2

State metrics are still reported, but they are secondary.

## Critical decision criterion

The behavioral predictor must be compared directly with copying.

A successful result is not merely:

higher cosine

It is:

better downstream behavior than copy

while ideally also maintaining useful state similarity.

Possible outcomes:

### A. Behavioral training beats copy

This supports the hypothesis that functional loss is a better objective for latent-state replacement.

### B. Behavioral training matches copy but does not improve it

The predictor has not yet demonstrated useful information beyond persistence.

### C. Behavioral training is worse than copy

The target state may not be predictable from the chosen source representation at the tested depth gap, or the predictor architecture may be inadequate.

### D. Behavioral training improves downstream behavior while state cosine decreases

This would be particularly informative. It would confirm that geometric hidden-state similarity is not the correct optimization target for functional replacement.

## Important limits

The base model computation is still performed before the state is replaced.

Therefore this experiment does not demonstrate a hardware speedup.

The training and evaluation are teacher-forced.

The smoke test is too small for generalization claims.

A successful behavioral predictor still requires autoregressive rollout and real timing before it can become an architectural component.

## Next step

If behavioral training beats source-state copying consistently, build the next experiment around:

predicted state
    |
confidence / uncertainty estimate
    |
downstream verification
    |
accept or recompute

That would turn latent prediction into an actual conditional-compute mechanism.
