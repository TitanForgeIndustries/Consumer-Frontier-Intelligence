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


## Smoke result: 2026-09-21

Command:

```powershell
python scripts/run_exp0009d.py --questions 4 --train-questions 2 --max-new-tokens 128 --epochs 4
```

The smoke run completed all three relations after fixing H36 trace retention.

### H30 -> H35

Held-out question 3:

- predictor: state cosine 0.8390, KL 1.5775, top-1 0.7578, target ratio 0.4364
- copy: KL 0.9020, top-1 0.7891

Held-out question 4:

- predictor: state cosine 0.8272, KL 1.2169, top-1 0.8047, target ratio 0.5195
- copy: KL 0.8672, top-1 0.8281

Aggregate:

- predictor KL 1.3972
- copy KL 0.8846
- predictor top-1 0.7813
- copy top-1 0.8086

The behavioral predictor is worse than copying.

### H35 -> H36

Held-out question 3:

- predictor: state cosine 0.5718, KL 0.1379, top-1 0.9453, target ratio 0.9928
- copy: KL 0.1637, top-1 0.9531

Held-out question 4:

- predictor: state cosine 0.6160, KL 0.1139, top-1 0.9219, target ratio 0.9738
- copy: KL 0.1101, top-1 0.9375

Aggregate:

- predictor KL 0.1259
- copy KL 0.1369
- predictor top-1 0.9336
- copy top-1 0.9453

This is mixed. KL improves slightly on average, while top-1 agreement declines. It is not sufficient evidence of a general behavioral improvement.

### H30 -> H36

Held-out question 3:

- predictor: state cosine 0.4403, KL 2.2465, top-1 0.6172, target ratio 0.6725
- copy: KL 2.3865, top-1 0.7188

Held-out question 4:

- predictor: state cosine 0.4671, KL 3.1335, top-1 0.6094, target ratio 0.5975
- copy: KL 2.7220, top-1 0.6562

Aggregate:

- predictor KL 2.6900
- copy KL 2.5543
- predictor top-1 0.6133
- copy top-1 0.6875

The behavioral predictor is worse than copying on both aggregate KL and top-1 agreement.

## Interpretation

EXP-0009D does not support the hypothesis that the current behavioral-distillation predictor consistently learns a better replacement state than source-state persistence.

The depth pattern is more important than the aggregate result:

- gap 1: approximately copy-level behavior, with a small KL improvement but lower top-1 agreement
- gap 5: clearly worse than copy
- gap 6: clearly worse than copy

Training loss decreases in every relation, so optimization is occurring. The failure is therefore not simply "the predictor did not learn." The learned residual is not reliably producing a downstream state that improves on the zero-learning copy baseline at larger depth gaps.

This is different from EXP-0009C. The result does not show that behavioral objectives are useless. It shows that the current formulation of behavioral distillation, predictor architecture, and/or trust region is not sufficient to produce stable improvement over persistence.

## Important implementation observation

The current behavioral loss uses the teacher's top-k probabilities after renormalizing within the selected top-k set, then scores only those selected candidate logits. It does not explicitly constrain probability mass outside the teacher top-k set.

That makes the objective incomplete as a full behavioral preservation target. A predictor can improve the relative fit among selected teacher tokens while moving probability mass elsewhere.

This is a concrete next-hypothesis candidate and should be tested before increasing predictor capacity.

## Research decision

Do not promote EXP-0009D as evidence for useful latent-state replacement.

Do not respond to this result by simply increasing bottleneck size, epochs, or training-set size.

The next controlled experiment should keep the predictor near the copy baseline and improve the behavioral objective itself.

Candidate EXP-0009E direction:

- train a bounded residual correction from source state
- anchor the correction to source-state persistence
- use a fuller behavioral loss rather than renormalized top-k-only matching
- explicitly measure whether the learned correction beats copy
- retain exact-state injection as a mandatory semantic control
- test H35->H36 first as the smallest gap where a signal exists, then H30->H35 and H30->H36

A particularly useful variant is a copy-anchored trust-region predictor whose update is penalized as it moves away from source state, with the behavioral loss applied to the downstream output. If that cannot consistently beat copy on held-out positions, the evidence increasingly points toward the need for additional source information rather than a stronger mapping from H30 alone.
