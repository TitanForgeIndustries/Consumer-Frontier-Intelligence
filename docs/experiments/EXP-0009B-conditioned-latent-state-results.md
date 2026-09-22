# EXP-0009B Results: Conditioned Predictive Latent-State Replacement

## Run

Smoke-test configuration:

- Qwen3-4B-Base
- 36 decoder layers
- RTX 3070
- 4-bit NF4 with double quantization
- BF16 compute
- 4 questions
- 2 training questions
- 2 held-out questions
- 128 maximum generated tokens
- 4 predictor epochs
- bottleneck 256
- seed base 42000

The exact-state injection control passed for every evaluated relation:

- KL = 0.000000
- top-1 agreement = 1.000000

This validates the state-injection boundary for the measured relations.

## Same-position depth prediction

### H30 -> H35

Held-out question 3:

- state cosine: 0.8520
- relative error: 0.6998
- downstream KL: 0.6865
- downstream top-1 agreement: 0.8125
- target-probability ratio: 0.7461

Copy baseline:

- state cosine: 0.8400
- relative error: 0.7112
- downstream KL: 0.9020
- downstream top-1 agreement: 0.7891

Held-out question 4:

- state cosine: 0.8399
- relative error: 0.7116
- downstream KL: 0.7202
- downstream top-1 agreement: 0.8281
- target-probability ratio: 0.7426

Copy baseline:

- state cosine: 0.8283
- relative error: 0.7224
- downstream KL: 0.8672
- downstream top-1 agreement: 0.8281

Aggregate across the two held-out questions:

- predictor state cosine: 0.8459
- predictor downstream top-1: 0.8203
- predictor downstream KL: 0.7034
- predictor target-probability ratio: 0.7444
- copy state cosine: approximately 0.8342
- copy downstream top-1: approximately 0.8086
- copy downstream KL: approximately 0.8846

The learned predictor therefore improves on source-state persistence in both hidden-state similarity and downstream KL in this pilot, and improves aggregate top-1 agreement by about 1.2 percentage points.

### H35 -> H36

Held-out question 3:

- state cosine: 0.6016
- relative error: 0.9207
- downstream KL: 0.1512
- downstream top-1 agreement: 0.9453
- target-probability ratio: 0.9926

Copy baseline:

- state cosine: 0.5712
- relative error: 0.9546
- downstream KL: 0.1637
- downstream top-1 agreement: 0.9531

Held-out question 4:

- state cosine: 0.6427
- relative error: 0.8778
- downstream KL: 0.1085
- downstream top-1 agreement: 0.9297
- target-probability ratio: 0.9855

Copy baseline:

- state cosine: 0.6155
- relative error: 0.9099
- downstream KL: 0.1101
- downstream top-1 agreement: 0.9375

Aggregate:

- predictor state cosine: 0.6222
- predictor downstream top-1: 0.9375
- predictor downstream KL: 0.1299
- predictor target-probability ratio: 0.9890
- copy state cosine: approximately 0.5934
- copy downstream top-1: approximately 0.9453
- copy downstream KL: approximately 0.1369

This relation produces an important result: relatively poor hidden-state cosine can coexist with very strong downstream preservation. It reinforces the CFI observation that representation distance and behavioral distance are not interchangeable metrics.

The predictor improves state similarity and slightly improves KL over copying, while copy has slightly higher top-1 agreement in this two-question sample.

## Token-conditioned temporal prediction

### H30 + token -> H30 next

Aggregate:

- state cosine: 0.7053
- relative error: 0.7646
- downstream top-1: 0.0000
- target-probability ratio: 0.0031
- KL: 16.2418

Copy baseline:

- state cosine: approximately 0.6993
- relative error: approximately 0.7741
- downstream top-1: 0.0000
- KL: approximately 16.5170

The token-conditioned predictor makes a small numerical improvement, but it does not preserve downstream behavior.

### H35 + token -> H35 next

Aggregate:

- state cosine: 0.8177
- relative error: 0.6087
- downstream top-1: 0.0000
- target-probability ratio: 0.0030
- KL: 16.3669

Copy baseline:

- state cosine: approximately 0.8149
- relative error: approximately 0.6134
- downstream top-1: 0.0000
- KL: approximately 16.5369

Again, numerical state prediction improves slightly, but downstream substitution remains unusable.

## Main result

EXP-0009B separates two hypotheses.

The same-position depth hypothesis has preliminary support:

- H30 -> H35 preserves about 82% of downstream top-1 decisions in this tiny held-out sample and beats simple source-state copying on aggregate KL.
- H35 -> H36 preserves about 94% of downstream top-1 decisions and keeps the baseline-token probability ratio near 0.99.
- Exact-state controls pass.

The temporal hypothesis remains unsupported in its current form:

- adding the next-token embedding produces small hidden-state improvements
- downstream top-1 agreement remains 0%
- KL remains extremely large
- the result is not a viable state-transition replacement

## Interpretation

The useful signal is not currently "predict the next hidden state from the current state."

The stronger signal is:

**Predict the representation that downstream computation would have produced for the same token position.**

This is much closer to replacing actual transformer computation.

The H35 -> H36 result is especially important because layer-36 behavior can remain close to the oracle even when hidden-state cosine is only about 0.62. This means future experiments should optimize and report downstream behavioral preservation directly, not rely on hidden-state similarity alone.

## Decision

Do not spend the next experiment trying to rescue the current token-conditioned temporal predictor.

Continue with predictive depth replacement.

The next experiment should broaden the depth test across more layer gaps and questions, establish how prediction quality changes with depth gap, and test whether a learned predictor consistently beats source-state persistence.

Only after that gate should the project move to sequential autoregressive replacement and real hardware timing.

## Limits

This is a two-question held-out smoke test. It is not sufficient to establish generalization.

The current predictor is trained only on hidden-state objectives. It is not optimized directly against downstream behavioral loss.

No speedup has been demonstrated. The current substitution runner still performs the original model computation before replacing states.

No architectural claim is made beyond the measured pilot behavior.
