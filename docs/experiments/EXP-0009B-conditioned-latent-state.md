# EXP-0009B: Conditioned Predictive Latent-State Replacement

## Why this follow-up exists

EXP-0009A established that the exact-state injection mechanism works, but a predictor given only H(t) was not sufficient to reproduce H(t+1).

That failure has a direct interpretation: the next-token transition contains information that is not present in the current hidden state alone. EXP-0009B therefore separates two prediction problems instead of treating them as one.

## Research questions

### 1. Can later-layer computation be predicted at the same token position?

Test:

- H30(t) -> H35(t)
- H35(t) -> H36(t)

This is the most direct compute-replacement test in this series.

The predictor is asked to estimate the representation that several downstream transformer layers would have produced for the same token.

### 2. Can temporal state transition be predicted when the next token is known?

Test:

- [H30(t), Emb(token[t+1])] -> H30(t+1)
- [H35(t), Emb(token[t+1])] -> H35(t+1)

This fixes the main information gap in EXP-0009A. The predictor receives the identity of the new token that drives the temporal transition.

## Controlled setup

The experiment keeps the established protocol:

- Qwen3-4B-Base
- 36 decoder layers
- 4-bit NF4 with double quantization
- BF16 compute
- RTX 3070 target
- CFI-Eval-0001-GSM8K
- temperature 0.6
- top-p 0.95
- top-k 20
- seed base 42000

The default run uses 6 questions:

- 4 training questions
- 2 held-out questions

Question-level separation prevents the predictor from training directly on the held-out trajectories.

## State capture

For each baseline trajectory the runner captures:

- hidden state after layer 30
- hidden state after layer 35
- hidden state after layer 36
- token embeddings
- token IDs
- prompt length
- baseline answer metadata
- generation time

The hidden-state indices are tied to the experimentally validated EXP-0009A convention.

## Predictor design

The predictor receives the source state and, for temporal relations, the next-token embedding.

It predicts an additive update:

predicted_state = source_state + learned_update

This gives every relation an explicit zero-learning starting point equal to source-state persistence.

For same-position depth prediction, there is no extra feature:

source = H30 or H35

For token-conditioned temporal prediction:

source = H30 or H35
feature = embedding(token[t+1])

The predictor is:

LayerNorm
  |
Linear
  |
GELU
  |
Linear
  |
learned update

Default bottleneck: 256.

The objective combines MSE and cosine-direction loss.

## Pair alignment

### Same-position depth

The source and target refer to the same token position:

H30(t) -> H35(t)
H35(t) -> H36(t)

Target positions run through the last position that has a following token, so downstream logits remain directly measurable against the captured sequence.

### Token-conditioned temporal

The source is at token position t and the target is at t+1:

[H(t), Emb(token[t+1])] -> H(t+1)

The final target position is excluded because there is no captured following token for its downstream logit.

## Downstream validation

State similarity is not sufficient.

For each held-out question:

1. Run the normal model to obtain oracle logits.
2. Predict the target state.
3. Inject the predicted target state into the corresponding decoder layer.
4. Run the real downstream network.
5. Compare candidate logits with the oracle logits.

The main downstream measurements are:

- top-1 agreement
- baseline-token probability ratio
- baseline-token log-probability change
- KL divergence
- logit L2 distance

State measurements are:

- cosine similarity
- relative L2 error
- MSE

## Exact-state control

Before trusting any predictor result, the runner injects the true target state.

The control must produce:

- top-1 agreement = 1.0 within numerical tolerance
- KL approximately zero

A failed control aborts interpretation of the predictor results.

This prevents state-indexing or hook-placement errors from masquerading as a research result.

## Copy baseline

The source state is also injected directly as a zero-extra-predictor baseline.

This answers:

Does the learned transformation add useful information beyond simply keeping the source representation?

For same-position depth relations, the copy baseline tests whether the learned predictor is actually approximating downstream computation rather than just carrying the earlier state forward.

For temporal relations, the copy baseline is intentionally weak because it ignores the next token. It remains useful as a reference for how much the token-conditioned predictor gains.

## First command

From the repository root:

    python scripts/run_exp0009b.py --questions 4 --train-questions 2 --max-new-tokens 128 --epochs 4

This is the smoke test.

The full first gate is:

    python scripts/run_exp0009b.py

## Outputs

Default output:

E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009B-Conditioned-Latent-State

Files:

- summary.json
- downstream_results.jsonl
- training_summary.json
- question_baselines.jsonl
- one predictor checkpoint per relation

## Interpretation gates

### Depth prediction succeeds

If H30 -> H35 or H35 -> H36 preserves downstream behavior, this is direct evidence that a portion of ordinary sequential transformer computation is predictable from an earlier internal representation.

That would justify a real compute-replacement experiment.

### Temporal prediction succeeds only after token conditioning

That would show that temporal state transition is predictable, but the predictor needs the incoming token identity as part of its state-transition input.

### State prediction succeeds but downstream validation fails

That means numerical state similarity is not sufficient for behavioral substitution. The next step should focus on better targets, uncertainty, local correction, or verification.

### Nothing succeeds

That would constrain the CFI approach and push the research toward different forms of computation reuse, such as subspace prediction, sparse correction, or learned state compression.

## Important limits

This is still a teacher-forced substitution experiment.

The runner computes the original model path before injecting the predicted state, so it does not demonstrate a hardware speedup.

The two held-out questions in the smoke test are a debugging gate, not statistically meaningful evidence about generalization.

A strong result must be reproduced with more questions before it should drive architectural changes.

## Relation to the CFI program

EXP-0007 measured temporal and within-layer structure.

EXP-0008 tested causal necessity.

EXP-0009A tested state prediction from the current hidden state alone and showed that this formulation was insufficient.

EXP-0009B now asks the more precise question:

Can we predict the state produced by downstream computation at the same token position, and can we predict temporal transitions once the incoming token is explicitly provided?

That is a much closer test of whether computation itself can be partially replaced by learned state transformation.
