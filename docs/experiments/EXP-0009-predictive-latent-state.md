# EXP-0009: Predictive Latent-State Replacement

## Research question

Can a small learned predictor forecast a future internal representation well enough that the model's real downstream computation still produces essentially the same next-token distribution?

EXP-0008 showed that local teacher-forced similarity does not guarantee stable autoregressive behavior. The next step is therefore not to skip a layer blindly, but to ask whether the layer's future state can be predicted and then verified by the computation that follows it.

## Core hypothesis

A useful internal state may be predictable from an earlier state at substantially lower computational cost than recomputing the entire path.

The first gate tests:

- H30(t) -> H30(t+1)
- H35(t) -> H35(t+1)
- H30(t) -> H35(t+1)

The same-layer cases test temporal state prediction. The cross-layer case tests a more aggressive hypothesis: whether several layers of computation can be replaced by forecasting a later latent state directly.

## Controlled setup

The experiment reuses the established EXP-0008 model and generation protocol:

- Qwen3-4B-Base
- 36 decoder layers
- 4-bit NF4 with double quantization
- BF16 compute
- RTX 3070 target
- CFI-Eval-0001-GSM8K
- sampled baseline generation with the established temperature/top-p/top-k settings
- fixed seed base of 42000
- baseline trajectories are captured before any predictor is trained

Questions are split by question, not by individual tokens. The leading questions are used for training and the held-out questions are used for evaluation.

This avoids training on the exact token states that are later scored.

## State collection

For every baseline trajectory the runner records:

- input token IDs
- prompt length
- hidden state after layer 30
- hidden state after layer 35
- generated token count
- baseline prediction and correctness
- baseline generation time

A state pair is formed as:

H_L(t) -> H_L(t+1)

The final state in a trace is excluded from downstream next-token evaluation because there is no following token inside the captured sequence.

## Predictor

The initial predictor is intentionally small:

LayerNorm
    |
Linear(hidden -> bottleneck)
    |
GELU
    |
Linear(bottleneck -> hidden)

For same-layer prediction, the predictor is initialized as an identity-plus-residual model:

predicted_state = current_state + learned_update

This makes simple state copying a zero-cost baseline inside the learned predictor.

For cross-layer prediction, the network predicts the target layer state directly.

The training objective combines:

- hidden-state mean squared error
- cosine-direction loss

The default bottleneck is 192 dimensions.

## Downstream validation

Hidden-state accuracy is not the main success criterion.

For each held-out question, predicted states are injected at their target layer while the rest of the original token sequence remains fixed.

The experiment then compares:

real state -> downstream network -> oracle logits

against:

predicted state -> downstream network -> candidate logits

The primary downstream metrics are:

- top-1 agreement
- baseline-token probability ratio
- change in baseline-token log probability
- KL divergence
- logit L2 distance

State-level metrics are also recorded:

- cosine similarity
- relative L2 error
- MSE

The downstream result is the important gate because a numerically close state can still land in a behaviorally different region after later computation.

## First command

From the repository root:

    python scripts/run_exp0009.py

Default first gate:

- 6 questions
- 4 training questions
- 2 held-out questions
- all three relations
- 512 maximum new tokens
- 12 predictor epochs
- 192-dimensional bottleneck

For a smaller smoke test:

    python scripts/run_exp0009.py --questions 4 --train-questions 2 --max-new-tokens 128 --epochs 4

For a broader first run after the smoke test:

    python scripts/run_exp0009.py --questions 10 --train-questions 7

## Outputs

Default output directory:

CFI_DATA_ROOT/Results/CFI-Eval-0009-Predictive-Latent-State

Expected files:

- summary.json
- downstream_results.jsonl
- training_summary.json
- question_baselines.jsonl
- one *_predictor.pt file per relation

JSON and JSONL are the source of truth. Predictor checkpoints are experiment artifacts, not part of the model itself.

## Interpretation gates

### Gate A: state prediction fails

If the predictor cannot achieve useful held-out state similarity, this route does not yet provide a viable computational substitute.

### Gate B: state prediction succeeds but downstream agreement fails

This means the representation has predictable structure, but later layers amplify the remaining error. The next step should focus on better prediction targets, uncertainty estimates, or correction mechanisms.

### Gate C: downstream agreement is high

A relation becomes a candidate for the next experiment: sequential autoregressive rollout with actual conditional replacement and hardware timing.

### Gate D: cross-layer prediction works

This would be substantially more interesting than same-layer prediction because it suggests that multiple layers of ordinary computation may be represented by a much cheaper predicted state.

## Important limits

This experiment does not demonstrate a speedup.

The model still computes the original layers before the experiment's hook replaces selected states. The purpose is to test whether the predicted representation is a valid substitute, not to claim that the current implementation saves compute.

It is also teacher-forced. The input token sequence is kept fixed, so this experiment should not be interpreted as proof of long-horizon autoregressive stability.

A successful result is evidence for the next systems experiment, not proof that the final CFI architecture should use this mechanism.

## Relation to EXP-0008

EXP-0008 established a critical distinction:

"high one-step distributional agreement can coexist with large free-running divergence."

EXP-0009 moves the target from the next-token distribution to the internal state that produces that distribution.

The central CFI question becomes:

Can computation be replaced by prediction of the state that computation would have produced, followed by verification of the resulting output?

That is the mechanism this experiment is designed to test.
