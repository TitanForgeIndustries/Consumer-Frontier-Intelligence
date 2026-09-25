# EXP-0008: Causal Sublayer Ablation on Late Qwen3 Layers

## Research question

Are the large late-layer attention and MLP transformations observed in EXP-0007 causally necessary for the model's token predictions?

EXP-0008 converts the EXP-0007 observations into interventions. It zeroes one selected attention or MLP sublayer while keeping the rest of the pretrained model unchanged.

The first gate focuses on layers 30, 35, and 36 because EXP-0007 showed strong temporal stability in the late network, substantial late-layer transformation, and a distinct final-layer transition.

## Hypothesis

Late-layer sublayers are not equally necessary.

Ablating some late sublayers may produce relatively small changes to the model's next-token distribution even when their raw activation magnitude is large. If so, CFI can investigate selective computation, predicted residual updates, or conditional sublayer execution instead of treating every late-layer operation as mandatory.

Ablation with large downstream disruption is also informative: it identifies computation that should not be approximated or skipped without a stronger replacement mechanism.

## Controlled setup

Defaults:

- Qwen3-4B-Base, 36 layers
- 4-bit NF4 with double quantization
- BF16 compute
- RTX 3070 target
- fixed CFI-Eval-0001-GSM8K dataset
- 3 questions for the first gate
- layers 30, 35, 36
- attention and MLP tested separately
- seed 42000 + question index
- do_sample=True for the baseline trajectory
- temperature 0.6
- top_p 0.95
- top_k 20
- max_new_tokens 512

## Primary causal measurement

For each question:

1. Generate one baseline sequence with the normal model.
2. Replay that exact baseline sequence under teacher forcing.
3. Zero one selected attention or MLP output.
4. Compare the counterfactual logits against the baseline logits for every generated token.

The primary measurements are:

- baseline/counterfactual top-1 agreement
- baseline target-token probability
- counterfactual target-token probability
- counterfactual/baseline target-probability ratio
- change in target log probability
- KL divergence from baseline to counterfactual
- logit L2 difference

This separates causal disruption from the confounding effect of letting a changed token sequence feed back into later steps.

## Optional free-running check

The runner can additionally regenerate the first N questions under each ablation:

python scripts/run_exp0008.py --free-run-questions 3

Free-running results report:

- predicted answer
- correctness
- generated length
- token agreement with the baseline generation
- generated output

Free-running generation is secondary. Teacher-forced counterfactual measurements are the primary causal comparison.

## First gate

Run:

python scripts/run_exp0008.py

The default produces:

- one baseline generation for 3 questions
- six causal ablations per question
- layers 30, 35, 36
- attention and MLP separately

This is intentionally narrower than a full layer sweep.

## Interpretation

The key question is not whether an ablated component has a large activation norm. The key question is whether removing it materially changes downstream predictions.

A potentially useful result looks like:

large measured activation
+
high baseline/counterfactual agreement
+
low KL
+
near-preserved target probability

That would justify testing a cheaper replacement or conditional execution.

A high-disruption result means the component carries important computation and should not be skipped directly.

A mixed result would support selective routing rather than uniform layer skipping.

## Decision gates

### Gate A: Low-disruption sublayers

A late attention or MLP ablation causes only modest distributional disruption. Test whether that computation can be approximated or executed conditionally.

### Gate B: High-disruption sublayers

Ablation substantially changes predictions. Treat the component as causally important in the tested regime and search for state prediction, compression, or cheaper equivalent computation.

### Gate C: Attention/MLP asymmetry

One component type is systematically less disruptive than the other. Prioritize that component for conditional computation.

### Gate D: Layer-36 special case

If layer 36 is substantially more disruptive than layers 30-35, preserve it as a separate finalization stage rather than treating it as an ordinary layer.

## Limits

This experiment does not prove that an ablated component can be skipped in a real inference engine. Zeroing a sublayer is a strong intervention, not a hardware optimization.

It also does not establish:

- causal importance across all tasks
- behavior under training
- a cheaper replacement mechanism
- a hardware speedup
- suitability for the eventual CFI architecture

The purpose is to identify which computations deserve the next round of approximation or conditional-execution research.

## Outputs

Default output:

CFI_DATA_ROOT/Results/CFI-Eval-0008-Causal-Sublayer-Ablation

Files:

- summary.json
- ablations.jsonl
- questions.jsonl

JSON and JSONL are the source of truth.
