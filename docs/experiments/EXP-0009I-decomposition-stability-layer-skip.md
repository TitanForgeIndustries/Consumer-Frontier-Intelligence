# EXP-0009I: Decomposition Stability and Whole-Layer-36 Skip Diagnostic

## Research question

After EXP-0009H established exact same-forward component replay, the next question is whether the observed internal L36 decomposition is actually stable across fresh forwards and whether L36 as a whole can be functionally removed.

This experiment tests both in one pass:

1. fresh-forward-to-fresh-forward stability of H35, L36 attention, L36 MLP, and H36
2. behavioral effect of functionally skipping the entire L36 transformation

## Fresh-vs-fresh diagnostic

The same generated sequence is evaluated twice with the untouched model.

For each forward the experiment captures:

- H35
- L36 attention output
- L36 MLP output
- H36
- final logits

The two forwards are compared independently.

The final logits are expected to remain exactly reproducible. Internal component differences are therefore interpreted separately from final behavioral stability.

### Interpretation

If attention/MLP vary substantially while H35/H36/logits remain stable, the internal decomposition is not a stable prediction target even though the model's externally observable computation is stable.

If all internal components are also stable, the cross-forward mismatch seen in EXP-0009H was likely capture-specific.

## Whole-layer-36 skip

A forward hook functionally replaces the output of L36 with its input H35.

This removes the L36 transformation from the downstream computation.

The model still executes L36 before the hook replaces its output, so this is a behavioral ablation only, not a speed measurement.

Metrics:

- oracle-to-skip KL
- top-1 agreement
- target-token probability ratio
- logit L2 distance

## Controlled setup

- Qwen3-4B-Base
- 36 decoder layers
- 4-bit NF4
- BF16 compute
- RTX 3070
- CFI-Eval-0001-GSM8K
- default 4 questions
- 128 maximum generated tokens
- temperature 0.6
- top-p 0.95
- top-k 20
- seed base 42000
- eager attention

Smoke command:

    python scripts/run_exp0009i.py --questions 4 --max-new-tokens 128 --output ".cfi-data/Results/CFI-Eval-0009I-Decomposition-Stability-Layer-Skip"

## Limits

This experiment does not demonstrate a speedup.

Whole-layer skip is a functional intervention after the layer executes. A future implementation would need to avoid executing L36 entirely or replace it with a cheaper transition to claim compute savings.

Fresh-vs-fresh internal differences do not by themselves prove mathematical non-uniqueness. They establish empirical instability of the captured component representation under the tested implementation.

## Decision rule

- Stable internal components -> investigate why H's earlier stored/fresh mismatch occurred and continue component-level analysis.
- Unstable internal components + stable final behavior -> stop treating raw attention/MLP outputs as primary prediction targets.
- Small whole-L36 behavioral impact -> test cheaper whole-layer replacement or adaptive L36 skipping.
- Large whole-L36 impact -> preserve L36 and investigate lower-cost approximate transitions rather than direct skipping.


## Smoke result: 2026-09-21

The 4-question smoke run completed with exact fresh-to-fresh reproducibility for every captured object:

- final logits: KL = 0, top-1 = 1.0
- H35: cosine = 1.0, mean absolute difference = 0
- L36 attention: cosine = 1.0, mean absolute difference = 0
- L36 MLP: cosine = 1.0, mean absolute difference = 0
- H36: cosine = 1.0, mean absolute difference = 0

This means the internal decomposition is reproducible when the same forward path is used.

The earlier EXP-0009H stored-vs-fresh attention mismatch should therefore not be interpreted as intrinsic decomposition instability. H stored its trace from a forward requesting \`output_hidden_states=True\`, while its later fresh component capture used a forward that did not request hidden states. The difference in forward configuration is now the primary implementation hypothesis to test explicitly.

Whole-L36 functional skip produced:

- mean top-1 agreement: 0.919922
- mean target probability ratio: 0.969845
- mean KL: 0.181536
- mean logit L2: 5226.49

L36 therefore has a measurable behavioral contribution. A complete skip is not behaviorally free, although the target-token probability remained relatively close on this smoke set.

## Decision

Do not pursue the "unstable decomposition" hypothesis.

Before training another sublayer predictor, isolate the capture-path difference explicitly by comparing:

1. forward A vs forward B with \`output_hidden_states=False\`
2. forward A vs forward B with \`output_hidden_states=True\`
3. \`output_hidden_states=True\` vs \`False\` on the same input

This should determine why EXP-0009H reported a large stored-vs-fresh component difference while EXP-0009I reports exact fresh-to-fresh equality.

After that control, the useful architectural question is no longer whether L36 can simply disappear. The current evidence supports investigating a cheaper approximation of the full L36 transition rather than raw attention/MLP reconstruction.
