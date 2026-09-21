# EXP-0007: Qwen3 Computation and Memory Instrumentation

## Research question

Where does a pretrained Qwen3-4B-Base actually spend its computation, and how redundant are its representations across layers and adjacent generated tokens?

EXP-0007 is a measurement experiment. It does not modify model weights and does not replace the Qwen3 generation path.

The first implementation measures:

1. Hidden-state changes through the 36-layer network.
2. Attention and MLP activity independently.
3. Representation similarity across adjacent generated tokens.
4. Intermediate predictive behavior during normal free-running generation.
5. Token-level difficulty signals.
6. Generation-level hardware telemetry.

## Controlled setup

Defaults match the established CFI evaluation protocol:

- Qwen3-4B-Base, 36 layers
- 4-bit NF4 with double quantization
- BF16 compute
- RTX 3070 target
- fixed CFI-Eval-0001-GSM8K dataset
- default gate: 5 questions
- seed: 42000 + question index
- do_sample=True
- temperature 0.6
- top_p 0.95
- top_k 20
- max_new_tokens 512
- stop strings: \nQuestion: and \nProblem:

The instrumented runtime is not a speed benchmark. Forward hooks and intermediate projections add measurement overhead. Its latency must not be compared directly with the clean EXP-0006 or baseline evaluator.

The first generated token is produced by the initial full-prompt (prefill) forward. The current decode instrumentation begins on the subsequent one-token forward steps, so a clean run should have exactly one fewer observed instrumentation step than generated tokens. Instrumentation samples are aligned to generated tokens 2..N, not token 1.

## Layer measurements

Every observed decode step records for every layer:

- hidden-state norm
- delta from the preceding layer
- relative delta from the preceding layer
- cosine similarity to the preceding layer
- relative delta from the same layer on the previous generated token
- cosine similarity to the same layer on the previous generated token

Selected layers also record:

- intermediate top-1 prediction
- intermediate top-1 confidence
- intermediate top-1 margin
- agreement with the final top-1 prediction

Default probe layers are 6, 12, 18, 24, and 30. Layer 36 is also recorded as the final reference.

## Sublayer measurements

For each decoder layer:

- attention output norm
- attention output-to-input norm ratio
- attention mean absolute activation
- attention exact-zero fraction
- attention fraction below 1e-3
- MLP output norm
- MLP output-to-input norm ratio
- MLP mean absolute activation
- MLP exact-zero fraction
- MLP fraction below 1e-3

These are activity and magnitude measurements, not causal importance scores.

## Token difficulty

Each decode step records:

- final top-1 probability
- final top-1/top-2 logit margin
- final entropy
- sampled token ID and text

These signals provide candidate controls for later adaptive-compute experiments.

## Output files

Default output:

E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0007-Instrumentation

Files:

- summary.json
- layer_summary.json
- sublayer_summary.json
- token_metrics.jsonl
- layer_metrics.jsonl
- sublayer_metrics.jsonl
- questions.jsonl
- hardware_samples.jsonl

JSON and JSONL are the source of truth. Hardware telemetry requires the repository package and its system dependency to be installed in the active environment.

## Five-question gate

Run from the CFI repository environment:

PowerShell:

$env:TEMP = "E:\Titan Forge Industries\CFI-Data\Temp"
$env:TMP = "E:\Titan Forge Industries\CFI-Data\Temp"
$env:HF_HOME = "E:\Titan Forge Industries\CFI-Data\HuggingFace"
$env:HF_HUB_CACHE = "E:\Titan Forge Industries\CFI-Data\HuggingFace\hub"

python scripts/measure_exp0007.py

Equivalent explicit run:

python scripts/measure_exp0007.py --questions 5 --probe-layers 6 12 18 24 30 --max-new-tokens 512 --seed-base 42000 --temperature 0.6 --top-p 0.95 --top-k 20

## Full fixed-set run

Run the 100-question version only after the 5-question gate completes cleanly, the expected one-token prefill offset is accounted for, and there are no unexpected instrumentation mismatches:

python -m pip install -e ".[system]"

python scripts/measure_exp0007.py --questions 100

## Interpretation

EXP-0007 is intended to find measurable redundancy and locality, not to declare that a layer or sublayer is safe to remove.

Potential useful outcome:

high fraction of tokens
to
small relative sublayer change
to
high adjacent-token similarity
to
candidate for conditional compute or reuse

Another useful outcome:

token or task state
to
repeatedly dominant layer ranges
to
candidate for task-conditioned routing

A null result is also useful because it eliminates a CFI hypothesis before a purpose-built architecture is built.

## Current limits

This first pass does not infer:

- exact VRAM/RAM/NVMe bytes moved per layer
- exact parameter bytes accessed
- exact KV-cache reuse ratio
- task-level expert locality in a model without explicit experts
- causal proof that measured low activity permits safe skipping without retraining

Those require dedicated lower-level experiments.

## Decision gates

### Gate A: Layer utility

Strong token-dependent redundancy or small later-layer changes should lead to dynamic depth or sublayer experiments.

### Gate B: Sublayer utility

Large attention-versus-MLP differences should lead to independent sublayer gating before whole-layer skipping.

### Gate C: Temporal redundancy

Strong adjacent-token hidden-state similarity should lead to representation reuse and neural computation cache experiments.

### Gate D: Intermediate prediction

Strong depth-dependent intermediate agreement should be tested as a training-free adaptive-compute signal.

### Gate E: No useful redundancy

If nearly every token requires broad high-magnitude computation, shift emphasis toward memory representation, routing, lookup computation, and alternative parameterization.

## Relation to EXP-0006

EXP-0006 showed that the 30/36-layer shared-weight token drafter was technically viable but too expensive to accelerate generation on the RTX 3070.

EXP-0007 therefore tests the narrower question underneath that failure:

Can useful intermediate or redundant computation be identified without repeatedly executing a nearly complete secondary token model?

That is why this experiment comes before another speculative-decoding optimization.
