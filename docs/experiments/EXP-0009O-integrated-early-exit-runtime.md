# EXP-0009O: Integrated Early-Exit Runtime Benchmark

## Purpose

Turn the EXP-0009N behavioral result into an actual runtime experiment.

EXP-0009N showed that a 663,168-parameter anchored residual transition can improve the behavioral match between H35 and the full Qwen3-4B output distribution on held-out traces. EXP-0009O tests whether that transition can replace the **actual execution of L36** during generation.

The base Qwen3-4B-Base model remains frozen.

## Runtime paths

### Full

```
Layers 1-35 -> L36 -> final RMSNorm -> LM head
```

### Raw H35

```
Layers 1-35 -> bypass L36 -> final RMSNorm -> LM head
```

### Adapted early exit

```
Layers 1-35 -> bypass L36 -> 663k transition -> final RMSNorm -> LM head
```

The raw H35 path establishes the runtime ceiling for a pure layer skip. The adapted path measures whether the learned transition preserves enough behavior to justify its additional computation.

## Critical implementation detail

The experiment temporarily replaces the installed Qwen3 L36 `forward()` method during generation.

This is intentionally different from the EXP-0009I/J functional skip, where L36 was still executed and its output was replaced afterward.

EXP-0009O therefore tests the actual computational claim:

> Can L36 be omitted from the forward pass while a much smaller learned transition supplies a useful approximation of its contribution?

The script first probes the installed L36 ABI and refuses to patch the layer if it does not return a tensor directly. This prevents an unknown cache/output ABI from being silently corrupted.

## Benchmark controls

- Qwen3-4B-Base
- 36 decoder layers
- 4-bit NF4 weights
- BF16 model compute
- RTX 3070
- layers 1-35 executed normally
- L36 physically bypassed in early-exit modes
- existing pretrained final RMSNorm and LM head remain frozen
- 7 questions used to train the transition
- 3 held-out questions used for runtime evaluation
- 128 generated tokens
- greedy deterministic decoding
- 16-token warmup
- 2 measured repeats per mode
- CUDA synchronization around measured generation

Greedy decoding is used here so the runtime A/B comparison is deterministic. This differs from the sampling configuration used by earlier behavioral experiments and should be reported as a benchmark-control change rather than treated as directly equivalent to prior generation results.

## Measurements

For each held-out question and each runtime mode:

- wall-clock generation time
- generated tokens
- tokens/second
- peak allocated GPU memory
- exact token agreement against the full model
- first generated-token position at which the candidate diverges
- parsed final answer when available

Aggregate speedup is:

```
speedup = full_generation_time / candidate_generation_time
```

A value greater than 1.0 means the candidate generated faster than the full 36-layer path under the same benchmark conditions.

## Interpretation rules

### Raw H35

A raw H35 speedup above 1.0 establishes that physically bypassing L36 produces measurable runtime savings on this implementation.

Its token agreement against full Qwen3 is the reference behavior cost of removing L36 without reconstruction.

### Adapted early exit

The adapted path must be considered jointly on:

1. runtime speed
2. token agreement
3. first-divergence position
4. final-answer behavior when parsing succeeds

The adapter adds computation, so it is expected to be somewhat slower than raw H35. The relevant question is whether it remains materially faster than the full 36-layer path while recovering behavior lost by the raw skip.

## Decision framework

### Case A: adapted path is faster and behaviorally improved

Proceed to a cleaner integrated architecture and larger runtime validation.

### Case B: adapted path is faster but behaviorally unstable

Investigate richer or conditional transitions, possibly using earlier depth information or adaptive routing.

### Case C: adapted behavior is good but there is little or no speedup

The transition or framework overhead is consuming the layer savings. Optimize execution structure before increasing model capacity.

### Case D: raw H35 is not meaningfully faster

The computational savings are not surviving the current framework/runtime implementation. Investigate kernel, cache, dispatch, and Python/model-wrapper overhead before claiming an architectural speed benefit.

### Case E: neither early-exit path is useful

Do not blindly increase transition capacity. Revisit the premise that L36 can be compressed from H35 alone and investigate additional source information or adaptive computation.

## Limitations

This is a prototype benchmark, not a production early-exit implementation.

The experiment still retrains the small transition from captured traces before benchmarking. Training time is therefore not included in generation throughput.

The benchmark also uses only three held-out questions. A favorable result should be followed by a larger evaluation before being presented as a general speedup result.

No claim of frontier-level capability, 4T parameter equivalence, or general model acceleration is made by this experiment.

## Run

```powershell
cd "E:\Titan Forge Industries\Consumer-Frontier-Intelligence"
.\.venv\Scripts\Activate.ps1
git switch exp-0009o-integrated-early-exit-runtime
git pull origin exp-0009o-integrated-early-exit-runtime

python scripts/run_exp0009o.py --questions 10 --train-questions 7 --max-new-tokens 128 --epochs 16 --bottleneck 128 --repeats 2 --warmup-tokens 16 --output "E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009O-Integrated-Early-Exit-Runtime"
```

## Status

Implementation committed. Runtime measurements pending.
