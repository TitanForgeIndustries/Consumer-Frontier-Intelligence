# EXP-0009Q: Cache-Safe L36 MLP Reconstruction

## Purpose

EXP-0009P established that the whole-L36 learned replacement improves teacher-forced similarity but does not remain stable during free-running generation.

A whole-layer adaptive fallback is also unsafe without preserving L36's KV cache. If L36 is skipped, its attention cache is not updated, so a later fallback cannot transparently resume the original L36 computation.

Q therefore takes a cache-safe decomposition:

- execute L36 self-attention normally
- preserve its real KV-cache update
- replace only the L36 MLP residual with a compact learned predictor

This tests whether the most problematic part of the whole-layer replacement is the removal of L36 attention and its recurrent state, rather than the learned reconstruction idea itself.

## Runtime paths

### Full

```
L1 -> ... -> L35 -> L36 attention -> L36 MLP -> final norm -> LM head
```

### MLP skip

```
L1 -> ... -> L35 -> L36 attention -> zero MLP -> final norm -> LM head
```

### MLP predicted

```
L1 -> ... -> L35 -> L36 attention -> learned MLP predictor -> final norm -> LM head
```

Only the MLP replacement differs. L36 self-attention and its KV cache remain genuine Qwen3 computation.

## Learned component

The replacement predicts the L36 MLP output from the normalized L36 MLP input:

```
MLP input
  -> Linear(hidden, 128)
  -> GELU
  -> Linear(128, hidden)
  -> predicted MLP residual
```

The base Qwen3-4B-Base model remains frozen.

## Training objective

The predictor is trained on matched greedy full-model trajectories using two signals:

1. full-vocabulary teacher KL through Qwen's existing final RMSNorm and LM head
2. normalized MLP-output reconstruction loss

The full-vocabulary behavioral objective is weighted directly; the MLP-output objective is an auxiliary reconstruction term.

## Setup

- Qwen3-4B-Base
- 36 decoder layers
- 4-bit NF4
- BF16 compute
- RTX 3070
- 7 training questions / 3 held out
- 128 generated tokens
- greedy deterministic decoding
- 64 training positions/question
- bottleneck 128
- 16 epochs
- two measured runtime repeats
- 16-token warmup
- CUDA synchronization around timing

## Measurements

Teacher-forced held-out:

- full-model KL
- top-1 agreement
- target-token probability ratio
- target log-probability delta
- logit L2 distance

Runtime:

- wall-clock generation time
- tokens/second
- speedup
- exact generated-token agreement with the full model
- first divergence position
- parsed answer when available

## Why this experiment matters

A whole-layer early exit changes both output behavior and the state maintained by L36 attention.

This experiment removes that ambiguity. If MLP reconstruction improves free-running agreement while retaining genuine L36 attention, that indicates L36 attention and its cache are a critical part of autoregressive stability.

If MLP reconstruction still diverges badly despite exact L36 attention, the learned replacement itself is the primary issue.

If behavior becomes stable but speed improvement is tiny, that points toward a more targeted computational-compression strategy rather than whole-layer skipping.

## Decision framework

- **High behavior + measurable speedup:** pursue sublayer compression and potentially apply the same approach to earlier expensive MLPs.
- **Improved behavior but negligible speedup:** focus on execution efficiency and lower-cost MLP approximators.
- **No behavioral improvement:** investigate conditioning, training objective, or richer state information before increasing capacity.
- **Runtime regression:** the predictor overhead is larger than the saved MLP computation under the current implementation.

## Limitations

Only three questions are held out for runtime evaluation.

The predictor is still trained from teacher-forced full-model trajectories rather than fully on-policy states.

This experiment does not establish whole-layer L36 skipping. It tests a cache-safe sublayer replacement.

## Run

```powershell
cd "E:\Titan Forge Industries\Consumer-Frontier-Intelligence"
git switch exp-0009q-cache-safe-mlp-reconstruction
git pull origin exp-0009q-cache-safe-mlp-reconstruction

python scripts/run_exp0009q.py --questions 10 --train-questions 7 --max-new-tokens 128 --epochs 16 --bottleneck 128 --repeats 2 --warmup-tokens 16 --output "E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009Q-Cache-Safe-MLP-Reconstruction"
```

## Status

Implementation committed. Runtime measurements pending.


## First Run Failure and Fix

The first execution completed all ten greedy trace captures and all 16 training epochs, then stopped during teacher-forced evaluation of held-out question 8.

Failure:

```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

The cause was a mixed-dtype evaluation path: the runtime copy of the learned MLP predictor was BF16, while the teacher-forced source tensor was explicitly cast to FP32 before entering the predictor.

The evaluation path was corrected to cast the predictor input to the predictor parameter dtype and cast the predicted MLP residual back to FP32 before combining it with the FP32 residual state.

No model weights, training objective, or runtime replacement architecture were changed.

The failed run is recorded as an implementation failure, not a scientific result.

The transition training itself completed successfully before the failure. The final training loss in the interrupted run was 0.058531.
