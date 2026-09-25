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

Run from the repository root with the experiment dependencies installed. This runner is included on `main`.

```powershell
python scripts/run_exp0009q.py --questions 10 --train-questions 7 --max-new-tokens 128 --epochs 16 --bottleneck 128 --repeats 2 --warmup-tokens 16 --output ".cfi-data/Results/CFI-Eval-0009Q-Cache-Safe-MLP-Reconstruction"
```

## Status

Completed on the established RTX 3070 setup. Two post-fix runs produced the
same held-out teacher-forced and generated-token results. The measured runtime
ratios varied; see the completed results and decision below.


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


### Second Implementation Failure and Fix

The corrected rerun then reached held-out evaluation but stopped with:

```
NameError: name 'distribution_metrics' is not defined
```

The evaluation function called the established `distribution_metrics()` helper but the import had been omitted from the final Q script.

The helper import is now restored from `run_exp0009n.py`.

No experiment logic or model behavior was changed.

This run is also recorded as an implementation failure, not a scientific result.


### Third Implementation Failure and Fix

The next corrected run successfully reached held-out Q8 teacher-forced evaluation:

```
MLP zero: KL=0.0967 top1=0.9297
MLP predicted: KL=0.0853 top1=0.9375
```

It then stopped when entering the runtime MLP-zero control with:

```
StopIteration
```

The cause was that the parameterless `ZeroMLP` control was passed through a generic wrapper that attempted to infer its dtype by reading its first parameter. Because the zero control intentionally has no parameters, that lookup failed.

The wrapper now falls back to the incoming hidden-state dtype when the replacement module has no parameters.

The teacher-forced values above are recorded as a partial diagnostic from the run, but the run is not treated as a complete scientific result because runtime evaluation did not finish.

## Completed Results (2026-09-22, local time)

The corrected script completed twice from fresh Python processes: an existing
post-fix run in
`CFI_DATA_ROOT/Results/CFI-Eval-0009Q-Cache-Safe-MLP-Reconstruction`
and a fresh rerun in
`CFI_DATA_ROOT/Results/CFI-Eval-0009Q-Cache-Safe-MLP-Reconstruction-rerun-20260922`.
Each directory retains its own `summary.json` and FP32 auxiliary-predictor
checkpoint; the rerun also retains `run.log`. The earlier completed result was
not overwritten. Both summaries report `status=completed`, ten matched greedy
traces, seven training questions, three held-out questions (8-10), 128 tokens,
64 training positions per question, 16 epochs, two measured repeats per mode,
and a 16-token warmup. The 658,048-parameter MLP predictor alone was trained;
Qwen3-4B-Base remained frozen. Training mean loss was 0.123276 in epoch 1 and
0.058531 in epoch 16, with a lower intermediate value; it was not monotonic.

Held-out teacher-forced metrics were identical across the two completed runs:

| Metric | Zero L36 MLP | Predicted L36 MLP |
|---|---:|---:|
| Mean KL to full-model teacher | 0.08565601 | **0.07878638** |
| Mean top-1 agreement | 0.94791667 | **0.95052083** |
| Mean target-token probability ratio | 1.01202110 | 1.01122487 |
| Mean target log-probability delta | -0.02593932 | -0.02582835 |
| Mean logit L2 distance | 5066.5073 | 4485.4642 |

The predictor reduced KL by 8.02% relative to the zero-MLP control and raised
top-1 agreement by 0.26 percentage points on these teacher-forced held-out
traces (one additional matching top-1 position out of 384). The target-token
probability ratio did not improve, and these numbers do not measure
free-running stability.

The runtime path temporarily replaces only `model.model.layers[35].mlp.forward`.
It physically omits the original L36 MLP computation while continuing to run
genuine L36 attention and its KV-cache updates. Each mode generated 128 tokens
per question. The measured mean seconds per question were:

| Run | Question | Full L36 | Zero MLP | Predicted MLP | Zero speedup | Predicted speedup |
|---|---:|---:|---:|---:|---:|---:|
| Existing completed run | 8 | 22.9057 | 23.2882 | 23.7779 | 0.9836x | 0.9633x |
| Existing completed run | 9 | 13.6126 | 12.1143 | 12.4206 | 1.1237x | 1.0960x |
| Existing completed run | 10 | 12.3075 | 12.8611 | 12.3208 | 0.9570x | 0.9989x |
| Fresh rerun | 8 | 13.1543 | 11.4553 | 10.5375 | 1.1483x | 1.2483x |
| Fresh rerun | 9 | 11.0418 | 10.9466 | 10.4581 | 1.0087x | 1.0558x |
| Fresh rerun | 10 | 11.1871 | 13.5577 | 12.8541 | 0.8251x | 0.8703x |

The script's arithmetic mean of per-question speedup ratios was 1.0214x for
zero and 1.0194x for predicted in the existing run, versus 0.9941x for zero
and 1.0582x for predicted in the fresh rerun. Per-question timing varied
substantially, particularly for question 8, and question 10 regressed in the
fresh rerun. With only three held-out questions, two repeats, and a fixed
full/zero/predicted measurement order, these timings do not establish a
reliable general speedup or a quality-preserving acceleration.
Mean measured peak GPU allocation in the fresh rerun was 2.561 GiB for full
and 2.560 GiB for each replacement mode; this is not a meaningful memory
reduction at the recorded precision.

Generated-token behavior was identical across completed runs:

| Question | Zero agreement with full | Predicted agreement with full | First divergence: zero / predicted |
|---|---:|---:|---:|
| 8 | 9/128 (7.03%) | 10/128 (7.81%) | 10 / 10 |
| 9 | 6/128 (4.69%) | 6/128 (4.69%) | 7 / 7 |
| 10 | 12/128 (9.38%) | 22/128 (17.19%) | 11 / 10 |
| Mean | 27/384 (7.03%) | 38/384 (9.90%) | — |

The mean improvement of 2.86 percentage points over zeroing is small relative
to the remaining sequence divergence. Retaining L36 attention and its cache
did not make this compact MLP replacement interchangeable with full L36 during
autoregressive generation. All full-model baseline generations in these
128-token runs had `predicted=None` under the GSM8K answer parser, so this
experiment does not establish an end-to-end question-accuracy comparison.

## Q Decision

**Closed for this configuration: local reconstruction signal, free-running
behavioral failure for direct MLP replacement, and inconclusive runtime
advantage.** The post-fix runs establish a completed result; the earlier
three crashes remain implementation failures, not negative scientific runs.
The result does not show that all cache-safe MLP replacements are impossible,
nor does it justify claiming a useful quality-preserving speedup. Do not infer
free-running quality from teacher-forced KL or enlarge the predictor blindly.
