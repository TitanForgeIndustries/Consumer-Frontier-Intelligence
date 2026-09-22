# EXP-0009P: Matched-Greedy Trajectory Early-Exit Validation

## Purpose

Resolve the principal training/evaluation confound identified in EXP-0009O.

EXP-0009N trained the anchored residual transition on sampled trajectories:

- temperature 0.6
- top-p 0.95
- top-k 20

EXP-0009O then benchmarked that transition using greedy generation.

P generates the training traces with the **same greedy decoding regime used by the runtime benchmark**, so the learned transition is trained and evaluated on a matched autoregressive trajectory distribution.

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

### Adapted

```
Layers 1-35 -> bypass L36 -> 663,168-parameter transition -> final RMSNorm -> LM head
```

L36 is physically bypassed in the early-exit paths.

## Hypothesis

The low free-running agreement in O may partly reflect a trajectory-distribution mismatch rather than only insufficient representational capacity.

P tests that hypothesis without increasing transition capacity:

> When the transition is trained on the same greedy trajectory regime used at runtime, does behavioral preservation improve while the measured L36 speed savings remain?

## Setup

- Qwen3-4B-Base
- 36 decoder layers
- 4-bit NF4 weights
- BF16 compute
- RTX 3070
- L36 physically bypassed for early-exit runtime modes
- Qwen weights frozen
- 7 training questions
- 3 held-out questions
- 128 generated tokens
- greedy deterministic decoding
- 64 training positions/question
- bottleneck 128
- 16 epochs
- two measured runtime repeats
- 16-token runtime warmup
- CUDA synchronization around timing

## Measurements

P records both teacher-forced behavioral metrics and actual runtime behavior.

Teacher-forced held-out metrics:

- KL to full-model teacher
- top-1 agreement
- target probability ratio
- target log-probability delta
- logit L2

Runtime metrics:

- wall-clock generation time
- tokens/second
- physical L36 bypass
- peak GPU memory
- exact token agreement against full generation
- first divergence position
- parsed answer when available

## Interpretation

A large increase in free-running agreement compared with O would indicate that the O mismatch was at least partly caused by training on sampled trajectories while evaluating greedily.

Little or no improvement would shift attention back toward:

- autoregressive error accumulation after the first divergence
- insufficient conditioning of the H35-only transition
- missing information from earlier layers or internal L36 state
- the need for verification/correction rather than one-shot replacement

P intentionally does **not** increase the transition width or depth. This isolates the trajectory-matching variable.

## Limitations

Only three held-out questions are used for runtime evaluation.

The transition is still trained with teacher-forced full-model token distributions along the matched full-model greedy traces. It is not yet trained fully on-policy from its own generated states.

The benchmark is therefore a targeted confound-control experiment, not a final quality validation.

No claim of production-level acceleration or general model acceleration is made.

## Run

```powershell
cd "E:\Titan Forge Industries\Consumer-Frontier-Intelligence"
git switch exp-0009p-matched-greedy-trajectory
git pull origin exp-0009p-matched-greedy-trajectory

python scripts/run_exp0009p.py --questions 10 --train-questions 7 --max-new-tokens 128 --epochs 16 --bottleneck 128 --repeats 2 --warmup-tokens 16 --output "E:\Titan Forge Industries\CFI-Data\Results\CFI-Eval-0009P-Matched-Greedy-Trajectory"
```


## Observed Run

The corrected P run completed all ten matched-greedy trace captures, all 16 training epochs, and all three held-out runtime evaluations.

Training loss decreased monotonically from 0.171319 at epoch 1 to 0.009709 at epoch 16.

Held-out teacher-forced mean:

| Metric | Raw H35 | Matched-greedy adapted |
|---|---:|---:|
| KL to full teacher | 0.16673453 | **0.13035729** |
| Top-1 agreement | 0.921875 | **0.937500** |

The adapted transition therefore reduced teacher-forced KL by approximately 21.8% relative to raw H35 and increased teacher-forced top-1 agreement by 1.56 percentage points.

Held-out runtime mean:

| Metric | Full 36-layer | Raw H35 | Adapted |
|---|---:|---:|---:|
| Generation time | 11.5450 s | 11.2194 s | **11.1912 s** |
| Tokens/s | 11.0934 | 11.4183 | **11.4445** |
| Speedup | 1.000x | **1.0293x** | **1.0318x** |
| Free-running token agreement vs full | 100% | 8.59% | **7.29%** |

Per-question adapted runtime speedups were 1.008x, 1.055x, and 1.033x for held-out questions 8, 9, and 10 respectively.

The matched trajectory regime did **not** recover free-running agreement. In fact, adapted agreement was slightly below the raw H35 baseline in this run. Therefore the O decoding mismatch hypothesis is not supported by P.

## P Decision

**Result: trajectory mismatch rejected as the primary explanation.**

The transition consistently improves teacher-forced token-distribution preservation, but that improvement does not survive autoregressive execution.

This separates two properties:

- **Local behavioral reconstruction:** supported.
- **Stable autoregressive replacement of L36:** not supported.

The remaining central problem is likely error accumulation after small token-level deviations, or missing information needed to determine when the learned replacement is safe.

Increasing adapter capacity blindly is not justified yet.

The next experiment should introduce **adaptive verification/gating**: use the cheap early-exit path when a learned gate predicts that it is safe, and execute L36 only when the gate predicts elevated risk. This trades some of the theoretical one-layer savings for the possibility of preserving the full model's trajectory when the approximation is unreliable.

## First Run Failure and Fix

The first P execution completed all ten matched-greedy trace captures and all 16 training epochs. It then stopped during held-out question 8 before producing evaluation results.

Failure:

```
RuntimeError: expected scalar type Float but found BFloat16
```

The cause was a dtype mismatch in the teacher-forced evaluation path. Captured H35 states are BF16, while the trained transition is maintained in FP32 for training. The runtime path in EXP-0009O already handled this conversion explicitly.

The evaluation path was corrected to cast H35 inputs to the trained transition's parameter dtype before the transition forward pass. No model weights or experiment architecture were changed.

The failed run is retained as an implementation failure, not treated as a scientific result.
