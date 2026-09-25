# EXP-0009R: Scorable Cache-Safe Evaluation

## Status

Completed for this limited configuration on 2026-09-22 local time. The baseline
gate passed and the comparison ran from a separate fresh Python process. Q is
closed separately; R does not revise its 128-token results or its three
documented implementation failures.

## Question

Q improved held-out teacher-forced KL but diverged early when generated freely.
At a forced 128-token limit, none of its full-model GSM8K answers parsed, so Q
could not measure end-to-end answer accuracy. R first tests whether allowing the
full frozen model to stop naturally within 512 tokens yields scorable answers.
Only if all three held-out answers parse does R compare the original Q predictor
with full L36 and a zero-MLP control at that horizon.

## Fixed setup

- Same local Qwen3-4B-Base snapshot, 4-bit NF4/BF16 compute, frozen weights,
  existing GSM8K dataset, held-out questions 8-10, prompt, numeric parser,
  and greedy seed base 42000 as Q.
- Natural-stop generation: `max_new_tokens=512`, no `min_new_tokens`. Record the
  number of tokens, whether EOS was reached, complete text, parsed numeric
  answer, correctness, and parsed answers at 128/256/512-token prefixes.
- Gate: all three full-model final answers must parse. If one does not, retain
  a `baseline_unscorable` summary and stop before running any replacement.
  Investigate prompt or scoring separately rather than silently changing R.
- On passing the gate, reuse the **existing EXP-0009Q auxiliary checkpoint**;
  do not train or tune a new predictor. Keep genuine L36 attention and its KV
  cache. Compare full L36, zeroed L36 MLP, and predicted L36 MLP under the
  same prompt and natural-stop cap. Record parsed answers, correctness,
  generation length, exact token agreement, and first divergence from full.
- Measure physical runtime separately with exactly 128 generated tokens,
  16-token warmup, two repeats per mode and question, CUDA synchronization,
  and a rotated measurement order across questions/repeats. This equal-token
  timing avoids mistaking shorter divergent answers for speedup. Record peak
  CUDA allocation. No claim of quality-preserving acceleration follows from
  timing alone.
- Both runs are fresh Python processes with separate local output directories.
  Never overwrite Q artifacts or the baseline gate's summary. Implementation
  exceptions are written to `implementation_failure.json`, distinct from
  scientific outcomes in `summary.json`.

## Interpretation

Three held-out questions are a diagnostic, not a reliable estimate of general
accuracy or speed. A parseable but incorrect full-model answer is still
scorable, though it cannot establish useful answer quality. If predicted MLP
diverges early or alters correctness, any fixed-token timing advantage is not
quality preserving. If the gate fails, R's replacement comparison remains
unrun, not a negative result for the predictor.

## Commands

Run the baseline gate first from the repository root, with the experiment dependencies installed and the required Q checkpoint available:

```powershell
$env:CFI_DATA_ROOT = Join-Path (Get-Location) ".cfi-data"
$env:HF_HOME = Join-Path $env:CFI_DATA_ROOT "HuggingFace"
python scripts/run_exp0009r.py --baseline-only --max-new-tokens 512 --output ".cfi-data/Results/CFI-Eval-0009R-Scoring-Gate"
```

Only after the gate summary reports `baseline_ready`, run the comparison
from a fresh process and a different directory:

```powershell
$env:CFI_DATA_ROOT = Join-Path (Get-Location) ".cfi-data"
$env:HF_HOME = Join-Path $env:CFI_DATA_ROOT "HuggingFace"
python scripts/run_exp0009r.py --max-new-tokens 512 --runtime-tokens 128 --warmup-tokens 16 --repeats 2 --output ".cfi-data/Results/CFI-Eval-0009R-Scorable-Cache-Safe-Evaluation"
```

## Completed results

The baseline gate summary is in
`CFI_DATA_ROOT/Results/CFI-Eval-0009R-Scoring-Gate`.
The separate comparison summary is in
`CFI_DATA_ROOT/Results/CFI-Eval-0009R-Scorable-Cache-Safe-Evaluation`.
Both retain complete generated text and token IDs; their corresponding
`-run.log` files are adjacent to the result directories. The full-model token
sequences were identical across the two fresh processes, as were the dataset
and Q-checkpoint SHA-256 hashes. Neither run recorded an implementation
failure. The Q predictor checkpoint was loaded for inference only; no model or
auxiliary-predictor weights were trained in R.

The natural-stop full baseline passed the gate. All three answers were parsed
correctly and ended with EOS. The existing parser could not score any
128-token prefix; Q8 and Q9 became scorable by the 256-token cap, and Q10
needed 337 tokens:

| Question | Expected | Full answer | Full generated tokens | Parsed at 128 / 256 / 512 |
|---|---:|---:|---:|---|
| 8 | 34 | 34 | 241 | no / yes / yes |
| 9 | 120 | 120 | 152 | no / yes / yes |
| 10 | 11 | 11 | 337 | no / no / yes |

The replacement paths also produced the correct parsed answer on all three
held-out questions, despite diverging from full-model token sequences early:

| Question | Zero answer / tokens | Predicted answer / tokens | Zero exact matches / shared tokens | Predicted exact matches / shared tokens | First divergence zero / predicted |
|---|---|---|---:|---:|---:|
| 8 | 34 / 231 | 34 / 190 | 10 / 231 | 12 / 190 | 10 / 10 |
| 9 | 120 / 159 | 120 / 192 | 7 / 152 | 6 / 152 | 7 / 7 |
| 10 | 11 / 335 | 11 / 265 | 14 / 335 | 22 / 265 | 11 / 10 |

The answer-level result is 3/3 for full, 3/3 for zero, and 3/3 for predicted
MLP on this small selection. Early exact-token divergence is therefore not
equivalent to losing these specific final numeric answers. Conversely, it is
not evidence that the replacement is generally behaviorally interchangeable:
the generated trajectories and lengths are substantially different.

The following measurements use the separate *fixed 128-token* timing path,
with two repeats and rotated order. Times are the mean seconds per question;
speedups divide that question's mean full time by its replacement mean time:

| Question | Full seconds | Zero seconds | Predicted seconds | Zero speedup | Predicted speedup |
|---|---:|---:|---:|---:|---:|
| 8 | 14.243 | 15.830 | 15.271 | 0.8998x | 0.9327x |
| 9 | 12.176 | 12.456 | 11.431 | 0.9776x | 1.0652x |
| 10 | 11.362 | 11.280 | 11.439 | 1.0073x | 0.9933x |

The arithmetic mean of per-question speedups is 0.9615x for zero and 0.9971x
for predicted MLP. Across the six timed 128-token runs per mode, mean elapsed
seconds were 12.5940 full, 13.1885 zero, and 12.7138 predicted. Mean peak
CUDA allocation was approximately 2.547 GiB for every mode. These are
descriptive timings on only three questions, not a robust speed estimate.
Even on these measurements there is no consistent physical speed advantage;
the predicted MLP was slower on questions 8 and 10.

## R Decision

**Closed for this exploratory configuration.** Removing the forced 128-token
stop made the original full-model baseline scorable. The reused Q predictor
retained the same final numeric answer on all three cases, but so did zeroing
the MLP; the tiny sample cannot distinguish their answer-level utility or
establish general quality retention. The fixed-token measurements do not show
an acceleration. Q's earlier teacher-forced and free-running findings remain
intact. Do not promote this replacement as a quality-preserving speedup or
start a larger follow-up without a separately declared design and sample.
