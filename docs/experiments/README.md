# Experiments

This directory contains detailed records of individual research experiments.

The repository includes an experiment logger under `src/cfi_experiment_logger/`. It stores each run as machine-readable JSON/JSONL, captures hardware telemetry, imports training metrics, records evaluations, and generates Markdown/CSV/Excel outputs.

## Research records

- [EXP-0007: Qwen3 computation and memory instrumentation](EXP-0007-instrumentation.md)
- [EXP-0008: causal late-layer sublayer ablation](EXP-0008-causal-sublayer-ablation.md) and [results](EXP-0008-causal-sublayer-ablation-results.md)
- [EXP-0009: predictive latent-state replacement](EXP-0009-predictive-latent-state.md)
- [EXP-0009B: conditioned latent state](EXP-0009B-conditioned-latent-state.md) and [results](EXP-0009B-conditioned-latent-state-results.md)
- [EXP-0009C: predictive depth sweep](EXP-0009C-predictive-depth-sweep.md) and [results](EXP-0009C-results.md)
- [EXP-0009D: behavioral latent predictor](EXP-0009D-behavioral-latent-predictor.md)
- [EXP-0009E: copy-anchored full behavior](EXP-0009E-copy-anchored-full-behavior.md)
- [EXP-0009F: causal context predictor](EXP-0009F-causal-context-predictor.md)
- [EXP-0009G: factorized transition predictor](EXP-0009G-factorized-transition-predictor.md)
- [EXP-0009H: sublayer predictability diagnostic](EXP-0009H-sublayer-predictability-diagnostic.md)
- [EXP-0009I: decomposition stability and layer skip](EXP-0009I-decomposition-stability-layer-skip.md)
- [EXP-0009J: capture-path control](EXP-0009J-capture-path-control.md)
- [EXP-0009K: direct behavioral exit-head protocol (no recorded result)](EXP-0009K-direct-behavioral-exit-head.md)
- [EXP-0009L: anchored early exit](EXP-0009L-anchored-early-exit.md)
- [EXP-0009M: boundary equivalence (one-question smoke only; full control outstanding)](EXP-0009M-layer-boundary-equivalence.md)
- [EXP-0009N: early-exit generalization](EXP-0009N-early-exit-generalization.md)
- [EXP-0009O: integrated early-exit runtime](EXP-0009O-integrated-early-exit-runtime.md)
- [EXP-0009P: matched greedy trajectories](EXP-0009P-matched-greedy-trajectory.md)
- [EXP-0009Q: cache-safe L36 MLP reconstruction](EXP-0009Q-cache-safe-mlp-reconstruction.md)
- [EXP-0009R: scorable cache-safe evaluation](EXP-0009R-scorable-cache-safe-evaluation.md)
- [EXP-0010: independent CFI v0 sparse-capacity smoke](EXP-0010-cfi-v0-smoke.md)
- [EXP-0011: route-collapse and useful-capacity control](EXP-0011-route-collapse-capacity-control.md)

## Record structure

```text
experiments/
└── EXP-0001/
    ├── config.json
    ├── events.jsonl
    ├── hardware_samples.jsonl
    ├── evaluation.json
    ├── summary.json
    └── report.md
```

JSON/JSONL is the source of truth. The spreadsheet is an export so experiment history stays diffable, scriptable, and reproducible.

## CLI

Install the repository:

```bash
pip install -e .
```

For Excel export and CPU/system-RAM sampling:

```bash
pip install -e ".[all]"
```

Initialize a run:

```bash
cfi-experiment init EXP-0001 \
  --model Qwen3-4B-Base \
  --method QLoRA \
  --dataset open-r1/DAPO-Math-17k-Processed \
  --max-steps 250 \
  --context-length 2048
```

Start hardware monitoring in another terminal while training:

```bash
cfi-experiment monitor EXP-0001 --interval 2
```

Stop the monitor with Ctrl+C when training ends.

Import framework metrics after the run:

```bash
cfi-experiment import-metrics EXP-0001 path/to/metrics.json
```

CSV, JSON, and JSONL are accepted. Common names such as `step`, `loss`, `learning_rate`, `grad_norm`, `tokens`, and throughput are normalized automatically.

Record held-out evaluation:

```bash
cfi-experiment record EXP-0001 --kind evaluation \
  --data '{"benchmark":"held-out-math","accuracy":0.42}'
```

Finalize the experiment:

```bash
cfi-experiment finalize EXP-0001 --status completed \
  --notes "Baseline complete."
```

Regenerate the project-wide registry:

```bash
cfi-experiment export
```

Outputs:

- `results/CFI_Experiment_Registry.csv`
- `results/CFI_Experiment_Registry.xlsx` when the Excel extra is installed

## What gets tracked

The logger is designed around the CFI research question, not just training loss. It can track model/method/dataset metadata, training steps and loss, tokens and throughput, GPU VRAM/utilization/power/temperature, system RAM, evaluation metrics, runtime, hypotheses, architectural changes, and notes.

Raw checkpoints, datasets, and very large logs should remain outside Git unless there is a deliberate reason to version them.
