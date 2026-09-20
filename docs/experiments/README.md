# Experiments

This directory contains detailed records of individual research experiments.

The repository includes an experiment logger under `src/cfi_experiment_logger/`. It stores each run as machine-readable JSON/JSONL, captures hardware telemetry, imports training metrics, records evaluations, and generates Markdown/CSV/Excel outputs.

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
