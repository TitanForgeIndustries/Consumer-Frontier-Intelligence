# EXP-0009N: Early-Exit Generalization with Matched Baselines

## Purpose

Validate the corrected EXP-0009L anchored early-exit result on a larger split and compare the learned exit against a functional whole-L36 skip under the same evaluation conditions.

Comparison:

1. full model oracle
2. raw H35 direct exit through the existing final RMSNorm + LM head
3. learned bounded residual H35 exit
4. functional whole-L36 skip

## Architecture

The learned transition is the validated EXP-0009L adapter:

\`LayerNorm -> Linear(hidden, 128) -> GELU -> Linear(128, hidden)\`

followed by a bounded residual added to H35.

The existing final RMSNorm and pretrained LM head remain frozen.

The base model is frozen and the transition is trained directly through the differentiable final output stack, using the full teacher token distribution.

## Validation goals

The key question is whether the improvement seen in EXP-0009L generalizes beyond two held-out questions.

All three reduced-compute candidates are evaluated on the same held-out traces:

- raw H35 exit
- learned adapted exit
- functional L36 skip

Metrics:

- KL divergence to full-model teacher
- top-1 agreement
- target-token probability ratio
- target log-probability delta
- logit L2 distance

## Planned setup

- Qwen3-4B-Base
- 36 decoder layers
- 4-bit NF4
- BF16 compute
- RTX 3070
- eager attention
- CFI-Eval-0001-GSM8K
- 10 questions
- 7 train / 3 held out
- 128 max new tokens
- temperature 0.6
- top-p 0.95
- top-k 20
- 64 training positions/question
- bottleneck 128
- maximum update ratio 0.5
- 16 epochs
- learning rate 1e-3
- weight decay 1e-5

Run:

    python scripts/run_exp0009n.py --questions 10 --train-questions 7 --max-new-tokens 128 --epochs 16 --bottleneck 128 --max-train-positions 64 --output "E:\\Titan Forge Industries\\CFI-Data\\Results\\CFI-Eval-0009N-Early-Exit-Generalization"

## Interpretation

A result is interesting only if the learned exit improves on the raw H35 exit on held-out traces and can be compared directly against the whole-L36 skip baseline.

This experiment is still behavioral. The functional skip executes L36 before replacing its output, and the exit head is currently evaluated from captured H35 states. Neither establishes runtime speed.

## Decision rule

- clear held-out improvement over raw H35 exit and competitive behavior with L36 skip -> pursue integrated early exit and runtime measurement
- improvement over raw H35 exit but substantially worse than L36 skip -> investigate richer/conditional transitions
- no held-out improvement -> stop increasing capacity blindly and investigate richer source information or adaptive routing


## Observed Run

The requested 10-question run completed successfully on the established RTX 3070 environment. The model remained frozen; only the 663,168-parameter anchored residual transition was trained.

Training:
- 7 training questions / 3 held out
- 16 epochs
- 64 training positions/question
- bottleneck 128
- training time: 1.83 seconds
- mean loss: 0.19314 -> 0.01348 by epoch 14, followed by an increase to 0.03288 at epoch 16

Held-out behavioral results:

| Metric | Raw H35 exit | Learned adapted exit | Whole-L36 skip |
|---|---:|---:|---:|
| Top-1 agreement | 0.92447917 | **0.95572917** | 0.92447917 |
| Target probability ratio | 0.97526407 | **0.99509468** | 0.97526475 |
| Target log-probability delta | -0.12146130 | **-0.07412996** | -0.12146066 |
| KL to full-model teacher | 0.16002291 | **0.10817067** | 0.16004192 |
| Logit L2 distance | 5169.1022 | **5130.6802** | 5169.1050 |

The learned exit reduced mean KL by approximately 32.4% relative to the raw H35 exit and increased top-1 agreement by approximately 3.13 percentage points on the three held-out traces.

The whole-L36 functional skip matched the raw H35 exit to numerical precision, as expected from the established H35/L36 boundary equivalence.

The generated GSM8K answers were not successfully parsed into final numeric answers in this run (`predicted=None` for all ten baseline generations). Therefore this run is evidence for behavioral/token-distribution preservation, not for an improvement in end-to-end GSM8K question accuracy.

This result supports continued investigation of learned late-layer reconstruction, but it does not establish runtime speedup. The current evaluation still captures H35 after the first 35 layers and applies the learned transition offline; an actual early-exit implementation must avoid executing L36 and measure wall-clock generation throughput under matched conditions.
