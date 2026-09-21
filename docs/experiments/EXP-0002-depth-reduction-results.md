# EXP-0002 Results: Dynamic Compute Feasibility via Depth Reduction

Initial 5-question sweep of fixed Qwen3-4B layer prefixes:

| Depth | Accuracy | Avg time/question | Throughput | Peak VRAM |
|---:|---:|---:|---:|---:|
| 12/36 | 0/5 | 16.87 s | 30.35 tok/s | 2.56 GiB |
| 18/36 | 0/5 | 20.90 s | 19.44 tok/s | 2.57 GiB |
| 24/36 | 0/5 | 4.67 s | 14.91 tok/s | 2.57 GiB |
| 30/36 | 0/5 | 22.86 s | 12.00 tok/s | 2.60 GiB |
| 36/36 | 4/5 | 19.99 s | 10.09 tok/s | 2.58 GiB |

The reduced-depth variants are substantially faster, but none produced a scored final answer under the initial evaluator. This is a fixed-prefix truncation feasibility result, not a trained early-exit result.

Do not scale this exact sweep to 100 questions until the saved outputs are rescored. If flexible rescoring also shows loss of usable answers, close fixed truncation as a negative result and redirect toward trained conditional computation.