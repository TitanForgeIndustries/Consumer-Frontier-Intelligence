# EXP-0004: Layerwise Logit Probe

## Research question
Does Qwen3-4B contain useful next-token predictive information at intermediate depths even when autoregressive early-exit generation fails?

## Method
Use the fixed 100-question CFI GSM8K set. Run the full frozen Qwen3-4B-Base once per example with hidden states enabled. At layers 12, 18, 24, 30, and 36, apply the original final RMSNorm and tied LM head to the hidden state and score the known GSM8K answer tokens under teacher forcing.

## Metrics
- Token-level accuracy
- Cross-entropy on answer tokens
- Agreement with the final layer's top-1 predictions

## Why this comes next
EXP-0002, EXP-0003A, and EXP-0003B all showed that naive or lightly adapted autoregressive exits at 30/36 layers fail. This probe determines whether the missing capability is already present in the intermediate representation but difficult to exploit autoregressively, or whether the predictive representation itself is substantially weaker.

## Decision rule
- If layer 30 retains substantial token-level predictive signal and high agreement with layer 36, investigate a token-level confidence controller and properly trained exit objective.
- If layer 30 is substantially weaker than layer 36, investigate trained shallow students or layer-specific distillation rather than another post-layer adapter.

No weights are modified. No generation is performed.
