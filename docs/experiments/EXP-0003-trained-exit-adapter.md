# EXP-0003: Trained Intermediate Exit Adapter

## Research question

Can a pretrained decoder-only transformer be made useful at an intermediate
depth by training only a lightweight exit adapter, allowing the model to stop
before its full depth?

## Hypothesis

The failure of naive fixed-depth truncation in EXP-0002 may be caused by the
intermediate representation not being aligned with the final language-model
head. A small trainable adapter placed after layer 30 may recover useful
next-token prediction while keeping the first 30 layers of the backbone
frozen.

## Controlled variables

- Qwen3-4B-Base
- 4-bit NF4 quantization
- BF16 compute
- RTX 3070 8 GB
- fixed CFI-Eval-0001-GSM8K evaluation set
- same prompt, sampling, seeds, and scoring protocol as EXP-0001
- same final normalization and tied LM head
- frozen transformer backbone

## Independent variable

A trainable intermediate exit adapter after layer 30.

The adapter is a small residual bottleneck MLP:

hidden -> 256 -> SiLU -> 256 -> hidden

Only the adapter parameters are trained. The 36-layer backbone is not updated.

## Training protocol

Initial feasibility configuration:

- GSM8K training split
- deterministic 512-example subset
- maximum sequence length: 512
- 300 optimizer steps
- AdamW
- learning rate: 2e-4
- gradient accumulation: 8
- loss only on the target answer/reasoning tokens

The adapter is attached after layer 30 and before the model's final RMSNorm and
language-model head.

## Evaluation

First validate on 5 questions. If the 30-layer trained exit produces usable
answers, run the full fixed 100-question CFI evaluation.

Compare:

- 36-layer full model
- 30-layer trained exit

Measure:

- GSM8K accuracy
- generation latency
- generated tokens
- tokens/second
- peak VRAM
- hardware telemetry
- executed depth

## Success criterion

EXP-0003 is interesting if the trained 30-layer exit recovers meaningful
capability while reducing generation cost compared with the 36-layer reference.

A faster model with unusable accuracy is not considered a successful
capability-efficiency result.

## Scientific context

Early-exit work on decoder-only transformers has shown that naive early exit
can be difficult and that training/alignment of intermediate representations
matters. This experiment therefore tests a lightweight trained exit rather than
simply deleting the final layers.

## Next experiment

If layer 30 recovers useful capability, test a second trained exit at layer 24
or add a learned controller that chooses between 24, 30, and 36 layers per
sequence.
