# EXP-0009K: Direct Behavioral Exit Head

## Research question

Can the model preserve downstream token behavior without reconstructing the internal H36 state?

EXP-0009C-G optimized latent-state replacement. The resulting predictors improved geometric state reconstruction more than they improved behavior. EXP-0009J closes the capture-path control issue. The next test therefore predicts the final token distribution directly from H35.

The hypothesis is that a compact exit head can learn the behaviorally relevant mapping:

\`H35 -> P(next token)\`

without reproducing the internal representation:

\`H35 -> H36 -> final norm -> LM head\`.

## Architecture

A small factorized head:

\`LayerNorm -> Linear(hidden, bottleneck) -> GELU -> Linear(bottleneck, vocab)\`

Default bottleneck: 256.

At Qwen3-4B-Base vocabulary size, this is much smaller than a full hidden-to-vocabulary projection.

Training target:

- full teacher token distribution from the untouched model
- KL divergence from teacher distribution to exit-head distribution

The base model is frozen.

## Evaluation

Held-out positions are compared against:

1. untouched model oracle
2. whole-L36 skip baseline
3. direct behavioral exit head

The direct exit head is evaluated from stored H35 states. The whole-L36 skip is evaluated through a functional L36-output replacement.

## Important limitation

The direct exit-head evaluation itself does not demonstrate runtime speed.

The current model-forward diagnostic still executes L36 before the skip hook. The learned head's parameter count is reported so its future compute cost can be compared with L36, but no timing claim is made.

A future integrated decoder would need to invoke the exit head directly after H35 and avoid both L36 and the normal final-vocabulary projection before measuring generation speed.

## Controlled setup

- Qwen3-4B-Base
- 36 layers
- 4-bit NF4
- BF16 compute
- RTX 3070
- eager attention
- CFI-Eval-0001-GSM8K
- default 4 questions
- 2 train / 2 held out
- 128 max new tokens
- temperature 0.6
- top-p 0.95
- top-k 20
- seed base 42000
- 32 train positions per training question
- 256 bottleneck
- 4 epochs
- learning rate 5e-4
- weight decay 1e-5

Run:

    python scripts/run_exp0009k.py --questions 4 --train-questions 2 --max-new-tokens 128 --epochs 4 --bottleneck 256 --output "E:\\Titan Forge Industries\\CFI-Data\\Results\\CFI-Eval-0009K-Direct-Behavioral-Exit-Head"

## Decision rule

- Exit head materially better than whole-L36 skip -> investigate integrated early-exit generation.
- Exit head close to skip -> direct behavior is not being recovered by this small head; try a richer source representation or conditional routing.
- Exit head worse than skip -> the compact behavioral mapping is insufficient at this capacity/training budget.

Regardless of outcome, this is a diagnostic of the direct-behavior hypothesis, not proof of a consumer-hardware speedup.
