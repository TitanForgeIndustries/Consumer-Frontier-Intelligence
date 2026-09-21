# EXP-0003 Results

## Result

EXP-0003A trained a residual bottleneck adapter after layer 30 for 300 steps
using GSM8K task loss.

5-question gate:

- Accuracy: 0/5
- 2/5 generations hit the 512-token limit without a parsed final answer
- Remaining scored outputs were incorrect
- Average generation time: 25.58 s/question
- Throughput: 14.30 tok/s
- Peak VRAM: approximately 2.60 GiB

EXP-0003B trained a teacher-aligned layer-30 adapter for 100 steps, matching
layer-30 representations to the full layer-36 representation while also using
an auxiliary LM loss.

5-question gate:

- Accuracy: 0/5
- 4/5 generations hit the 512-token limit or failed to produce a parseable
  answer
- 1 scored output was incorrect
- Average generation time: 31.60 s/question
- Throughput: 13.29 tok/s
- Peak VRAM: approximately 2.60 GiB

## Conclusion

Neither a direct GSM8K-trained exit adapter nor the teacher-aligned adapter
produced usable 30-layer autoregressive reasoning in the initial feasibility
gates.

The experiments demonstrate that reducing execution from 36 to 30 layers can
increase generation throughput, but the tested post-layer adapters were not
sufficient to preserve task capability.

This does not establish that 30-layer early exit is impossible. It establishes
that the tested lightweight adapters and training objectives were insufficient.

EXP-0003 is therefore closed as a negative feasibility result.

## Next direction

EXP-0004 measures intermediate-layer next-token information directly using
teacher-forced sequences. This separates a representation problem from an
autoregressive exposure problem and avoids another expensive training cycle
until we know whether intermediate layers contain useful predictive signal.
