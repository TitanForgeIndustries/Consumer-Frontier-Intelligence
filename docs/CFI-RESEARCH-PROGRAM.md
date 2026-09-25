# CFI: Decoupled Intelligence Research Program

Status: CFI v0 smoke and routing controls completed, 2026-09-23. EXP-0010 established independent training and generation but found route collapse. [EXP-0011](experiments/EXP-0011-route-collapse-capacity-control.md) prevented collapse with stronger balancing across three seeds; the sparse model still lost on runtime and allocated memory, and a parameter-matched dense control achieved lower held-out NLL. No original CFI checkpoint has demonstrated general capability or consumer-hardware frontier performance.

## Thesis and boundaries

CFI asks whether **total learned capacity**, **active computation**, **resident memory**, **context state**, and **reasoning depth** can be managed independently. Fewer active parameters can still be slower because routing, memory traffic, kernels, or transfers dominate. The eventual consumer target is an RTX 3070-class 8 GB GPU, 64 GB RAM, desktop CPU, and local storage. The speculative 4T-capacity system remains a hypothesis, not an existing model.

The first milestone is an original CFI-defined model that trains and generates without Qwen weights. The 2–4 week checkpoint target is an engineering milestone, not a frontier-capability forecast. A small smoke checkpoint may precede a well-trained 100M–500M candidate; its text quality must not be overstated.

## What prior experiments establish

The [experiment index](experiments/README.md) preserves negative results. EXP-0001 through EXP-0006 show that naive depth truncation and early exits can lose quality; the 30/36-layer shared-weight draft was technically viable but slower. EXP-0007 and EXP-0008 instrumented and causally ablated late sublayers. EXP-0009 through P studied late-state prediction and exposed a teacher-forced/free-running gap. EXP-0009Q preserved L36 attention/KV and trained only a 658,048-parameter MLP predictor: local KL improved, but free-running agreement was 38/384, with no reliable acceleration. EXP-0009R is a separate, closed three-question scoring control: full, zero-MLP, and predicted-MLP each scored 3/3; fixed-length predicted speed ratio averaged 0.9971x. Neither Q nor R is an independently trained CFI model.

The Workspace Brain and full conversation archive preserve the filing-cabinet metaphor, expert locality, computational paging, shared memory, and adaptive reasoning. These are hypotheses, not CFI measurements. Current code and exact experiment artifacts outrank older chat; the archive's description of Q as active is superseded by Q's completed record and R's committed result.

## Research families and decision gates

| Family | Falsifiable question and control | Status |
|---|---|---|
| Sparse capacity and hierarchical routing | Can total capacity grow while per-token expert computation stays bounded without losing held-out quality? Same-active-width and parameter-matched dense FFNs; report NLL, routed load, wall time, memory | v0 smoke and route controls completed; capacity-efficiency advantage not established |
| Predictive expert locality and paging | Are adjacent routes predictable enough to reduce stalls versus LRU/on-demand? Measure trace transitions, cache hit, transferred bytes, and PCIe/NVMe time | Later; v0 all-resident |
| Explicit knowledge/lookup | Does learned retrieval beat matched-compute dense capacity on held-out facts? Include index size and lookup latency | Later |
| Shared/alternative context | Can exact recent plus compressed or recurrent state replace per-layer KV? Exact-KV long-context control | Later |
| Reconstructable state | When is storage, compression, lookup, reuse, or reconstruction cheapest under a quality constraint? | Open after Q/R |
| Adaptive reasoning/verification | Does budget allocation improve quality per resource over fixed compute? | Later |

External papers and engineering claims are classified in [systems precedents](research/CFI-SYSTEMS-PRECEDENTS.md). They do not automatically transfer to the RTX 3070.

## CFI v0 candidate

Start with a from-scratch causal byte-language model. Every token passes through a shared attention/FFN core; a two-stage learned router chooses one domain group and one expert in that group for an additional residual computation. **Only selected experts execute.** The router and all expert weights remain GPU-resident in v0. This tests *total capacity versus active expert computation*, **not** resident-memory paging, shared context, or general intelligence. Retaining ordinary causal attention avoids confounding the first sparse-capacity test with a new state representation.

Use the same byte corpus, train/validation split, seed, sampled windows, optimizer steps, and decoding for a same-active-width dense control. Record total and estimated per-token active parameters, routed load/switching, held-out byte NLL, real tokens/second, peak GPU memory, and generated examples. A smoke run establishes only execution/training/serialization/generation. Meaningful conclusions require more data, seeds, a parameter-matched control, and profiling. Qwen can remain a future teacher but is not required for CFI v0 inference.

## Research discipline

1. Preserve failures separately from scientific results; restart Python after a CUDA device-side assert.
2. Bound initial jobs; keep corpora, checkpoints, caches, and generated results under a local data root and out of tracked Git files.
3. Do not claim wall-clock or VRAM wins from parameter arithmetic; all-resident sparse experts can cost *more* memory and latency.
4. Before promoting paging, measure whether route locality, transfers, and hit rate actually support it.
5. A tiny-corpus checkpoint is not evidence of general reasoning or frontier intelligence. The 2–4 week milestone can succeed while the broader thesis remains unresolved.
