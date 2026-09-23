# CFI Systems and Architecture Precedents

Reviewed 2026-09-23 against linked primary papers, proceedings, and official implementations. Paper abstracts and project READMEs are **author-reported**, not CFI replications. Fields not established from a source are unknown; model parameter counts cannot be converted into a VRAM or throughput claim without runtime, precision, context, and hardware details.

| Source / evidence type | What the source establishes or reports | Relevance and limit |
|---|---|---|
| [DeepSeekMoE](https://arxiv.org/abs/2401.06066), author preprint | Fine-grained routed experts plus shared experts; reports 2B/16B/145B studies and compute-matched comparisons. | Shared core and expert specialization are precedents, not a 3070 benchmark. |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437), author technical report | 671B total, 37B activated per token, MLA, MoE, multi-token prediction; H800 training. | Separate total versus active counts; 37B active does **not** mean 37B total memory or practical consumer inference. |
| [DeepSeek-V2 / MLA](https://arxiv.org/abs/2405.04434), author preprint | 236B total, 21B activated; compressed KV latent and 128K-context design. | Future CFI context-state experiments, not v0. |
| [Engram](https://arxiv.org/abs/2601.07372) and [official code](https://github.com/deepseek-ai/Engram), preprint/project | Conditional lookup memory separates static pattern storage from computation. | Quality, IO, footprint, and latency need matched consumer measurement. |
| [Memory Layers at Scale](https://arxiv.org/abs/2412.09764), author preprint | Trainable key/value lookup memory; study reports up to **128B memory parameters** with up to 8B base models. | **Not** evidence of a 128B model generating on a 12 GB GPU. |
| [Mixture of Lookup Experts](https://proceedings.mlr.press/v267/jie25b.html), peer-reviewed ICML 2025 | Reparameterizes trained experts into lookup structures for inference. | Compute-versus-storage option dependent on representation and lookup bandwidth. |
| [Oracle-MoE](https://proceedings.mlr.press/v267/zhou25b.html), peer-reviewed ICML 2025 | Attention-derived locality-preserving routing; reports GPT-2-scale 200M–2B tests. | Measure switching before prefetch; no guaranteed CFI locality. |
| [FloE](https://proceedings.mlr.press/v267/zhou25j.html), peer-reviewed ICML 2025 | On-the-fly expert execution targets PCIe bottlenecks on memory-constrained GPUs. | Runtime depends on model, quantization, memory budget, hardware. |
| [ExpertFlow](https://arxiv.org/abs/2410.17954), author preprint | Predictive expert caching/token scheduling for offloaded MoE. | Compare predicted and oracle hit rates, bytes, stalls. |
| [Colibrì](https://github.com/JustVugg/colibri), evolving project README/code | Engineering implementation of memory-tier expert streaming. | Ability to load is not practical generation; CFI has not reproduced throughput or GPU/RAM/NVMe/KV footprint. |
| [Mamba](https://arxiv.org/abs/2312.00752) and [Samba](https://arxiv.org/abs/2406.07522), author preprints | Selective recurrent state and hybrid sliding-window attention. | Alternatives to unbounded KV need recall/quality and runtime controls. |
| [EAGLE](https://arxiv.org/abs/2401.15077), author preprint | Feature-level speculative drafting with target verification. | EXP-0006 does not refute all latent speculation. |
| [FlashAttention](https://arxiv.org/abs/2205.14135), paper, and [FlashInfer](https://github.com/flashinfer-ai/flashinfer), official project | IO-aware attention algorithms and serving kernels. | Benchmark actual Windows/3070 runtime, not theoretical FLOPs. |
| [TinyStories](https://arxiv.org/abs/2305.07759), author paper | Small models can generate on a deliberately simplified story distribution with suitable training. | Tiny-corpus output is not general intelligence. |

## Unverified or pending

- The archived **"128B model on 12 GB GPU"** lead has no confidently identified model, quantization, GPU/RAM/NVMe, context, generation throughput, or reproducible run. The 128B *memory-layer* result above does not establish it. Remains unresolved.
- Archived **DeepSeek-V4/V4.1** descriptions and Colibrì numerical throughput were not promoted to fact. Verify a release-dated official source and pinned runtime setup first.
- PyramidKV, H2O, Quest, KIVI, Kimi Linear, Infini-attention, Titans, KTransformers, llama.cpp, D3, QMoE, AQLM, SpQR, AWQ, BitNet, T-MAC, QServe, and cross-layer/quantized KV remain a review queue; this prioritized survey does not imply complete review or replication.

For each future benchmark record source and peer-review status, total/active parameters, architecture, precision, GPU/VRAM, CPU/RAM/NVMe, context and KV, batch, prefill/decode protocol, throughput and latency, quality impact, and exact control. Unknown is preferable to extrapolation. CFI measurements belong in separately indexed experiment records.
