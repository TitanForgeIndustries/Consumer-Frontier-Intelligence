# Research Program

## Consumer Frontier Intelligence

This document defines the research direction, hypotheses, experimental standards, and technical questions for Consumer Frontier Intelligence.

The project investigates whether frontier-level AI capability can be achieved with dramatically lower compute and memory requirements than current large-model approaches by changing how intelligence is represented, activated, learned, remembered, and computed.

The goal is not to make a smaller copy of an existing frontier model. The goal is to investigate whether a fundamentally different system can produce comparable capabilities while remaining practical on ordinary consumer hardware.

**Near-term test:** an original, independently trained model that writes working code on local consumer hardware. Its coding ability must be evaluated on held-out executable tasks and weighed against measured memory, latency, and training cost. The EXP-0010/0011 byte-model controls do not yet demonstrate this ability or a sparse-efficiency advantage; the long-term frontier target remains open.

## 1. Core Research Question

**Can a highly capable general-purpose AI system approach frontier-model capability while operating within the compute, memory, power, and cost constraints of consumer hardware?**

The primary hardware reference for this project is a system comparable to:

- NVIDIA RTX 3070 class GPU with 8 GB VRAM
- 64 GB system RAM
- Consumer desktop CPU
- Consumer-grade storage
- No dependence on a permanent cloud inference service for core operation

The hardware is a constraint, not the definition of the architecture. A successful approach should explain why it remains efficient as capability increases.

## 2. Research Thesis

Current large AI systems place a substantial amount of intelligence into large collections of learned parameters and then activate a relatively similar computational structure for many tasks.

This project investigates a different possibility:

> **High capability may come less from making one static neural network larger and more from making a system dynamically decide what to compute, what to remember, what to retrieve, what to verify, and what to learn.**

This suggests that intelligence could be distributed across multiple interacting mechanisms rather than represented primarily by a monolithic parameter set.

Potential components include:

- Sparse and conditional computation
- Dynamic routing
- Mixture-of-experts architectures
- Specialized modules and reusable skills
- Adaptive computation depth
- Weight sharing and recurrent computation
- Persistent external memory
- Episodic and procedural memory
- Fast weights or test-time adaptation
- Retrieval systems
- Neural and symbolic computation together
- Tool-assisted reasoning
- Search and planning during inference
- Verification and self-correction
- Synthetic curriculum generation
- Continual learning
- Automated architecture search

These are research directions, not assumptions that any individual technique will produce frontier-level capability.

## 3. What This Project Is Not

This project is not based on the assumption that simply quantizing an existing frontier model will make it equivalent on consumer hardware.

It is also not limited to building a useful assistant from an existing open model.

Compression, quantization, distillation, offloading, and inference optimization are valuable tools, but they are treated as enabling techniques rather than the central scientific claim.

The central question is whether **architectural and algorithmic efficiency can change the amount of compute required to produce a given level of intelligence.**

## 4. Working Hypotheses

### H1. Conditional computation can increase capability per unit of compute

A model that activates only the modules relevant to a task may achieve higher effective capacity without paying the full cost of evaluating the entire network every time.

Questions:

- How much computation can be skipped without reducing quality?
- Can routing become task-aware at increasingly fine granularity?
- Can experts specialize without becoming brittle?
- Can unused capacity remain dormant without losing useful knowledge?

### H2. Intelligence can be represented across multiple computational timescales

A single static forward pass may not be the most efficient representation of general intelligence.

A system could combine:

- Fast local inference
- Short-term working state
- Persistent episodic memory
- Learned procedures
- Slow background learning
- On-demand search

Questions:

- Which information belongs in parameters versus memory?
- Can persistent experience reduce repeated computation?
- Can the system accumulate useful skills without continuously retraining the full model?

### H3. Adaptive computation can replace unnecessary work

Different problems require different amounts of reasoning.

The system should be able to decide when to stop, when to think longer, and when to invoke specialized computation.

Questions:

- Can a learned controller allocate compute dynamically?
- Can easy tasks terminate early?
- Can difficult tasks receive additional reasoning steps?
- Can the system estimate when its answer requires verification?

### H4. Recurrent or shared-weight systems may provide depth without proportional parameter growth

Instead of storing every layer independently, a system could repeatedly apply shared computation with changing internal state.

Questions:

- How much useful depth can be obtained through recurrence?
- When does recurrence outperform simply adding layers?
- Can state carry useful intermediate abstractions across repeated passes?

### H5. External memory can reduce the need to encode everything in weights

A model does not necessarily need to memorize every useful fact or prior interaction inside its parameters.

Questions:

- Can a small core model plus high-quality memory approach the usefulness of a much larger static model?
- What memory representations are most efficient?
- How should memory be written, consolidated, corrected, and forgotten?
- Can procedural memory store reusable solution strategies rather than only text?

### H6. Verification may be cheaper than producing a perfect first-pass answer

A system may become more reliable by generating candidate solutions and selectively verifying them rather than using maximal compute on every generation.

Questions:

- Can specialized verifiers detect reasoning errors cheaply?
- Can the system choose when verification is worth the cost?
- Can multiple weak processes combine into a stronger verified result?

### H7. Search and planning can substitute for some parameter scale

For certain tasks, additional inference-time search may provide useful capability without permanently increasing model size.

Questions:

- When does search outperform additional model parameters?
- Can the system selectively search only ambiguous branches?
- Can reusable search results become long-term memory?

### H8. A heterogeneous architecture may be more efficient than a single universal network

General intelligence may be better implemented as a coordinated system containing multiple specialized mechanisms.

Possible subsystems include:

- Language reasoning
- Vision or multimodal processing
- Retrieval
- Memory
- Planning
- Verification
- Tool execution
- Code execution
- Structured reasoning
- Skill modules

The key research question is whether coordination overhead remains lower than the compute required by a monolithic model with equivalent capability.

## 5. Research Areas

### Architecture

Investigate sparse, modular, recurrent, hierarchical, and dynamically routed architectures.

### Memory

Investigate short-term state, episodic memory, semantic memory, procedural memory, retrieval, consolidation, and forgetting.

### Reasoning

Investigate adaptive depth, test-time computation, search, planning, decomposition, verification, and self-correction.

### Learning

Investigate continual learning, synthetic curricula, self-generated tasks, fast adaptation, and selective parameter updates.

### Efficiency

Investigate quantization, sparsity, caching, parameter sharing, CPU/RAM offload, memory locality, and low-cost inference.

### System Design

Investigate how multiple computational mechanisms can operate as one coherent model without creating excessive routing and orchestration overhead.

### Evaluation

Develop benchmarks that measure capability relative to actual compute and memory expenditure rather than model size alone.

## 6. The Central Metric: Capability per Resource

Traditional benchmarks often report capability without measuring the full cost required to obtain it.

This project will track at least:

- Task accuracy or quality
- Tokens per second
- Latency
- Peak VRAM
- System RAM usage
- GPU utilization
- CPU utilization
- Energy where measurable
- Total inference compute
- Training compute where measurable
- Cost per solved task
- Recovery or verification cost

The goal is not simply to maximize benchmark scores.

The goal is to maximize **useful capability per unit of resource.**

## 7. Baselines

Every major experiment should compare against one or more clear baselines.

Examples:

- Dense transformer baseline
- Same parameter count without sparsity
- Same model with and without adaptive computation
- Same model with and without persistent memory
- Same capability target at different quantization levels
- Local system versus cloud-assisted system where relevant

A claimed improvement should identify what changed and what resource budget was held constant.

## 8. Experimental Principles

### Measure, do not assume

A technique should be judged by controlled measurements rather than reputation or intuition.

### Hardware is part of the experiment

The target machine is not merely a deployment environment. Memory limits, bandwidth, thermals, and latency are architectural constraints.

### Separate capability from scaffolding

Results should make clear whether improvement came from the base model, memory, retrieval, tools, search, verification, or some combination.

### Reproduce before extending

A result should be repeatable before it becomes a dependency for later experiments.

### Failures are results

Negative findings should be recorded because they eliminate unproductive directions and improve future architecture decisions.

### Avoid benchmark overfitting

A system should not be considered substantially more capable because it improves on a narrow benchmark while degrading elsewhere.

### Track total cost

A method that saves GPU memory but requires extreme CPU computation or massive retrieval overhead is not automatically more efficient.

## 9. Experiment Structure

Each experiment should record:

```text
Experiment ID:
Date:
Hypothesis:
Change:
Baseline:
Hardware:
Software:
Dataset / Benchmark:
Training Budget:
Inference Budget:
Metrics:
Results:
Failure Modes:
Interpretation:
Next Experiment:
```

Where possible, experiments should include deterministic seeds, configuration files, versioned datasets, and machine-readable result output.

## 10. Breakthrough Criteria

A meaningful breakthrough would require more than a smaller model or faster inference.

Examples of evidence that would matter:

- A smaller system matching a substantially larger baseline at the same resource budget
- A system gaining capability primarily by dynamically allocating computation rather than increasing parameters
- Persistent memory producing large capability gains without proportionally increasing model size
- Stronger reasoning achieved through selective search and verification at acceptable cost
- Continual learning that adds skills without catastrophic degradation
- Architecture changes that produce repeated gains across unrelated tasks
- A general system that maintains high capability while fitting practical consumer-hardware constraints

The ultimate target is a system whose capability is competitive with frontier-class AI while its core inference can remain practical on consumer hardware.

That target is a research objective, not a current claim.

## 11. Major Risks

### Routing collapse

Dynamic expert systems may learn to overuse a small subset of modules.

### Memory pollution

Poor memories can reinforce incorrect or low-value information.

### Continual-learning instability

Incremental updates may damage previously learned capabilities.

### Verification loops

Adding verification or search can improve quality while making the total system too expensive.

### Coordination overhead

A highly modular system can become slower than a simpler dense model if communication costs dominate.

### Benchmark illusion

Improvements may be specific to the selected evaluation suite rather than general capability.

### Hardware bottlenecks

VRAM capacity may not be the limiting factor. Memory bandwidth, system RAM bandwidth, CPU latency, storage, or inter-process communication may dominate.

## 12. Research Sequence

The research should progress from measurable efficiency improvements toward increasingly general systems.

1. Establish strong low-cost baselines.
2. Measure parameter, memory, and compute efficiency independently.
3. Add one efficiency mechanism at a time.
4. Combine mechanisms only after their isolated effects are understood.
5. Introduce persistent memory and adaptive computation.
6. Introduce planning, verification, and search.
7. Introduce continual learning and automated curriculum generation.
8. Explore architecture evolution and self-improvement.
9. Evaluate the complete system against larger and more expensive baselines.

## 13. Long-Term Question

The deepest question in this project is not simply how to shrink a large language model.

It is:

> **What is the minimum computation required to produce general intelligence, and does modern AI leave a large amount of that computation unused or inefficiently organized?**

Consumer hardware provides a concrete limit that forces this question to be answered experimentally.

