# Roadmap

## Consumer Frontier Intelligence

This roadmap is intentionally research-driven. The project should move from measurable foundations to increasingly ambitious architectural experiments without assuming the final approach is known in advance.

**Current prototype focus:** establish a reproducible, independently trained code-writing model with useful held-out executable-task performance and consumer-local inference. No language, release date, or architecture is guaranteed by the existing tiny-corpus smoke. Record data provenance, model/training cost, quality, latency, and memory before proposing promotion. The phases below describe research directions, not completed capabilities or a requirement to scale the unsuccessful sparse candidate.

## Phase 0: Repository and Research Infrastructure

**Goal:** Make the project reproducible before attempting major model changes.

### Deliverables

- Repository structure
- Apache 2.0 source license
- Research log format
- Experiment configuration format
- Benchmark runner
- Resource measurement tools
- Hardware profile collection
- Reproducible environment setup
- Result storage and comparison tools
- Initial technical documentation

### Exit criteria

A new experiment can be configured, executed, measured, and compared with a baseline using the repository alone.

---

## Phase 1: Consumer Baselines

**Goal:** Establish exactly what can be achieved on the target hardware before introducing novel architecture.

### Tasks

- Select several small and medium open model baselines
- Establish 8 GB VRAM deployment paths
- Measure 4-bit, 8-bit, and higher precision configurations where practical
- Measure CPU/RAM offload
- Benchmark latency and throughput
- Establish a common evaluation suite
- Record memory and compute usage
- Measure quality per resource budget

### Key question

What is the strongest useful baseline we can establish on the target machine under strict resource limits?

### Exit criteria

The project has a reproducible baseline and a resource-normalized evaluation framework.

---

## Phase 2: Dynamic Compute

**Goal:** Test whether a system can avoid unnecessary computation on easy tasks while allocating more computation to difficult tasks.

### Research directions

- Adaptive layer depth
- Early exiting
- Dynamic token computation
- Conditional branches
- Learned compute controllers
- Recurrent refinement
- Compute-budget-aware inference

### Experiments

Compare fixed-compute inference against adaptive-compute inference at matched average resource budgets.

### Exit criteria

At least one adaptive mechanism demonstrates a repeatable improvement in capability per unit of compute, or produces a useful negative result that informs the next architecture.

---

## Phase 3: Sparse and Modular Intelligence

**Goal:** Increase effective capacity without requiring every component to execute on every input.

### Research directions

- Mixture-of-experts
- Expert routing
- Hierarchical routing
- Specialized skill modules
- Shared-weight modules
- Sparse activation
- Module caching

### Key experiments

- Dense versus sparse compute at matched hardware budgets
- Expert specialization quality
- Router stability
- Communication overhead
- Parameter count versus active parameter count

### Exit criteria

The architecture demonstrates whether modularity provides a measurable advantage once real routing and memory costs are included.

---

## Phase 4: Persistent Memory

**Goal:** Move useful information and learned experience outside the core parameter set when doing so is more efficient.

### Research directions

- Working memory
- Episodic memory
- Semantic retrieval
- Procedural memory
- Memory consolidation
- Memory correction
- Memory decay and forgetting
- Skill storage

### Experiments

Measure performance on tasks requiring long-term continuity, repeated learning, and reuse of previous solutions.

### Exit criteria

Memory produces measurable capability gains without introducing unacceptable retrieval or latency overhead.

---

## Phase 5: Reasoning, Search, and Verification

**Goal:** Increase reasoning quality through selective inference-time computation.

### Research directions

- Structured decomposition
- Candidate generation
- Search trees
- Planning
- Verifiers
- Critic modules
- Self-consistency
- Selective rechecking
- Tool execution

### Key principle

Do not spend maximum reasoning compute on every task. The system should learn when additional computation is worth paying for.

### Exit criteria

The system can allocate extra reasoning to difficult problems while preserving low cost on easy problems.

---

## Phase 6: Learning from Experience

**Goal:** Allow the system to improve from interaction and task history without repeatedly retraining the entire model.

### Research directions

- Continual learning
- Fast adaptation
- Lightweight parameter updates
- Fast weights
- Skill acquisition
- Synthetic curriculum generation
- Self-generated evaluation tasks
- Automatic failure replay

### Key questions

- Can the system permanently acquire useful skills?
- Can it improve without catastrophic forgetting?
- Can learning be limited to small modules rather than the entire model?

### Exit criteria

The system demonstrates repeatable skill acquisition over time while retaining previous capabilities.

---

## Phase 7: Heterogeneous Cognitive Architecture

**Goal:** Combine the strongest mechanisms into one coordinated system.

### Possible architecture

```text
                   ┌──────────────────┐
                   │   Input / Task   │
                   └────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │   Task Router     │
                  └───────┬─┬─┬───────┘
                          │ │ │
            ┌─────────────┘ │ └──────────────┐
            ▼               ▼                ▼
       Core Model        Memory         Skill Modules
            │               │                │
            └───────────────┼────────────────┘
                            ▼
                    Reasoning / Search
                            │
                            ▼
                        Verifier
                            │
                            ▼
                       Final Output
```

The exact architecture is expected to change as experiments eliminate weaker approaches.

### Exit criteria

The combined system outperforms its individual components under the same resource budget on multiple unrelated task categories.

---

## Phase 8: Automated Architecture Research

**Goal:** Reduce dependence on manually designed architecture choices.

### Research directions

- Neural architecture search
- Module evolution
- Learned routing policies
- Automated ablation
- Hyperparameter search
- Synthetic benchmark generation
- Self-generated experiments
- Compute-aware architecture optimization

### Constraint

Automated search must optimize useful capability under a real hardware budget, not benchmark score alone.

### Exit criteria

The research system can discover at least one configuration that humans did not manually specify and that provides a measurable improvement.

---

## Phase 9: Frontier Capability Testing

**Goal:** Determine whether the complete architecture can approach frontier-class capability under consumer-hardware constraints.

### Evaluation

Test across broad capability categories, including:

- Reasoning
- Coding
- Mathematics
- General knowledge
- Long-context tasks
- Planning
- Instruction following
- Structured output
- Tool use
- Learning from examples
- Novel task adaptation

Compare against substantially larger baselines while reporting the full resource budget.

### Required reporting

- Model parameters
- Active parameters
- VRAM
- RAM
- Compute
- Latency
- Throughput
- Energy where measurable
- External tool cost
- Retrieval cost
- Search cost
- Benchmark performance

### Exit criteria

A clear empirical answer exists about how far the architecture can push capability under consumer constraints.

---

## Phase 10: Consumer Deployment

**Goal:** Make the resulting system practical for ordinary users rather than only researchers.

### Targets

- Simple local installation
- Consumer GPU support
- CPU fallback
- Persistent local memory
- Offline core functionality
- Optional cloud acceleration
- Model and memory portability
- Efficient updates
- Transparent resource controls

### Exit criteria

A non-specialist can install and operate the system on supported consumer hardware without requiring a permanent cloud subscription.

---

# Ongoing Workstreams

These tracks continue across phases rather than being isolated milestones.

## Benchmarking

Continuously improve evaluations so improvements cannot hide behind a narrow benchmark.

## Systems Optimization

Profile memory bandwidth, CPU overhead, storage, kernel launch overhead, data movement, and other real bottlenecks.

## Research Reproduction

Reproduce important external findings before incorporating them into the main architecture.

## Documentation

Record successful, failed, and inconclusive experiments.

## Open Research

Publish useful findings, benchmarks, tools, and negative results where practical so the project can benefit from external replication and criticism.

# Decision Gates

Major architecture changes should pass through these questions:

1. Does the change improve capability, efficiency, or both?
2. Was the comparison made against an appropriate baseline?
3. Was total system cost measured rather than one bottleneck alone?
4. Does the improvement generalize beyond one benchmark?
5. Can the result be reproduced?
6. Does the change help the long-term consumer-hardware objective?

A technically impressive optimization that does not improve the overall research objective should remain documented but should not automatically become part of the main architecture.

# Ultimate Objective

Build and experimentally validate a general AI architecture that can deliver frontier-class capabilities while operating within practical consumer hardware constraints.

The roadmap is successful only if it produces evidence. The implementation is expected to change as the evidence changes.
