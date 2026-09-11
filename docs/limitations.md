# Limitations & Threat Model

## 1. Scope and Threats to Validity

This repository documents an empirical investigation into internal representations and behavioral compliance of instruction-tuned language models acting in software development tool loops. The following constraints and threats to validity should be considered when interpreting findings:

### 1.1 Model Scale and Architecture
- **Target Model:** Primary evaluations utilize `Qwen/Qwen2.5-0.5B-Instruct`. While lightweight and fast for mechanistic analysis, small models may exhibit different activation geometry and instruction adherence dynamics compared to frontier models (e.g., Qwen-2.5-72B, Claude 3.5 Sonnet, GPT-4o).
- **Architecture Specifics:** Qwen2.5 utilizes specific chat template control tokens (`<|im_start|>`, `<|im_end|>`). Findings regarding token-level role boundaries must be validated across other chat formatting standards (e.g., Llama-3 headers, ChatML).

### 1.2 Probe Representation vs. Causal Mechanism
- **Linear Decodability is Not Causality:** A linear probe achieving high accuracy demonstrates that role information is linearly accessible in the residual stream. It does not prove the model downstream computation actually relies on this direction to make behavioral decisions.
- **Probe Generalization:** Probes trained on standard role wrappers may pick up low-level token features rather than abstract authority semantics unless strict out-of-distribution and source-controlled tests are applied (as addressed in Experiment 02).

### 1.3 Behavioral Sandbox Bounds
- **Simulated File Tool:** The agent environment implements a controlled `write_file` tool call abstraction. Complex agentic scenarios involving multi-step bash execution, linting feedback, or iterative git actions are out of scope for this focused pilot.
- **Sample Sizes:** Behavioral runs are conducted at $N=30$ per condition. While suitable for identifying primary effects with Fisher's exact tests ($p < 0.05$), subtler interaction effects require larger sample regimes ($N \ge 200$).

### 1.4 Compute Constraints
- **Hardware:** CPU-only execution limits the throughput of activation caching and batch inference. Layer probes are restricted to selected representative depths (layers 5, 11, 17, 23 of 24) rather than exhaustive every-layer extraction.
