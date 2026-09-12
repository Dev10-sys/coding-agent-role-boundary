# Limitations & Threat Model

## 1. Scope and Threats to Validity

This repository documents an empirical investigation into internal representations and behavioral compliance of instruction-tuned language models acting in software development tool loops. The following constraints and threats to validity should be considered when interpreting findings:

### 1.1 Model Scale and Architecture
- **Target Model:** Primary evaluations utilize `Qwen/Qwen2.5-0.5B-Instruct`. While lightweight and practical for controlled mechanistic analysis, small models may exhibit different activation geometry and instruction adherence dynamics compared to frontier coding models (e.g., Qwen-2.5-72B, Claude 3.5 Sonnet, GPT-4o).
- **Architecture Specifics:** Qwen2.5 utilizes specific chat template control tokens (`<|im_start|>`, `<|im_end|>`). Findings regarding token-level role boundaries must be validated across other chat formatting standards (e.g., Llama-3 headers, ChatML).

### 1.2 Probe Representation vs. Causal Mechanism
- **Linear Decodability is Not Causality:** A linear probe achieving high accuracy demonstrates that role or source-label information is linearly accessible in the residual stream. It does not prove that downstream generation circuits rely on this direction to make behavioral authorization decisions.
- **Probe Generalization and Token Features:** Probes trained on standardized formatting (`SOURCE: ...`) decode explicit textual metadata labels rather than an inferred provenance feature. In Experiment 02, `content_only` pooling excludes ChatML role delimiters but retains the literal metadata tokens in the text body.
- **Intervention Limitations:** In Experiment 05, steering along the 1D probe direction was conducted at an exploratory scale ($N=10$ per cell). The lack of monotonic steering effect indicates that the 1D linear readout is not a simple control dial for compliance, but does not rule out distributed or multi-layer representations.

### 1.3 Prompt Realism & Provenance Abstraction
- **Abstract Source Metadata:** The fixed `SOURCE: {label}\nCONTENT:\n{instruction}` wrapper isolates source metadata under controlled formatting, but does not replicate actual agent framework provenance mechanisms. Real-world systems often separate untrusted content through distinct API message channels, system prompts, or multi-turn tool message streams. Generalization from text-level source labeling to native agent provenance mechanisms remains unestablished.

### 1.4 Tool Realism & Agentic Scope
- **Narrow Tool Abstraction:** The evaluation sandbox implements a single focused action: calling a `write_file` tool to mutate a targeted file (`forbidden.txt`). Real autonomous coding agents (e.g., Devin, OpenHands, SWE-agent) operate with multi-step workflows, interactive bash shells, linting feedback, and complex planning horizons. Generalization to general agentic tool-use environments requires broader benchmark validation.

### 1.5 Sample Sizes and Statistical Power
- **Behavioral Sample Sizes:** Behavioral experiments (Experiments 03 and 04) are conducted at $N=30$ per condition. While sufficient to identify large effects with Fisher's exact tests ($p < 0.05$, as in the keyword salience ablation), subtler interaction effects require larger sample regimes ($N \ge 200$).
- **Null Hypothesis Interpretation:** In Experiment 04, the comparison between `system_policy` (70.0%) and `repo_file` (73.3%) yielded $p = 1.00$. Statistically, this represents a failure to reject the null hypothesis in this sample, which does not constitute proof of equivalence.

### 1.6 Compute Constraints
- **Hardware:** CPU-only execution limits activation extraction throughput. Probes and interventions are restricted to representative architectural depths (layers 5, 11, 17, 23 of 24) rather than exhaustive every-layer extraction.
