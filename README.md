# Coding Agent Role Boundary: A Corrected Pilot Study

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

**Author:** Dev ([@Dev10-sys](https://github.com/Dev10-sys))  
**Repository:** [https://github.com/Dev10-sys/coding-agent-role-boundary](https://github.com/Dev10-sys/coding-agent-role-boundary)

---

## 🔬 Research Question

> **Does a coding model internally represent persistent repository instructions as a higher-authority role because of their wording or presentation, even when the actual source is lower-trust workspace content?**

In autonomous coding agents (e.g., Devin, OpenHands, SWE-agent), instructions found in repository files (such as `AGENTS.md`, `.cursorrules`, or `README.md`) are routinely injected into the model's context window. This study investigates whether models represent such repository instructions with system-like authority, or whether prompt injection vulnerabilities in coding agents stem from architectural role confusion versus standard instruction adherence.

---

## 📋 Key Pilot Audit Findings

A rigorous audit of earlier pilot experiments identified critical methodological confounds:

1. **Formatting vs. Semantics Confound:** The pilot's linear role probe achieved 100% accuracy at layer 23 primarily because special chat control tokens (`<|im_start|>`, `<|im_end|>`) were included in the activation representations, trivializing role discrimination without reflecting semantic authority.
2. **Train/Test Data Leakage:** Stratifying samples by role rather than base text allowed the same text passage to appear in both training and test partitions across different role wrappers.
3. **Behavioral Metric Inaccuracy:** The pilot relied on regex keyword matching (`forbidden.txt`, `INJECTED`), counting benign model discussions or explicit refusals as unsafe compliance.
4. **Provenance Warning Salience Artifact:** An elevated unsafe rate under provenance warnings was confounded by repeating trigger keywords in the warning prompt itself.

For the full detailed audit of all 10 identified issues, see [`docs/pilot_audit.md`](docs/pilot_audit.md).

---

## 🛠️ Corrected Experimental Framework

This repository provides an end-to-end, reproducible suite addressing each confound:

| Experiment | Focus | Key Methodology / Controls |
| :--- | :--- | :--- |
| **`01_role_probe.py`** | Role Representation | Text-level disjoint split, 3 pooling strategies (`mean_all`, `content_only`, `last_content`), scramble control |
| **`02_source_ablation.py`** | Source Attribution | Fixed outer wrapper, varying only source metadata to test genuine authority encoding |
| **`03_warning_control.py`** | Warning Salience | 5 matched-length conditions to isolate warning wording from length and keyword echoing |
| **`04_behavior.py`** | Tool Execution Sandbox | Actual filesystem mutation in ephemeral directory, logging tool calls and classifying outcomes |
| **`05_intervention.py`** | Activation Steering | Layer-specific steering along probe direction with orthogonal control (conditional) |

Detailed protocol specifications are documented in [`docs/experiment_design.md`](docs/experiment_design.md).

---

## 📊 Empirical Results

### Experiment 01: Corrected Role Representation Probe

Linear logistic regression probes evaluated on held-out base texts with strict disjoint partitions (96 train texts / 384 samples, 24 test texts / 96 samples; Chance = 25.0%):

| Layer | Depth Fraction | `mean_all` (Seed 42) | `content_only` (No Role Tokens) | `last_content` (Single End Token) | `scramble_control` (`ROLE_A..D`) | `mean_all` (Seed 99) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Layer 5** | ~25% | **96.88%** | **89.58%** | **77.08%** | **77.08%** | **94.79%** |
| **Layer 11** | ~50% | **92.71%** | **84.38%** | **80.21%** | **78.12%** | **92.71%** |
| **Layer 17** | ~75% | **93.75%** | **81.25%** | **70.83%** | **77.08%** | **90.62%** |
| **Layer 23** | ~100% | **93.75%** | **88.54%** | **69.79%** | **87.50%** | **94.79%** |

<p align="center">
  <img src="figures/probe_accuracy.png" alt="Probe Accuracy across Layers" width="700">
</p>

#### Key Mechanistic Takeaways:
1. **Resolution of Pilot 100% Confound:** In the pilot, Layer 23 showed 100.0% accuracy due to base-text data leakage and explicit `<|im_start|>`/`<|im_end|>` boundary tokens. Under strict zero-leakage base-text splitting, baseline accuracy settles at **93.75%**.
2. **Boundary Token vs. Semantic Signal:** Stripping explicit role wrapper tokens (`content_only`) reduces accuracy by 5.2%–12.5% across layers, confirming that boundary tokens directly encode lexical identity. Crucially, accuracy remains high (**81.25%–89.58%**), demonstrating that self-attention successfully context-injects role identity into internal content representations.
3. **Layer Specialization:** On isolated final content tokens (`last_content`), decodability peaks at mid-depth (Layer 11: **80.21%**) before degrading at the final layer (Layer 23: **69.79%**), indicating that representations specialize towards task output rather than role wrappers as depth increases.

---

## 📂 Repository Structure

```text
coding-agent-role-boundary/
├── CITATION.cff             # Citation metadata
├── LICENSE                  # MIT License
├── README.md                # Research overview and reproduction guide
├── pyproject.toml           # Build and dependency configuration
├── .gitignore
├── data/                    # Benchmark text datasets
├── docs/
│   ├── pilot_audit.md       # Audit of pilot flaws and confounds
│   ├── experiment_design.md # Formal experiment specifications
│   └── limitations.md       # Threats to validity and limitations
├── experiments/
│   ├── 01_role_probe.py     # Corrected role probe experiment
│   ├── 02_source_ablation.py# Source attribution probe
│   ├── 03_warning_control.py# Warning salience control experiment
│   ├── 04_behavior.py       # Behavioral sandbox execution
│   └── 05_intervention.py   # Activation steering intervention
├── src/
│   ├── activations.py       # Hidden state extraction & pooling
│   ├── analysis.py          # Statistical tests (Fisher exact, Wilson CIs)
│   ├── behavior.py          # Execution sandbox & tool tracking
│   ├── data.py              # Data partitioning with zero text-leakage
│   ├── model.py             # Model loader (CPU/GPU fallback)
│   ├── probes.py            # Linear probe training & evaluation
│   └── prompts.py           # Controlled prompt templates
└── tests/
    ├── test_probe_pipeline.py
    ├── test_prompt_construction.py
    └── test_result_parsing.py
```

---

## 🚀 Quickstart & Reproducibility

### 1. Environment Setup

```bash
git clone https://github.com/Dev10-sys/coding-agent-role-boundary.git
cd coding-agent-role-boundary

# Install dependencies
pip install -e .
pip install pytest scipy scikit-learn transformers torch
```

### 2. Run Test Suite

Verify prompt structure invariants, zero-leakage split guarantees, and output parser correctness:

```bash
python -m pytest tests/ -v
```

### 3. Run Experiments

```bash
# Experiment 01: Role probe with pooling ablations
python experiments/01_role_probe.py

# Experiment 02: Source ablation representation probe
python experiments/02_source_ablation.py

# Experiment 03: Warning salience ablation
python experiments/03_warning_control.py

# Experiment 04: Behavioral sandbox with write tool
python experiments/04_behavior.py

# Experiment 05: Activation steering (conditional on Exp 02)
python experiments/05_intervention.py
```

---

## 📚 Citation

If you use this codebase or build upon these findings, please cite:

```bibtex
@software{dev2026codingagentroleboundary,
  author = {Dev},
  title = {Coding Agent Role Boundary: A Corrected Pilot Study},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Dev10-sys/coding-agent-role-boundary}}
}
```

See [`CITATION.cff`](CITATION.cff) for full machine-readable metadata.
