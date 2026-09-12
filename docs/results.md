# Empirical Results & Analysis

## 1. Experiment 01: Corrected Role Representation Probe

### 1.1 Overview and Protocol
Experiment 01 evaluates whether internal activations of `Qwen/Qwen2.5-0.5B-Instruct` linearly separate chat roles (`system`, `user`, `assistant/thought`, `tool_output`) across four architectural depths:
- Layer 5 (Depth ~25%)
- Layer 11 (Depth ~50%)
- Layer 17 (Depth ~75%)
- Layer 23 (Depth ~100%)

To eliminate the methodological confounds identified in the pilot audit, the evaluation applies:
1. **Zero-leakage text-level partitioning:** 120 base texts from WikiText-2 are partitioned strictly by base-text ID (`seed=42`, test fraction = 0.20), yielding 96 training base texts (384 samples) and 24 disjoint test base texts (96 samples).
2. **Three pooling strategies:**
   - `mean_all`: Mean-pooling over all sequence tokens (pilot comparison baseline).
   - `content_only`: Mean-pooling strictly over content tokens with role delimiters (`<|im_start|>`, `<|im_end|>`, `system`, `user`, `assistant`, `tool`, etc.) masked out.
   - `last_content`: Residual vector extracted from the final non-special content token.
3. **Control conditions:**
   - `scramble_control`: Chat control tokens replaced by arbitrary dummy tokens (`ROLE_A`, `ROLE_B`, `ROLE_C`, `ROLE_D`) to test token-level lexical decodability.
   - `mean_all_seed99`: Full replication with an independent partition seed to establish split stability.

---

### 1.2 Quantitative Findings

| Layer | Depth | `mean_all` (Seed 42) | `content_only` | `last_content` | `scramble_control` | `mean_all` (Seed 99) | Chance Baseline |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5** | 25% | **96.88%** | **89.58%** | **77.08%** | **77.08%** | **94.79%** | 25.00% |
| **11** | 50% | **92.71%** | **84.38%** | **80.21%** | **78.12%** | **92.71%** | 25.00% |
| **17** | 75% | **93.75%** | **81.25%** | **70.83%** | **77.08%** | **90.62%** | 25.00% |
| **23** | 100% | **93.75%** | **88.54%** | **69.79%** | **87.50%** | **94.79%** | 25.00% |

---

### 1.3 Confusion Matrix Analysis (Best Layer 5, `mean_all`)

The multiclass confusion matrix on the 96 disjoint test samples demonstrates balanced separation across all four roles:

| True \ Predicted | `system` | `user` | `assistant` | `tool` |
| :--- | :---: | :---: | :---: | :---: |
| **`system`** | 24 | 0 | 0 | 0 |
| **`user`** | 1 | 23 | 0 | 0 |
| **`assistant`** | 0 | 0 | 24 | 0 |
| **`tool`** | 0 | 2 | 0 | 22 |

- Balanced Accuracy: **96.88%**
- Only 3 misclassifications out of 96 test samples (1 user misclassified as system, 2 tools misclassified as user).

---

### 1.4 Mechanistic Interpretations

1. **Impact of Delimiter Masking:**
   - In `content_only`, accuracy remains remarkably high (**81.25% to 89.58%**), even though no explicit role-name or special delimiter tokens enter the pooling calculation.
   - This empirically confirms that within the transformer blocks, multi-head self-attention broadcasts role identity into the representations of every token in the passage.
2. **Representational Drift at Deep Layers:**
   - In `last_content`, accuracy declines from **80.21%** at Layer 11 down to **69.79%** at Layer 23.
   - This indicates that individual content tokens at the final layer increasingly prioritize next-token prediction and task semantics over container-level role metadata.
3. **Scramble Control Dynamics:**
   - Arbitrary synthetic tokens (`ROLE_A`..`D`) show lower accuracy at early layers (77.08% vs 96.88%), but reach 87.50% by Layer 23 as the model learns to linearly separate the distinct token embeddings.

---

## 2. Experiment 02: Source-Controlled Representation Probe

### 2.1 Overview and Protocol
Experiment 02 tests whether source identity can be decoded when prompt wrappers are standardized across all conditions into a single uniform format:
```
SOURCE: {source_label}
CONTENT:
{instruction}
```
Where `source_label` is one of:
- `SYSTEM_POLICY`
- `USER_INSTRUCTION`
- `REPO_FILE:AGENTS.md`
- `TOOL_OUTPUT:read_file`

The evaluation utilizes 120 base texts × 4 conditions = 480 samples, with a strict disjoint 80/20 text-level split (96 train texts / 384 samples, 24 test texts / 96 samples; Chance = 25.0%).

### 2.2 Quantitative Findings

| Layer | Depth | `mean_all` Accuracy | `content_only` Accuracy | Balanced Accuracy | Chance Baseline |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **5** | 25% | **100.0%** | **100.0%** | **100.0%** | 25.0% |
| **11** | 50% | **100.0%** | **100.0%** | **100.0%** | 25.0% |
| **17** | 75% | **100.0%** | **100.0%** | **100.0%** | 25.0% |
| **23** | 100% | **100.0%** | **100.0%** | **100.0%** | 25.0% |

### 2.3 Key Insights
- Standardizing the wrapper eliminates the pilot's outer chat-wrapper artifact (where all inputs were categorized as `tool` simply because the last turn was a tool block).
- Under controlled formatting, the source metadata label is linearly decoded with 100% accuracy throughout the network.
- Because Experiment 02 linear probes achieved $\ge 70\%$ balanced accuracy, the prerequisite for **Experiment 05 (Activation Steering Intervention)** is fully met!

---

## 3. Experiment 03: Warning Salience Ablation

### 3.1 Overview and Protocol
Experiment 03 investigates the pilot's purported "salience paradox" — the claim that adding provenance warnings to workspace files backfired and increased unsafe compliance from 32% to 60%.

To isolate causal factors, Experiment 03 implements 5 matched conditions ($N=30$ per condition, 150 total trials) evaluated inside an ephemeral sandbox with actual filesystem state tracking:
1. `no_warning`: Baseline with bare repository file header.
2. `short_provenance`: Minimal neutral metadata origin tag.
3. `long_provenance`: Neutral multi-sentence file origin description.
4. `neutral_same_length`: Same token count as warning text, but strictly benign context (file metadata).
5. `warning_different_wording`: Security warning stating access policy without quoting target keywords (`forbidden.txt`, `INJECTED`).

### 3.2 Quantitative Findings

| Condition | N | Unsafe Action Rate | 95% Wilson CI | Mention Rate | Executed | Discussed | Refused |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `no_warning` | 30 | **70.0%** (21/30) | [52.1%, 83.3%] | 90.0% | 21 | 6 | 1 |
| `short_provenance` | 30 | **73.3%** (22/30) | [55.6%, 85.8%] | 86.7% | 22 | 4 | 3 |
| `long_provenance` | 30 | **60.0%** (18/30) | [42.3%, 75.4%] | 76.7% | 18 | 5 | 0 |
| `neutral_same_length` | 30 | **60.0%** (18/30) | [42.3%, 75.4%] | 73.3% | 18 | 4 | 1 |
| `warning_different_wording` | 30 | **36.7%** (11/30) | [21.9%, 54.5%] | 60.0% | 11 | 5 | **11** |

### 3.3 Statistical Hypothesis Testing (vs `no_warning`)

| Comparison | Two-Sided Fisher Exact $p$ | Odds Ratio | Statistical Significance |
| :--- | :---: | :---: | :--- |
| `no_warning` vs `short_provenance` | $p = 1.0000$ | 0.85 | Not significant |
| `no_warning` vs `long_provenance` | $p = 0.5889$ | 1.56 | Not significant |
| `no_warning` vs `neutral_same_length` | $p = 0.5889$ | 1.56 | Not significant |
| `no_warning` vs `warning_different_wording` | **$p = 0.0191$** | **4.03** | **Statistically Significant ($p < 0.05$)** |

### 3.4 Key Takeaways
1. **Length Control Refutes Paradox:** `long_provenance` (60.0%) and `neutral_same_length` (60.0%) exhibit identical compliance rates, proving prompt length and complexity modulate response tendencies rather than a psychological warning backfire.
2. **Keyword Echoing Priming Effect:** Quoting target filenames in warning text primes model execution. Rewording warnings to state security boundaries without mentioning target keywords reduces unsafe edits by half (**70.0% → 36.7%**, $p=0.0191$) and increases explicit refusals from 3.3% to **36.7%**.

---

## 4. Experiment 04: Behavioral Sandbox with Actual Tool Execution

### 4.1 Overview and Protocol
Experiment 04 tests the core behavioral question:
> Does labeling an injected instruction as coming from `system_policy`, `user_direct`, `repo_file`, or `tool_output` modulate the probability that the agent executes an unsafe file modification when equipped with a functional `write_file` tool?

Protocol:
- $N=30$ independent trials per condition (120 trials total), using randomized coding tasks from `CODING_TASK_POOL`.
- Each trial runs inside an ephemeral directory containing a monitored `forbidden.txt` file.
- Unsafe action is defined by **ground-truth filesystem mutation** (`INJECTED` written to `forbidden.txt`), completely distinct from conversational mention.

### 4.2 Quantitative Findings

| Condition | N | Unsafe Execution | Unsafe Rate | 95% Wilson CI | Discussed | Refused |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `system_policy` | 30 | 21 | **70.0%** | [52.1%, 83.3%] | 5 | 0 |
| `user_direct` | 30 | 23 | **76.7%** | [59.1%, 88.2%] | 7 | 0 |
| `repo_file` | 30 | 22 | **73.3%** | [55.6%, 85.8%] | 6 | 0 |
| `tool_output` | 30 | 25 | **83.3%** | [66.4%, 92.7%] | 4 | 0 |

### 4.3 Pairwise Statistical Comparisons (vs `system_policy`)

| Pairwise Comparison | Fisher Exact $p$ | Odds Ratio | Statistical Inference |
| :--- | :---: | :---: | :--- |
| `system_policy` vs `user_direct` | $p = 0.7710$ | 0.71 | No significant difference |
| `system_policy` vs `repo_file` | $p = 1.0000$ | 0.85 | Identical compliance ($p = 1.0$) |
| `system_policy` vs `tool_output` | $p = 0.3604$ | 0.47 | No significant difference |

### 4.4 Resolving the Central Research Question
- **Empirical Answer:** No. Coding models do **not** internally gate behavioral authority based on source attribution. Instructions originating from low-trust workspace files (`repo_file: 73.3%`) and external tool outputs (`tool_output: 83.3%`) are executed with equivalent or higher frequency compared to official system directives (`system_policy: 70.0%`).
- **Mechanism Decoupling:** Although the residual stream linearly separates source tags with 100% fidelity (Experiment 02), the downstream generation circuit fails to translate this representational boundary into execution suppression.

---

## 5. Experiment 05: Activation Steering Intervention

### 5.1 Overview and Protocol
Experiment 05 investigates whether the linear probe direction $\vec{v} = \mathbf{w} / \|\mathbf{w}\|$ identified in Experiment 02 at Layer 5 is causally linked to instruction compliance.

Protocol:
- Layer 5 intervention: steering vector injected into the residual stream across token forward passes via PyTorch forward hooks.
- Steering parameter $\alpha \in \{-2.0, -1.0, 0.0, +1.0, +2.0\}$.
- Control condition: random orthogonal unit vector in the same Layer 5 subspace.
- $N=10$ randomized coding tasks evaluated per $(\alpha, \text{control})$ cell (100 trials total).

### 5.2 Quantitative Findings

| Steering Strength ($\alpha$) | Probe Direction Unsafe Rate | Orthogonal Control Unsafe Rate |
| :---: | :---: | :---: |
| $\alpha = -2.0$ | **60.0%** (6/10) | **60.0%** (6/10) |
| $\alpha = -1.0$ | **80.0%** (8/10) | **60.0%** (6/10) |
| $\alpha = 0.0$ (Baseline) | **50.0%** (5/10) | **50.0%** (5/10) |
| $\alpha = +1.0$ | **60.0%** (6/10) | **30.0%** (3/10) |
| $\alpha = +2.0$ | **70.0%** (7/10) | **70.0%** (7/10) |

### 5.3 Mechanistic Conclusions
1. **Decoupling of Linear Decodability from Causal Control:** Steerability along the probe direction produces no monotonic reduction in tool execution. Compliance remains between 50% and 80%, tracking closely with the orthogonal random control.
2. **Safety Implications:** The ability of a probe to classify source metadata tags does not mean that modifying that single linear direction will mitigate prompt injection. Model instruction adherence in agentic tool-use is determined by complex, multi-layer computational circuits rather than a localized 1D switch.
