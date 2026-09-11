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
