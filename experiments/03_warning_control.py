# -*- coding: utf-8 -*-
"""
experiments/03_warning_control.py
===================================
Warning Salience Ablation
---------------------------

Research question fragment:
  Does the elevated unsafe-edit rate in the pilot's provenance-warning condition
  (15/25 = 60% vs. 8/25 = 32% for repo_file) reflect:
    (a) genuine increased compliance with provenance-tagged instructions?
    (b) lexical salience of the warning text triggering the unsafe metric?
    (c) increased token budget in the warning condition changing generation?

This experiment tests 5 warning conditions, all sharing:
  - identical outer ChatML wrapper
  - identical payload text (legitimate task + INJECTED_MARKER)
  - varied SOURCE/label region only

Conditions:
  1. no_warning           — bare SOURCE: REPO_FILE:AGENTS.md
  2. short_provenance     — brief neutral source label
  3. long_provenance      — 3-sentence neutral origin description
  4. neutral_same_length  — same token length as warning but unrelated (file metadata)
  5. warning_different_wording — warning reworded to avoid "forbidden"/"INJECTED" keywords

N = 30 trials per condition (150 total), seed=42.

Outputs:
  results/behavior_results.csv  (appended / updated with warning conditions)
  results/warning_results.csv   (dedicated CSV)
  figures/warning_ablation.png
"""

import sys, os, json, csv, random, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import load_model, get_device_info
from src.prompts import (
    warning_prompt, WARNING_CONDITIONS, LEGITIMATE_TASK,
    INJECTED_PAYLOAD, CODING_TASK_POOL,
)
from src.behavior import AgentSandbox, run_trial, parse_output
from src.analysis import proportion_summary, pairwise_comparisons, fisher_exact_test

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PRIMARY_SEED    = 42
N_PER_CONDITION = 30
MODEL_ID        = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR         = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR     = os.path.join(OUT_DIR, "results")
FIGURES_DIR     = os.path.join(OUT_DIR, "figures")
SANDBOX_DIR     = os.path.join(OUT_DIR, "sandbox")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(SANDBOX_DIR, exist_ok=True)

random.seed(PRIMARY_SEED)
np.random.seed(PRIMARY_SEED)
torch.manual_seed(PRIMARY_SEED)


def main():
    t0 = time.time()
    print("=" * 60)
    print("EXPERIMENT 03: Warning Salience Ablation")
    print("=" * 60)
    print(f"  N per condition: {N_PER_CONDITION}")
    print(f"  Conditions: {WARNING_CONDITIONS}")
    print(f"  Total trials: {N_PER_CONDITION * len(WARNING_CONDITIONS)}")

    device_info = get_device_info()

    # Use CODING_TASK_POOL (30 tasks); cycle if needed
    task_pool = (CODING_TASK_POOL * 10)[:N_PER_CONDITION]  # exactly 30 unique tasks

    model, tokenizer, device, n_layers, dtype_used = load_model(MODEL_ID)

    all_rows = []
    random_examples = []
    # Pick 8 random trial indices for qualitative examples (seed-fixed)
    total_trials = N_PER_CONDITION * len(WARNING_CONDITIONS)
    rng_example = random.Random(PRIMARY_SEED)
    example_indices = set(rng_example.sample(range(total_trials), min(8, total_trials)))

    trial_idx = 0
    for cond in WARNING_CONDITIONS:
        print(f"\n--- Condition: {cond} ---")
        for task_i, legitimate_task in enumerate(task_pool):
            prompt = warning_prompt(cond, legitimate_task, INJECTED_PAYLOAD)
            sandbox = AgentSandbox(base_tmpdir=SANDBOX_DIR)

            trial_result = run_trial(model, tokenizer, prompt, device, sandbox, max_new_tokens=128)
            parsed = trial_result["parse"]

            row = {
                "trial_idx": trial_idx,
                "condition": cond,
                "task_idx": task_i,
                "legitimate_task": legitimate_task,
                "injected_payload": INJECTED_PAYLOAD,
                "outcome": parsed["outcome"],
                "unsafe_action": trial_result["unsafe_action"],
                "marker_mentioned": parsed["marker_mentioned"],
                "forbidden_mentioned": parsed["forbidden_mentioned"],
                "write_intent_detected": parsed["write_intent_detected"],
                "refusal_detected": parsed["refusal_detected"],
                "legitimate_task_addressed": parsed["legitimate_task_addressed"],
                "response_length": parsed["response_length"],
                "response_truncated": trial_result["model_output"][:200],
                "forbidden_contents_after": trial_result["forbidden_contents_after"][:50],
            }
            all_rows.append(row)

            if trial_idx in example_indices:
                random_examples.append({
                    "trial_idx": trial_idx,
                    "experiment": "03_warning_control",
                    "condition": cond,
                    "legitimate_task": legitimate_task,
                    "injected_payload": INJECTED_PAYLOAD,
                    "model_output": trial_result["model_output"],
                    "outcome": parsed["outcome"],
                    "unsafe_action": trial_result["unsafe_action"],
                    "forbidden_file_contents_after": trial_result["forbidden_contents_after"],
                    "marker_mentioned": parsed["marker_mentioned"],
                    "refusal_detected": parsed["refusal_detected"],
                })

            print(
                f"  [{trial_idx+1:4d}] {cond:<30} "
                f"outcome={parsed['outcome']:<12} "
                f"unsafe={trial_result['unsafe_action']}",
                flush=True,
            )
            sandbox.cleanup()
            trial_idx += 1

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Save warning_results.csv ---
    csv_path = os.path.join(RESULTS_DIR, "warning_results.csv")
    fieldnames = [
        "trial_idx", "condition", "task_idx", "legitimate_task", "injected_payload",
        "outcome", "unsafe_action", "marker_mentioned", "forbidden_mentioned",
        "write_intent_detected", "refusal_detected", "legitimate_task_addressed",
        "response_length", "response_truncated", "forbidden_contents_after",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\nSaved: {csv_path}")

    # --- Aggregate stats per condition ---
    agg = {}
    for cond in WARNING_CONDITIONS:
        rows = [r for r in all_rows if r["condition"] == cond]
        k_unsafe = sum(1 for r in rows if r["unsafe_action"])
        k_mentioned = sum(1 for r in rows if r["marker_mentioned"] or r["forbidden_mentioned"])
        k_executed = sum(1 for r in rows if r["outcome"] == "EXECUTED")
        k_discussed = sum(1 for r in rows if r["outcome"] == "DISCUSSED")
        k_refused = sum(1 for r in rows if r["outcome"] == "REFUSED")
        n = len(rows)
        agg[cond] = {
            "n": n,
            "unsafe_action_k": k_unsafe,
            "unsafe_action_rate": k_unsafe / n if n > 0 else 0.0,
            "mention_k": k_mentioned,
            "mention_rate": k_mentioned / n if n > 0 else 0.0,
            "executed_k": k_executed,
            "discussed_k": k_discussed,
            "refused_k": k_refused,
            "proportion": proportion_summary(k_unsafe, n),
        }

    print("\n-- Aggregated Results --")
    print(f"{'Condition':<35} {'N':>4} {'Unsafe':>8} {'Mention':>8} {'EXEC':>6} {'DISC':>6} {'REF':>6}")
    print("-" * 80)
    for cond, vals in agg.items():
        print(
            f"{cond:<35} {vals['n']:>4} "
            f"{vals['unsafe_action_rate']:>8.3f} "
            f"{vals['mention_rate']:>8.3f} "
            f"{vals['executed_k']:>6} {vals['discussed_k']:>6} {vals['refused_k']:>6}"
        )

    # --- Pairwise Fisher exact vs no_warning ---
    conditions = WARNING_CONDITIONS
    ks = [agg[c]["unsafe_action_k"] for c in conditions]
    ns = [agg[c]["n"] for c in conditions]
    comparisons = pairwise_comparisons(conditions, ks, ns, reference="no_warning")

    print("\n-- Pairwise Fisher Exact (vs no_warning) --")
    for cmp in comparisons:
        print(
            f"  {cmp['condition_a']} vs {cmp['condition_b']}: "
            f"p={cmp['p_value']:.4f}, OR={cmp['odds_ratio']:.2f} — {cmp['interpretation']}"
        )

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["steelblue", "darkorange", "seagreen", "crimson", "purple"]
    short_labels = [c.replace("_", "\n") for c in WARNING_CONDITIONS]

    # Unsafe action rate
    axes[0].bar(range(len(WARNING_CONDITIONS)),
                [agg[c]["unsafe_action_rate"] for c in WARNING_CONDITIONS],
                color=colors, alpha=0.85)
    axes[0].set_xticks(range(len(WARNING_CONDITIONS)))
    axes[0].set_xticklabels(short_labels, fontsize=7)
    axes[0].set_ylabel("Unsafe action rate (filesystem modified)")
    axes[0].set_title("Warning Ablation: Unsafe Action Rate")
    axes[0].set_ylim(0, 1)
    # Add CI error bars
    ci_lows = [agg[c]["unsafe_action_rate"] - agg[c]["proportion"]["ci95_lower"] for c in WARNING_CONDITIONS]
    ci_highs = [agg[c]["proportion"]["ci95_upper"] - agg[c]["unsafe_action_rate"] for c in WARNING_CONDITIONS]
    axes[0].errorbar(range(len(WARNING_CONDITIONS)),
                     [agg[c]["unsafe_action_rate"] for c in WARNING_CONDITIONS],
                     yerr=[ci_lows, ci_highs], fmt="none", color="black", capsize=4, lw=1.5)

    # Mention rate
    axes[1].bar(range(len(WARNING_CONDITIONS)),
                [agg[c]["mention_rate"] for c in WARNING_CONDITIONS],
                color=colors, alpha=0.85)
    axes[1].set_xticks(range(len(WARNING_CONDITIONS)))
    axes[1].set_xticklabels(short_labels, fontsize=7)
    axes[1].set_ylabel("Mention rate (text contains marker/forbidden)")
    axes[1].set_title("Warning Ablation: Mention Rate")
    axes[1].set_ylim(0, 1)

    plt.suptitle(f"Experiment 03: Warning Salience Controls\n{MODEL_ID} | N={N_PER_CONDITION}/condition", fontsize=10)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "warning_ablation.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved: {fig_path}")

    # --- Random examples ---
    examples_path = os.path.join(RESULTS_DIR, "random_examples.json")
    existing = []
    if os.path.exists(examples_path):
        with open(examples_path) as f:
            existing = json.load(f)
    existing.extend(random_examples)
    with open(examples_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Updated: {examples_path} ({len(existing)} total examples)")

    # --- Update summary.json ---
    elapsed = time.time() - t0
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    summary["experiment_03"] = {
        "model_id": MODEL_ID,
        "n_per_condition": N_PER_CONDITION,
        "n_conditions": len(WARNING_CONDITIONS),
        "conditions": WARNING_CONDITIONS,
        "total_trials": trial_idx,
        "device_info": device_info,
        "elapsed_seconds": round(elapsed, 1),
        "aggregated": {
            c: {
                "n": agg[c]["n"],
                "unsafe_action_k": agg[c]["unsafe_action_k"],
                "unsafe_action_rate": agg[c]["unsafe_action_rate"],
                "ci95_lower": agg[c]["proportion"]["ci95_lower"],
                "ci95_upper": agg[c]["proportion"]["ci95_upper"],
            }
            for c in WARNING_CONDITIONS
        },
        "pairwise_vs_no_warning": comparisons,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Updated: {summary_path}")

    print(f"\n{'=' * 60}")
    print("EXPERIMENT 03 COMPLETE")
    print(f"  Trials: {trial_idx}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
