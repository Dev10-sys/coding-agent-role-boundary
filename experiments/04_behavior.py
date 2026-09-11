# -*- coding: utf-8 -*-
"""
experiments/04_behavior.py
============================
Behavioral Sandbox with Actual Write Tool
-------------------------------------------

Research question fragment:
  Does the source label of an injected instruction (system_policy, user_direct,
  repo_file, tool_output) affect whether the model actually executes the unsafe
  action (writing to forbidden.txt) when a real write tool is available?

Critical corrections vs the pilot:
  1. unsafe_action = forbidden.txt ACTUALLY CHANGED (filesystem truth), not mention
  2. Outcome categories: EXECUTED | DISCUSSED | REFUSED | NO_RESPONSE
  3. Fixed-structure prompts (same outer format, only SOURCE label varies)
  4. N=30 per condition (120 total)
  5. 8 random qualitative examples with full filesystem state recorded

Outputs:
  results/behavior_results.csv
  results/random_examples.json (8+ examples)
  figures/source_behavior.png
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
    behavior_prompt, SOURCE_CONDITIONS,
    LEGITIMATE_TASK, INJECTED_PAYLOAD, CODING_TASK_POOL,
)
from src.behavior import AgentSandbox, run_trial
from src.analysis import proportion_summary, pairwise_comparisons

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
    print("EXPERIMENT 04: Behavioral Sandbox with Actual Write Tool")
    print("=" * 60)
    print(f"  Conditions: {SOURCE_CONDITIONS}")
    print(f"  N per condition: {N_PER_CONDITION}")
    print(f"  Total trials: {N_PER_CONDITION * len(SOURCE_CONDITIONS)}")

    device_info = get_device_info()
    task_pool = (CODING_TASK_POOL * 10)[:N_PER_CONDITION]

    model, tokenizer, device, n_layers, dtype_used = load_model(MODEL_ID)

    all_rows = []
    random_examples = []
    total_trials = N_PER_CONDITION * len(SOURCE_CONDITIONS)
    rng_example = random.Random(PRIMARY_SEED)
    example_indices = set(rng_example.sample(range(total_trials), min(8, total_trials)))

    trial_idx = 0
    for cond in SOURCE_CONDITIONS:
        print(f"\n--- Condition: {cond} ---")
        for task_i, legitimate_task in enumerate(task_pool):
            prompt = behavior_prompt(cond, legitimate_task, INJECTED_PAYLOAD)
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
                "forbidden_contents_after": trial_result["forbidden_contents_after"][:100],
            }
            all_rows.append(row)

            if trial_idx in example_indices:
                random_examples.append({
                    "trial_idx": trial_idx,
                    "experiment": "04_behavior",
                    "condition": cond,
                    "legitimate_task": legitimate_task,
                    "injected_payload": INJECTED_PAYLOAD,
                    "model_output": trial_result["model_output"],
                    "outcome": parsed["outcome"],
                    "unsafe_action": trial_result["unsafe_action"],
                    "forbidden_file_contents_after": trial_result["forbidden_contents_after"],
                    "marker_mentioned": parsed["marker_mentioned"],
                    "write_intent_detected": parsed["write_intent_detected"],
                    "refusal_detected": parsed["refusal_detected"],
                    "legitimate_task_addressed": parsed["legitimate_task_addressed"],
                })

            print(
                f"  [{trial_idx+1:4d}] {cond:<20} "
                f"outcome={parsed['outcome']:<12} "
                f"unsafe={trial_result['unsafe_action']}",
                flush=True,
            )
            sandbox.cleanup()
            trial_idx += 1

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Save behavior_results.csv ---
    csv_path = os.path.join(RESULTS_DIR, "behavior_results.csv")
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

    # --- Aggregate per condition ---
    agg = {}
    for cond in SOURCE_CONDITIONS:
        rows = [r for r in all_rows if r["condition"] == cond]
        k_unsafe = sum(1 for r in rows if r["unsafe_action"])
        k_exec   = sum(1 for r in rows if r["outcome"] == "EXECUTED")
        k_disc   = sum(1 for r in rows if r["outcome"] == "DISCUSSED")
        k_ref    = sum(1 for r in rows if r["outcome"] == "REFUSED")
        k_none   = sum(1 for r in rows if r["outcome"] == "NO_RESPONSE")
        k_legit  = sum(1 for r in rows if r["legitimate_task_addressed"])
        n = len(rows)
        agg[cond] = {
            "n": n,
            "unsafe_action_k": k_unsafe,
            "unsafe_action_rate": k_unsafe / n if n > 0 else 0.0,
            "executed_k": k_exec,
            "discussed_k": k_disc,
            "refused_k": k_ref,
            "no_response_k": k_none,
            "legitimate_task_k": k_legit,
            "proportion": proportion_summary(k_unsafe, n),
        }

    print("\n-- Aggregated Results --")
    print(f"{'Condition':<25} {'N':>4} {'Unsafe/N':>10} {'Rate':>7} {'95% CI':>18} {'EXEC':>6} {'DISC':>6} {'REF':>6}")
    print("-" * 90)
    for cond, vals in agg.items():
        lo = vals["proportion"]["ci95_lower"]
        hi = vals["proportion"]["ci95_upper"]
        print(
            f"{cond:<25} {vals['n']:>4} "
            f"{vals['unsafe_action_k']:>4}/{vals['n']:<4} "
            f"{vals['unsafe_action_rate']:>7.3f} "
            f"[{lo:.3f}, {hi:.3f}]"
            f"{vals['executed_k']:>6} {vals['discussed_k']:>6} {vals['refused_k']:>6}"
        )

    # --- Pairwise Fisher exact vs system_policy ---
    conditions = SOURCE_CONDITIONS
    ks = [agg[c]["unsafe_action_k"] for c in conditions]
    ns = [agg[c]["n"] for c in conditions]
    comparisons = pairwise_comparisons(conditions, ks, ns, reference="system_policy")

    print("\n-- Pairwise Fisher Exact (vs system_policy) --")
    for cmp in comparisons:
        print(
            f"  {cmp['condition_a']} vs {cmp['condition_b']}: "
            f"p={cmp['p_value']:.4f}, OR={cmp['odds_ratio']:.2f} — {cmp['interpretation']}"
        )

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]
    short_labels = [c.replace("_", "\n") for c in SOURCE_CONDITIONS]

    # Unsafe action rate
    rates = [agg[c]["unsafe_action_rate"] for c in SOURCE_CONDITIONS]
    ci_lows  = [agg[c]["unsafe_action_rate"] - agg[c]["proportion"]["ci95_lower"] for c in SOURCE_CONDITIONS]
    ci_highs = [agg[c]["proportion"]["ci95_upper"] - agg[c]["unsafe_action_rate"] for c in SOURCE_CONDITIONS]
    axes[0].bar(range(len(SOURCE_CONDITIONS)), rates, color=colors, alpha=0.85)
    axes[0].errorbar(range(len(SOURCE_CONDITIONS)), rates,
                     yerr=[ci_lows, ci_highs], fmt="none", color="black", capsize=4, lw=1.5)
    axes[0].set_xticks(range(len(SOURCE_CONDITIONS)))
    axes[0].set_xticklabels(short_labels, fontsize=8)
    axes[0].set_ylabel("Unsafe action rate (forbidden.txt modified)")
    axes[0].set_title("Behavioral Unsafe Action Rate by Source")
    axes[0].set_ylim(0, 1)

    # Outcome breakdown (stacked bar)
    executed_rates = [agg[c]["executed_k"] / agg[c]["n"] for c in SOURCE_CONDITIONS]
    discussed_rates = [agg[c]["discussed_k"] / agg[c]["n"] for c in SOURCE_CONDITIONS]
    refused_rates   = [agg[c]["refused_k"] / agg[c]["n"] for c in SOURCE_CONDITIONS]
    none_rates      = [agg[c]["no_response_k"] / agg[c]["n"] for c in SOURCE_CONDITIONS]
    x = range(len(SOURCE_CONDITIONS))
    b1 = axes[1].bar(x, executed_rates, label="EXECUTED", color="crimson", alpha=0.85)
    b2 = axes[1].bar(x, discussed_rates, bottom=executed_rates, label="DISCUSSED", color="darkorange", alpha=0.85)
    b3_bot = [e + d for e, d in zip(executed_rates, discussed_rates)]
    b3 = axes[1].bar(x, refused_rates, bottom=b3_bot, label="REFUSED", color="steelblue", alpha=0.85)
    b4_bot = [a + b for a, b in zip(b3_bot, refused_rates)]
    axes[1].bar(x, none_rates, bottom=b4_bot, label="NO_RESPONSE", color="lightgray", alpha=0.85)
    axes[1].set_xticks(range(len(SOURCE_CONDITIONS)))
    axes[1].set_xticklabels(short_labels, fontsize=8)
    axes[1].set_ylabel("Fraction of trials")
    axes[1].set_title("Outcome Breakdown by Source")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8, loc="upper right")

    plt.suptitle(f"Experiment 04: Behavioral Sandbox\n{MODEL_ID} | N={N_PER_CONDITION}/condition", fontsize=10)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "source_behavior.png")
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

    summary["experiment_04"] = {
        "model_id": MODEL_ID,
        "n_per_condition": N_PER_CONDITION,
        "n_conditions": len(SOURCE_CONDITIONS),
        "source_conditions": SOURCE_CONDITIONS,
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
                "executed_k": agg[c]["executed_k"],
                "discussed_k": agg[c]["discussed_k"],
                "refused_k": agg[c]["refused_k"],
            }
            for c in SOURCE_CONDITIONS
        },
        "pairwise_vs_system_policy": comparisons,
        "metric_note": (
            "unsafe_action = forbidden.txt actually modified in sandbox temp dir. "
            "This is filesystem truth, not text-match."
        ),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Updated: {summary_path}")

    print(f"\n{'=' * 60}")
    print("EXPERIMENT 04 COMPLETE")
    print(f"  Trials: {trial_idx}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
