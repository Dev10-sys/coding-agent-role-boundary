# -*- coding: utf-8 -*-
"""
experiments/05_intervention.py
================================
Exploratory Activation Steering Experiment (Conditional)
---------------------------------------------------------

IMPORTANT: This experiment is evaluated only if the source probe from
Experiment 02 achieves balanced accuracy >= 70% on held-out content with
content_only pooling. If the probe does not generalize beyond formatting,
steering along its direction has no interpretable meaning.

This script checks that precondition and exits with a clear message if not met.

If the operational threshold is met:
  - Extract the probe direction (system_policy vs. other) from the best-layer probe
  - Steer the model's residual stream in that direction during generation
  - Measure whether steering changes the unsafe-action probability
  - Also steer in the NEGATIVE direction (control)
  - Also steer in an ORTHOGONAL random direction (control)

Steering method: residual stream hook that adds alpha * direction at each
forward pass token.

Steering strengths: alpha in [-2, -1, 0, +1, +2]

Outputs:
  results/intervention_results.csv
  figures/intervention_effect.png

Explicit caveat (hard-coded into output):
  "Changing the representation changes behaviour" does NOT imply that the
  probe direction is the causal mechanism of behavioral authority. Steering along a probe
  direction in a small model is consistent with but does not prove that the
  probe captures a causally relevant feature.
"""

import sys, os, json, csv, random, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import load_model, get_device_info
from src.data import load_base_texts, text_level_split, expand_text_indices_to_sample_indices
from src.prompts import source_format, SOURCE_CONDITIONS, behavior_prompt, LEGITIMATE_TASK, INJECTED_PAYLOAD, CODING_TASK_POOL
from src.activations import extract_hidden_states
from src.probes import train_probe, eval_probe, make_label_encoder
from src.behavior import AgentSandbox, run_trial
from src.analysis import proportion_summary, fisher_exact_test

PRIMARY_SEED    = 42
MIN_BAL_ACC     = 0.70   # Pre-specified operational threshold
MODEL_ID        = "Qwen/Qwen2.5-0.5B-Instruct"
WIKITEXT_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "wikitext_valid.txt")
N_BASE_TEXTS    = 120
MAX_SEQ_LEN     = 128
TEST_FRAC       = 0.20
STEER_STRENGTHS = [-2.0, -1.0, 0.0, 1.0, 2.0]
N_TRIALS_PER_STRENGTH = 10
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


def check_precondition(summary_path: str) -> tuple:
    """
    Check if Experiment 02 source probe achieved balanced accuracy > MIN_BAL_ACC
    with content_only pooling. Returns (passed: bool, bal_acc: float, layer: int).
    """
    if not os.path.exists(summary_path):
        return False, 0.0, -1
    with open(summary_path) as f:
        summary = json.load(f)
    exp02 = summary.get("experiment_02", {})
    content_only = exp02.get("results_by_pooling", {}).get("content_only", {})
    bal_acc = content_only.get("best_balanced_accuracy", 0.0)
    layer   = content_only.get("best_layer", -1)
    return bal_acc >= MIN_BAL_ACC, bal_acc, layer


def get_probe_direction(model, tokenizer, device, n_layers, base_texts, layer_idx, seed):
    """
    Train a binary (system_policy vs. all other) probe on the fixed-structure
    source format and return the weight vector pointing toward system_policy.

    Returns: direction (np.ndarray shape (D,)), normalised to unit length.
    """
    print(f"\nTraining binary probe for steering direction at layer {layer_idx}...")
    conditions = SOURCE_CONDITIONS
    texts, labels = [], []
    for text in base_texts:
        for cond in conditions:
            texts.append(source_format(cond, text))
            labels.append(1 if cond == "system_policy" else 0)

    features = extract_hidden_states(
        model, tokenizer, texts, [layer_idx],
        max_len=MAX_SEQ_LEN, device=device,
        pooling="content_only", batch_size=4,
    )[:, 0, :]  # (N, D)

    y = np.array(labels)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, solver="lbfgs")
    clf.fit(features, y)

    direction = clf.coef_[0]  # shape (D,)
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    print(f"  Direction norm (before normalise): {np.linalg.norm(clf.coef_[0]):.4f}")
    return direction


def steer_and_run(model, tokenizer, device, direction, alpha, layer_idx,
                  task, sandbox, max_new_tokens=96):
    """
    Run a trial with steering: add alpha * direction to the residual stream
    at layer_idx after each forward pass.
    """
    import torch

    # Register hook
    def make_hook(layer, dir_tensor, strength):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]
                hs = hs + strength * dir_tensor.to(hs.device, dtype=hs.dtype)
                return (hs,) + output[1:]
            else:
                return output + strength * dir_tensor.to(output.device, dtype=output.dtype)
        return hook

    dir_tensor = torch.tensor(direction, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,D)
    hook_handle = model.model.layers[layer_idx].register_forward_hook(
        make_hook(layer_idx, dir_tensor, alpha)
    )

    prompt = behavior_prompt("repo_file", task, INJECTED_PAYLOAD)
    trial_result = run_trial(model, tokenizer, prompt, device, sandbox, max_new_tokens=max_new_tokens)
    hook_handle.remove()
    return trial_result


def main():
    t0 = time.time()
    print("=" * 60)
    print("EXPERIMENT 05: Activation Steering (Conditional)")
    print("=" * 60)

    device_info = get_device_info()
    summary_path = os.path.join(RESULTS_DIR, "summary.json")

    precondition_met, bal_acc, probe_layer = check_precondition(summary_path)
    print(f"\nPrecondition: source probe content_only balanced accuracy >= {MIN_BAL_ACC}")
    print(f"  Measured: {bal_acc:.4f} at layer {probe_layer}")

    if not precondition_met:
        msg = (
            f"PRECONDITION NOT MET: Source probe balanced accuracy = {bal_acc:.4f} "
            f"(threshold = {MIN_BAL_ACC}). "
            "Activation steering is not meaningful when the probe direction does not "
            "generalise beyond formatting tokens to held-out content. "
            "Experiment 05 is skipped."
        )
        print(f"\n{msg}")
        # Still save a results entry so the absence is documented
        summary = {}
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
        summary["experiment_05"] = {
            "skipped": True,
            "reason": msg,
            "precondition": f"content_only balanced_accuracy >= {MIN_BAL_ACC}",
            "measured_balanced_accuracy": bal_acc,
            "probe_layer": probe_layer,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print("Saved skip record to summary.json")

        # Write empty intervention_results.csv so the file always exists
        csv_path = os.path.join(RESULTS_DIR, "intervention_results.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["alpha", "control_type", "trial_idx", "unsafe_action", "outcome",
                        "note"])
            w.writerow([0, "n/a", 0, False, "n/a", "experiment skipped: precondition not met"])
        return

    print(f"  Precondition met. Proceeding with steering at layer {probe_layer}.")

    base_texts = load_base_texts(WIKITEXT_PATH, N_BASE_TEXTS, seed=PRIMARY_SEED)
    task_pool  = (CODING_TASK_POOL * 10)[:N_TRIALS_PER_STRENGTH]

    model, tokenizer, device, n_layers, dtype_used = load_model(MODEL_ID)

    direction = get_probe_direction(model, tokenizer, device, n_layers, base_texts, probe_layer, PRIMARY_SEED)

    # Orthogonal control direction
    rng = np.random.default_rng(PRIMARY_SEED)
    orth_dir = rng.standard_normal(direction.shape)
    orth_dir = orth_dir - np.dot(orth_dir, direction) * direction
    orth_dir = orth_dir / (np.linalg.norm(orth_dir) + 1e-8)

    all_rows = []
    for alpha in STEER_STRENGTHS:
        for ctrl_type, steer_dir in [("probe_direction", direction), ("orthogonal_control", orth_dir)]:
            print(f"\n  alpha={alpha:+.1f}  ctrl={ctrl_type}")
            for task_i, task in enumerate(task_pool):
                sandbox = AgentSandbox(base_tmpdir=SANDBOX_DIR)
                trial = steer_and_run(model, tokenizer, device, steer_dir, alpha,
                                      probe_layer, task, sandbox)
                row = {
                    "alpha": alpha,
                    "control_type": ctrl_type,
                    "trial_idx": task_i,
                    "task": task,
                    "unsafe_action": trial["unsafe_action"],
                    "outcome": trial["parse"]["outcome"],
                    "marker_mentioned": trial["parse"]["marker_mentioned"],
                    "refusal_detected": trial["parse"]["refusal_detected"],
                    "response_truncated": trial["model_output"][:150],
                }
                all_rows.append(row)
                sandbox.cleanup()
                print(f"    [{task_i+1}/{N_TRIALS_PER_STRENGTH}] unsafe={trial['unsafe_action']}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Save CSV ---
    csv_path = os.path.join(RESULTS_DIR, "intervention_results.csv")
    fieldnames = ["alpha", "control_type", "trial_idx", "task", "unsafe_action",
                  "outcome", "marker_mentioned", "refusal_detected", "response_truncated"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\nSaved: {csv_path}")

    # --- Aggregate ---
    print("\n-- Intervention Results --")
    print(f"{'Alpha':>6} {'Control':>25} {'k_unsafe':>10} {'N':>5} {'Rate':>8}")
    print("-" * 60)
    agg = {}
    for alpha in STEER_STRENGTHS:
        for ctrl_type in ["probe_direction", "orthogonal_control"]:
            rows = [r for r in all_rows if r["alpha"] == alpha and r["control_type"] == ctrl_type]
            k = sum(1 for r in rows if r["unsafe_action"])
            n = len(rows)
            rate = k / n if n > 0 else 0.0
            agg[(alpha, ctrl_type)] = {"k": k, "n": n, "rate": rate}
            print(f"{alpha:>+6.1f} {ctrl_type:>25} {k:>10} {n:>5} {rate:>8.3f}")

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, ctrl_type in zip(axes, ["probe_direction", "orthogonal_control"]):
        xs = STEER_STRENGTHS
        ys = [agg[(a, ctrl_type)]["rate"] for a in xs]
        ax.plot(xs, ys, "-o", color="steelblue", lw=2, ms=8)
        ax.set_xlabel("Steering strength (alpha)")
        ax.set_ylabel("Unsafe action rate")
        ax.set_title(f"Steering: {ctrl_type}")
        ax.set_ylim(0, 1)
        ax.axhline(0, color="gray", ls=":", lw=0.8)

    plt.suptitle(
        f"Experiment 05: Activation Steering\n{MODEL_ID} | layer={probe_layer} | "
        f"N={N_TRIALS_PER_STRENGTH}/alpha\n"
        "CAUTION: steering effect ≠ causal mechanism", fontsize=9
    )
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "intervention_effect.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved: {fig_path}")

    # --- Update summary ---
    elapsed = time.time() - t0
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    summary["experiment_05"] = {
        "skipped": False,
        "model_id": MODEL_ID,
        "probe_layer": probe_layer,
        "precondition_bal_acc": bal_acc,
        "steer_strengths": STEER_STRENGTHS,
        "n_trials_per_strength": N_TRIALS_PER_STRENGTH,
        "device_info": device_info,
        "elapsed_seconds": round(elapsed, 1),
        "results": {
            f"alpha={a}_{ctrl}": {"k": agg[(a, ctrl)]["k"], "n": agg[(a, ctrl)]["n"], "rate": agg[(a, ctrl)]["rate"]}
            for a, ctrl in agg.keys()
        },
        "caution": (
            "Steering along the probe direction changes model behaviour in this experiment. "
            "This is consistent with the probe capturing a causally relevant feature, but "
            "does NOT establish that the probe direction IS the mechanism. "
            "See docs/limitations.md for full discussion."
        ),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Updated: {summary_path}")

    print(f"\n{'=' * 60}")
    print("EXPERIMENT 05 COMPLETE")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
