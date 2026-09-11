# -*- coding: utf-8 -*-
"""
experiments/01_role_probe.py
=============================
Corrected Role Probe Experiment
---------------------------------

Research question fragment:
  Does Qwen2.5-0.5B-Instruct encode role identity in its residual stream,
  and if so, is that signal driven by role-name tokens (formatting) or by
  semantic content?

Corrections vs the pilot
  1. Text-level train/test split — zero base-text leakage
  2. Three pooling strategies: mean_all | content_only | last_content
  3. Scramble-control: role name tokens replaced with ROLE_A/B/C/D
  4. Balanced accuracy alongside raw accuracy
  5. Second random seed (seed=99) robustness check
  6. Layers: 5, 11, 17, 23 (same as pilot for comparability)

Expected finding:
  - mean_all probe: high accuracy (replicates pilot) — driven by role tokens
  - content_only / last_content: lower accuracy if role tokens drove pilot result
  - scramble_control probe: near-chance if role tokens are necessary
  - If content_only accuracy > 70%, the model also encodes semantic role signal
    beyond just role-name token identity

Outputs
  results/probe_results.csv          — per-layer accuracy for each pooling mode
  results/probe_scramble.json        — scramble control results
  figures/probe_accuracy.png         — accuracy vs layer (3 pooling modes)
  figures/probe_confusion.png        — confusion matrix at best layer (mean_all)
"""

import sys, os, json, csv, random, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Make src importable from experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import load_base_texts, text_level_split, expand_text_indices_to_sample_indices
from src.model import load_model, get_device_info
from src.prompts import pilot_role_wrap, scramble_role_wrap, PILOT_ROLES
from src.activations import extract_hidden_states
from src.probes import train_probe, eval_probe, make_label_encoder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PRIMARY_SEED  = 42
ROBUST_SEED   = 99
N_BASE_TEXTS  = 120
MAX_SEQ_LEN   = 128
TEST_FRAC     = 0.20
MODEL_ID      = "Qwen/Qwen2.5-0.5B-Instruct"
WIKITEXT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "wikitext_valid.txt")
OUT_DIR       = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR   = os.path.join(OUT_DIR, "results")
FIGURES_DIR   = os.path.join(OUT_DIR, "figures")
POOLING_MODES = ["mean_all", "content_only", "last_content"]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

random.seed(PRIMARY_SEED)
np.random.seed(PRIMARY_SEED)
torch.manual_seed(PRIMARY_SEED)


def build_dataset(base_texts, roles, wrap_fn):
    """Build (texts, labels) by wrapping each base text in each role."""
    texts, labels = [], []
    for text in base_texts:
        for role in roles:
            texts.append(wrap_fn(role, text))
            labels.append(role)
    return texts, labels


def run_probe_for_pooling(
    model, tokenizer, device, n_layers,
    base_texts, layers, pooling, seed,
    wrap_fn=None, label="",
):
    """Train and evaluate probes at each layer for a given pooling mode."""
    if wrap_fn is None:
        wrap_fn = pilot_role_wrap

    print(f"\n  [{label or pooling}] Extracting hidden states...")
    all_wrapped, all_labels = build_dataset(base_texts, PILOT_ROLES, wrap_fn)

    features = extract_hidden_states(
        model, tokenizer, all_wrapped, layers,
        max_len=MAX_SEQ_LEN, device=device,
        pooling=pooling, batch_size=4,
    )
    print(f"  Feature shape: {features.shape}")

    le = make_label_encoder(PILOT_ROLES)
    y = le.transform(all_labels)
    n_texts = len(base_texts)

    # Text-level split (key correction)
    train_text_idx, test_text_idx = text_level_split(base_texts, test_frac=TEST_FRAC, seed=seed)
    train_sample_idx = expand_text_indices_to_sample_indices(train_text_idx, len(PILOT_ROLES))
    test_sample_idx  = expand_text_indices_to_sample_indices(test_text_idx,  len(PILOT_ROLES))

    print(f"  Train texts: {len(train_text_idx)}, Test texts: {len(test_text_idx)}")
    print(f"  Train samples: {len(train_sample_idx)}, Test samples: {len(test_sample_idx)}")
    assert set(train_sample_idx) & set(test_sample_idx) == set(), "LEAKAGE DETECTED"

    layer_results = []
    for li, layer_idx in enumerate(layers):
        X = features[:, li, :]
        X_train = X[train_sample_idx]
        X_test  = X[test_sample_idx]
        y_train = y[train_sample_idx]
        y_test  = y[test_sample_idx]

        probe = train_probe(X_train, y_train, seed=seed)
        result = eval_probe(probe, X_test, y_test, label_names=PILOT_ROLES)
        result["layer"] = layer_idx
        result["layer_fraction"] = round((layer_idx + 1) / n_layers, 3)
        result["pooling"] = pooling
        result["seed"] = seed
        result["label"] = label or pooling
        print(f"    Layer {layer_idx:3d} | acc={result['accuracy']:.4f} | "
              f"bal_acc={result['balanced_accuracy']:.4f}")
        layer_results.append(result)

    return layer_results


def run_scramble_control(model, tokenizer, device, n_layers, base_texts, layers, seed):
    """Run probe on scramble-controlled data (role tokens replaced with ROLE_A/B/C/D)."""
    print("\n  [scramble_control] Running scramble control probe...")
    results = run_probe_for_pooling(
        model, tokenizer, device, n_layers,
        base_texts, layers, pooling="mean_all",
        seed=seed, wrap_fn=scramble_role_wrap, label="scramble_control",
    )
    return results


def plot_probe_accuracy(all_results_by_label, layers, n_layers, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "mean_all":       ("-o", "steelblue"),
        "content_only":   ("-s", "darkorange"),
        "last_content":   ("-^", "seagreen"),
        "scramble_control": ("--x", "crimson"),
    }
    for label, results in all_results_by_label.items():
        xs = [r["layer"] for r in results]
        ys = [r["accuracy"] for r in results]
        ls, color = styles.get(label, ("-o", "gray"))
        ax.plot(xs, ys, ls, color=color, lw=2, ms=8, label=label)

    # Chance baseline
    chance = 1.0 / len(PILOT_ROLES)
    ax.axhline(chance, color="black", ls=":", lw=1.5, label=f"Chance ({chance:.2f})")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Test accuracy (4-class)")
    ax.set_title(f"Role Probe Accuracy vs. Layer\n{MODEL_ID} | N={120} base texts")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_confusion_matrix(result, label_names, out_path):
    cm = np.array(result["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45)
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"Confusion Matrix — Layer {result['layer']} ({result['label']})\n"
        f"acc={result['accuracy']:.3f} bal_acc={result['balanced_accuracy']:.3f}"
    )
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    t0 = time.time()
    print("=" * 60)
    print("EXPERIMENT 01: Corrected Role Probe")
    print("=" * 60)

    device_info = get_device_info()
    print("\nEnvironment:")
    for k, v in device_info.items():
        print(f"  {k}: {v}")

    # Check wikitext file location
    if not os.path.exists(WIKITEXT_PATH):
        # Try adjacent to script (pilot location)
        alt_path = os.path.join(os.path.dirname(__file__), "..", "mats_pilot", "wikitext_valid.txt")
        pilot_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "mats_role_boundary_experiment", "wikitext_valid.txt"
        )
        for candidate in [alt_path, pilot_path]:
            if os.path.exists(candidate):
                wikitext_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"wikitext_valid.txt not found at {WIKITEXT_PATH}. "
                "Run: copy the file to data/wikitext_valid.txt"
            )
    else:
        wikitext_path = WIKITEXT_PATH

    print(f"\nLoading {N_BASE_TEXTS} base texts from {wikitext_path}")
    base_texts = load_base_texts(wikitext_path, N_BASE_TEXTS, seed=PRIMARY_SEED)
    print(f"Loaded: {len(base_texts)} texts")

    model, tokenizer, device, n_layers, dtype_used = load_model(MODEL_ID)

    # Determine layers to probe (same as pilot: 5, 11, 17, 23 for a 24-layer model)
    seen = set()
    layers = []
    for l in [n_layers // 4 - 1, n_layers // 2 - 1, 3 * n_layers // 4 - 1, n_layers - 1]:
        if l >= 0 and l not in seen:
            layers.append(l)
            seen.add(l)
    print(f"\nLayers to probe: {layers}")

    all_results_by_label = {}

    # --- Primary seed, all three pooling modes ---
    for pooling in POOLING_MODES:
        results = run_probe_for_pooling(
            model, tokenizer, device, n_layers,
            base_texts, layers, pooling=pooling,
            seed=PRIMARY_SEED,
        )
        all_results_by_label[pooling] = results

    # --- Scramble control ---
    scramble_results = run_scramble_control(
        model, tokenizer, device, n_layers, base_texts, layers, PRIMARY_SEED
    )
    all_results_by_label["scramble_control"] = scramble_results

    # --- Robustness: seed=99, mean_all only ---
    print(f"\n  [robustness seed={ROBUST_SEED}] Running mean_all probe...")
    robust_results = run_probe_for_pooling(
        model, tokenizer, device, n_layers,
        base_texts, layers, pooling="mean_all",
        seed=ROBUST_SEED, label=f"mean_all_seed{ROBUST_SEED}",
    )
    all_results_by_label[f"mean_all_seed{ROBUST_SEED}"] = robust_results

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Save probe_results.csv ---
    csv_path = os.path.join(RESULTS_DIR, "probe_results.csv")
    fieldnames = ["layer", "layer_fraction", "pooling", "label", "seed",
                  "accuracy", "balanced_accuracy", "chance_accuracy",
                  "above_chance", "n_test"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for label, results in all_results_by_label.items():
            for r in results:
                w.writerow({k: r[k] for k in fieldnames})
    print(f"\nSaved: {csv_path}")

    # --- Save confusion matrices JSON ---
    cm_data = {}
    for label, results in all_results_by_label.items():
        best_layer_result = max(results, key=lambda r: r["accuracy"])
        cm_data[label] = {
            "best_layer": best_layer_result["layer"],
            "best_accuracy": best_layer_result["accuracy"],
            "best_balanced_accuracy": best_layer_result["balanced_accuracy"],
            "confusion_matrix": best_layer_result["confusion_matrix"],
            "label_names": PILOT_ROLES,
        }
    cm_path = os.path.join(RESULTS_DIR, "probe_confusion_matrices.json")
    with open(cm_path, "w") as f:
        json.dump(cm_data, f, indent=2)
    print(f"Saved: {cm_path}")

    # --- Figures ---
    plot_probe_accuracy(
        all_results_by_label, layers, n_layers,
        os.path.join(FIGURES_DIR, "probe_accuracy.png"),
    )
    # Confusion matrix for best mean_all layer
    best_mean_all = max(all_results_by_label["mean_all"], key=lambda r: r["accuracy"])
    plot_confusion_matrix(
        best_mean_all, PILOT_ROLES,
        os.path.join(FIGURES_DIR, "probe_confusion.png"),
    )

    # --- Summary printout ---
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("EXPERIMENT 01 SUMMARY")
    print(f"{'=' * 60}")
    for label, results in all_results_by_label.items():
        best = max(results, key=lambda r: r["accuracy"])
        print(f"  {label:<30} best_layer={best['layer']:3d}  "
              f"acc={best['accuracy']:.4f}  bal_acc={best['balanced_accuracy']:.4f}")
    print(f"\n  Elapsed: {elapsed:.1f}s")

    # --- Save summary to results/summary.json (initialise or update) ---
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}

    summary["experiment_01"] = {
        "model_id": MODEL_ID,
        "n_base_texts": len(base_texts),
        "n_roles": len(PILOT_ROLES),
        "layers_probed": layers,
        "n_layers_total": n_layers,
        "test_frac": TEST_FRAC,
        "primary_seed": PRIMARY_SEED,
        "robust_seed": ROBUST_SEED,
        "device_info": device_info,
        "elapsed_seconds": round(elapsed, 1),
        "results_by_pooling": {
            label: {
                "best_layer": max(rs, key=lambda r: r["accuracy"])["layer"],
                "best_accuracy": max(rs, key=lambda r: r["accuracy"])["accuracy"],
                "best_balanced_accuracy": max(rs, key=lambda r: r["balanced_accuracy"])["balanced_accuracy"],
                "chance": 1.0 / len(PILOT_ROLES),
            }
            for label, rs in all_results_by_label.items()
        },
        "confound_note": (
            "mean_all accuracy is expected to be high (≥90%) due to role-name tokens. "
            "content_only and last_content accuracies isolate semantic role signal. "
            "scramble_control isolates token-only signal."
        ),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")
    print(f"{'=' * 60}")

    return summary["experiment_01"]


if __name__ == "__main__":
    main()
