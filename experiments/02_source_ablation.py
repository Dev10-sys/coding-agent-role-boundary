# -*- coding: utf-8 -*-
"""
experiments/02_source_ablation.py
===================================
Source-Controlled Representation Probe
-----------------------------------------

Research question fragment:
  After controlling for prompt formatting (same outer structure across all
  conditions), can source/provenance identity be recovered from the model's
  internal activations at the layer where the injected instruction appears?

The key correction over the pilot:
  The pilot applied the role probe to different ChatML wrappers (system turn
  vs. user turn vs. tool turn) and found all conditions classified as "tool".
  This was because the probe read the multi-turn wrapper, not the source label.

  This experiment uses a FIXED outer structure across all conditions:

    SOURCE: {SYSTEM_POLICY | USER_INSTRUCTION | REPO_FILE:AGENTS.md | TOOL_OUTPUT:read_file}
    CONTENT:
    {same instruction text}

  and trains a 4-class source probe on this controlled representation.

  The probe is trained on held-out base texts (different from those used in
  the behaviour experiments) to test cross-content generalization.

Key questions:
  Q1: Can source label be recovered from mean-pooled activations?
  Q2: Is that recovery reduced when content-only pooling is used?
  Q3: Does the probe generalise to unseen content (held-out texts)?

Outputs:
  results/source_probe_results.csv
  figures/source_role_scores.png
"""

import sys, os, json, csv, random, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import load_base_texts, text_level_split, expand_text_indices_to_sample_indices
from src.model import load_model, get_device_info
from src.prompts import source_format, SOURCE_CONDITIONS
from src.activations import extract_multi_pooling
from src.probes import train_probe, eval_probe, make_label_encoder, probe_score_distributions

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PRIMARY_SEED  = 42
N_BASE_TEXTS  = 120   # same as pilot for comparability
MAX_SEQ_LEN   = 128
TEST_FRAC     = 0.20
MODEL_ID      = "Qwen/Qwen2.5-0.5B-Instruct"
WIKITEXT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "wikitext_valid.txt")
OUT_DIR       = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR   = os.path.join(OUT_DIR, "results")
FIGURES_DIR   = os.path.join(OUT_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

random.seed(PRIMARY_SEED)
np.random.seed(PRIMARY_SEED)
torch.manual_seed(PRIMARY_SEED)


def build_source_dataset(base_texts, conditions):
    """Wrap each base text in each source condition."""
    texts, labels = [], []
    for text in base_texts:
        for cond in conditions:
            texts.append(source_format(cond, text))
            labels.append(cond)
    return texts, labels


def main():
    t0 = time.time()
    print("=" * 60)
    print("EXPERIMENT 02: Source-Controlled Representation Probe")
    print("=" * 60)

    device_info = get_device_info()

    base_texts = load_base_texts(WIKITEXT_PATH, N_BASE_TEXTS, seed=PRIMARY_SEED)
    print(f"Loaded {len(base_texts)} base texts.")

    model, tokenizer, device, n_layers, dtype_used = load_model(MODEL_ID)

    seen = set()
    layers = []
    for l in [n_layers // 4 - 1, n_layers // 2 - 1, 3 * n_layers // 4 - 1, n_layers - 1]:
        if l >= 0 and l not in seen:
            layers.append(l)
            seen.add(l)
    print(f"Probing layers: {layers}")

    all_wrapped, all_labels = build_source_dataset(base_texts, SOURCE_CONDITIONS)
    print(f"Dataset: {len(base_texts)} texts × {len(SOURCE_CONDITIONS)} conditions = {len(all_wrapped)} samples")

    # Print 3 random examples
    print("\n-- 3 random source-format examples --")
    for idx in random.sample(range(len(all_wrapped)), 3):
        print(f"\n[cond={all_labels[idx]}]\n{all_wrapped[idx][:300]}")

    # Extract with mean_all and content_only in a single pass
    print("\nExtracting hidden states (mean_all + content_only single pass)...")
    features_dict = extract_multi_pooling(
        model, tokenizer, all_wrapped, layers,
        poolings=["mean_all", "content_only"],
        max_len=MAX_SEQ_LEN, device=device,
        batch_size=4,
    )

    le = make_label_encoder(SOURCE_CONDITIONS)
    y = le.transform(all_labels)

    # Text-level split
    train_text_idx, test_text_idx = text_level_split(base_texts, test_frac=TEST_FRAC, seed=PRIMARY_SEED)
    train_samples = expand_text_indices_to_sample_indices(train_text_idx, len(SOURCE_CONDITIONS))
    test_samples  = expand_text_indices_to_sample_indices(test_text_idx,  len(SOURCE_CONDITIONS))
    assert set(train_samples) & set(test_samples) == set(), "Leakage!"

    results_all = {}
    for pooling in ["mean_all", "content_only"]:
        features = features_dict[pooling]
        print(f"\nEvaluating probes for pooling={pooling} (Feature shape: {features.shape})...")

        layer_results = []
        for li, layer_idx in enumerate(layers):
            X = features[:, li, :]
            probe = train_probe(X[train_samples], y[train_samples], seed=PRIMARY_SEED)
            result = eval_probe(probe, X[test_samples], y[test_samples], label_names=SOURCE_CONDITIONS)
            result["layer"] = layer_idx
            result["layer_fraction"] = round((layer_idx + 1) / n_layers, 3)
            result["pooling"] = pooling
            result["seed"] = PRIMARY_SEED
            result["n_conditions"] = len(SOURCE_CONDITIONS)
            print(f"  Layer {layer_idx:3d} | acc={result['accuracy']:.4f} | "
                  f"bal_acc={result['balanced_accuracy']:.4f} | chance={result['chance_accuracy']:.3f}")
            layer_results.append(result)

        results_all[pooling] = layer_results

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Save CSV ---
    csv_path = os.path.join(RESULTS_DIR, "source_probe_results.csv")
    fieldnames = ["layer", "layer_fraction", "pooling", "seed", "n_conditions",
                  "accuracy", "balanced_accuracy", "chance_accuracy", "above_chance", "n_test"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for pooling, results in results_all.items():
            for r in results:
                w.writerow({k: r[k] for k in fieldnames})
    print(f"\nSaved: {csv_path}")

    # --- Figure: accuracy vs layer, both poolings ---
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"mean_all": "steelblue", "content_only": "darkorange"}
    for pooling, results in results_all.items():
        xs = [r["layer"] for r in results]
        ys = [r["accuracy"] for r in results]
        ax.plot(xs, ys, "-o", color=colors.get(pooling, "gray"), lw=2, ms=8, label=pooling)
    chance = 1.0 / len(SOURCE_CONDITIONS)
    ax.axhline(chance, color="black", ls=":", lw=1.5, label=f"Chance ({chance:.2f})")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Test accuracy (4-class source)")
    ax.set_title(f"Source Probe Accuracy vs. Layer\n{MODEL_ID} | fixed-structure format")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "source_role_scores.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved: {fig_path}")

    # --- Update summary.json ---
    elapsed = time.time() - t0
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    exp02_summary = {}
    for pooling, results in results_all.items():
        best = max(results, key=lambda r: r["accuracy"])
        exp02_summary[pooling] = {
            "best_layer": best["layer"],
            "best_accuracy": best["accuracy"],
            "best_balanced_accuracy": best["balanced_accuracy"],
            "chance": chance,
        }

    summary["experiment_02"] = {
        "model_id": MODEL_ID,
        "n_base_texts": len(base_texts),
        "n_source_conditions": len(SOURCE_CONDITIONS),
        "source_conditions": SOURCE_CONDITIONS,
        "format": "SOURCE: {label}\\nCONTENT:\\n{text}",
        "layers_probed": layers,
        "primary_seed": PRIMARY_SEED,
        "device_info": device_info,
        "elapsed_seconds": round(elapsed, 1),
        "results_by_pooling": exp02_summary,
        "interpretation_note": (
            "If mean_all >> content_only accuracy, the source label signal is driven by "
            "label tokens (SYSTEM_POLICY, USER_INSTRUCTION, etc.), not content representations. "
            "If both are >> chance, the model encodes source identity in content activations too."
        ),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Updated: {summary_path}")

    print(f"\n{'=' * 60}")
    print("EXPERIMENT 02 SUMMARY")
    for pooling, data in exp02_summary.items():
        print(f"  {pooling:<20} best_layer={data['best_layer']:3d}  "
              f"acc={data['best_accuracy']:.4f}  bal_acc={data['best_balanced_accuracy']:.4f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
