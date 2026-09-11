# -*- coding: utf-8 -*-
"""
tests/test_probe_pipeline.py
=============================
Verify probe training, text-level split logic, and data integrity.
Uses synthetic data only (no model required).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from src.data import load_base_texts, text_level_split, expand_text_indices_to_sample_indices
from src.probes import train_probe, eval_probe, make_label_encoder


# ---------------------------------------------------------------------------
# Text-level split: zero base-text leakage
# ---------------------------------------------------------------------------

class TestTextLevelSplit:
    def test_no_overlap(self):
        texts = [f"text_{i}" for i in range(100)]
        train_idx, test_idx = text_level_split(texts, test_frac=0.2, seed=42)
        assert set(train_idx) & set(test_idx) == set(), \
            "Base text indices overlap between train and test!"

    def test_correct_sizes(self):
        texts = [f"t{i}" for i in range(100)]
        train_idx, test_idx = text_level_split(texts, test_frac=0.2, seed=42)
        assert len(test_idx) == 20
        assert len(train_idx) == 80

    def test_deterministic(self):
        texts = [f"t{i}" for i in range(50)]
        t1, e1 = text_level_split(texts, test_frac=0.2, seed=42)
        t2, e2 = text_level_split(texts, test_frac=0.2, seed=42)
        assert t1 == t2 and e1 == e2

    def test_different_seeds_differ(self):
        texts = [f"t{i}" for i in range(50)]
        _, e1 = text_level_split(texts, test_frac=0.2, seed=42)
        _, e2 = text_level_split(texts, test_frac=0.2, seed=99)
        assert e1 != e2, "Different seeds produced identical splits (unlikely unless n is tiny)"

    def test_expand_to_sample_indices_no_leakage(self):
        """After expansion, train and test sample indices must not overlap."""
        texts = [f"t{i}" for i in range(20)]
        n_roles = 4
        train_idx, test_idx = text_level_split(texts, test_frac=0.2, seed=42)
        train_samples = expand_text_indices_to_sample_indices(train_idx, n_roles)
        test_samples  = expand_text_indices_to_sample_indices(test_idx,  n_roles)
        assert set(train_samples) & set(test_samples) == set(), \
            "Sample indices (after expansion) still overlap!"

    def test_expand_covers_all_samples(self):
        texts = [f"t{i}" for i in range(10)]
        n_roles = 4
        train_idx, test_idx = text_level_split(texts, test_frac=0.2, seed=42)
        train_s = expand_text_indices_to_sample_indices(train_idx, n_roles)
        test_s  = expand_text_indices_to_sample_indices(test_idx,  n_roles)
        all_samples = sorted(train_s + test_s)
        assert all_samples == list(range(len(texts) * n_roles))


# ---------------------------------------------------------------------------
# Probe training on synthetic data
# ---------------------------------------------------------------------------

class TestProbeSynthetic:
    @pytest.fixture
    def linearly_separable_data(self):
        """4-class linearly separable synthetic data (easy case for probe)."""
        np.random.seed(0)
        n_per_class = 50
        D = 16
        centers = np.eye(4, D)  # orthogonal class centroids
        X, y = [], []
        for ci, center in enumerate(centers):
            X.append(center + 0.01 * np.random.randn(n_per_class, D))
            y.extend([ci] * n_per_class)
        return np.vstack(X), np.array(y)

    def test_probe_trains_without_error(self, linearly_separable_data):
        X, y = linearly_separable_data
        probe = train_probe(X, y, seed=42)
        assert probe is not None

    def test_probe_high_accuracy_on_separable(self, linearly_separable_data):
        X, y = linearly_separable_data
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
        probe = train_probe(X_tr, y_tr, seed=42)
        result = eval_probe(probe, X_te, y_te, label_names=["a","b","c","d"])
        assert result["accuracy"] > 0.95, f"Accuracy too low: {result['accuracy']}"

    def test_eval_returns_required_keys(self, linearly_separable_data):
        X, y = linearly_separable_data
        probe = train_probe(X, y, seed=42)
        result = eval_probe(probe, X[:10], y[:10], label_names=["a","b","c","d"])
        required_keys = [
            "accuracy", "balanced_accuracy", "n_test",
            "chance_accuracy", "confusion_matrix", "label_names"
        ]
        for k in required_keys:
            assert k in result, f"Missing key: {k}"

    def test_confusion_matrix_shape(self, linearly_separable_data):
        X, y = linearly_separable_data
        probe = train_probe(X, y, seed=42)
        result = eval_probe(probe, X, y, label_names=["a","b","c","d"])
        cm = result["confusion_matrix"]
        n_classes = len(np.unique(y))
        assert len(cm) == n_classes
        assert all(len(row) == n_classes for row in cm)

    def test_chance_accuracy_correct(self, linearly_separable_data):
        X, y = linearly_separable_data
        probe = train_probe(X, y, seed=42)
        result = eval_probe(probe, X, y)
        n_classes = len(np.unique(y))
        expected_chance = 1.0 / n_classes
        assert abs(result["chance_accuracy"] - expected_chance) < 1e-9

    def test_label_encoder_roundtrip(self):
        roles = ["system", "user", "cot", "tool"]
        le = make_label_encoder(roles)
        encoded = le.transform(["system", "tool", "user", "cot"])
        decoded = le.inverse_transform(encoded)
        assert list(decoded) == ["system", "tool", "user", "cot"]


# ---------------------------------------------------------------------------
# load_base_texts (basic smoke test without a real file)
# ---------------------------------------------------------------------------

class TestLoadBaseTexts:
    def test_load_from_real_wikitext(self, tmp_path):
        # Create a minimal fake wikitext file
        content = "\n".join([
            "= Section Heading =",  # should be excluded
            "This is a normal sentence with enough characters to pass the filter.",
            "Another valid sentence that exceeds the minimum character requirement.",
            "Short",  # too short, excluded
            "A" * 301,  # too long, excluded
            "Third valid sentence with appropriate length for the experiment.",
        ])
        fpath = tmp_path / "wikitext.txt"
        fpath.write_text(content, encoding="utf-8")

        texts = load_base_texts(str(fpath), n=10, min_len=40, max_len=300, seed=42)
        assert len(texts) == 3  # exactly 3 valid lines
        for t in texts:
            assert 40 <= len(t) <= 300
            assert not t.startswith("=")
