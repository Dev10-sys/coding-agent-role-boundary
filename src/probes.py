# -*- coding: utf-8 -*-
"""
src/probes.py
=============
Linear probe training and evaluation utilities.

Design:
  - LogisticRegression as primary (matches pilot; interpretable weights)
  - Balanced accuracy alongside raw accuracy (important for imbalanced label sets)
  - Confusion matrix always returned for qualitative inspection
  - Probe score distributions per class (for cross-condition transfer analysis)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    seed: int = 42,
) -> LogisticRegression:
    """
    Fit a multinomial logistic regression probe.

    Parameters
    ----------
    X_train  : (N, D) feature matrix
    y_train  : (N,) integer label vector
    C        : regularisation strength (lower = more regularised)
    max_iter : solver iterations
    seed     : random state for reproducibility

    Returns
    -------
    Fitted LogisticRegression instance.
    """
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    return clf


def eval_probe(
    probe: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate a fitted probe on a test set.

    Returns
    -------
    dict with keys:
      accuracy           : fraction correct
      balanced_accuracy  : macro-averaged per-class recall
      n_test             : number of test samples
      chance_accuracy    : 1 / n_classes
      chance_balanced    : 1 / n_classes (same for balanced multi-class)
      confusion_matrix   : list-of-lists (true × predicted)
      label_names        : class name list (if provided)
      predictions        : list of predicted class indices
    """
    y_pred = probe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    n_classes = len(np.unique(y_test))
    chance = 1.0 / n_classes

    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "n_test": int(len(y_test)),
        "chance_accuracy": float(chance),
        "chance_balanced": float(chance),
        "above_chance": bool(acc > chance),
        "confusion_matrix": cm.tolist(),
        "label_names": label_names if label_names is not None else [],
        "predictions": y_pred.tolist(),
    }


def probe_score_distributions(
    probe: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    label_names: List[str],
) -> Dict:
    """
    Compute the softmax probability distribution of each probe class for
    each sample, grouped by true label. Useful for cross-condition transfer
    and for visualising whether the probe separates classes cleanly.

    Returns
    -------
    dict mapping label_name -> list of float probabilities for that class
    across all samples in X whose true label matches.

    Example:
      result["system"] = [0.98, 0.87, 0.95, ...]  # p(system) when true=system
    """
    proba = probe.predict_proba(X)  # (N, n_classes)
    result = {}
    for ci, cls_name in enumerate(label_names):
        class_idx = list(probe.classes_).index(ci) if ci in probe.classes_ else None
        result[cls_name] = {}
        for label_idx, label_name in enumerate(label_names):
            mask = y == label_idx
            if mask.sum() > 0 and class_idx is not None:
                result[cls_name][f"when_true_{label_name}"] = proba[mask, class_idx].tolist()
            else:
                result[cls_name][f"when_true_{label_name}"] = []
    return result


def make_label_encoder(roles: List[str]) -> LabelEncoder:
    """Fit and return a LabelEncoder on the given roles list."""
    le = LabelEncoder()
    le.fit(roles)
    return le
