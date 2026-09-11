# -*- coding: utf-8 -*-
"""
src/data.py
===========
Deterministic text loading and splitting utilities.

Key design decisions vs the pilot:
  - Text-level train/test split: all four role-variants of the same base text
    are guaranteed to appear in the same partition (train OR test), never split
    across partitions. This prevents data leakage via shared base text.
  - Seed is always required; no silent defaults in functions that affect splits.
"""

import random
from typing import List, Tuple


def load_base_texts(
    path: str,
    n: int,
    min_len: int = 40,
    max_len: int = 300,
    seed: int = 42,
) -> List[str]:
    """
    Load up to `n` base texts from a plain-text file (one line per entry).
    Lines starting with '=' (WikiText headings) are excluded.

    Parameters
    ----------
    path    : absolute path to the text file
    n       : maximum number of texts to return
    min_len : minimum character length of accepted lines
    max_len : maximum character length of accepted lines
    seed    : random seed used to shuffle before selection (reproducible)

    Returns
    -------
    List of at most n stripped text strings.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    candidates = [
        l.strip()
        for l in lines
        if l.strip() and not l.strip().startswith("=")
        and min_len <= len(l.strip()) <= max_len
    ]

    # Shuffle deterministically before trimming to n so the selection is
    # seed-reproducible regardless of insertion order in the source file.
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n]


def text_level_split(
    base_texts: List[str],
    test_frac: float = 0.20,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Return (train_text_indices, test_text_indices) such that NO base text index
    appears in both sets.

    When this function is used to construct (features, labels) for probe training:
        - expand train_text_indices into sample indices (multiply by n_roles)
        - expand test_text_indices into sample indices
    This guarantees zero base-text leakage between train and test splits.

    Parameters
    ----------
    base_texts : list of raw base text strings
    test_frac  : fraction of texts to hold out for testing
    seed       : random seed

    Returns
    -------
    (train_indices, test_indices) as lists of integers into base_texts
    """
    n = len(base_texts)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_test = max(1, round(n * test_frac))
    test_indices = sorted(indices[:n_test])
    train_indices = sorted(indices[n_test:])
    return train_indices, test_indices


def expand_text_indices_to_sample_indices(
    text_indices: List[int],
    n_roles: int,
) -> List[int]:
    """
    Given text-level indices and the number of role variants per text,
    return the flat sample indices into the full (texts × roles) dataset.

    Example: text_index=3, n_roles=4 → sample indices [12, 13, 14, 15]
    """
    sample_indices = []
    for ti in text_indices:
        for ri in range(n_roles):
            sample_indices.append(ti * n_roles + ri)
    return sorted(sample_indices)
