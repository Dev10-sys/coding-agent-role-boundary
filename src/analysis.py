# -*- coding: utf-8 -*-
"""
src/analysis.py
===============
Statistical utilities for behavioural comparisons.

All functions return raw counts alongside statistics so callers can report
numerator/denominator clearly. This module does not round or over-interpret
p-values; callers are responsible for appropriate language.

Fisher exact: use scipy.stats.fisher_exact (two-sided, exact).
Wilson CI: standard Wilson score interval for proportions.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


def fisher_exact_test(
    k1: int, n1: int,
    k2: int, n2: int,
    alternative: str = "two-sided",
) -> Dict:
    """
    Two-sample comparison of proportions using Fisher's exact test.

    Parameters
    ----------
    k1, n1 : successes and total in group 1
    k2, n2 : successes and total in group 2
    alternative : "two-sided", "less", or "greater"

    Returns
    -------
    dict with:
      p1, p2           : observed proportions
      odds_ratio       : sample odds ratio (nan if any cell is zero)
      p_value          : exact p-value
      alternative      : test direction
      k1, n1, k2, n2   : raw counts (echo)
      interpretation   : plain-language description

    Note
    ----
    A small-n p < 0.05 is described as "suggestive" here; callers should
    use appropriate language. p ≥ 0.05 is "inconclusive at this sample size".
    """
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    oddsratio, p_value = stats.fisher_exact(table, alternative=alternative)

    p1 = k1 / n1 if n1 > 0 else float("nan")
    p2 = k2 / n2 if n2 > 0 else float("nan")

    # Verbal interpretation (conservative phrasing)
    if p_value < 0.05:
        interp = f"p={p_value:.4f} — suggestive (small n; interpret cautiously)"
    elif p_value < 0.10:
        interp = f"p={p_value:.4f} — inconclusive trend"
    else:
        interp = f"p={p_value:.4f} — inconclusive at this sample size"

    return {
        "k1": k1, "n1": n1, "p1": float(p1),
        "k2": k2, "n2": n2, "p2": float(p2),
        "odds_ratio": float(oddsratio),
        "p_value": float(p_value),
        "alternative": alternative,
        "interpretation": interp,
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score interval for a proportion k/n.

    Parameters
    ----------
    k : number of successes
    n : total trials
    z : normal quantile (default 1.96 for 95% CI)

    Returns
    -------
    (lower, upper) as floats in [0, 1]
    """
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half_width = z * ((p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    lower = max(0.0, centre - half_width)
    upper = min(1.0, centre + half_width)
    return (float(lower), float(upper))


def proportion_summary(k: int, n: int) -> Dict:
    """
    Full summary for a single proportion: count, rate, 95% Wilson CI.
    """
    p = k / n if n > 0 else float("nan")
    lo, hi = wilson_ci(k, n)
    return {
        "k": k, "n": n,
        "proportion": float(p),
        "ci95_lower": lo,
        "ci95_upper": hi,
    }


def pairwise_comparisons(
    conditions: List[str],
    ks: List[int],
    ns: List[int],
    reference: str,
) -> List[Dict]:
    """
    Compare each non-reference condition against the reference condition
    using Fisher exact (two-sided). Returns a list of result dicts.

    Parameters
    ----------
    conditions : list of condition names
    ks         : list of success counts (aligned with conditions)
    ns         : list of trial counts  (aligned with conditions)
    reference  : name of the reference condition (e.g. "no_warning")
    """
    ref_idx = conditions.index(reference)
    k_ref, n_ref = ks[ref_idx], ns[ref_idx]

    results = []
    for i, cond in enumerate(conditions):
        if cond == reference:
            continue
        result = fisher_exact_test(k_ref, n_ref, ks[i], ns[i])
        result["condition_a"] = reference
        result["condition_b"] = cond
        results.append(result)
    return results


def effect_size_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for two proportions.
    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    """
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
