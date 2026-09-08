"""Error metrics and the significance machinery used to back the claims.

Reporting "model A beat model B by X%" on a single split is not evidence, so
every comparison in this repo goes through one of the tests below.  The default
headline test is a **paired** test on per-observation squared errors: the two
models see identical inputs, so pairing removes the sample-to-sample difficulty
variation and is far more powerful than an unpaired comparison of RMSEs.

Note on the paired-t assumption: squared errors are heavily right-skewed, so
the accompanying Wilcoxon signed-rank test and the bootstrap confidence
interval on the RMSE ratio are reported alongside; agreement between all three
is what the conclusion actually rests on.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import stats


# ----------------------------------------------------------------- metrics
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    ss_res = np.sum((y_true - np.asarray(y_pred)) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def ndei(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Non-dimensional error index: RMSE divided by the target's std.

    The standard yardstick for the Mackey-Glass benchmark, which makes results
    comparable across papers that scale the series differently.
    """
    return rmse(y_true, y_pred) / float(max(np.std(np.asarray(y_true)), 1e-12))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "ndei": ndei(y_true, y_pred),
    }


# ------------------------------------------------------------ significance
def paired_error_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    alternative: str = "less",
) -> Dict[str, float]:
    """Compare model A against model B on per-observation squared errors.

    ``alternative="less"`` tests the directional hypothesis that A's squared
    errors are smaller than B's.
    """
    y_true = np.asarray(y_true)
    err_a = (y_true - np.asarray(pred_a)) ** 2
    err_b = (y_true - np.asarray(pred_b)) ** 2
    diff = err_a - err_b
    t_stat, t_p = stats.ttest_rel(err_a, err_b, alternative=alternative)
    try:
        w_stat, w_p = stats.wilcoxon(err_a, err_b, alternative=alternative)
    except ValueError:  # pragma: no cover - all-zero differences
        w_stat, w_p = np.nan, np.nan
    sd = float(np.std(diff, ddof=1))
    return {
        "mean_sq_err_a": float(err_a.mean()),
        "mean_sq_err_b": float(err_b.mean()),
        "t_stat": float(t_stat),
        "t_pvalue": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pvalue": float(w_p),
        "cohens_d": float(diff.mean() / sd) if sd > 0 else 0.0,
        "n": int(y_true.size),
    }


def bootstrap_rmse_ratio(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap CI for the relative RMSE reduction of A over B.

    Resampling is paired -- the same bootstrap indices are applied to both
    residual vectors -- so the interval reflects uncertainty about the
    *difference*, not about each model separately.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    ra = (y_true - np.asarray(pred_a)) ** 2
    rb = (y_true - np.asarray(pred_b)) ** 2
    n = y_true.size
    ratios = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        ratios[b] = 1.0 - np.sqrt(ra[idx].mean()) / np.sqrt(max(rb[idx].mean(), 1e-16))
    lo, hi = np.percentile(ratios, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = 1.0 - np.sqrt(ra.mean()) / np.sqrt(max(rb.mean(), 1e-16))
    return {
        "rmse_reduction": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": int(n_boot),
    }


def paired_fold_test(
    scores_a: Sequence[float], scores_b: Sequence[float], alternative: str = "less"
) -> Dict[str, float]:
    """Paired t-test across cross-validation folds (lower score is better)."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    t_stat, p = stats.ttest_rel(a, b, alternative=alternative)
    diff = a - b
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "t_stat": float(t_stat),
        "pvalue": float(p),
        "cohens_d": float(diff.mean() / sd) if sd > 0 else 0.0,
        "n_folds": int(a.size),
    }


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Return a boolean mask of hypotheses that survive BH-FDR control.

    The sweep runs 90 configurations; testing each against the same baseline
    without correction would produce false positives by construction.
    """
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    order = np.argsort(p)
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresholds
    mask = np.zeros(n, dtype=bool)
    if passed.any():
        k = np.max(np.nonzero(passed)[0])
        mask[order[: k + 1]] = True
    return mask


def format_pvalue(p: float) -> str:
    """Render a p-value the way it should be reported in a results table."""
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-16:
        return "p < 1e-16"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"
