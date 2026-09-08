"""Rule-base construction and premise initialisation.

Two ways of laying rules over the input space are supported.

**Grid partition** (Jang 1993).  Each of the ``d`` inputs is covered by ``M``
membership functions and the rule base is their full Cartesian product, giving
``R = M**d`` rules.  Interpretable and exhaustive, but the rule count explodes
with dimension.

**Scatter partition.**  Rules are placed at fuzzy c-means cluster prototypes, so
``R`` is chosen directly and is decoupled from ``d``.  Each rule owns a private
membership function per input (``M = R`` with a diagonal rule index), which is
what lets the sweep hit arbitrary rule counts such as 24 or 96.

Both routines return the ``rule_index`` matrix of shape ``[R, d]``: entry
``(r, i)`` is the index of the MF that rule ``r`` uses for input ``i``.
"""

from __future__ import annotations

import itertools
from typing import Optional, Tuple

import numpy as np
import torch

try:  # optional dependency, only used for the reference FCM implementation
    import skfuzzy as fuzz

    _HAS_SKFUZZY = True
except Exception:  # pragma: no cover - exercised only when skfuzzy is absent
    _HAS_SKFUZZY = False


def grid_rule_index(n_inputs: int, n_mfs: int) -> torch.Tensor:
    """Cartesian product rule index of shape ``[n_mfs ** n_inputs, n_inputs]``."""
    combos = list(itertools.product(range(n_mfs), repeat=n_inputs))
    return torch.tensor(combos, dtype=torch.long)


def scatter_rule_index(n_rules: int, n_inputs: int) -> torch.Tensor:
    """Diagonal rule index: rule ``r`` uses MF ``r`` on every input."""
    idx = torch.arange(n_rules, dtype=torch.long).unsqueeze(1)
    return idx.repeat(1, n_inputs)


def grid_premise_init(
    X: np.ndarray, n_mfs: int, overlap: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniformly spread MF centres across the observed range of each input.

    Widths are set to ``overlap * spacing / 2`` so that adjacent MFs cross near
    a membership of ``0.5`` -- the usual "sum to roughly one" starting point
    that keeps the normalisation layer well conditioned.
    """
    X = np.asarray(X, dtype=np.float64)
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo, hi = lo - 0.05 * span, hi + 0.05 * span
    centers = np.stack([np.linspace(lo[i], hi[i], n_mfs) for i in range(X.shape[1])])
    if n_mfs > 1:
        spacing = (hi - lo) / (n_mfs - 1)
    else:
        spacing = hi - lo
    widths = np.repeat((overlap * spacing / 2.0)[:, None], n_mfs, axis=1)
    widths = np.maximum(widths, 1e-3)
    return (
        torch.tensor(centers, dtype=torch.float32),
        torch.tensor(widths, dtype=torch.float32),
    )


def _fcm_numpy(
    X: np.ndarray, n_clusters: int, m: float = 2.0, n_iter: int = 150,
    tol: float = 1e-6, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Plain-NumPy fuzzy c-means, used when scikit-fuzzy is unavailable."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    U = rng.random((n_clusters, n))
    U /= U.sum(axis=0, keepdims=True)
    centers = np.zeros((n_clusters, X.shape[1]))
    for _ in range(n_iter):
        Um = U ** m
        centers = (Um @ X) / np.maximum(Um.sum(axis=1, keepdims=True), 1e-12)
        dist = np.linalg.norm(X[None, :, :] - centers[:, None, :], axis=2)
        dist = np.maximum(dist, 1e-10)
        inv = dist ** (-2.0 / (m - 1.0))
        U_new = inv / inv.sum(axis=0, keepdims=True)
        if np.linalg.norm(U_new - U) < tol:
            U = U_new
            break
        U = U_new
    return centers, U


def fcm_premise_init(
    X: np.ndarray,
    n_rules: int,
    m: float = 2.0,
    seed: int = 0,
    width_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Place one MF per rule per input at fuzzy c-means prototypes.

    Returns ``(centers, widths)`` of shape ``[d, n_rules]``.  The width of the
    MF for rule ``r`` on input ``i`` is the membership-weighted standard
    deviation of that input within cluster ``r``, which gives narrow MFs in
    dense regions and broad ones in sparse regions.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    n_rules = int(min(n_rules, max(2, n)))
    if _HAS_SKFUZZY:
        centers, U, *_ = fuzz.cluster.cmeans(
            X.T, c=n_rules, m=m, error=1e-6, maxiter=150, seed=seed
        )
    else:  # pragma: no cover
        centers, U = _fcm_numpy(X, n_rules, m=m, seed=seed)

    Um = U ** m  # [R, N]
    denom = np.maximum(Um.sum(axis=1, keepdims=True), 1e-12)
    var = (Um @ (X ** 2)) / denom - ((Um @ X) / denom) ** 2
    widths = width_scale * np.sqrt(np.maximum(var, 1e-6))
    global_scale = X.std(axis=0, keepdims=True) / max(n_rules, 1) ** (1.0 / max(d, 1))
    widths = np.maximum(widths, 0.25 * np.maximum(global_scale, 1e-3))
    return (
        torch.tensor(centers.T, dtype=torch.float32),
        torch.tensor(widths.T, dtype=torch.float32),
    )


def build_partition(
    X: np.ndarray,
    mode: str = "grid",
    n_mfs: int = 2,
    n_rules: Optional[int] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(rule_index, centers, widths)`` for the requested partition."""
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]
    if mode == "grid":
        centers, widths = grid_premise_init(X, n_mfs)
        return grid_rule_index(d, n_mfs), centers, widths
    if mode == "scatter":
        if n_rules is None:
            raise ValueError("scatter partition needs n_rules")
        centers, widths = fcm_premise_init(X, n_rules, seed=seed)
        return scatter_rule_index(centers.shape[1], d), centers, widths
    raise ValueError(f"unknown partition mode {mode!r}")


def rules_for_grid(n_inputs: int, n_mfs: int) -> int:
    return n_mfs ** n_inputs
