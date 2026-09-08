"""Figures for the accuracy / interpretability trade-off.

Every function takes a fitted object, returns a Matplotlib ``Figure`` and does
not call ``plt.show``, so the same code works in a notebook and in the headless
experiment scripts.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .interpret import _label_map, rule_importance
from .model import ANFIS

PALETTE = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#b83280", "#4a5568"]


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


@torch.no_grad()
def plot_membership_functions(
    model: ANFIS,
    X: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    n_points: int = 400,
) -> "matplotlib.figure.Figure":
    """One panel per input showing its learned linguistic terms."""
    X = np.asarray(X)
    d = model.n_inputs
    names = list(feature_names or [f"x{i+1}" for i in range(d)])
    labels = _label_map(model)
    ncols = min(d, 4)
    nrows = int(np.ceil(d / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), squeeze=False)

    for i in range(d):
        ax = axes[i // ncols][i % ncols]
        lo, hi = X[:, i].min(), X[:, i].max()
        pad = 0.1 * (hi - lo + 1e-9)
        grid = np.linspace(lo - pad, hi + pad, n_points)
        probe = np.tile(X.mean(axis=0), (n_points, 1))
        probe[:, i] = grid
        mu = model.layer1_memberships(torch.as_tensor(probe, dtype=torch.float32))
        mu = mu[:, i, :].cpu().numpy()
        for j in range(mu.shape[1]):
            ax.plot(grid, mu[:, j], lw=2, color=PALETTE[j % len(PALETTE)],
                    label=str(labels[i, j]))
        ax.hist(X[:, i], bins=40, density=True, alpha=0.12, color="grey")
        ax.set_title(names[i], fontsize=10)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel("value")
        ax.set_ylabel("membership")
        if mu.shape[1] <= 6:
            ax.legend(fontsize=7, frameon=False, ncol=2)
        _style(ax)

    for k in range(d, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"Learned membership functions ({model.mf_name})", fontsize=12)
    fig.tight_layout()
    return fig


@torch.no_grad()
def plot_rule_activation_surface(
    model: ANFIS,
    X: np.ndarray,
    dims: tuple = (0, 1),
    n_rules_shown: int = 6,
    resolution: int = 120,
    feature_names: Optional[Sequence[str]] = None,
) -> "matplotlib.figure.Figure":
    """Normalised firing strength of the most important rules over a 2-D slice.

    The remaining inputs are held at their median, so each panel answers: "for
    which region of the input space is this rule responsible?"  Sharp, disjoint
    patches mean the partition is readable; overlapping mush means the model has
    stopped being a fuzzy system and become a smoother.
    """
    X = np.asarray(X)
    i, j = dims
    names = list(feature_names or [f"x{k+1}" for k in range(model.n_inputs)])
    imp = rule_importance(model, X)
    top = np.argsort(imp)[::-1][:n_rules_shown]

    gi = np.linspace(X[:, i].min(), X[:, i].max(), resolution)
    gj = np.linspace(X[:, j].min(), X[:, j].max(), resolution)
    GI, GJ = np.meshgrid(gi, gj)
    probe = np.tile(np.median(X, axis=0), (resolution * resolution, 1))
    probe[:, i] = GI.ravel()
    probe[:, j] = GJ.ravel()
    wbar = model.normalised_firing(torch.as_tensor(probe, dtype=torch.float32)).cpu().numpy()

    ncols = min(n_rules_shown, 3)
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows), squeeze=False)
    for k, r in enumerate(top):
        ax = axes[k // ncols][k % ncols]
        surf = wbar[:, r].reshape(resolution, resolution)
        im = ax.contourf(GI, GJ, surf, levels=12, cmap="viridis")
        ax.set_title(f"rule {r} ({imp[r]:.1%} of output)", fontsize=9)
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
    for k in range(len(top), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("Rule activation surfaces (other inputs at median)", fontsize=12)
    fig.tight_layout()
    return fig


def plot_pareto(
    df,
    x: str = "n_params",
    y: str = "rmse_mean",
    hue: str = "mf",
    annotate: Optional[str] = "n_rules",
) -> "matplotlib.figure.Figure":
    """Accuracy against model size, with the Pareto frontier highlighted."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for k, (name, grp) in enumerate(df.groupby(hue)):
        ax.scatter(grp[x], grp[y], s=34, alpha=0.75, label=str(name),
                   color=PALETTE[k % len(PALETTE)], edgecolor="white", linewidth=0.5)

    pts = df[[x, y]].to_numpy()
    order = np.argsort(pts[:, 0])
    frontier, best = [], np.inf
    for idx in order:
        if pts[idx, 1] < best:
            best = pts[idx, 1]
            frontier.append(pts[idx])
    frontier = np.array(frontier)
    ax.step(frontier[:, 0], frontier[:, 1], where="post", color="black",
            lw=1.4, alpha=0.8, label="Pareto frontier")

    if annotate and annotate in df.columns:
        for k, (_, row) in enumerate(df.nsmallest(3, y).iterrows()):
            ax.annotate(
                f"{int(row[annotate])} rules",
                (row[x], row[y]),
                textcoords="offset points",
                xytext=(8, 10 + 14 * k),
                fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="grey"),
            )
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel("cross-validated RMSE")
    ax.set_title("Accuracy vs model size")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_rmse_heatmap(
    df, index: str = "mf", columns: str = "n_rules", values: str = "rmse_mean"
) -> "matplotlib.figure.Figure":
    """Grid of CV RMSE across membership family and rule count."""
    pivot = df.pivot_table(index=index, columns=columns, values=values, aggfunc="min")
    size = (1.0 + 0.72 * len(pivot.columns), 0.75 + 0.62 * len(pivot.index))
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(pivot.to_numpy(), cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    for a in range(pivot.shape[0]):
        for b in range(pivot.shape[1]):
            v = pivot.to_numpy()[a, b]
            if np.isfinite(v):
                shade = "white" if v > np.nanmedian(pivot.to_numpy()) else "black"
                ax.text(b, a, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color=shade)
    ax.set_xlabel(columns)
    ax.set_ylabel(index)
    ax.set_title(f"{values} by {index} and {columns}")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return fig


def plot_interpretability_tradeoff(df) -> "matplotlib.figure.Figure":
    """RMSE and 'rules a human must read' as the rule base grows."""
    agg = df.groupby("n_rules").agg(
        rmse=("rmse_mean", "min"),
        effective=("effective_rules", "mean"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax1.plot(agg["n_rules"], agg["rmse"], "o-", color=PALETTE[0], lw=2, label="best CV RMSE")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("rules in the rule base (log scale)")
    ax1.set_ylabel("cross-validated RMSE", color=PALETTE[0])
    ax1.tick_params(axis="y", labelcolor=PALETTE[0])
    _style(ax1)

    ax2 = ax1.twinx()
    ax2.plot(agg["n_rules"], agg["effective"], "s--", color=PALETTE[1], lw=2,
             label="effective rules (firing perplexity)")
    ax2.set_ylabel("effective rules actually used", color=PALETTE[1])
    ax2.tick_params(axis="y", labelcolor=PALETTE[1])
    ax2.spines["top"].set_visible(False)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, frameon=False, loc="upper center")
    ax1.set_title("Interpretability / accuracy trade-off")
    fig.tight_layout()
    return fig


def plot_training_curves(histories: Dict[str, Dict[str, list]]) -> "matplotlib.figure.Figure":
    """Validation MSE per epoch for each optimisation regime."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for k, (name, hist) in enumerate(histories.items()):
        vals = hist["val_mse"]
        ax.plot(np.arange(1, len(vals) + 1), vals, lw=2,
                color=PALETTE[k % len(PALETTE)], label=name)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MSE (log scale)")
    ax.set_title("Convergence by optimiser")
    ax.legend(fontsize=9, frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_predictions(
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    n_show: int = 200,
    title: str = "Predictions on held-out data",
) -> "matplotlib.figure.Figure":
    """Overlay of held-out predictions plus a residual panel."""
    n = min(n_show, len(y_true))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.0, 5.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax1.plot(y_true[:n], color="black", lw=1.8, label="actual")
    for k, (name, p) in enumerate(preds.items()):
        ax1.plot(p[:n], lw=1.3, alpha=0.9, color=PALETTE[k % len(PALETTE)], label=name)
    ax1.set_ylabel("target")
    ax1.set_title(title)
    ax1.legend(fontsize=8, frameon=False, ncol=len(preds) + 1)
    _style(ax1)

    for k, (name, p) in enumerate(preds.items()):
        ax2.plot(np.asarray(y_true[:n]) - np.asarray(p[:n]), lw=1.1,
                 color=PALETTE[k % len(PALETTE)], label=name)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("residual")
    ax2.set_xlabel("test sample")
    _style(ax2)
    fig.tight_layout()
    return fig


def save(fig, path: str, dpi: int = 160) -> str:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
