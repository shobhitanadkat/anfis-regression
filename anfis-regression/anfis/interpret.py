"""Turning a trained ANFIS back into something a human can read.

The point of a neuro-fuzzy model is that the fitted parameters *are* the
explanation.  This module converts them into three artefacts:

``extract_rules``
    A pandas table of IF-THEN rules with linguistic antecedents, the consequent
    coefficients, and how often each rule actually fires.
``rule_importance``
    Contribution of each rule to the model output, so a 256-rule model can be
    truncated to the handful of rules that carry it.
``local_explanation``
    Per-prediction attribution: which rules fired for *this* input and what each
    contributed.  This is the property MLPs cannot offer at any parameter count.

Linguistic labels come from ranking each input's MF centres; with ``M`` terms
the labels are drawn from a scale of the appropriate length (``low/high`` for
two, ``low/medium/high`` for three, and so on).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .model import ANFIS

LABEL_SCALES: Dict[int, List[str]] = {
    1: ["any"],
    2: ["low", "high"],
    3: ["low", "medium", "high"],
    4: ["very low", "low", "high", "very high"],
    5: ["very low", "low", "medium", "high", "very high"],
    6: ["very low", "low", "med-low", "med-high", "high", "very high"],
    7: ["very low", "low", "med-low", "medium", "med-high", "high", "very high"],
}


def term_labels(n_terms: int) -> List[str]:
    """Ordered linguistic labels for ``n_terms`` membership functions."""
    if n_terms in LABEL_SCALES:
        return LABEL_SCALES[n_terms]
    return [f"T{i+1}" for i in range(n_terms)]


def _label_map(model: ANFIS) -> np.ndarray:
    """``[d, M]`` array of labels, assigned by the rank of each MF centre."""
    centers = model.mf.centers().cpu().numpy()  # [d, M]
    d, m = centers.shape
    labels = np.empty((d, m), dtype=object)
    scale = term_labels(m)
    for i in range(d):
        order = np.argsort(centers[i])
        for rank, j in enumerate(order):
            labels[i, j] = scale[rank]
    return labels


@torch.no_grad()
def extract_rules(
    model: ANFIS,
    X: Optional[np.ndarray] = None,
    feature_names: Optional[Sequence[str]] = None,
    target_name: str = "y",
    top_k: Optional[int] = None,
) -> "object":
    """Return the rule base as a ``pandas.DataFrame``.

    Columns: the linguistic antecedent, the consequent expression, mean and max
    normalised firing strength on ``X``, and the share of the output the rule
    accounts for.
    """
    import pandas as pd

    names = list(feature_names or [f"x{i+1}" for i in range(model.n_inputs)])
    labels = _label_map(model)
    centers = model.mf.centers().cpu().numpy()
    rule_index = model.rule_index.cpu().numpy()
    coef = model.consequent.detach().cpu().numpy()

    if X is not None:
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        wbar = model.normalised_firing(Xt).cpu().numpy()
        _, internals = model(Xt, return_internals=True)
        contrib = np.abs(internals["weighted"].cpu().numpy()).mean(axis=0)
        share = contrib / max(contrib.sum(), 1e-12)
        mean_fire, max_fire = wbar.mean(axis=0), wbar.max(axis=0)
    else:
        mean_fire = max_fire = share = np.full(model.n_rules, np.nan)

    rows = []
    for r in range(model.n_rules):
        antecedent = " AND ".join(
            f"{names[i]} is {labels[i, rule_index[r, i]]} (~{centers[i, rule_index[r, i]]:.2f})"
            for i in range(model.n_inputs)
        )
        if model.order == 1:
            terms = " + ".join(f"{coef[r, i]:+.3f}*{names[i]}" for i in range(model.n_inputs))
            consequent = f"{target_name} = {terms} {coef[r, -1]:+.3f}"
        else:
            consequent = f"{target_name} = {coef[r, 0]:+.3f}"
        rows.append(
            {
                "rule": r,
                "IF": antecedent,
                "THEN": consequent,
                "mean_firing": float(mean_fire[r]),
                "max_firing": float(max_fire[r]),
                "output_share": float(share[r]),
            }
        )

    df = pd.DataFrame(rows).sort_values("output_share", ascending=False, ignore_index=True)
    return df.head(top_k) if top_k else df


@torch.no_grad()
def rule_importance(model: ANFIS, X: np.ndarray) -> np.ndarray:
    """Share of the output magnitude attributable to each rule."""
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    _, internals = model(Xt, return_internals=True)
    contrib = np.abs(internals["weighted"].cpu().numpy()).mean(axis=0)
    return contrib / max(contrib.sum(), 1e-12)


@torch.no_grad()
def coverage_report(model: ANFIS, X: np.ndarray, threshold: float = 0.05) -> Dict[str, float]:
    """How concentrated the rule base is -- the interpretability side of the trade-off.

    ``rules_for_90pct`` is the number of rules needed to explain 90% of the
    output magnitude, i.e. the size of the rule base a human would actually
    have to read.
    """
    imp = np.sort(rule_importance(model, X))[::-1]
    cum = np.cumsum(imp)
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    stats = model.firing_stats(Xt)
    return {
        **stats,
        "rules_for_90pct": float(np.searchsorted(cum, 0.90) + 1),
        "rules_above_threshold": float((imp > threshold).sum()),
        "top_rule_share": float(imp[0]),
    }


@torch.no_grad()
def local_explanation(
    model: ANFIS,
    x: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    target_name: str = "y",
    top_k: int = 3,
) -> "object":
    """Explain one prediction as a weighted vote of the rules that fired."""
    import pandas as pd

    x = np.asarray(x, dtype=np.float64).reshape(1, -1)
    xt = torch.as_tensor(x, dtype=torch.float32)
    pred, internals = model(xt, return_internals=True)
    wbar = internals["wbar"].cpu().numpy()[0]
    weighted = internals["weighted"].cpu().numpy()[0]

    names = list(feature_names or [f"x{i+1}" for i in range(model.n_inputs)])
    labels = _label_map(model)
    rule_index = model.rule_index.cpu().numpy()
    order = np.argsort(wbar)[::-1][:top_k]

    rows = []
    for r in order:
        antecedent = " AND ".join(
            f"{names[i]} is {labels[i, rule_index[r, i]]}" for i in range(model.n_inputs)
        )
        rows.append(
            {
                "rule": int(r),
                "IF": antecedent,
                "firing_strength": float(wbar[r]),
                "contribution": float(weighted[r]),
                "pct_of_prediction": float(weighted[r] / max(abs(float(pred)), 1e-12) * 100),
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["prediction"] = float(pred)
    df.attrs["target_name"] = target_name
    return df


def rules_to_text(df, max_rules: int = 10) -> str:
    """Pretty-print an extracted rule table as plain IF-THEN sentences."""
    lines = []
    for _, row in df.head(max_rules).iterrows():
        lines.append(
            f"R{int(row['rule']):>3}  IF {row['IF']}\n"
            f"      THEN {row['THEN']}\n"
            f"      (fires on {row['mean_firing']:.1%} of mass, "
            f"{row['output_share']:.1%} of output)"
        )
    return "\n".join(lines)
