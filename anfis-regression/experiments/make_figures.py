"""Regenerate every figure and the extracted rule base from saved results.

    python -m experiments.make_figures --dataset mackey_glass

Reads ``results/headline_<dataset>.json`` and ``results/sweep_<dataset>.csv``
when they exist, refits the headline model to produce the membership-function
and rule-activation plots, and writes PNGs to ``results/figures/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anfis import (  # noqa: E402
    ANFIS,
    Standardizer,
    TrainConfig,
    coverage_report,
    extract_rules,
    fit,
    get_dataset,
    local_explanation,
    rules_to_text,
    viz,  # noqa: E402
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(RESULTS, "figures")


def fit_headline(ds, mf: str, n_mfs: int, epochs: int, seed: int = 0):
    tr, te = ds.split if ds.split is not None else (
        np.arange(int(0.7 * ds.n_samples)), np.arange(int(0.7 * ds.n_samples), ds.n_samples)
    )
    scaler = Standardizer().fit(ds.X[tr], ds.y[tr])
    Xtr, ytr = scaler.transform(ds.X[tr], ds.y[tr])
    Xte, _ = scaler.transform(ds.X[te], ds.y[te])
    model = ANFIS.from_data(Xtr, mf=mf, n_mfs=n_mfs, partition="grid", order=1, seed=seed)
    histories = {}
    res = fit(model, Xtr, ytr, TrainConfig(epochs=epochs, method="hybrid", seed=seed))
    histories["hybrid (LSE + gradient)"] = res.history
    for method in ("gradient",):
        alt = ANFIS.from_data(Xtr, mf=mf, n_mfs=n_mfs, partition="grid", order=1, seed=seed)
        histories["gradient only"] = fit(
            alt, Xtr, ytr, TrainConfig(epochs=epochs, method=method, seed=seed)
        ).history
    return model, scaler, (tr, te), (Xtr, ytr, Xte), histories


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="mackey_glass")
    p.add_argument("--mf", default="gaussian")
    p.add_argument("--n-mfs", type=int, default=2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--dims", type=int, nargs=2, default=[0, 3])
    args = p.parse_args()

    os.makedirs(FIGDIR, exist_ok=True)
    ds = get_dataset(args.dataset)
    model, scaler, (tr, te), (Xtr, ytr, Xte), histories = fit_headline(
        ds, args.mf, args.n_mfs, args.epochs
    )
    written = []

    written.append(viz.save(
        viz.plot_membership_functions(model, Xtr, ds.feature_names),
        os.path.join(FIGDIR, f"{args.dataset}_membership_functions.png"),
    ))
    written.append(viz.save(
        viz.plot_rule_activation_surface(
            model, Xtr, dims=tuple(args.dims), feature_names=ds.feature_names
        ),
        os.path.join(FIGDIR, f"{args.dataset}_rule_activation.png"),
    ))
    written.append(viz.save(
        viz.plot_training_curves(histories),
        os.path.join(FIGDIR, f"{args.dataset}_convergence.png"),
    ))

    head_path = os.path.join(RESULTS, f"headline_{args.dataset}.json")
    if os.path.exists(head_path):
        with open(head_path) as f:
            payload = json.load(f)
        y_true = np.array(payload["y_true_seed0"])
        preds = {k: np.array(v) for k, v in payload["predictions_seed0"].items()
                 if k in ("anfis", "linear", "mlp")}
        label = {"anfis": "ANFIS (96 params)", "linear": "linear", "mlp": "MLP 64-32"}
        preds = {label.get(k, k): v for k, v in preds.items()}
        written.append(viz.save(
            viz.plot_predictions(y_true, preds, title=f"Held-out predictions: {args.dataset}"),
            os.path.join(FIGDIR, f"{args.dataset}_predictions.png"),
        ))

    sweep_path = os.path.join(RESULTS, f"sweep_{args.dataset}.csv")
    if not os.path.exists(sweep_path):
        candidates = [f for f in os.listdir(RESULTS)
                      if f.startswith("sweep_") and f.endswith(".csv")]
        sweep_path = os.path.join(RESULTS, candidates[0]) if candidates else ""
    if sweep_path and os.path.exists(sweep_path):
        import pandas as pd

        df = pd.read_csv(sweep_path)
        tag = os.path.basename(sweep_path).replace("sweep_", "").replace(".csv", "")
        written.append(viz.save(viz.plot_pareto(df), os.path.join(FIGDIR, f"{tag}_pareto.png")))
        written.append(viz.save(viz.plot_rmse_heatmap(df),
                                os.path.join(FIGDIR, f"{tag}_heatmap.png")))
        written.append(viz.save(
            viz.plot_interpretability_tradeoff(df),
            os.path.join(FIGDIR, f"{tag}_tradeoff.png"),
        ))

    # ---- the rule base itself, as text -----------------------------------
    rules = extract_rules(model, Xtr, ds.feature_names, ds.target_name)
    rules.to_csv(os.path.join(RESULTS, f"{args.dataset}_rules.csv"), index=False)
    cov = coverage_report(model, Xtr)
    expl = local_explanation(model, Xte[0], ds.feature_names, ds.target_name)

    lines = [
        f"# Extracted rule base: {args.dataset}",
        "",
        f"Model: {model.n_rules} rules, {model.n_params} parameters "
        f"({model.n_premise_params} premise "
        f"+ {model.n_consequent_params} consequent), "
        f"{args.mf} membership functions.",
        "",
        "Rules needed to account for 90% of the output: "
        f"{int(cov['rules_for_90pct'])} of {model.n_rules}.",
        f"Effective rules (firing perplexity): {cov['effective_rules']:.1f}. "
        f"Dead rules: {int(cov['dead_rules'])}.",
        "",
        "## Top rules by output share",
        "",
        "```",
        rules_to_text(rules, max_rules=8),
        "```",
        "",
        "## Local explanation for one held-out prediction",
        "",
        f"Prediction (standardised): {expl.attrs['prediction']:.4f}",
        "",
        expl.to_markdown(index=False),
    ]
    rules_md = os.path.join(RESULTS, f"{args.dataset}_rules.md")
    with open(rules_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    written.append(rules_md)

    print("wrote:")
    for w in written:
        print("  " + os.path.relpath(w, ROOT))


if __name__ == "__main__":
    main()
