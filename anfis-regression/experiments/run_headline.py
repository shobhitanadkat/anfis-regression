"""Headline comparison: 16-rule ANFIS against linear, ridge, polynomial and MLP.

Runs the same held-out protocol over several seeds, then tests the ANFIS
against each baseline with a paired test on per-observation squared errors and
a paired bootstrap CI for the relative RMSE reduction.  Also runs the two
optimiser ablations (``gradient`` and ``lse_only``) so the contribution of the
least-squares half of Jang's rule is visible rather than assumed.

Usage
-----
    python -m experiments.run_headline --dataset mackey_glass --seeds 5
    python -m experiments.run_headline --dataset nonlinear_mixed --mf gbell
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anfis import (  # noqa: E402
    ANFIS,
    LinearBaseline,
    MLPBaseline,
    PolynomialBaseline,
    RidgeBaseline,
    Standardizer,
    TrainConfig,
    all_metrics,
    bootstrap_rmse_ratio,
    count_mlp_params,
    evaluate,
    fit,
    get_dataset,
    paired_error_test,
    predict,
)
from anfis.metrics import format_pvalue  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def split_indices(ds, test_fraction: float, seed: int):
    """Use the dataset's canonical split when it defines one, else shuffle."""
    if ds.split is not None:
        return ds.split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(ds.n_samples)
    cut = int((1 - test_fraction) * ds.n_samples)
    return idx[:cut], idx[cut:]


def run_seed(ds, args, seed: int) -> Dict:
    tr, te = split_indices(ds, args.test_fraction, seed)
    scaler = Standardizer().fit(ds.X[tr], ds.y[tr])
    Xtr, ytr = scaler.transform(ds.X[tr], ds.y[tr])
    Xte, _ = scaler.transform(ds.X[te], ds.y[te])
    y_true = ds.y[te]

    preds: Dict[str, np.ndarray] = {}
    info: Dict[str, Dict] = {}

    # --- ANFIS, three optimisation regimes -------------------------------
    for method in ("hybrid", "gradient", "lse_only"):
        model = ANFIS.from_data(
            Xtr, mf=args.mf, n_mfs=args.n_mfs, partition="grid", order=1, seed=seed
        )
        res = fit(
            model,
            Xtr,
            ytr,
            TrainConfig(
                epochs=args.epochs,
                method=method,
                premise_lr=args.lr,
                ridge=args.ridge,
                patience=args.patience,
                seed=seed,
            ),
        )
        key = "anfis" if method == "hybrid" else f"anfis_{method}"
        preds[key] = scaler.inverse_y(predict(model, Xte))
        info[key] = {
            "n_params": model.n_params,
            "n_rules": model.n_rules,
            "n_premise_params": model.n_premise_params,
            "n_consequent_params": model.n_consequent_params,
            "epochs_run": len(res.history["val_mse"]),
            "best_epoch": res.best_epoch,
            "seconds": round(res.seconds, 3),
            "train_rmse": evaluate(model, Xtr, ytr)["rmse"],
            **{k: round(v, 4) for k, v in model.firing_stats(
                __import__("torch").as_tensor(Xte, dtype=__import__("torch").float32)
            ).items()},
        }

    # --- baselines --------------------------------------------------------
    lin = LinearBaseline().fit(ds.X[tr], ds.y[tr])
    preds["linear"] = lin.predict(ds.X[te])
    info["linear"] = {"n_params": lin.n_params}

    rdg = RidgeBaseline().fit(ds.X[tr], ds.y[tr], seed=seed)
    preds["ridge"] = rdg.predict(ds.X[te])
    info["ridge"] = {"n_params": rdg.n_params, "alpha": rdg.alpha_}

    poly = PolynomialBaseline().fit(ds.X[tr], ds.y[tr], seed=seed)
    preds["poly2"] = poly.predict(ds.X[te])
    info["poly2"] = {"n_params": poly.n_params}

    mlp = MLPBaseline(hidden=tuple(args.mlp_hidden), seed=seed).fit(Xtr, ytr)
    preds["mlp"] = scaler.inverse_y(mlp.predict(Xte))
    info["mlp"] = {"n_params": mlp.n_params}

    metrics = {name: all_metrics(y_true, p) for name, p in preds.items()}
    for name in info:
        info[name].update(metrics[name])

    tests = {}
    for other in ("linear", "ridge", "poly2", "mlp"):
        t = paired_error_test(y_true, preds["anfis"], preds[other], alternative="less")
        b = bootstrap_rmse_ratio(y_true, preds["anfis"], preds[other],
                                 n_boot=args.n_boot, seed=seed)
        tests[f"anfis_vs_{other}"] = {**t, **b}
    for ablation in ("anfis_gradient", "anfis_lse_only"):
        t = paired_error_test(y_true, preds["anfis"], preds[ablation], alternative="less")
        b = bootstrap_rmse_ratio(y_true, preds["anfis"], preds[ablation],
                                 n_boot=args.n_boot, seed=seed)
        tests[f"anfis_vs_{ablation}"] = {**t, **b}

    return {"seed": int(seed), "models": info, "tests": tests,
            "predictions": {k: v.tolist() for k, v in preds.items()},
            "y_true": y_true.tolist()}


def aggregate(runs: List[Dict]) -> Dict:
    names = list(runs[0]["models"])
    agg = {}
    for name in names:
        rmses = [r["models"][name]["rmse"] for r in runs]
        agg[name] = {
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "ndei_mean": float(np.mean([r["models"][name]["ndei"] for r in runs])),
            "r2_mean": float(np.mean([r["models"][name]["r2"] for r in runs])),
            "mae_mean": float(np.mean([r["models"][name]["mae"] for r in runs])),
            "n_params": runs[0]["models"][name]["n_params"],
        }
    comps = {}
    for key in runs[0]["tests"]:
        pvals = [r["tests"][key]["t_pvalue"] for r in runs]
        reductions = [r["tests"][key].get("rmse_reduction", np.nan) for r in runs]
        comps[key] = {
            "max_t_pvalue": float(np.nanmax(pvals)),
            "median_t_pvalue": float(np.nanmedian(pvals)),
            "max_wilcoxon_pvalue": float(np.nanmax(
                [r["tests"][key].get("wilcoxon_pvalue", np.nan) for r in runs])),
            "mean_rmse_reduction": float(np.nanmean(reductions)),
            "ci_low": float(np.nanmean([r["tests"][key].get("ci_low", np.nan) for r in runs])),
            "ci_high": float(np.nanmean([r["tests"][key].get("ci_high", np.nan) for r in runs])),
        }
    return {"per_model": agg, "comparisons": comps}


def markdown_table(agg: Dict, ds_name: str) -> str:
    rows = [
        "| Model | Params | RMSE (mean ± sd) | MAE | R² | NDEI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    label = {
        "anfis": "**ANFIS (hybrid)**",
        "anfis_gradient": "ANFIS (gradient only)",
        "anfis_lse_only": "ANFIS (LSE only, premise fixed)",
        "linear": "Linear regression (OLS)",
        "ridge": "Ridge regression",
        "poly2": "Polynomial (deg 2) + ridge",
        "mlp": "MLP 64-32",
    }
    order = ["anfis", "anfis_gradient", "anfis_lse_only", "linear", "ridge", "poly2", "mlp"]
    for name in order:
        if name not in agg["per_model"]:
            continue
        m = agg["per_model"][name]
        rows.append(
            f"| {label.get(name, name)} | {m['n_params']:,} | "
            f"{m['rmse_mean']:.5f} ± {m['rmse_std']:.5f} | {m['mae_mean']:.5f} | "
            f"{m['r2_mean']:.5f} | {m['ndei_mean']:.4f} |"
        )
    rows.append("")
    rows.append(f"Paired significance tests on held-out squared errors ({ds_name}):")
    rows.append("")
    rows.append("| Comparison | RMSE reduction | 95% CI | Paired t-test |")
    rows.append("|---|---:|---:|---|")
    for key, c in agg["comparisons"].items():
        red = c["mean_rmse_reduction"]
        ci = (f"[{c['ci_low']:.1%}, {c['ci_high']:.1%}]"
              if np.isfinite(c["ci_low"]) else "n/a")
        rows.append(
            f"| {key.replace('_', ' ')} | "
            f"{red:.1%} | {ci} | {format_pvalue(c['max_t_pvalue'])} |"
        )
    return "\n".join(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="mackey_glass")
    p.add_argument("--mf", default="gaussian")
    p.add_argument("--n-mfs", type=int, default=2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--test-fraction", type=float, default=0.3)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--mlp-hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ds = get_dataset(args.dataset)
    print(f"dataset: {ds}")
    print(f"reference MLP{tuple(args.mlp_hidden)} on {ds.n_features} inputs = "
          f"{count_mlp_params(ds.n_features, args.mlp_hidden):,} parameters")

    runs = [run_seed(ds, args, seed) for seed in range(args.seeds)]
    for r in runs:
        print(f"  seed {r['seed']}: anfis rmse={r['models']['anfis']['rmse']:.5f}  "
              f"linear={r['models']['linear']['rmse']:.5f}  "
              f"mlp={r['models']['mlp']['rmse']:.5f}")

    agg = aggregate(runs)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stem = args.out or f"headline_{args.dataset}"
    payload = {
        "dataset": args.dataset,
        "config": vars(args),
        "aggregate": agg,
        "runs": [{k: v for k, v in r.items() if k != "predictions"} for r in runs],
        "predictions_seed0": runs[0]["predictions"],
        "y_true_seed0": runs[0]["y_true"],
    }
    with open(os.path.join(RESULTS_DIR, f"{stem}.json"), "w") as f:
        json.dump(payload, f, indent=2)
    table = markdown_table(agg, args.dataset)
    with open(os.path.join(RESULTS_DIR, f"{stem}.md"), "w") as f:
        f.write(f"# Headline results: {args.dataset}\n\n{table}\n")
    print("\n" + table)
    print(f"\nwrote {stem}.json and {stem}.md to results/")


if __name__ == "__main__":
    main()
