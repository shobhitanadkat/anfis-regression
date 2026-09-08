"""Cross-validated sweep over the ANFIS design space.

The grid is 5 membership families x 9 rule counts x 2 consequent orders = **90
configurations**, each evaluated with K-fold cross-validation (450 fits at the
default K=5).  Rules are placed by fuzzy c-means scatter partitioning, which
decouples the rule count from the input dimension and lets the sweep hit
counts like 24 or 48 that a grid partition (``M**d``) can never produce.

For every configuration the sweep records accuracy (RMSE / MAE / R2 / NDEI),
size (premise and consequent parameters separately) and three interpretability
diagnostics:

``effective_rules``
    ``exp(H)`` of the mean firing distribution -- how many rules the model
    actually leans on.
``dead_rules``
    Rules that never fire above threshold anywhere in the test fold.
``rules_for_90pct``
    How many rules a reader must go through to account for 90% of the output.

Configurations are also tested against the linear baseline on the same folds,
with Benjamini-Hochberg FDR control across all 90 tests -- without correction,
testing 90 hypotheses at alpha=0.05 is expected to produce false positives.

Usage
-----
    python -m experiments.run_sweep --dataset nonlinear_mixed --folds 5
    python -m experiments.run_sweep --quick          # 2 folds, fewer epochs
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anfis import (  # noqa: E402
    LinearBaseline,
    MLPBaseline,
    Standardizer,
    TrainConfig,
    cross_validate,
    get_dataset,
    rmse,
)
from anfis.membership import MF_FAMILIES  # noqa: E402
from anfis.metrics import benjamini_hochberg, paired_fold_test  # noqa: E402
from anfis.train import kfold_indices  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
RULE_COUNTS = [4, 8, 16, 24, 32, 48, 64, 128, 256]
ORDERS = [0, 1]


def baseline_fold_scores(ds, folds, mlp_hidden=(64, 32)) -> Dict[str, List[float]]:
    """Per-fold RMSE for the reference models, on identical folds."""
    lin, mlp = [], []
    for tr, te in folds:
        lin.append(rmse(ds.y[te], LinearBaseline().fit(ds.X[tr], ds.y[tr]).predict(ds.X[te])))
        sc = Standardizer().fit(ds.X[tr], ds.y[tr])
        Xtr, ytr = sc.transform(ds.X[tr], ds.y[tr])
        Xte, _ = sc.transform(ds.X[te], ds.y[te])
        m = MLPBaseline(hidden=mlp_hidden, seed=0).fit(Xtr, ytr)
        mlp.append(rmse(ds.y[te], sc.inverse_y(m.predict(Xte))))
    return {"linear": lin, "mlp": mlp}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="nonlinear_mixed")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rules", type=int, nargs="+", default=RULE_COUNTS)
    p.add_argument("--mfs", nargs="+", default=MF_FAMILIES)
    p.add_argument("--orders", type=int, nargs="+", default=ORDERS)
    p.add_argument("--quick", action="store_true", help="2 folds and 30 epochs")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.quick:
        args.folds, args.epochs, args.patience = 2, 30, 10

    ds = get_dataset(args.dataset)
    configs = list(itertools.product(args.mfs, args.rules, args.orders))
    print(f"dataset: {ds}")
    print(f"sweep: {len(args.mfs)} membership families x {len(args.rules)} rule counts "
          f"x {len(args.orders)} orders = {len(configs)} configurations, "
          f"{args.folds}-fold CV = {len(configs) * args.folds} fits")

    folds = kfold_indices(ds.n_samples, args.folds, seed=args.seed)
    base = baseline_fold_scores(ds, folds)
    print(f"baselines: linear RMSE {np.mean(base['linear']):.4f}, "
          f"MLP RMSE {np.mean(base['mlp']):.4f}")

    rows: List[Dict] = []
    t0 = time.time()
    for i, (mf, n_rules, order) in enumerate(configs, 1):
        cv = cross_validate(
            ds.X,
            ds.y,
            model_kwargs={
                "mf": mf,
                "partition": "scatter",
                "n_rules": n_rules,
                "order": order,
            },
            train_config=TrainConfig(
                epochs=args.epochs,
                method="hybrid",
                premise_lr=args.lr,
                ridge=args.ridge,
                patience=args.patience,
                seed=args.seed,
            ),
            k=args.folds,
            seed=args.seed,
        )
        fold_rmse = [f["rmse"] for f in cv["per_fold"]]
        vs_lin = paired_fold_test(fold_rmse, base["linear"], alternative="less")
        vs_mlp = paired_fold_test(fold_rmse, base["mlp"], alternative="less")
        row = {
            "mf": mf,
            "order": order,
            **cv["summary"],
            # firing_stats also reports n_rules as a float; the nominal count wins
            "n_rules": int(n_rules),
            "rmse_reduction_vs_linear": 1.0 - np.mean(fold_rmse) / np.mean(base["linear"]),
            "rmse_reduction_vs_mlp": 1.0 - np.mean(fold_rmse) / np.mean(base["mlp"]),
            "pvalue_vs_linear": vs_lin["pvalue"],
            "pvalue_vs_mlp": vs_mlp["pvalue"],
            "fold_rmse": fold_rmse,
        }
        rows.append(row)
        elapsed = time.time() - t0
        print(
            f"[{i:3d}/{len(configs)}] {mf:12s} R={n_rules:3d} order={order} "
            f"params={row['n_params']:5d} rmse={row['rmse_mean']:.4f}±{row['rmse_std']:.4f} "
            f"eff_rules={row['effective_rules']:6.1f} "
            f"vs_lin={row['rmse_reduction_vs_linear']:+.1%} "
            f"[{elapsed/i*len(configs)/60:.0f} min est]"
        )

    pvals = [r["pvalue_vs_linear"] for r in rows]
    passed = benjamini_hochberg(pvals, alpha=0.05)
    for r, ok in zip(rows, passed):
        r["significant_vs_linear_bh"] = bool(ok)

    import pandas as pd

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stem = args.out or f"sweep_{args.dataset}"
    df.drop(columns=["fold_rmse"]).to_csv(os.path.join(RESULTS_DIR, f"{stem}.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, f"{stem}_meta.json"), "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "config": vars(args),
                "baselines": {
                    "linear_rmse_mean": float(np.mean(base["linear"])),
                    "mlp_rmse_mean": float(np.mean(base["mlp"])),
                    "fold_rmse": {k: list(map(float, v)) for k, v in base.items()},
                },
                "n_configs": len(rows),
                "n_significant_bh": int(passed.sum()),
                "total_minutes": round((time.time() - t0) / 60, 2),
            },
            f,
            indent=2,
        )

    best = df.loc[df["rmse_mean"].idxmin()]
    frugal = df[df["rmse_mean"] <= best["rmse_mean"] * 1.05].nsmallest(1, "n_params")
    print("\n" + "=" * 78)
    print(f"best RMSE      : {best['mf']} R={int(best['n_rules'])} order={int(best['order'])} "
          f"-> {best['rmse_mean']:.4f} with {int(best['n_params'])} params")
    if len(frugal):
        fr = frugal.iloc[0]
        print(f"smallest within 5% of best: {fr['mf']} R={int(fr['n_rules'])} "
              f"order={int(fr['order'])} -> {fr['rmse_mean']:.4f} "
              f"with {int(fr['n_params'])} params")
    print(f"configurations beating linear after BH-FDR: {int(passed.sum())}/{len(rows)}")
    print(f"wrote {stem}.csv to results/  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
