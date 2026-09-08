"""Merge sweep shards into one table and recompute FDR across all of them.

``run_sweep.py`` can be sharded by membership family so each shard fits in a
short job.  Benjamini-Hochberg control has to be applied over the *whole*
family of tests, though, so it is redone here rather than kept per shard.

    python scripts/merge_sweep.py --pattern "sweep_part_*.csv" --out sweep_mackey_glass.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anfis.metrics import benjamini_hochberg  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pattern", default="sweep_part_*.csv")
    p.add_argument("--out", default="sweep_mackey_glass.csv")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--keep-parts", action="store_true")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(RESULTS, args.pattern)))
    if not paths:
        raise SystemExit(f"no shards matched {args.pattern!r} in {RESULTS}")

    df = pd.concat([pd.read_csv(f) for f in paths], ignore_index=True)
    df = df.sort_values(["mf", "n_rules", "order"], ignore_index=True)
    df["significant_vs_linear_bh"] = benjamini_hochberg(
        df["pvalue_vs_linear"].to_numpy(), alpha=args.alpha
    )
    df.to_csv(os.path.join(RESULTS, args.out), index=False)

    metas = sorted(glob.glob(os.path.join(RESULTS, args.pattern.replace(".csv", "_meta.json"))))
    if metas:
        merged = [json.load(open(m)) for m in metas]
        summary = {
            "shards": [os.path.basename(m) for m in metas],
            "n_configs": len(df),
            "n_significant_bh": int(df["significant_vs_linear_bh"].sum()),
            "linear_rmse_mean": merged[0]["baselines"]["linear_rmse_mean"],
            "mlp_rmse_mean": merged[0]["baselines"]["mlp_rmse_mean"],
            "total_minutes": round(sum(m["total_minutes"] for m in merged), 2),
        }
        with open(os.path.join(RESULTS, args.out.replace(".csv", "_meta.json")), "w") as f:
            json.dump(summary, f, indent=2)

    if not args.keep_parts:
        for f in paths + metas:
            os.remove(f)

    print(f"merged {len(paths)} shards -> {args.out}: {len(df)} configurations, "
          f"{int(df['significant_vs_linear_bh'].sum())} beat linear after BH-FDR")


if __name__ == "__main__":
    main()
