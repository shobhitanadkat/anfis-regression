# Committed results

Everything here is produced by `bash scripts/reproduce_all.sh` and is checked in so the
README's numbers can be verified without re-running anything.

| File | What it is |
|---|---|
| `headline_mackey_glass.{json,md}` | ANFIS vs 4 baselines + 2 ablations on the Mackey-Glass benchmark, 5 seeds, with paired tests and bootstrap CIs. The JSON also stores seed-0 predictions so the figures can be redrawn without refitting. |
| `headline_nonlinear_mixed.{json,md}` | The same protocol on the second benchmark. |
| `sweep_mackey_glass.csv` | One row per configuration for all 90 (5 membership families x 9 rule counts x 2 consequent orders), each 5-fold cross-validated. |
| `sweep_mackey_glass_meta.json` | Sweep provenance: baselines, shard list, FDR count, runtime. |
| `mackey_glass_rules.{csv,md}` | The extracted rule base of the headline 16-rule model, plus a worked local explanation. |
| `figures/` | Every figure referenced in the README. |

## Columns in the sweep CSV

`mf`, `n_rules`, `order` identify the configuration. `rmse_mean` / `rmse_std` (and the
same for `mae`, `r2`, `ndei`) are across CV folds. `n_params` is the total trainable
count. `effective_rules`, `dead_rules`, `active_rules`, `firing_entropy` and
`mean_max_activation` are the interpretability diagnostics averaged over folds.
`rmse_reduction_vs_linear` / `_vs_mlp` are relative improvements over the baselines
evaluated on identical folds, with `pvalue_vs_linear` / `_vs_mlp` from paired
across-fold t-tests and `significant_vs_linear_bh` the Benjamini-Hochberg decision
at alpha = 0.05 applied across all 90 tests at once.
