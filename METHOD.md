# Method notes

Design decisions that are not obvious from the code, and the reasoning behind them.

## Why the T-norm runs in log space

Layer 2 computes a rule's firing strength as a product of `d` membership degrees. For a Gaussian MF, a point three widths from the centre contributes about `1e-2`; four inputs like that give `1e-8`, and in float32 a point far from *every* MF gives exactly `0.0`. Layer 3 then computes `0 / 0` and the model returns NaN — the failure mode most hand-rolled ANFIS implementations hit as soon as the rule count grows or the premise drifts during training.

Since

```
w_r = Π_i μ_i   ⟹   log w_r = Σ_i log μ_i   ⟹   w̄ = softmax(log w)
```

the log-space path is algebraically identical for the product T-norm, is immune to underflow, and reuses PyTorch's numerically stable `softmax` (which subtracts the row max internally). `test_log_space_firing_survives_far_out_inputs` pins this behaviour with inputs 500 σ away from every MF.

The min T-norm is implemented the same way. It is available but not the default: `min` has zero gradient with respect to every input except the argmin, so premise learning is much slower.

## Why premise parameters are reparameterised

Widths, slopes and plateau lengths must stay positive. Three options:

1. **Clip after every step.** Introduces a non-differentiable boundary the optimiser repeatedly slams into, and Adam's moment estimates keep pointing into the wall.
2. **Penalise negativity.** Adds a hyperparameter and only discourages the failure.
3. **Reparameterise:** store `ρ` freely and use `σ = softplus(ρ) + ε`.

Option 3 is used throughout. Beyond positivity it removes the `σ → 0` attractor: a membership function collapsing onto a single training point drives the loss down and the gradient is happy to go there, but under a softplus parameterisation reaching `σ = 0` requires `ρ → −∞`, which gradient descent will not do in finite steps. The `+ε` floor makes this exact rather than asymptotic.

The same trick enforces *ordering* where the shape needs it: `dsig` stores an inflection-point gap as `softplus(ρ_gap)` so `c₂ > c₁` always holds and the difference of sigmoids stays a single bump; `trapezoidal` stores its plateau length the same way so `b ≤ c` cannot be violated.

## Why the least-squares solve is ridge-damped

At `R = 256` rules with `d = 4` inputs, a first-order Sugeno model has `256 × 5 = 1,280` consequent coefficients. Cross-validation on 1,000 samples gives 800 training rows, so `AᵀA` is singular and the unregularised normal equations have infinitely many exact solutions — most with enormous coefficient norms that generalise terribly.

The solver damps by `λ · mean(diag(AᵀA)) · I`, scaling the penalty with the problem so `λ` means the same thing across configurations. Cholesky is tried first (it is the fastest route for a symmetric positive-definite system and *fails* informatively when the matrix is not), falling back to `lstsq` and then to an explicit pseudo-inverse.

A consequence worth stating plainly: the high-rule configurations in the sweep are regularised, so their poor showing is not an artefact of an unregularised solve blowing up. They are genuinely overfitting.

## Why interpretability is measured, not asserted

"Fuzzy models are interpretable" is a claim about a model instance, not a model class. A 256-rule ANFIS with overlapping Gaussians is not interpretable in any useful sense. Three diagnostics make this measurable:

* **Effective rules** — the perplexity `exp(H)` of the mean normalised firing distribution. If ten rules carry all the mass, perplexity is about ten regardless of how many rules the model nominally has.
* **`rules_for_90pct`** — sort rules by contribution to output magnitude, count how many are needed to reach 90%. This is the size of the rule base a human actually has to read.
* **Dead rules** — rules never firing above threshold on held-out data. Parameters that cost capacity and explain nothing.

The sweep shows effective rules growing roughly as the square root of nominal rules, so the readable regime tops out well before 256.

## Why the significance testing is structured the way it is

* **Paired, not unpaired.** Both models see identical inputs; pairing removes sample-difficulty variance and is far more powerful.
* **Per-observation squared errors, not fold RMSEs, for the headline test.** 500 paired observations rather than 5 paired folds.
* **Wilcoxon alongside the *t*-test.** Squared errors are strongly right-skewed, so the *t*-test's normality assumption is shaky; agreement between the two is what the conclusion rests on.
* **Paired bootstrap for the effect size.** A *p*-value says the difference is not zero. The bootstrap CI on relative RMSE reduction says how big it is, which is the part anyone actually cares about.
* **Benjamini-Hochberg across the sweep.** Ninety tests against one baseline at α = 0.05 would produce false positives by construction.
* **Fold-level tests in the sweep.** With 5-fold CV every observation is used exactly once out-of-fold, so per-observation pairing across configurations would reuse the same data; fold-level pairing is the honest unit there.

## Why standardisation is fitted inside each fold

The fuzzy partition is initialised *from the data* — grid partitioning spreads MF centres across the observed range, and FCM places them at cluster prototypes. Fitting the scaler on the full dataset would leak test-fold range and cluster structure into the premise parameters before training begins. `cross_validate` therefore fits a fresh `Standardizer` on each training fold, and `test_standardizer_never_sees_the_test_fold` guards it.

## Parameter accounting

For grid partitioning with `d` inputs, `M` MFs per input and a first-order consequent:

```
premise      = d × M × (params per MF)
consequent   = M^d × (d + 1)
```

The headline configuration is `d = 4`, `M = 2`, Gaussian (2 params/MF):

```
premise      = 4 × 2 × 2  =  16
consequent   = 16 × 5     =  80
total        =               96
```

The reference MLP, `4 → 64 → 32 → 1`:

```
(4×64 + 64) + (64×32 + 32) + (32×1 + 1) = 320 + 2080 + 33 = 2433
```

Both are asserted in `tests/test_model.py::test_headline_parameter_counts`, so the comparison cannot silently drift as the code changes.
