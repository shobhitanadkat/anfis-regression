# Neuro-Fuzzy System for Explainable Regression and Decision Support

[![CI](https://github.com/your-username/anfis-regression/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/anfis-regression/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-64%20passed-brightgreen.svg)](tests/)

A five-layer **Adaptive Neuro-Fuzzy Inference System (ANFIS)** implemented from scratch in PyTorch, with Jang's hybrid least-squares / gradient-descent optimiser, five differentiable membership-function families, and a 90-configuration cross-validated study of the interpretability–accuracy trade-off.

The point of the project: a model whose fitted parameters *are* the explanation. On the standard Mackey-Glass benchmark this implementation reaches **lower held-out error than a 64-32 MLP while using 96 parameters instead of 2,433**, and every prediction decomposes into a handful of IF-THEN rules over named variables.

```
IF x(t-18) is high AND x(t-12) is high AND x(t-6) is low AND x(t) is low
THEN x(t+6) = -0.369*x(t-18) - 0.490*x(t-12) + 0.836*x(t-6) - 0.354*x(t) - 1.632
     (fires on 14.6% of the input mass, accounts for 19.1% of the output)
```

---

## Results

### Mackey-Glass chaotic time series

Jang's protocol: predict `x(t+6)` from `x(t-18), x(t-12), x(t-6), x(t)`; 500 training and 500 held-out points; mean ± sd over 5 seeds. ANFIS is 2 Gaussian MFs per input on a grid partition → 16 rules, 96 parameters.

| Model | Params | RMSE (mean ± sd) | MAE | R² | NDEI |
|---|---:|---:|---:|---:|---:|
| **ANFIS (hybrid)** | **96** | **0.00201 ± 0.00003** | 0.00144 | 0.99992 | 0.0089 |
| ANFIS (gradient only) | 96 | 0.01547 ± 0.00039 | 0.01212 | 0.99528 | 0.0687 |
| ANFIS (LSE only, premise frozen) | 96 | 0.00313 ± 0.00001 | 0.00233 | 0.99981 | 0.0139 |
| Linear regression (OLS) | 5 | 0.09569 ± 0.00000 | 0.07919 | 0.81961 | 0.4247 |
| Ridge regression | 5 | 0.09570 ± 0.00002 | 0.07922 | 0.81957 | 0.4248 |
| Polynomial (deg 2) + ridge | 15 | 0.03047 ± 0.00002 | 0.02364 | 0.98171 | 0.1352 |
| MLP 64-32 | 2,433 | 0.00434 ± 0.00017 | 0.00323 | 0.99963 | 0.0193 |

Paired tests on per-observation held-out squared errors, with a paired bootstrap CI (5,000 resamples) on the relative RMSE reduction:

| Comparison | RMSE reduction | 95% CI | Paired *t*-test |
|---|---:|---:|---|
| ANFIS vs linear regression | **97.9%** | [97.7%, 98.1%] | *p* < 1e-16 |
| ANFIS vs ridge | 97.9% | [97.7%, 98.1%] | *p* < 1e-16 |
| ANFIS vs degree-2 polynomial | 93.4% | [92.5%, 94.2%] | *p* < 1e-16 |
| ANFIS vs MLP 64-32 (25× the parameters) | **53.5%** | [48.3%, 58.3%] | *p* < 1e-16 |
| Hybrid vs gradient-descent-only ANFIS | 87.0% | [85.5%, 88.4%] | *p* < 1e-16 |
| Hybrid vs least-squares-only ANFIS | 35.6% | [30.2%, 40.8%] | *p* < 0.001 |

NDEI 0.0089 with a 16-rule model is in the range Jang reports for the same architecture and protocol, which is the main external check that the implementation is faithful rather than merely convergent.

The last two rows are the ablations that matter: solving the consequents in closed form is worth an 87% error reduction over training the same 96 parameters with Adam alone, and letting gradients reshape the premise is worth a further 36% over leaving the initial fuzzy partition in place.

### Second benchmark (`nonlinear_mixed`, 4 inputs)

A surface built so that roughly two thirds of the target variance is linearly explainable — OLS is a real baseline here, not a straw man — with the rest in an interaction, a saturating response and a soft threshold.

| Model | Params | RMSE | R² | Reduction vs ANFIS |
|---|---:|---:|---:|---:|
| **ANFIS (hybrid)** | **96** | **0.16037 ± 0.00868** | 0.99707 | — |
| Linear regression | 5 | 1.72650 ± 0.03801 | 0.66022 | 90.7% (*p* < 1e-16) |
| Polynomial (deg 2) + ridge | 15 | 0.56556 ± 0.01313 | 0.96349 | 71.6% (*p* < 1e-16) |
| MLP 64-32 | 2,433 | 0.22174 ± 0.01908 | 0.99433 | 27.3% (*p* < 0.001) |

### The 90-configuration sweep

5 membership families × 9 rule counts (4 … 256) × 2 consequent orders, each with 5-fold CV — 450 model fits. Rules are placed by fuzzy c-means scatter partitioning, which decouples the rule count from the input dimension so counts like 24 and 48 are reachable (a grid partition can only produce `M**d`).

![Interpretability / accuracy trade-off](results/figures/mackey_glass_tradeoff.png)

The curve is the headline finding: **accuracy peaks at 32 rules and then degrades**, while the number of rules a human would have to read keeps climbing. At 256 nominal rules the firing perplexity is 142 — the model has stopped being a rule base and become a soft nearest-neighbour smoother, and its CV RMSE is 8× worse than the 32-rule model.

| Rules | Best CV RMSE | Params | Effective rules (firing perplexity) |
|---:|---:|---:|---:|
| 4 | 0.0141 | 52 | 3.6 |
| 8 | 0.0050 | 104 | 7.4 |
| 16 | 0.0036 | 208 | 14.7 |
| 24 | 0.0033 | 312 | 21.9 |
| **32** | **0.0031** | 416 | 28.4 |
| 48 | 0.0042 | 624 | 41.3 |
| 64 | 0.0046 | 832 | 53.3 |
| 128 | 0.0082 | 1,664 | 92.3 |
| 256 | 0.0249 | 3,328 | 142.2 |

Other findings from the sweep:

* **87 of 90** configurations beat linear regression after Benjamini-Hochberg FDR control across all 90 tests. Testing 90 hypotheses at α = 0.05 without correction would be expected to manufacture false positives, so the correction is applied rather than assumed away.
* **13 of 90** beat the 2,433-parameter MLP.
* First-order (affine) consequents average 0.0112 RMSE against 0.0405 for constant consequents — but zeroth-order rules read as `THEN y = 0.83`, which is a genuinely simpler explanation. That is the trade-off in its cleanest form.
* Gaussian and generalised-bell MFs dominate; the piecewise-linear families (triangular, trapezoidal) lose roughly 2× accuracy, plausibly because their zero-gradient regions stall premise learning.

| Figure | |
|---|---|
| ![Learned membership functions](results/figures/mackey_glass_membership_functions.png) | Learned linguistic terms per input, over the data histogram |
| ![Rule activation surfaces](results/figures/mackey_glass_rule_activation.png) | Where each rule is responsible — sharp, near-disjoint regions |
| ![Pareto frontier](results/figures/mackey_glass_pareto.png) | Accuracy vs model size across all 90 configurations |
| ![Convergence](results/figures/mackey_glass_convergence.png) | Hybrid vs gradient-only convergence at an equal epoch budget |

---

## Install

```bash
git clone https://github.com/<you>/anfis-regression.git
cd anfis-regression
pip install -e ".[dev]"
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0 (CPU is fine — the whole study above runs in about 10 minutes on a laptop). `scikit-fuzzy` is optional; a NumPy fuzzy c-means implementation is used when it is absent.

## Quickstart

```python
from anfis import ANFIS, TrainConfig, fit, evaluate, get_dataset, extract_rules
from anfis.data import Standardizer

ds = get_dataset("mackey_glass")
train, test = ds.split

scaler = Standardizer().fit(ds.X[train], ds.y[train])
Xtr, ytr = scaler.transform(ds.X[train], ds.y[train])
Xte, yte = scaler.transform(ds.X[test], ds.y[test])

# 2 Gaussian MFs per input on 4 inputs -> 16 rules, 96 parameters
model = ANFIS.from_data(Xtr, mf="gaussian", n_mfs=2, partition="grid", order=1)
fit(model, Xtr, ytr, TrainConfig(epochs=200, method="hybrid"))

print(evaluate(model, Xte, yte))          # {'rmse': ..., 'r2': ..., 'ndei': ...}
print(extract_rules(model, Xtr, ds.feature_names).head())
```

Explain a single prediction:

```python
from anfis import local_explanation
print(local_explanation(model, Xte[0], ds.feature_names, top_k=3))
#   rule                                    IF  firing_strength  contribution  pct_of_prediction
#      3   x(t-18) is low AND x(t-12) is ...            0.412         0.271               41.6
```

## Reproducing everything

```bash
bash scripts/reproduce_all.sh          # ~15 min on CPU: headline runs, sweep, figures
```

or step by step:

```bash
python -m experiments.run_headline --dataset mackey_glass --seeds 5
python -m experiments.run_sweep     --dataset mackey_glass --folds 5    # 90 configs
python -m experiments.make_figures  --dataset mackey_glass
```

The sweep can be sharded by membership family and merged, which is what the committed results were produced with:

```bash
for M in gaussian gbell dsig triangular trapezoidal; do
  python -m experiments.run_sweep --dataset mackey_glass --mfs $M --out sweep_part_$M
done
python scripts/merge_sweep.py        # concatenates and re-applies BH-FDR across all 90
```

---

## How it works

### The five layers

| Layer | Name | Operation | Shape |
|---|---|---|---|
| 1 | Fuzzification | `μ_ij = MF_ij(x_i)` | `[N, d, M]` |
| 2 | Rule firing (T-norm) | `w_r = Π_i μ_{i, idx(r,i)}` | `[N, R]` |
| 3 | Normalisation | `w̄_r = w_r / Σ_s w_s` | `[N, R]` |
| 4 | Consequent | `w̄_r · (pᵣᵀx + qᵣ)` | `[N, R]` |
| 5 | Aggregation | `y = Σ_r` | `[N]` |

Layer 2 runs **in log space**. A product of `d` memberships underflows to exactly zero once an input sits in the tail of every MF, at which point layer 3 divides `0/0` and the model returns NaN. Summing `log μ` and applying a softmax across rules is algebraically identical for the product T-norm, cannot underflow, and hands layer 3 an exact Jacobian. This is what makes 256-rule models trainable at all.

### Membership families

| Family | Params / MF | Notes |
|---|---:|---|
| `gaussian` | 2 | Smooth; the default in Jang's papers |
| `gbell` | 3 | Generalised bell; adjustable shoulder sharpness |
| `dsig` | 4 | Difference of sigmoids; asymmetric, ordered inflection points |
| `triangular` | 3 | Piecewise linear, differentiable a.e.; `smooth=True` gives a C^∞ surrogate |
| `trapezoidal` | 4 | Piecewise linear with a plateau; same smooth option |

Every positivity constraint (widths, slopes, plateau lengths) is enforced by a `softplus` reparameterisation rather than by clipping. Premise parameters then live on an unbounded manifold where plain Adam can move freely, and the degenerate `σ → 0` spikes that make naive ANFIS implementations diverge are unreachable by construction.

### Jang's hybrid rule

The output is **linear in the consequent parameters** and **non-linear in the premise parameters**. Each epoch:

1. **Forward pass** — freeze the premise, build the layer-4 design matrix `A`, and solve `θ* = argmin ‖Aθ − y‖² + λ‖θ‖²` in closed form by Cholesky factorisation of the damped normal equations. Half the parameter vector jumps to its global optimum instead of crawling there.
2. **Backward pass** — freeze `θ*` and take one gradient step on the premise only, back-propagating through layers 1–3.

Ridge damping is not decoration: at 256 rules a first-order model has 1,280 consequent coefficients against 800 training rows, so the unregularised system is underdetermined. A recursive-least-squares solver with a forgetting factor (`lse="rlse"`) is also provided for mini-batch and streaming use.

### Interpretability diagnostics

Beyond the rule table, three numbers quantify how readable a fitted model actually is:

* **`effective_rules`** — `exp(H)` of the mean firing distribution: how many rules the model genuinely leans on, usually far below the nominal count.
* **`rules_for_90pct`** — how many rules a reader must go through to account for 90% of the output magnitude. The headline 16-rule model needs 12.
* **`dead_rules`** — rules that never fire above threshold anywhere in the test set; pure parameter waste.

---

## Repository layout

```
anfis/
  membership.py    five differentiable MF families, softplus-reparameterised
  model.py         the five-layer Sugeno network, log-space T-norm, design matrix
  hybrid.py        Jang's hybrid rule: ridge LSE + RLSE + premise optimiser
  partition.py     grid and fuzzy-c-means scatter rule-base construction
  train.py         training loop, early stopping, K-fold cross-validation
  baselines.py     OLS, ridge, degree-2 polynomial, 64-32 MLP
  metrics.py       RMSE/MAE/R²/NDEI, paired tests, bootstrap CIs, BH-FDR
  interpret.py     rule extraction, importance, coverage, local explanations
  viz.py           MFs, rule activation surfaces, Pareto, heatmaps
experiments/
  run_headline.py  ANFIS vs four baselines + two ablations, with significance tests
  run_sweep.py     the 90-configuration cross-validated study
  make_figures.py  regenerates every figure and the rule dump
tests/             64 tests: numerics, invariants, and the claims above
results/           committed CSV/JSON/Markdown outputs and figures
```

## Tests

```bash
pytest -q          # 64 passed
```

The suite checks the properties the results depend on, not just that the code runs: that layer 3 sums to one, that `y == A @ vec(θ)` exactly (the identity the hybrid rule relies on), that the log-space T-norm survives inputs 500 σ from every MF, that RLSE converges to the batch solution, that the LSE solve is optimal for the current partition (no random perturbation of the consequents lowers the loss), that the backward pass moves premise parameters while leaving consequents untouched, and that the headline parameter counts really are 96 and 2,433.

## Limitations

* **Grid partitioning scales as `M^d`.** Beyond ~6 inputs, use `partition="scatter"`. Rule-count explosion is a property of the model class, not of this implementation.
* **Interpretability is not free at scale.** As the sweep shows, a 256-rule ANFIS is not meaningfully more explainable than an MLP — the readable regime here is roughly 8–32 rules.
* **Linguistic labels are automatic, not domain-validated.** Terms are assigned by ranking MF centres; a domain expert should confirm that "high `x(t-12)`" means what the label implies.
* **Benchmarks are synthetic or semi-synthetic.** `load_csv` is provided for real tabular data, but the numbers reported here come from generated benchmarks with known structure.
* Reported centres are in standardised units, since models are fitted on standardised inputs.

## References

1. J.-S. R. Jang, "ANFIS: Adaptive-Network-Based Fuzzy Inference System", *IEEE Transactions on Systems, Man, and Cybernetics* 23(3), 1993.
2. T. Takagi and M. Sugeno, "Fuzzy identification of systems and its applications to modeling and control", *IEEE TSMC* 15(1), 1985.
3. M. C. Mackey and L. Glass, "Oscillation and chaos in physiological control systems", *Science* 197, 1977.
4. Y. Benjamini and Y. Hochberg, "Controlling the false discovery rate", *JRSS B* 57(1), 1995.

## License

MIT — see [LICENSE](LICENSE).
