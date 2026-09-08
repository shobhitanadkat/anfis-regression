# Headline results: nonlinear_mixed

| Model | Params | RMSE (mean ± sd) | MAE | R² | NDEI |
|---|---:|---:|---:|---:|---:|
| **ANFIS (hybrid)** | 96 | 0.16037 ± 0.00868 | 0.12986 | 0.99707 | 0.0541 |
| ANFIS (gradient only) | 96 | 0.43525 ± 0.03426 | 0.34588 | 0.97819 | 0.1470 |
| ANFIS (LSE only, premise fixed) | 96 | 0.22141 ± 0.01069 | 0.17631 | 0.99439 | 0.0747 |
| Linear regression (OLS) | 5 | 1.72650 ± 0.03801 | 1.31911 | 0.66022 | 0.5826 |
| Ridge regression | 5 | 1.72517 ± 0.03871 | 1.31702 | 0.66082 | 0.5821 |
| Polynomial (deg 2) + ridge | 15 | 0.56556 ± 0.01313 | 0.45907 | 0.96349 | 0.1909 |
| MLP 64-32 | 2,433 | 0.22174 ± 0.01908 | 0.17467 | 0.99433 | 0.0749 |

Paired significance tests on held-out squared errors (nonlinear_mixed):

| Comparison | RMSE reduction | 95% CI | Paired t-test |
|---|---:|---:|---|
| anfis vs linear | 90.7% | [89.6%, 91.7%] | p < 1e-16 |
| anfis vs ridge | 90.7% | [89.6%, 91.7%] | p < 1e-16 |
| anfis vs poly2 | 71.6% | [68.7%, 74.3%] | p < 1e-16 |
| anfis vs mlp | 27.3% | [20.6%, 33.5%] | p < 0.001 |
| anfis vs anfis gradient | 63.0% | [58.7%, 66.8%] | p < 0.001 |
| anfis vs anfis lse only | 27.5% | [21.9%, 32.8%] | p < 0.001 |
