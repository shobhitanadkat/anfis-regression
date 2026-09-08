# Headline results: mackey_glass

| Model | Params | RMSE (mean ± sd) | MAE | R² | NDEI |
|---|---:|---:|---:|---:|---:|
| **ANFIS (hybrid)** | 96 | 0.00201 ± 0.00003 | 0.00144 | 0.99992 | 0.0089 |
| ANFIS (gradient only) | 96 | 0.01547 ± 0.00039 | 0.01212 | 0.99528 | 0.0687 |
| ANFIS (LSE only, premise fixed) | 96 | 0.00313 ± 0.00001 | 0.00233 | 0.99981 | 0.0139 |
| Linear regression (OLS) | 5 | 0.09569 ± 0.00000 | 0.07919 | 0.81961 | 0.4247 |
| Ridge regression | 5 | 0.09570 ± 0.00002 | 0.07922 | 0.81957 | 0.4248 |
| Polynomial (deg 2) + ridge | 15 | 0.03047 ± 0.00002 | 0.02364 | 0.98171 | 0.1352 |
| MLP 64-32 | 2,433 | 0.00434 ± 0.00017 | 0.00323 | 0.99963 | 0.0193 |

Paired significance tests on held-out squared errors (mackey_glass):

| Comparison | RMSE reduction | 95% CI | Paired t-test |
|---|---:|---:|---|
| anfis vs linear | 97.9% | [97.7%, 98.1%] | p < 1e-16 |
| anfis vs ridge | 97.9% | [97.7%, 98.1%] | p < 1e-16 |
| anfis vs poly2 | 93.4% | [92.5%, 94.2%] | p < 1e-16 |
| anfis vs mlp | 53.5% | [48.3%, 58.3%] | p < 1e-16 |
| anfis vs anfis gradient | 87.0% | [85.5%, 88.4%] | p < 1e-16 |
| anfis vs anfis lse only | 35.6% | [30.2%, 40.8%] | p < 0.001 |
