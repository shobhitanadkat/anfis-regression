"""Explainable neuro-fuzzy regression: a five-layer Sugeno ANFIS in PyTorch.

Quick start
-----------
>>> from anfis import ANFIS, TrainConfig, fit, evaluate, get_dataset
>>> ds = get_dataset("mackey_glass")
>>> model = ANFIS.from_data(ds.X[:500], mf="gaussian", n_mfs=2)   # 16 rules, 96 params
>>> _ = fit(model, ds.X[:500], ds.y[:500], TrainConfig(epochs=200))
>>> evaluate(model, ds.X[500:], ds.y[500:])["rmse"]               # doctest: +SKIP
"""

from .baselines import (
    LinearBaseline,
    MLPBaseline,
    PolynomialBaseline,
    RidgeBaseline,
    count_mlp_params,
)
from .data import DATASETS, Dataset, Standardizer, get_dataset
from .hybrid import HybridConfig, HybridOptimizer, RecursiveLSE, ridge_lstsq, solve_consequents
from .interpret import (
    coverage_report,
    extract_rules,
    local_explanation,
    rule_importance,
    rules_to_text,
)
from .membership import MF_FAMILIES, MF_REGISTRY, build_mf
from .metrics import all_metrics, bootstrap_rmse_ratio, ndei, paired_error_test, rmse
from .model import ANFIS
from .partition import build_partition, grid_rule_index, scatter_rule_index
from .train import TrainConfig, TrainResult, cross_validate, evaluate, fit, predict

__version__ = "0.1.0"

__all__ = [
    "ANFIS",
    "TrainConfig",
    "TrainResult",
    "fit",
    "evaluate",
    "predict",
    "cross_validate",
    "HybridOptimizer",
    "HybridConfig",
    "RecursiveLSE",
    "ridge_lstsq",
    "solve_consequents",
    "MF_FAMILIES",
    "MF_REGISTRY",
    "build_mf",
    "build_partition",
    "grid_rule_index",
    "scatter_rule_index",
    "Dataset",
    "DATASETS",
    "get_dataset",
    "Standardizer",
    "LinearBaseline",
    "RidgeBaseline",
    "PolynomialBaseline",
    "MLPBaseline",
    "count_mlp_params",
    "rmse",
    "ndei",
    "all_metrics",
    "paired_error_test",
    "bootstrap_rmse_ratio",
    "extract_rules",
    "rule_importance",
    "coverage_report",
    "local_explanation",
    "rules_to_text",
    "__version__",
]
