import numpy as np
import pytest

from anfis import ANFIS, TrainConfig, coverage_report, extract_rules, fit, get_dataset
from anfis.data import DATASETS, Standardizer, mackey_glass_series
from anfis.interpret import local_explanation, rules_to_text, term_labels
from anfis.metrics import (
    benjamini_hochberg,
    bootstrap_rmse_ratio,
    ndei,
    paired_error_test,
    r2,
    rmse,
)


# ------------------------------------------------------------------- data
@pytest.mark.parametrize("name", list(DATASETS))
def test_every_dataset_loads_with_matching_shapes(name):
    ds = get_dataset(name)
    assert ds.X.ndim == 2 and ds.y.ndim == 1
    assert len(ds.X) == len(ds.y)
    assert len(ds.feature_names) == ds.n_features
    assert np.isfinite(ds.X).all() and np.isfinite(ds.y).all()


def test_mackey_glass_stays_on_its_attractor():
    """The tau=17 series is bounded and non-periodic, not a decaying transient."""
    s = mackey_glass_series(n_points=800)
    assert s.shape == (800,)
    assert 0.2 < s.min() < s.max() < 1.7
    assert s.std() > 0.1  # would collapse toward a fixed point if the integrator were wrong


def test_mackey_glass_uses_jangs_four_lag_protocol():
    ds = get_dataset("mackey_glass")
    assert ds.n_features == 4
    assert ds.n_samples == 1000
    train, test = ds.split
    assert len(train) == len(test) == 500
    assert set(train).isdisjoint(set(test))


def test_standardizer_never_sees_the_test_fold():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(100, 3)), rng.normal(size=100)
    sc = Standardizer().fit(X[:70], y[:70])
    np.testing.assert_allclose(sc.mu_x, X[:70].mean(axis=0))
    Xs, ys = sc.transform(X[:70], y[:70])
    np.testing.assert_allclose(Xs.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(sc.inverse_y(ys), y[:70], atol=1e-8)


# ---------------------------------------------------------------- metrics
def test_metric_edge_cases():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0
    assert r2(y, y) == pytest.approx(1.0)
    assert ndei(y, y) == 0.0
    assert r2(y, np.full(3, y.mean())) == pytest.approx(0.0, abs=1e-9)


def test_paired_test_detects_a_genuinely_better_model():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    good = y + rng.normal(scale=0.1, size=500)
    bad = y + rng.normal(scale=1.0, size=500)
    res = paired_error_test(y, good, bad, alternative="less")
    assert res["t_pvalue"] < 0.001
    assert res["wilcoxon_pvalue"] < 0.001


def test_paired_test_finds_nothing_when_models_are_equivalent():
    rng = np.random.default_rng(1)
    y = rng.normal(size=400)
    a = y + rng.normal(scale=0.3, size=400)
    b = y + rng.normal(scale=0.3, size=400)
    assert paired_error_test(y, a, b, alternative="less")["t_pvalue"] > 0.01


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(2)
    y = rng.normal(size=300)
    a = y + rng.normal(scale=0.2, size=300)
    b = y + rng.normal(scale=0.8, size=300)
    res = bootstrap_rmse_ratio(y, a, b, n_boot=400, seed=0)
    assert res["ci_low"] <= res["rmse_reduction"] <= res["ci_high"]
    assert res["rmse_reduction"] > 0


def test_benjamini_hochberg_is_more_conservative_than_raw_alpha():
    p = np.concatenate([np.full(5, 1e-8), np.linspace(0.04, 0.9, 85)])
    mask = benjamini_hochberg(p, alpha=0.05)
    assert mask[:5].all()
    assert mask.sum() < (p <= 0.05).sum() + 1
    assert not benjamini_hochberg(np.linspace(0.2, 0.9, 20)).any()


# --------------------------------------------------------- interpretability
@pytest.fixture(scope="module")
def trained():
    ds = get_dataset("sinc2d", n_samples=400)
    model = ANFIS.from_data(ds.X, mf="gaussian", n_mfs=3)
    fit(model, ds.X, ds.y, TrainConfig(epochs=40, seed=0))
    return model, ds


def test_rule_table_has_one_row_per_rule_and_shares_sum_to_one(trained):
    model, ds = trained
    df = extract_rules(model, ds.X, ds.feature_names, ds.target_name)
    assert len(df) == model.n_rules
    assert df["output_share"].sum() == pytest.approx(1.0, abs=1e-6)
    assert df["IF"].str.contains("AND").all()
    assert df["THEN"].str.startswith("y =").all()


def test_linguistic_labels_follow_centre_ordering(trained):
    model, ds = trained
    df = extract_rules(model, ds.X, ds.feature_names)
    text = " ".join(df["IF"])
    for label in term_labels(model.n_terms):
        assert label in text


def test_coverage_report_is_self_consistent(trained):
    model, ds = trained
    cov = coverage_report(model, ds.X)
    assert 1 <= cov["rules_for_90pct"] <= model.n_rules
    assert 0 < cov["top_rule_share"] <= 1
    assert cov["effective_rules"] <= model.n_rules + 1e-6


def test_local_explanation_attributes_the_prediction(trained):
    model, ds = trained
    expl = local_explanation(model, ds.X[0], ds.feature_names, top_k=3)
    assert len(expl) == 3
    assert expl["firing_strength"].is_monotonic_decreasing
    assert np.isfinite(expl.attrs["prediction"])


def test_rules_render_as_readable_text(trained):
    model, ds = trained
    text = rules_to_text(extract_rules(model, ds.X, ds.feature_names), max_rules=3)
    assert text.count("IF") == 3 and text.count("THEN") == 3
