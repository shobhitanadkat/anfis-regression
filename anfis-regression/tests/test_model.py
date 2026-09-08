import numpy as np
import pytest
import torch

from anfis import ANFIS, count_mlp_params
from anfis.membership import MF_FAMILIES


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(200, 4))
    y = np.sin(X[:, 0]) + X[:, 1] * X[:, 2] - X[:, 3]
    return X, y


def test_grid_rule_count_is_m_to_the_d(data):
    X, _ = data
    for m in (2, 3):
        model = ANFIS.from_data(X, n_mfs=m, partition="grid")
        assert model.n_rules == m ** X.shape[1]


def test_normalised_firing_sums_to_one(data):
    X, _ = data
    model = ANFIS.from_data(X, n_mfs=3)
    wbar = model.normalised_firing(torch.as_tensor(X, dtype=torch.float32))
    np.testing.assert_allclose(wbar.sum(1).detach().numpy(), 1.0, rtol=1e-5)
    assert torch.all(wbar >= 0)


def test_output_is_linear_in_the_consequents(data):
    """``y == A @ vec(theta)`` -- the identity Jang's hybrid rule relies on."""
    X, _ = data
    model = ANFIS.from_data(X, n_mfs=2)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    with torch.no_grad():
        torch.nn.init.normal_(model.consequent, std=0.5)
        direct = model(Xt).numpy()
        via_design = (model.design_matrix(Xt) @ model.consequent.reshape(-1)).numpy()
    np.testing.assert_allclose(direct, via_design, rtol=1e-4, atol=1e-5)


def test_headline_parameter_counts(data):
    """The claim under test: 96 ANFIS parameters vs 2,433 for a 64-32 MLP."""
    X, _ = data
    model = ANFIS.from_data(X, mf="gaussian", n_mfs=2, partition="grid", order=1)
    assert model.n_rules == 16
    assert model.n_premise_params == 16       # 4 inputs x 2 MFs x 2 params
    assert model.n_consequent_params == 80    # 16 rules x (4 + 1)
    assert model.n_params == 96
    assert count_mlp_params(4, (64, 32)) == 2433


def test_zeroth_order_consequent_shrinks_the_model(data):
    X, _ = data
    first = ANFIS.from_data(X, n_mfs=2, order=1)
    zeroth = ANFIS.from_data(X, n_mfs=2, order=0)
    assert zeroth.n_consequent_params == first.n_rules
    assert zeroth.n_params < first.n_params


def test_scatter_partition_hits_the_requested_rule_count(data):
    X, _ = data
    for r in (4, 24, 48):
        model = ANFIS.from_data(X, partition="scatter", n_rules=r)
        assert model.n_rules == r


@pytest.mark.parametrize("family", MF_FAMILIES)
def test_forward_is_finite_for_every_family(family, data):
    X, _ = data
    model = ANFIS.from_data(X, mf=family, n_mfs=2)
    out = model(torch.as_tensor(X, dtype=torch.float32))
    assert out.shape == (len(X),)
    assert torch.isfinite(out).all()


def test_log_space_firing_survives_far_out_inputs(data):
    """Product T-norms underflow in linear space; the log-space path must not."""
    X, _ = data
    model = ANFIS.from_data(X, n_mfs=2)
    far = torch.full((8, 4), 500.0)  # every membership is numerically zero
    wbar = model.normalised_firing(far)
    assert torch.isfinite(wbar).all()
    np.testing.assert_allclose(wbar.sum(1).detach().numpy(), 1.0, rtol=1e-5)


def test_min_tnorm_is_supported(data):
    X, _ = data
    model = ANFIS.from_data(X, n_mfs=2, tnorm="min")
    out = model(torch.as_tensor(X, dtype=torch.float32))
    assert torch.isfinite(out).all()


def test_firing_stats_reports_effective_rules(data):
    X, _ = data
    model = ANFIS.from_data(X, n_mfs=2)
    stats = model.firing_stats(torch.as_tensor(X, dtype=torch.float32))
    assert stats["n_rules"] == 16
    assert 1.0 <= stats["effective_rules"] <= 16.0
    assert stats["active_rules"] + stats["dead_rules"] == 16


def test_internals_are_exposed_for_interpretation(data):
    X, _ = data
    model = ANFIS.from_data(X, n_mfs=2)
    y, internals = model(torch.as_tensor(X, dtype=torch.float32), return_internals=True)
    assert set(internals) == {"mu", "log_w", "wbar", "weighted"}
    np.testing.assert_allclose(
        internals["weighted"].sum(1).detach().numpy(), y.detach().numpy(), rtol=1e-5
    )
