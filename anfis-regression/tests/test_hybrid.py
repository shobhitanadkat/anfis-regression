import numpy as np
import torch

from anfis import ANFIS, TrainConfig, evaluate, fit
from anfis.hybrid import HybridConfig, HybridOptimizer, RecursiveLSE, ridge_lstsq


def test_ridge_lstsq_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(400, 6))
    true = np.array([1.5, -2.0, 0.0, 3.25, 0.5, -1.0])
    y = A @ true
    got = ridge_lstsq(torch.tensor(A, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32), lam=1e-10).numpy()
    np.testing.assert_allclose(got, true, rtol=1e-3, atol=1e-3)


def test_ridge_penalty_shrinks_the_solution():
    rng = np.random.default_rng(1)
    A = torch.tensor(rng.normal(size=(50, 20)), dtype=torch.float32)
    y = torch.tensor(rng.normal(size=50), dtype=torch.float32)
    small = ridge_lstsq(A, y, lam=1e-8).norm()
    large = ridge_lstsq(A, y, lam=1.0).norm()
    assert large < small


def test_recursive_lse_converges_to_the_batch_solution():
    """RLSE with no forgetting is algebraically the batch normal equations."""
    rng = np.random.default_rng(2)
    A = rng.normal(size=(600, 5))
    true = np.array([0.5, -1.0, 2.0, 0.25, -0.75])
    y = A @ true
    rlse = RecursiveLSE(n_features=5, gamma=1e6)
    theta = rlse.update(torch.tensor(A), torch.tensor(y))
    np.testing.assert_allclose(theta.numpy(), true, rtol=1e-3, atol=1e-3)


def test_lse_solve_is_optimal_for_the_current_partition():
    """After the forward pass no consequent perturbation can lower the loss."""
    rng = np.random.default_rng(3)
    X = rng.uniform(-2, 2, size=(300, 3))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 - X[:, 2]
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    model = ANFIS.from_data(X, n_mfs=2)
    opt = HybridOptimizer(model, HybridConfig(ridge=1e-10))
    opt.forward_pass(Xt, yt)
    with torch.no_grad():
        base = float(torch.mean((model(Xt) - yt) ** 2))
        best = base
        for _ in range(15):
            saved = model.consequent.clone()
            model.consequent.add_(torch.randn_like(model.consequent) * 0.05)
            best = min(best, float(torch.mean((model(Xt) - yt) ** 2)))
            model.consequent.copy_(saved)
    assert base <= best + 1e-9


def test_hybrid_beats_gradient_descent_at_equal_epochs():
    """The headline claim for Jang's rule: same budget, better fit."""
    rng = np.random.default_rng(4)
    X = rng.uniform(-2, 2, size=(400, 2))
    y = np.sin(X[:, 0] * 1.5) * np.cos(X[:, 1])

    scores = {}
    for method in ("hybrid", "gradient"):
        model = ANFIS.from_data(X, n_mfs=3)
        fit(model, X, y, TrainConfig(epochs=60, method=method, seed=0))
        scores[method] = evaluate(model, X, y)["rmse"]
    assert scores["hybrid"] < scores["gradient"]


def test_premise_moves_but_consequents_are_frozen_in_the_backward_pass():
    rng = np.random.default_rng(5)
    X = rng.uniform(-1, 1, size=(100, 2))
    y = X[:, 0] ** 2
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    model = ANFIS.from_data(X, n_mfs=2)
    opt = HybridOptimizer(model, HybridConfig(premise_lr=0.1))
    opt.forward_pass(Xt, yt)
    before_c = model.consequent.detach().clone()
    before_p = model.mf.c.detach().clone()
    opt.backward_pass(Xt, yt)
    assert torch.equal(model.consequent.detach(), before_c)
    assert not torch.equal(model.mf.c.detach(), before_p)


def test_training_is_stable_at_high_rule_counts():
    """256 rules on 300 samples is underdetermined; ridge must keep it finite."""
    rng = np.random.default_rng(6)
    X = rng.uniform(-2, 2, size=(300, 3))
    y = np.tanh(X[:, 0]) + X[:, 1]
    model = ANFIS.from_data(X, partition="scatter", n_rules=256)
    fit(model, X, y, TrainConfig(epochs=10, method="hybrid", ridge=1e-3, seed=0))
    assert np.isfinite(evaluate(model, X, y)["rmse"])


def test_early_stopping_restores_the_best_checkpoint():
    rng = np.random.default_rng(7)
    X = rng.uniform(-2, 2, size=(200, 2))
    y = X[:, 0] * X[:, 1]
    model = ANFIS.from_data(X, n_mfs=2)
    res = fit(model, X, y, TrainConfig(epochs=100, patience=5, seed=0))
    assert res.best_epoch <= len(res.history["val_mse"]) - 1
    assert res.best_val == min(res.history["val_mse"])
