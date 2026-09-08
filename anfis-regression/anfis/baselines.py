"""Reference models the ANFIS is measured against.

* :class:`LinearBaseline` -- ordinary least squares, the headline comparison.
* :class:`RidgeBaseline` -- OLS with an L2 penalty tuned by inner CV; a
  stronger linear baseline that rules out "the linear model was just
  under-regularised".
* :class:`PolynomialBaseline` -- degree-2 features plus ridge; a nonlinear but
  non-interpretable-at-scale reference.
* :class:`MLPBaseline` -- the 64-32 ReLU network used for the parameter-count
  comparison.  On a 4-input problem it carries 2,433 weights.

All four share the ``fit``/``predict``/``n_params`` interface so the experiment
scripts can treat them interchangeably.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class LinearBaseline:
    """Ordinary least squares via the pseudo-inverse (no iterative solver)."""

    name = "linear"

    def __init__(self) -> None:
        self.coef_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearBaseline":
        A = np.hstack([X, np.ones((len(X), 1))])
        self.coef_ = np.linalg.lstsq(A, y, rcond=None)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        A = np.hstack([X, np.ones((len(X), 1))])
        return A @ self.coef_

    @property
    def n_params(self) -> int:
        return int(self.coef_.size) if self.coef_ is not None else 0


class RidgeBaseline:
    """Ridge regression with the penalty chosen by held-out validation."""

    name = "ridge"

    def __init__(self, alphas: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)) -> None:
        self.alphas = list(alphas)
        self.alpha_: float = self.alphas[0]
        self.coef_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 0) -> "RidgeBaseline":
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        cut = max(1, int(0.8 * len(X)))
        tr, va = idx[:cut], idx[cut:]
        best, best_err = self.alphas[0], np.inf
        for a in self.alphas:
            coef = self._solve(X[tr], y[tr], a)
            pred = np.hstack([X[va], np.ones((len(va), 1))]) @ coef
            err = np.mean((y[va] - pred) ** 2)
            if err < best_err:
                best, best_err = a, err
        self.alpha_ = best
        self.coef_ = self._solve(X, y, best)
        return self

    @staticmethod
    def _solve(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        A = np.hstack([X, np.ones((len(X), 1))])
        p = A.shape[1]
        reg = alpha * np.eye(p)
        reg[-1, -1] = 0.0  # never penalise the intercept
        return np.linalg.solve(A.T @ A + reg, A.T @ y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([X, np.ones((len(X), 1))]) @ self.coef_

    @property
    def n_params(self) -> int:
        return int(self.coef_.size) if self.coef_ is not None else 0


class PolynomialBaseline(RidgeBaseline):
    """Degree-2 polynomial expansion followed by ridge regression."""

    name = "poly2"

    def __init__(self, degree: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.degree = degree

    def _expand(self, X: np.ndarray) -> np.ndarray:
        feats: List[np.ndarray] = [X]
        d = X.shape[1]
        for i in range(d):
            for j in range(i, d):
                feats.append((X[:, i] * X[:, j])[:, None])
        return np.hstack(feats)

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 0) -> "PolynomialBaseline":
        return super().fit(self._expand(X), y, seed=seed)  # type: ignore[return-value]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return super().predict(self._expand(X))


class MLP(nn.Module):
    """Plain ReLU MLP; ``hidden=(64, 32)`` is the reference architecture."""

    def __init__(self, n_inputs: int, hidden: Sequence[int] = (64, 32)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = n_inputs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPBaseline:
    """Early-stopped Adam training for the reference network."""

    name = "mlp"

    def __init__(
        self,
        hidden: Sequence[int] = (64, 32),
        epochs: int = 500,
        lr: float = 1e-2,
        patience: int = 40,
        val_fraction: float = 0.15,
        seed: int = 0,
    ) -> None:
        self.hidden = tuple(hidden)
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.val_fraction = val_fraction
        self.seed = seed
        self.model: Optional[MLP] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPBaseline":
        torch.manual_seed(self.seed)
        Xt = torch.as_tensor(X, dtype=torch.float32)
        yt = torch.as_tensor(y, dtype=torch.float32)
        n_val = max(1, int(len(Xt) * self.val_fraction))
        perm = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(self.seed))
        va, tr = perm[:n_val], perm[n_val:]
        Xv, yv, Xt_, yt_ = Xt[va], yt[va], Xt[tr], yt[tr]

        self.model = MLP(X.shape[1], self.hidden)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best, best_state, stale = np.inf, None, 0
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((self.model(Xt_) - yt_) ** 2)
            loss.backward()
            opt.step()
            with torch.no_grad():
                val = float(torch.mean((self.model(Xv) - yv) ** 2))
            if val < best - 1e-6:
                best, stale = val, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.as_tensor(X, dtype=torch.float32)).numpy()

    @property
    def n_params(self) -> int:
        if self.model is None:
            return count_mlp_params(0, self.hidden)
        return sum(p.numel() for p in self.model.parameters())


def count_mlp_params(n_inputs: int, hidden: Sequence[int] = (64, 32)) -> int:
    """Weight count of an MLP without building it (useful for tables)."""
    total, prev = 0, n_inputs
    for h in hidden:
        total += prev * h + h
        prev = h
    return total + prev + 1


BASELINES = {
    "linear": LinearBaseline,
    "ridge": RidgeBaseline,
    "poly2": PolynomialBaseline,
    "mlp": MLPBaseline,
}
