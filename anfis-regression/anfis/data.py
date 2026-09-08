"""Regression benchmarks with a nonlinear structure that linear models miss.

Each loader returns a :class:`Dataset` with ``X``, ``y`` and human-readable
feature names, so the rules extracted later read as sentences about real
variables rather than about ``x[2]``.

``mackey_glass``
    The chaotic delay-differential benchmark from Jang's original ANFIS paper.
    Four lagged observations predict the series six steps ahead, which is the
    exact protocol that makes the 16-rule / 96-parameter model comparable to
    published numbers.
``jang_nonlinear``
    Three-input synthetic surface ``(1 + x^0.5 + y^-1 + z^-1.5)^2``.
``sinc2d``
    Two-input ``sin(x)/x * sin(y)/y``; small enough to visualise rule surfaces.
``friedman1``
    Friedman's five-informative-plus-noise-features benchmark, with a
    sinusoidal interaction term and additive Gaussian noise.
``nonlinear_mixed``
    Four-input surface combining an interaction, a saturating term and a
    threshold effect, with a deliberately strong linear component so that
    ordinary least squares is a genuinely competitive baseline rather than a
    straw man.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    name: str = "dataset"
    target_name: str = "y"
    split: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None, repr=False)

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Dataset(name={self.name!r}, n={self.n_samples}, d={self.n_features}, "
            f"features={self.feature_names})"
        )


# --------------------------------------------------------------------------
# Mackey-Glass
# --------------------------------------------------------------------------
def mackey_glass_series(
    n_points: int = 1200,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n_exp: int = 10,
    x0: float = 1.2,
    dt: float = 0.1,
    burn_in: int = 200,
) -> np.ndarray:
    """Integrate ``dx/dt = beta x(t-tau) / (1 + x(t-tau)^n) - gamma x(t)``.

    Fourth-order Runge-Kutta on a step of ``dt`` with the delayed term taken
    from the stored history, then sampled at unit time steps.  With
    ``tau = 17`` the attractor is chaotic, which is what makes one-step linear
    prediction hard.
    """
    steps_per_unit = int(round(1.0 / dt))
    total_steps = (n_points + burn_in + tau + 1) * steps_per_unit
    delay_steps = tau * steps_per_unit
    hist = np.empty(total_steps + delay_steps + 1, dtype=np.float64)
    hist[: delay_steps + 1] = x0

    def f(x_now: float, x_del: float) -> float:
        return beta * x_del / (1.0 + x_del ** n_exp) - gamma * x_now

    for i in range(delay_steps, total_steps + delay_steps):
        x = hist[i]
        xd = hist[i - delay_steps]           # x(t - tau)
        xd_next = hist[i - delay_steps + 1]  # x(t - tau + dt)
        xd_half = 0.5 * (xd + xd_next)       # linear interpolation of the delayed state
        k1 = f(x, xd)
        k2 = f(x + 0.5 * dt * k1, xd_half)
        k3 = f(x + 0.5 * dt * k2, xd_half)
        k4 = f(x + dt * k3, xd_next)
        hist[i + 1] = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    sampled = hist[delay_steps :: steps_per_unit]
    return sampled[burn_in : burn_in + n_points]


def load_mackey_glass(
    n_samples: int = 1000, lags: Tuple[int, ...] = (18, 12, 6, 0), horizon: int = 6
) -> Dataset:
    """Predict ``x(t + horizon)`` from four lagged observations.

    Jang's protocol: 1000 input-output pairs starting at ``t = 118``, the first
    500 for training and the last 500 for testing.
    """
    max_lag = max(lags)
    series = mackey_glass_series(n_points=n_samples + max_lag + horizon + 130)
    start = max(max_lag, 118)
    idx = np.arange(start, start + n_samples)
    X = np.stack([series[idx - lag] for lag in lags], axis=1)
    y = series[idx + horizon]
    names = [f"x(t-{lag})" if lag else "x(t)" for lag in lags]
    ds = Dataset(X, y, names, name="mackey_glass", target_name=f"x(t+{horizon})")
    ds.split = (np.arange(n_samples // 2), np.arange(n_samples // 2, n_samples))
    return ds


# --------------------------------------------------------------------------
# Synthetic surfaces
# --------------------------------------------------------------------------
def load_jang_nonlinear(n_samples: int = 1000, seed: int = 0) -> Dataset:
    """``y = (1 + x^0.5 + y^-1 + z^-1.5)^2`` on ``[1, 6]^3``."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 6.0, size=(n_samples, 3))
    y = (1.0 + X[:, 0] ** 0.5 + X[:, 1] ** -1.0 + X[:, 2] ** -1.5) ** 2
    return Dataset(X, y, ["x", "y", "z"], name="jang_nonlinear")


def load_sinc2d(n_samples: int = 900, seed: int = 0, grid: bool = True) -> Dataset:
    """``y = sinc(x) * sinc(y)`` on ``[-10, 10]^2``."""
    if grid:
        side = int(np.sqrt(n_samples))
        g = np.linspace(-10, 10, side)
        xx, yy = np.meshgrid(g, g)
        X = np.stack([xx.ravel(), yy.ravel()], axis=1)
    else:
        rng = np.random.default_rng(seed)
        X = rng.uniform(-10, 10, size=(n_samples, 2))
    def sinc(t: np.ndarray) -> np.ndarray:
        safe = np.where(np.abs(t) < 1e-9, 1.0, t)
        return np.where(np.abs(t) < 1e-9, 1.0, np.sin(t) / safe)

    y = sinc(X[:, 0]) * sinc(X[:, 1])
    return Dataset(X, y, ["x1", "x2"], name="sinc2d")


def load_friedman1(n_samples: int = 1000, noise: float = 1.0, seed: int = 0) -> Dataset:
    """Friedman #1: five informative inputs, sinusoidal interaction, plus noise."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_samples, 5))
    y = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
        + noise * rng.standard_normal(n_samples)
    )
    return Dataset(X, y, [f"x{i+1}" for i in range(5)], name="friedman1")


def load_nonlinear_mixed(n_samples: int = 1000, noise: float = 0.15, seed: int = 0) -> Dataset:
    """Four-input surface with a strong linear part plus nonlinear structure.

    Roughly two thirds of the target variance is linearly explainable, so OLS
    is a fair baseline; the remaining third lives in an interaction, a
    saturating response and a soft threshold that only a nonlinear model can
    recover.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n_samples, 4))
    x1, x2, x3, x4 = X.T
    y = (
        1.5 * x1
        + 1.0 * x2
        - 0.8 * x3
        + 0.6 * x4
        + 1.2 * x1 * x2
        + 1.5 * np.tanh(2.0 * x3)
        + 0.9 * np.sin(1.5 * x4)
        + noise * rng.standard_normal(n_samples)
    )
    return Dataset(X, y, ["temp", "pressure", "flow", "load"], name="nonlinear_mixed")


def load_csv(
    path: str, target: str, features: Optional[List[str]] = None, name: str = "csv"
) -> Dataset:
    """Load an arbitrary tabular regression problem from disk."""
    import pandas as pd

    df = pd.read_csv(path)
    cols = features or [c for c in df.columns if c != target]
    X = df[cols].to_numpy(dtype=np.float64)
    y = df[target].to_numpy(dtype=np.float64)
    return Dataset(X, y, list(cols), name=name, target_name=target)


DATASETS: Dict[str, Callable[..., Dataset]] = {
    "mackey_glass": load_mackey_glass,
    "jang_nonlinear": load_jang_nonlinear,
    "sinc2d": load_sinc2d,
    "friedman1": load_friedman1,
    "nonlinear_mixed": load_nonlinear_mixed,
}


def get_dataset(name: str, **kwargs) -> Dataset:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    return DATASETS[name](**kwargs)


class Standardizer:
    """Zero-mean unit-variance scaling fitted on training data only."""

    def __init__(self) -> None:
        self.mu_x: Optional[np.ndarray] = None
        self.sd_x: Optional[np.ndarray] = None
        self.mu_y: float = 0.0
        self.sd_y: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Standardizer":
        self.mu_x = X.mean(axis=0)
        self.sd_x = np.maximum(X.std(axis=0), 1e-8)
        self.mu_y = float(y.mean())
        self.sd_y = float(max(y.std(), 1e-8))
        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        Xs = (X - self.mu_x) / self.sd_x
        if y is None:
            return Xs
        return Xs, (y - self.mu_y) / self.sd_y

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.sd_y + self.mu_y
