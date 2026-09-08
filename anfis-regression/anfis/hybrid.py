"""Jang's hybrid learning rule: least squares forward, gradient descent backward.

The ANFIS output is **linear in the consequent parameters** and **non-linear in
the premise parameters**.  Jang's hybrid rule exploits that split by alternating
two passes per epoch:

1. *Forward pass.*  Freeze the premise parameters, build the layer-4 design
   matrix ``A`` and solve the linear least-squares problem
   ``theta* = argmin ||A theta - y||^2 + lambda ||theta||^2`` in closed form.
   The consequents jump straight to their global optimum for the current
   fuzzy partition instead of crawling there by gradient descent.
2. *Backward pass.*  Hold ``theta*`` fixed and take one (or a few) gradient
   steps on the premise parameters only, back-propagating through layers 1-3.

The result converges in far fewer epochs than plain gradient descent because
half the parameter vector is solved exactly at every step.  Two solvers are
provided:

``batch``
    Cholesky solve of the ridge-regularised normal equations, with an automatic
    fall back to ``lstsq``/pseudo-inverse when ``A^T A`` is ill conditioned.
    Regularisation matters here: at 256 rules a first-order model has more
    consequent coefficients than a typical training set has rows, so the
    unregularised system is underdetermined.

``rlse``
    Recursive least squares with covariance ``S`` and an optional forgetting
    factor, i.e. the sequential form given in the original paper.  Useful for
    mini-batch or streaming updates and for large ``R`` where forming
    ``A^T A`` in one shot is expensive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from .model import ANFIS


def ridge_lstsq(
    A: torch.Tensor, y: torch.Tensor, lam: float = 1e-6
) -> torch.Tensor:
    """Solve ``min ||A theta - y||^2 + lam ||theta||^2`` for ``theta``.

    Uses a Cholesky factorisation of the (symmetric positive definite) damped
    normal matrix, falling back to a least-norm ``lstsq`` solution if the
    factorisation fails.
    """
    A = A.double()
    y = y.double().reshape(-1, 1)
    n_features = A.shape[1]
    AtA = A.T @ A
    Aty = A.T @ y
    eye = torch.eye(n_features, dtype=A.dtype, device=A.device)
    scale = float(torch.diagonal(AtA).mean().clamp_min(1e-12))
    damped = AtA + lam * scale * eye
    try:
        L = torch.linalg.cholesky(damped)
        theta = torch.cholesky_solve(Aty, L)
    except Exception:  # pragma: no cover - numerical fallback
        theta = torch.linalg.lstsq(damped, Aty).solution
        if not torch.isfinite(theta).all():
            theta = torch.linalg.pinv(damped) @ Aty
    return theta.reshape(-1).float()


class RecursiveLSE:
    """Sequential least-squares estimator with an optional forgetting factor.

    ``S`` is initialised to ``gamma * I`` with a large ``gamma``, matching the
    diffuse prior used in Jang's formulation.
    """

    def __init__(
        self,
        n_features: int,
        gamma: float = 1e4,
        forgetting: float = 1.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.n_features = int(n_features)
        self.forgetting = float(forgetting)
        self.dtype = dtype
        self.S = gamma * torch.eye(n_features, dtype=dtype)
        self.theta = torch.zeros(n_features, 1, dtype=dtype)

    def update(self, A: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Absorb a batch of rows one at a time and return the current estimate."""
        A = A.to(self.dtype)
        y = y.to(self.dtype).reshape(-1, 1)
        lam = self.forgetting
        for i in range(A.shape[0]):
            a = A[i : i + 1].T  # [p, 1]
            Sa = self.S @ a
            denom = lam + float((a.T @ Sa).squeeze())
            if denom <= 0 or not torch.isfinite(torch.tensor(denom)):
                continue
            self.S = (self.S - (Sa @ Sa.T) / denom) / lam
            self.theta = self.theta + Sa / denom * (y[i : i + 1] - a.T @ self.theta)
        return self.theta.reshape(-1).float()

    def reset(self, gamma: float = 1e4) -> None:
        self.S = gamma * torch.eye(self.n_features, dtype=self.dtype)
        self.theta = torch.zeros(self.n_features, 1, dtype=self.dtype)


@dataclass
class HybridConfig:
    """Settings for :class:`HybridOptimizer`."""

    lse: Literal["batch", "rlse"] = "batch"
    ridge: float = 1e-6
    forgetting: float = 1.0
    rlse_gamma: float = 1e4
    premise_lr: float = 1e-2
    premise_optimizer: Literal["adam", "sgd", "rmsprop"] = "adam"
    momentum: float = 0.9
    grad_clip: Optional[float] = 5.0


class HybridOptimizer:
    """Alternating LSE / gradient optimiser for :class:`~anfis.model.ANFIS`.

    Example
    -------
    >>> opt = HybridOptimizer(model, HybridConfig(premise_lr=1e-2))
    >>> for epoch in range(epochs):
    ...     opt.forward_pass(X_train, y_train)      # exact consequent solve
    ...     loss = opt.backward_pass(X_train, y_train)  # premise gradient step
    """

    def __init__(self, model: ANFIS, config: Optional[HybridConfig] = None) -> None:
        self.model = model
        self.config = config or HybridConfig()
        params = list(model.premise_parameters())
        if self.config.premise_optimizer == "adam":
            self.premise_opt = torch.optim.Adam(params, lr=self.config.premise_lr)
        elif self.config.premise_optimizer == "sgd":
            self.premise_opt = torch.optim.SGD(
                params, lr=self.config.premise_lr, momentum=self.config.momentum
            )
        else:
            self.premise_opt = torch.optim.RMSprop(params, lr=self.config.premise_lr)

        self._rlse: Optional[RecursiveLSE] = None
        if self.config.lse == "rlse":
            self._rlse = RecursiveLSE(
                model.n_consequent_params,
                gamma=self.config.rlse_gamma,
                forgetting=self.config.forgetting,
            )

    # ------------------------------------------------------------- pass one
    @torch.no_grad()
    def forward_pass(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Solve for the consequent parameters with the premise held fixed."""
        A = self.model.design_matrix(X)
        if self.config.lse == "batch":
            theta = ridge_lstsq(A, y, lam=self.config.ridge)
        else:
            assert self._rlse is not None
            theta = self._rlse.update(A, y)
        if torch.isfinite(theta).all():
            self.model.set_consequent_vector(theta)

    # ------------------------------------------------------------- pass two
    def backward_pass(
        self, X: torch.Tensor, y: torch.Tensor, loss_fn=None
    ) -> float:
        """One gradient step on the premise parameters (consequents frozen)."""
        loss_fn = loss_fn or torch.nn.functional.mse_loss
        self.model.consequent.requires_grad_(False)
        self.premise_opt.zero_grad(set_to_none=True)
        pred = self.model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.model.premise_parameters()), self.config.grad_clip
            )
        self.premise_opt.step()
        self.model.consequent.requires_grad_(True)
        return float(loss.detach())

    # -------------------------------------------------------------- helpers
    def step(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """One full hybrid epoch: LSE solve then premise gradient step."""
        self.forward_pass(X, y)
        return self.backward_pass(X, y)

    def reset_rlse(self) -> None:
        if self._rlse is not None:
            self._rlse.reset(self.config.rlse_gamma)


def solve_consequents(
    model: ANFIS, X: torch.Tensor, y: torch.Tensor, ridge: float = 1e-6
) -> None:
    """Convenience one-shot LSE solve (the forward half of the hybrid rule)."""
    with torch.no_grad():
        theta = ridge_lstsq(model.design_matrix(X), y, lam=ridge)
        if torch.isfinite(theta).all():
            model.set_consequent_vector(theta)
