"""Differentiable membership-function (MF) families for the fuzzification layer.

Every family is an ``nn.Module`` mapping a crisp input batch ``x`` of shape
``[N, d]`` to membership degrees ``mu`` of shape ``[N, d, M]``, where ``M`` is
the number of linguistic terms ("low", "medium", ...) per input variable.

Shape parameters that must stay positive (widths, slopes) are stored as
unconstrained "raw" tensors and pushed through ``softplus`` in the forward
pass.  This keeps the premise parameters on an unbounded manifold so plain
SGD/Adam can move them freely without projection steps, and it removes the
degenerate ``sigma -> 0`` solutions that make naive ANFIS implementations
produce NaNs after a few hundred steps.

Families
--------
``gaussian``     smooth, 2 params/MF, the default in Jang's papers
``gbell``        generalised bell, 3 params/MF, adjustable shoulder sharpness
``dsig``         difference of two sigmoids, 4 params/MF, asymmetric
``triangular``   piecewise linear, 3 params/MF, differentiable a.e.
``trapezoidal``  piecewise linear with a plateau, 4 params/MF

The two piecewise-linear families are differentiable almost everywhere; their
kinks form a measure-zero set, so autograd's subgradient is well defined.  Both
also accept ``smooth=True``, which replaces the hard ``min``/``max`` with
temperature-controlled soft equivalents and yields an everywhere-C-infinity
surrogate (useful when second-order optimisers or Hessian diagnostics are
needed).
"""

from __future__ import annotations

from typing import Dict, List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Inverse of ``softplus``, used to initialise raw (unconstrained) params."""
    y = y.clamp_min(1e-6)
    return y + torch.log(-torch.expm1(-y))


def _soft_min(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    return -tau * torch.logaddexp(-a / tau, -b / tau)


def _soft_max(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    return tau * torch.logaddexp(a / tau, b / tau)


class MembershipFunction(nn.Module):
    """Base class for a bank of ``d * M`` membership functions.

    Parameters
    ----------
    n_inputs:
        Number of input variables ``d``.
    n_mfs:
        Number of linguistic terms ``M`` per input variable.
    """

    #: number of learnable scalars per single membership function
    n_params_per_mf: int = 0
    #: short name used by the registry and the CLI
    name: str = "base"

    def __init__(self, n_inputs: int, n_mfs: int) -> None:
        super().__init__()
        self.n_inputs = int(n_inputs)
        self.n_mfs = int(n_mfs)

    # ------------------------------------------------------------------ utils
    @property
    def n_premise_params(self) -> int:
        return self.n_inputs * self.n_mfs * self.n_params_per_mf

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def initialize(self, centers: torch.Tensor, widths: torch.Tensor) -> None:
        """Set the MFs from ``centers``/``widths`` of shape ``[d, M]``."""
        raise NotImplementedError

    def centers(self) -> torch.Tensor:
        """Representative location of each MF, shape ``[d, M]``."""
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"n_inputs={self.n_inputs}, n_mfs={self.n_mfs}, params={self.n_premise_params}"


class GaussianMF(MembershipFunction):
    r"""``mu(x) = exp(-0.5 ((x - c) / sigma)^2)`` with ``sigma = softplus(rho)``."""

    n_params_per_mf = 2
    name = "gaussian"

    def __init__(self, n_inputs: int, n_mfs: int) -> None:
        super().__init__(n_inputs, n_mfs)
        self.c = nn.Parameter(torch.zeros(n_inputs, n_mfs))
        self.rho = nn.Parameter(torch.zeros(n_inputs, n_mfs))

    def initialize(self, centers: torch.Tensor, widths: torch.Tensor) -> None:
        with torch.no_grad():
            self.c.copy_(centers)
            self.rho.copy_(_inv_softplus(widths))

    def sigma(self) -> torch.Tensor:
        return F.softplus(self.rho) + 1e-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x.unsqueeze(-1) - self.c) / self.sigma()
        return torch.exp(-0.5 * z * z)

    def centers(self) -> torch.Tensor:
        return self.c.detach()


class GeneralizedBellMF(MembershipFunction):
    r"""``mu(x) = 1 / (1 + |(x - c) / a|^{2b})``.

    ``b`` controls how square the shoulders are; it is kept above ``0.5`` so the
    MF cannot collapse into a spike.
    """

    n_params_per_mf = 3
    name = "gbell"

    def __init__(self, n_inputs: int, n_mfs: int) -> None:
        super().__init__(n_inputs, n_mfs)
        self.c = nn.Parameter(torch.zeros(n_inputs, n_mfs))
        self.rho_a = nn.Parameter(torch.zeros(n_inputs, n_mfs))
        self.rho_b = nn.Parameter(torch.zeros(n_inputs, n_mfs))

    def initialize(self, centers: torch.Tensor, widths: torch.Tensor) -> None:
        with torch.no_grad():
            self.c.copy_(centers)
            self.rho_a.copy_(_inv_softplus(widths))
            self.rho_b.fill_(float(_inv_softplus(torch.tensor(1.5))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.softplus(self.rho_a) + 1e-4
        b = F.softplus(self.rho_b) + 0.5
        z = torch.abs((x.unsqueeze(-1) - self.c) / a) + 1e-8
        return 1.0 / (1.0 + torch.pow(z, 2.0 * b))

    def centers(self) -> torch.Tensor:
        return self.c.detach()


class SigmoidDifferenceMF(MembershipFunction):
    r"""``mu(x) = sigma(a1 (x - c1)) - sigma(a2 (x - c2))`` with ``c2 = c1 + gap``.

    Ordering the two inflection points via a positive ``gap`` guarantees a
    single bump.  The result is clamped at zero, which the difference form can
    otherwise dip below in the far tails when ``a1`` and ``a2`` differ a lot.
    """

    n_params_per_mf = 4
    name = "dsig"

    def __init__(self, n_inputs: int, n_mfs: int) -> None:
        super().__init__(n_inputs, n_mfs)
        self.c1 = nn.Parameter(torch.zeros(n_inputs, n_mfs))
        self.rho_gap = nn.Parameter(torch.zeros(n_inputs, n_mfs))
        self.rho_a1 = nn.Parameter(torch.zeros(n_inputs, n_mfs))
        self.rho_a2 = nn.Parameter(torch.zeros(n_inputs, n_mfs))

    def initialize(self, centers: torch.Tensor, widths: torch.Tensor) -> None:
        with torch.no_grad():
            self.c1.copy_(centers - widths)
            self.rho_gap.copy_(_inv_softplus(2.0 * widths))
            slope = 3.0 / widths.clamp_min(1e-3)
            self.rho_a1.copy_(_inv_softplus(slope))
            self.rho_a2.copy_(_inv_softplus(slope))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a1 = F.softplus(self.rho_a1) + 1e-3
        a2 = F.softplus(self.rho_a2) + 1e-3
        c1 = self.c1
        c2 = c1 + F.softplus(self.rho_gap) + 1e-3
        xe = x.unsqueeze(-1)
        mu = torch.sigmoid(a1 * (xe - c1)) - torch.sigmoid(a2 * (xe - c2))
        return mu.clamp_min(0.0)

    def centers(self) -> torch.Tensor:
        with torch.no_grad():
            return self.c1 + 0.5 * (F.softplus(self.rho_gap) + 1e-3)


class TriangularMF(MembershipFunction):
    r"""``mu(x) = max(0, min((x-a)/(b-a), (c-x)/(c-b)))``.

    Parameterised by the peak ``b`` and two positive half-widths so the vertex
    ordering ``a < b < c`` holds by construction.
    """

    n_params_per_mf = 3
    name = "triangular"

    def __init__(self, n_inputs: int, n_mfs: int, smooth: bool = False, tau: float = 0.05) -> None:
        super().__init__(n_inputs, n_mfs)
        self.smooth = bool(smooth)
        self.tau = float(tau)
        self.b = nn.Parameter(torch.zeros(n_inputs, n_mfs))
        self.rho_l = nn.Parameter(torch.zeros(n_inputs, n_mfs))
        self.rho_r = nn.Parameter(torch.zeros(n_inputs, n_mfs))

    def initialize(self, centers: torch.Tensor, widths: torch.Tensor) -> None:
        with torch.no_grad():
            self.b.copy_(centers)
            self.rho_l.copy_(_inv_softplus(2.0 * widths))
            self.rho_r.copy_(_inv_softplus(2.0 * widths))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = F.softplus(self.rho_l) + 1e-3
        right = F.softplus(self.rho_r) + 1e-3
        xe = x.unsqueeze(-1)
        rising = (xe - (self.b - left)) / left
        falling = ((self.b + right) - xe) / right
        if self.smooth:
            return _soft_max(_soft_min(rising, falling, self.tau),
                             torch.zeros_like(rising), self.tau)
        return torch.clamp(torch.minimum(rising, falling), min=0.0)

    def centers(self) -> torch.Tensor:
        return self.b.detach()


class TrapezoidalMF(MembershipFunction):
    r"""Trapezoid with feet ``a < b <= c < d`` and a flat top on ``[b, c]``."""

    n_params_per_mf = 4
    name = "trapezoidal"

    def __init__(self, n_inputs: int, n_mfs: int, smooth: bool = False, tau: float = 0.05) -> None:
        super().__init__(n_inputs, n_mfs)
        self.smooth = bool(smooth)
        self.tau = float(tau)
        self.b = nn.Parameter(torch.zeros(n_inputs, n_mfs))
        self.rho_plateau = nn.Parameter(torch.zeros(n_inputs, n_mfs))
        self.rho_l = nn.Parameter(torch.zeros(n_inputs, n_mfs))
        self.rho_r = nn.Parameter(torch.zeros(n_inputs, n_mfs))

    def initialize(self, centers: torch.Tensor, widths: torch.Tensor) -> None:
        with torch.no_grad():
            self.b.copy_(centers - 0.5 * widths)
            self.rho_plateau.copy_(_inv_softplus(widths))
            self.rho_l.copy_(_inv_softplus(widths))
            self.rho_r.copy_(_inv_softplus(widths))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = F.softplus(self.rho_l) + 1e-3
        right = F.softplus(self.rho_r) + 1e-3
        plateau = F.softplus(self.rho_plateau) + 1e-3
        b = self.b
        c = b + plateau
        xe = x.unsqueeze(-1)
        rising = (xe - (b - left)) / left
        falling = ((c + right) - xe) / right
        one = torch.ones_like(rising)
        if self.smooth:
            inner = _soft_min(_soft_min(rising, falling, self.tau), one, self.tau)
            return _soft_max(inner, torch.zeros_like(inner), self.tau)
        inner = torch.minimum(torch.minimum(rising, falling), one)
        return torch.clamp(inner, min=0.0)

    def centers(self) -> torch.Tensor:
        with torch.no_grad():
            return self.b + 0.5 * (F.softplus(self.rho_plateau) + 1e-3)


MF_REGISTRY: Dict[str, Type[MembershipFunction]] = {
    GaussianMF.name: GaussianMF,
    GeneralizedBellMF.name: GeneralizedBellMF,
    SigmoidDifferenceMF.name: SigmoidDifferenceMF,
    TriangularMF.name: TriangularMF,
    TrapezoidalMF.name: TrapezoidalMF,
}

MF_FAMILIES: List[str] = list(MF_REGISTRY)


def build_mf(name: str, n_inputs: int, n_mfs: int, **kwargs) -> MembershipFunction:
    """Instantiate a membership-function bank by family name."""
    key = name.lower()
    if key not in MF_REGISTRY:
        raise KeyError(f"unknown membership family {name!r}; choose from {MF_FAMILIES}")
    cls = MF_REGISTRY[key]
    if cls in (TriangularMF, TrapezoidalMF):
        return cls(n_inputs, n_mfs, **kwargs)
    kwargs.pop("smooth", None)
    kwargs.pop("tau", None)
    return cls(n_inputs, n_mfs, **kwargs)
