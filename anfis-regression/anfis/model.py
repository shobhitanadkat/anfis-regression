"""Five-layer Takagi-Sugeno-Kang ANFIS in PyTorch.

Layer map (Jang, *ANFIS: Adaptive-Network-Based Fuzzy Inference System*,
IEEE TSMC 23(3), 1993):

===== ============================ ===========================================
Layer Name                         Operation
===== ============================ ===========================================
1     fuzzification                ``mu_ij = MF_ij(x_i)``            -> [N,d,M]
2     rule firing (T-norm)         ``w_r = prod_i mu_{i, idx(r,i)}`` -> [N,R]
3     normalisation                ``wbar_r = w_r / sum_s w_s``      -> [N,R]
4     consequent                   ``wbar_r * (p_r^T x + q_r)``      -> [N,R]
5     aggregation                  ``y = sum_r ...``                 -> [N]
===== ============================ ===========================================

Two implementation details matter for training stability at 128-256 rules:

* Layer 2 runs in **log space**.  A product of ``d`` memberships underflows to
  exactly ``0`` once the inputs sit in the tail of every MF, and layer 3 then
  divides ``0/0``.  Summing ``log mu`` and applying a ``softmax`` across rules
  is algebraically identical for the product T-norm, never underflows, and
  gives layer 3 a numerically exact Jacobian for free.
* Rules that receive no support anywhere in the batch are detected rather than
  silently producing NaNs; ``firing_stats`` reports them so the sweep can log
  the effective (as opposed to nominal) rule count.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .membership import MembershipFunction, build_mf
from .partition import build_partition

LOG_EPS = 1e-16


class ANFIS(nn.Module):
    """Adaptive neuro-fuzzy inference system with a Sugeno consequent.

    Parameters
    ----------
    n_inputs:
        Input dimension ``d``.
    mf:
        Membership family name (see :mod:`anfis.membership`).
    n_mfs:
        Terms per input, used by the grid partition.
    n_rules:
        Rule count, used by the scatter partition.
    partition:
        ``"grid"`` (``R = n_mfs ** d``) or ``"scatter"`` (``R = n_rules``).
    order:
        ``1`` for a first-order Sugeno consequent (affine in ``x``), ``0`` for a
        constant per rule.
    tnorm:
        ``"product"`` (differentiable everywhere, the default) or ``"min"``.
    """

    def __init__(
        self,
        n_inputs: int,
        mf: str = "gaussian",
        n_mfs: int = 2,
        n_rules: Optional[int] = None,
        partition: str = "grid",
        order: int = 1,
        tnorm: str = "product",
        rule_index: Optional[torch.Tensor] = None,
        mf_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if order not in (0, 1):
            raise ValueError("order must be 0 or 1")
        if tnorm not in ("product", "min"):
            raise ValueError("tnorm must be 'product' or 'min'")

        self.n_inputs = int(n_inputs)
        self.order = int(order)
        self.tnorm = tnorm
        self.partition_mode = partition

        if rule_index is None:
            if partition == "grid":
                from .partition import grid_rule_index

                rule_index = grid_rule_index(n_inputs, n_mfs)
                n_terms = n_mfs
            elif partition == "scatter":
                from .partition import scatter_rule_index

                if n_rules is None:
                    raise ValueError("scatter partition needs n_rules")
                rule_index = scatter_rule_index(n_rules, n_inputs)
                n_terms = n_rules
            else:
                raise ValueError(f"unknown partition {partition!r}")
        else:
            rule_index = torch.as_tensor(rule_index, dtype=torch.long)
            n_terms = int(rule_index.max().item()) + 1

        self.register_buffer("rule_index", rule_index)
        self.n_rules = int(rule_index.shape[0])
        self.n_terms = int(n_terms)

        self.mf: MembershipFunction = build_mf(
            mf, n_inputs, n_terms, **(mf_kwargs or {})
        )
        self.mf_name = mf

        n_coef = self.n_inputs + 1 if self.order == 1 else 1
        self.consequent = nn.Parameter(torch.zeros(self.n_rules, n_coef))
        nn.init.normal_(self.consequent, std=0.01)

    # ------------------------------------------------------------- properties
    @property
    def n_premise_params(self) -> int:
        return self.mf.n_premise_params

    @property
    def n_consequent_params(self) -> int:
        return int(self.consequent.numel())

    @property
    def n_params(self) -> int:
        return self.n_premise_params + self.n_consequent_params

    def premise_parameters(self):
        return self.mf.parameters()

    # ------------------------------------------------------------ init helper
    @classmethod
    def from_data(
        cls,
        X: np.ndarray,
        mf: str = "gaussian",
        n_mfs: int = 2,
        n_rules: Optional[int] = None,
        partition: str = "grid",
        order: int = 1,
        tnorm: str = "product",
        seed: int = 0,
        mf_kwargs: Optional[dict] = None,
    ) -> "ANFIS":
        """Build a model whose premise parameters already cover the data."""
        rule_index, centers, widths = build_partition(
            X, mode=partition, n_mfs=n_mfs, n_rules=n_rules, seed=seed
        )
        model = cls(
            n_inputs=X.shape[1],
            mf=mf,
            partition=partition,
            order=order,
            tnorm=tnorm,
            rule_index=rule_index,
            mf_kwargs=mf_kwargs,
        )
        model.mf.initialize(centers, widths)
        return model

    # --------------------------------------------------------------- layers
    def layer1_memberships(self, x: torch.Tensor) -> torch.Tensor:
        """Fuzzification -> ``[N, d, M]``."""
        return self.mf(x)

    def layer2_firing(self, mu: torch.Tensor) -> torch.Tensor:
        """Rule firing strengths in log space -> ``[N, R]``."""
        idx = self.rule_index.T  # [d, R]
        # gather MF used by each rule for each input: [N, d, R]
        picked = torch.gather(
            mu, 2, idx.unsqueeze(0).expand(mu.shape[0], -1, -1)
        )
        if self.tnorm == "product":
            return torch.log(picked.clamp_min(LOG_EPS)).sum(dim=1)
        return torch.log(picked.clamp_min(LOG_EPS)).min(dim=1).values

    def layer3_normalise(self, log_w: torch.Tensor) -> torch.Tensor:
        """Normalised firing strengths -> ``[N, R]`` rows summing to one."""
        return torch.softmax(log_w, dim=1)

    def layer4_consequent(self, x: torch.Tensor, wbar: torch.Tensor) -> torch.Tensor:
        """Weighted rule outputs -> ``[N, R]``."""
        if self.order == 1:
            f = x @ self.consequent[:, :-1].T + self.consequent[:, -1]
        else:
            f = self.consequent[:, 0].unsqueeze(0).expand(x.shape[0], -1)
        return wbar * f

    def layer5_aggregate(self, weighted: torch.Tensor) -> torch.Tensor:
        return weighted.sum(dim=1)

    # --------------------------------------------------------------- forward
    def forward(
        self, x: torch.Tensor, return_internals: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mu = self.layer1_memberships(x)
        log_w = self.layer2_firing(mu)
        wbar = self.layer3_normalise(log_w)
        weighted = self.layer4_consequent(x, wbar)
        y = self.layer5_aggregate(weighted)
        if return_internals:
            return y, {"mu": mu, "log_w": log_w, "wbar": wbar, "weighted": weighted}
        return y

    # ---------------------------------------------------- least-squares hooks
    def normalised_firing(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer3_normalise(self.layer2_firing(self.layer1_memberships(x)))

    def design_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Layer-4 regressor matrix ``A`` with ``y = A @ vec(consequent)``.

        Shape ``[N, R * (d + 1)]`` for a first-order model.  Because layers 1-3
        do not involve the consequent parameters, the model output is *linear*
        in ``A`` -- this is exactly the structure Jang's hybrid rule exploits.
        """
        wbar = self.normalised_firing(x)
        if self.order == 1:
            ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
            xb = torch.cat([x, ones], dim=1)  # [N, d+1]
            A = wbar.unsqueeze(2) * xb.unsqueeze(1)  # [N, R, d+1]
            return A.reshape(x.shape[0], -1)
        return wbar

    def set_consequent_vector(self, theta: torch.Tensor) -> None:
        with torch.no_grad():
            self.consequent.copy_(theta.reshape_as(self.consequent))

    # ----------------------------------------------------------- diagnostics
    @torch.no_grad()
    def firing_stats(self, x: torch.Tensor, threshold: float = 1e-3) -> Dict[str, float]:
        """Rule-usage diagnostics used to quantify interpretability.

        ``effective_rules`` is the perplexity ``exp(H)`` of the mean normalised
        firing distribution: the number of rules the model actually leans on,
        which is often far below the nominal rule count.
        """
        wbar = self.normalised_firing(x)
        mean_w = wbar.mean(dim=0)
        p = mean_w / mean_w.sum().clamp_min(1e-12)
        entropy = float(-(p * torch.log(p.clamp_min(1e-12))).sum())
        max_w = wbar.max(dim=0).values
        return {
            "n_rules": float(self.n_rules),
            "active_rules": float((max_w > threshold).sum()),
            "dead_rules": float((max_w <= threshold).sum()),
            "effective_rules": float(np.exp(entropy)),
            "firing_entropy": entropy,
            "mean_max_activation": float(wbar.max(dim=1).values.mean()),
        }

    def extra_repr(self) -> str:
        return (
            f"n_inputs={self.n_inputs}, n_rules={self.n_rules}, mf={self.mf_name}, "
            f"order={self.order}, partition={self.partition_mode}, "
            f"params={self.n_params} ({self.n_premise_params} premise "
            f"+ {self.n_consequent_params} consequent)"
        )
