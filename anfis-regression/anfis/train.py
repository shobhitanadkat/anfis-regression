"""Training loops, early stopping and cross-validation for ANFIS models.

``fit`` supports three optimisation regimes so the sweep can attribute credit
correctly:

``hybrid``
    Jang's rule -- exact least-squares solve for the consequents, gradient step
    for the premise.
``gradient``
    Every parameter trained by Adam.  The control condition that shows what the
    least-squares half of the hybrid rule is actually buying.
``lse_only``
    One least-squares solve with the initial (data-driven) fuzzy partition left
    untouched.  The floor that premise learning has to beat.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

from .data import Standardizer
from .hybrid import HybridConfig, HybridOptimizer, solve_consequents
from .metrics import all_metrics
from .model import ANFIS


@dataclass
class TrainConfig:
    """Everything that defines a training run."""

    epochs: int = 200
    method: Literal["hybrid", "gradient", "lse_only"] = "hybrid"
    premise_lr: float = 1e-2
    premise_optimizer: str = "adam"
    consequent_lr: float = 1e-2  # only used by method="gradient"
    ridge: float = 1e-6
    lse: Literal["batch", "rlse"] = "batch"
    batch_size: Optional[int] = None
    val_fraction: float = 0.15
    patience: int = 30
    min_delta: float = 1e-6
    grad_clip: Optional[float] = 5.0
    seed: int = 0
    verbose: bool = False


@dataclass
class TrainResult:
    model: ANFIS = field(repr=False)
    history: Dict[str, List[float]] = field(repr=False)
    best_epoch: int = 0
    best_val: float = float("inf")
    seconds: float = 0.0
    n_params: int = 0


def _to_tensor(a: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), dtype=dtype)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))


def fit(
    model: ANFIS,
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[TrainConfig] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> TrainResult:
    """Train ``model`` in place and return the best checkpoint by validation MSE."""
    cfg = config or TrainConfig()
    set_seed(cfg.seed)
    start = time.time()

    Xt, yt = _to_tensor(X), _to_tensor(y)
    if X_val is None and cfg.val_fraction > 0:
        n_val = max(1, int(len(Xt) * cfg.val_fraction))
        perm = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(cfg.seed))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        Xv, yv = Xt[val_idx], yt[val_idx]
        Xt, yt = Xt[tr_idx], yt[tr_idx]
    elif X_val is not None:
        Xv, yv = _to_tensor(X_val), _to_tensor(y_val)
    else:
        Xv, yv = Xt, yt

    history: Dict[str, List[float]] = {"train_mse": [], "val_mse": []}
    best_state = copy.deepcopy(model.state_dict())
    best_val, best_epoch, stale = float("inf"), 0, 0
    # `ref_val` drives patience: only an improvement larger than `min_delta`
    # resets the counter, while *any* improvement still updates the checkpoint.
    ref_val = float("inf")

    if cfg.method == "lse_only":
        solve_consequents(model, Xt, yt, ridge=cfg.ridge)
        with torch.no_grad():
            tr = float(torch.mean((model(Xt) - yt) ** 2))
            va = float(torch.mean((model(Xv) - yv) ** 2))
        history["train_mse"].append(tr)
        history["val_mse"].append(va)
        return TrainResult(model, history, 0, va, time.time() - start, model.n_params)

    if cfg.method == "hybrid":
        opt = HybridOptimizer(
            model,
            HybridConfig(
                lse=cfg.lse,
                ridge=cfg.ridge,
                premise_lr=cfg.premise_lr,
                premise_optimizer=cfg.premise_optimizer,
                grad_clip=cfg.grad_clip,
            ),
        )
        stepper = None
    else:
        opt = None
        stepper = torch.optim.Adam(
            [
                {"params": list(model.premise_parameters()), "lr": cfg.premise_lr},
                {"params": [model.consequent], "lr": cfg.consequent_lr},
            ]
        )

    n = len(Xt)
    batch = cfg.batch_size or n
    gen = torch.Generator().manual_seed(cfg.seed)

    for epoch in range(cfg.epochs):
        model.train()
        if cfg.method == "hybrid":
            # LSE on the full training set, then one premise step per mini-batch
            opt.forward_pass(Xt, yt)
            if batch >= n:
                train_loss = opt.backward_pass(Xt, yt)
            else:
                perm = torch.randperm(n, generator=gen)
                losses = []
                for i in range(0, n, batch):
                    idx = perm[i : i + batch]
                    losses.append(opt.backward_pass(Xt[idx], yt[idx]))
                train_loss = float(np.mean(losses))
        else:
            if batch >= n:
                stepper.zero_grad(set_to_none=True)
                loss = torch.mean((model(Xt) - yt) ** 2)
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                stepper.step()
                train_loss = float(loss.detach())
            else:
                perm = torch.randperm(n, generator=gen)
                losses = []
                for i in range(0, n, batch):
                    idx = perm[i : i + batch]
                    stepper.zero_grad(set_to_none=True)
                    loss = torch.mean((model(Xt[idx]) - yt[idx]) ** 2)
                    loss.backward()
                    if cfg.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    stepper.step()
                    losses.append(float(loss.detach()))
                train_loss = float(np.mean(losses))

        model.eval()
        with torch.no_grad():
            val_loss = float(torch.mean((model(Xv) - yv) ** 2))
        history["train_mse"].append(train_loss)
        history["val_mse"].append(val_loss)

        if not np.isfinite(val_loss):
            break
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if val_loss < ref_val - cfg.min_delta:
            ref_val, stale = val_loss, 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
        if cfg.verbose and epoch % 25 == 0:
            print(f"  epoch {epoch:4d}  train {train_loss:.6f}  val {val_loss:.6f}")

    model.load_state_dict(best_state)
    return TrainResult(model, history, best_epoch, best_val, time.time() - start, model.n_params)


def evaluate(model: ANFIS, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred = model(_to_tensor(X)).numpy()
    return all_metrics(y, pred)


def predict(model: ANFIS, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(_to_tensor(X)).numpy()


def kfold_indices(n: int, k: int, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    out = []
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((train, test))
    return out


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_kwargs: Dict,
    train_config: Optional[TrainConfig] = None,
    k: int = 5,
    seed: int = 0,
    standardize: bool = True,
    collect_predictions: bool = False,
) -> Dict:
    """K-fold CV returning per-fold metrics plus out-of-fold predictions.

    Standardisation is fitted inside each fold, never on the full dataset, so
    no test-fold statistics leak into the fuzzy partition.
    """
    cfg = train_config or TrainConfig()
    folds = kfold_indices(len(X), k, seed=seed)
    per_fold: List[Dict[str, float]] = []
    oof = np.full(len(X), np.nan)
    stats: List[Dict[str, float]] = []
    seconds = 0.0
    n_params = 0

    for fold, (tr, te) in enumerate(folds):
        scaler = Standardizer().fit(X[tr], y[tr]) if standardize else None
        if scaler is not None:
            Xtr, ytr = scaler.transform(X[tr], y[tr])
            Xte = scaler.transform(X[te])
        else:
            Xtr, ytr, Xte = X[tr], y[tr], X[te]

        fold_cfg = TrainConfig(**{**asdict(cfg), "seed": cfg.seed + fold})
        model = ANFIS.from_data(Xtr, seed=cfg.seed + fold, **model_kwargs)
        res = fit(model, Xtr, ytr, fold_cfg)
        seconds += res.seconds
        n_params = model.n_params

        pred_s = predict(model, Xte)
        pred = scaler.inverse_y(pred_s) if scaler is not None else pred_s
        oof[te] = pred
        per_fold.append(all_metrics(y[te], pred))
        stats.append(model.firing_stats(_to_tensor(Xte)))

    summary = {
        f"{m}_mean": float(np.mean([f[m] for f in per_fold])) for m in per_fold[0]
    }
    summary.update(
        {f"{m}_std": float(np.std([f[m] for f in per_fold])) for m in per_fold[0]}
    )
    summary.update(
        {k_: float(np.mean([s[k_] for s in stats])) for k_ in stats[0]}
    )
    summary["n_params"] = int(n_params)
    summary["seconds"] = float(seconds)
    out = {"summary": summary, "per_fold": per_fold, "firing": stats}
    if collect_predictions:
        out["oof_predictions"] = oof
    return out
