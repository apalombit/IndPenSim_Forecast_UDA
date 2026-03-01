"""Loss functions and training loops for modular PieceLog sub-models.

Phase A: Independent sequential training of TimingModel → GrowthModel → DeclineModel
Phase B: Optional joint fine-tuning of CompositeModel end-to-end
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import pandas as pd

from .modular_piecelog import (
    CompositeModel,
    DeclineModel,
    GrowthModel,
    SplitDeclineModel,
    TimingModel,
    UngatedDeclineModel,
    initialize_decline_head,
    initialize_growth_head,
    initialize_split_decline_head,
    initialize_timing_head,
    initialize_ungated_decline_head,
)
from .piecelog_model import PARAM_NAMES, piecelog_torch
from .train import EarlyStopping


# ---------------------------------------------------------------------------
# Curve evaluation helpers
# ---------------------------------------------------------------------------

def _jittered_grid(
    t_lo: torch.Tensor,
    t_hi: torch.Tensor,
    n_points: int,
    training: bool,
) -> torch.Tensor:
    """Create a per-sample jittered time grid in [t_lo, t_hi].

    Args:
        t_lo: Lower bound, shape (B,).
        t_hi: Upper bound, shape (B,).
        n_points: Number of grid points.
        training: If True, add stratified jitter; else deterministic.

    Returns:
        Time grid, shape (B, N).
    """
    B = t_lo.shape[0]
    device = t_lo.device
    base = torch.linspace(0.0, 1.0, n_points, device=device).unsqueeze(0).expand(B, n_points)
    if training and n_points > 1:
        spacing = 1.0 / (n_points - 1)
        jitter = (torch.rand(B, n_points, device=device) - 0.5) * spacing
        base = (base + jitter).clamp(0.0, 1.0)
    span = (t_hi - t_lo).unsqueeze(1)  # (B, 1)
    return t_lo.unsqueeze(1) + base * span  # (B, N)


def _curve_mse(
    params_pred: dict[str, torch.Tensor],
    params_true_tensor: torch.Tensor,
    t_grid: torch.Tensor,
    conc_scale: float | None = None,
) -> torch.Tensor:
    """MSE between predicted and true piece-log curves on a time grid.

    Args:
        params_pred: Predicted parameters dict, each (B,).
        params_true_tensor: True parameters, shape (B, 7).
        t_grid: Time grid, shape (B, N).
        conc_scale: If provided, normalize both curves by this value.

    Returns:
        Scalar MSE loss.
    """
    B, N = t_grid.shape
    t_flat = t_grid.reshape(B * N)

    def _expand(p):
        return p.unsqueeze(1).expand(B, N).reshape(B * N)

    P_pred = piecelog_torch(
        t_flat,
        _expand(params_pred["K"]), _expand(params_pred["r"]),
        _expand(params_pred["t0"]), _expand(params_pred["lam"]),
        _expand(params_pred["t_lag"]), _expand(params_pred["t_break"]),
        _expand(params_pred["slope"]),
    )

    K_t, r_t, t0_t, lam_t, tlag_t, tbreak_t, slope_t = [
        params_true_tensor[:, i].detach() for i in range(7)
    ]
    P_true = piecelog_torch(
        t_flat,
        _expand(K_t), _expand(r_t), _expand(t0_t), _expand(lam_t),
        _expand(tlag_t), _expand(tbreak_t), _expand(slope_t),
    )

    if conc_scale is not None:
        P_pred = P_pred / conc_scale
        P_true = P_true / conc_scale

    return nn.functional.mse_loss(P_pred, P_true)


# ---------------------------------------------------------------------------
# Phase-specific loss functions
# ---------------------------------------------------------------------------

def timing_loss(
    model: nn.Module,
    x: torch.Tensor,
    params_true: torch.Tensor,
    alpha: float = 0.1,
    param_stats: dict | None = None,
    n_curve_points: int = 32,
    margin_frac: float = 0.1,
    conc_scale: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Loss for TimingModel: curve on [0, t_lag_true + margin] + param MSE for t_lag.

    Args:
        model: TimingModel.
        x: Input, shape (B, T, C).
        params_true: Fitted params, shape (B, 7).
        alpha: Weight for parameter loss.
        param_stats: For z-score normalization of t_lag.
        n_curve_points: Grid points for curve loss.
        margin_frac: Fractional margin beyond t_lag_true.
        conc_scale: Concentration scale for normalization.

    Returns:
        (total_loss, {"curve": float, "param": float})
    """
    if param_stats is None:
        warnings.warn(
            "param_stats is None — parameter loss will use raw (unnormalized) values. "
            "Pass param_stats from create_piecelog_dataloaders() for balanced training.",
            stacklevel=2,
        )

    timing_params = model(x)
    t_lag_pred = timing_params["t_lag"]
    T_max = model.head.T_max

    # For curve evaluation, use true params for growth/decline, predicted t_lag
    t_lag_true = params_true[:, 4]
    margin = margin_frac * T_max
    t_hi = (t_lag_true + margin).clamp(max=T_max)
    t_lo = torch.zeros_like(t_hi)

    # Build full param dict: predicted t_lag, true everything else
    full_params_pred = {
        "K": params_true[:, 0].detach(),
        "r": params_true[:, 1].detach(),
        "t0": params_true[:, 2].detach(),
        "lam": params_true[:, 3].detach(),
        "t_lag": t_lag_pred,
        "t_break": params_true[:, 5].detach(),
        "slope": params_true[:, 6].detach(),
    }

    t_grid = _jittered_grid(t_lo, t_hi, n_curve_points, model.training)
    L_curve = _curve_mse(full_params_pred, params_true, t_grid, conc_scale)

    # Parameter MSE for t_lag
    pred_i = t_lag_pred
    true_i = t_lag_true
    if param_stats is not None and "t_lag" in param_stats:
        mean = param_stats["t_lag"]["mean"]
        std = param_stats["t_lag"]["std"]
        pred_i = (pred_i - mean) / std
        true_i = (true_i - mean) / std
    L_param = nn.functional.mse_loss(pred_i, true_i)

    total = L_curve + alpha * L_param
    return total, {"curve": L_curve.item(), "param": L_param.item()}


def growth_loss(
    model: nn.Module,
    x: torch.Tensor,
    params_true: torch.Tensor,
    t_lag_input: torch.Tensor,
    alpha: float = 0.1,
    param_stats: dict | None = None,
    n_curve_points: int = 32,
    conc_scale: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Loss for GrowthModel: curve on [t_lag, t_break_true] + param MSE for K, r, t0, lam.

    Args:
        model: GrowthModel.
        x: Input, shape (B, T, C).
        params_true: Fitted params, shape (B, 7).
        t_lag_input: t_lag values from TimingModel (detached), shape (B,).
        alpha: Weight for parameter loss.
        param_stats: For z-score normalization.
        n_curve_points: Grid points for curve loss.
        conc_scale: Concentration scale.

    Returns:
        (total_loss, {"curve": float, "param": float})
    """
    if param_stats is None:
        warnings.warn(
            "param_stats is None — parameter loss will use raw (unnormalized) values. "
            "Pass param_stats from create_piecelog_dataloaders() for balanced training.",
            stacklevel=2,
        )

    growth_params = model(x)
    T_max = model.head.T_max

    t_break_true = params_true[:, 5]
    t_lo = t_lag_input.detach()
    t_hi = t_break_true.detach().clamp(max=T_max)
    # Ensure t_lo < t_hi
    t_hi = torch.max(t_hi, t_lo + 1.0)

    full_params_pred = {
        "K": growth_params["K"],
        "r": growth_params["r"],
        "t0": growth_params["t0"],
        "lam": growth_params["lam"],
        "t_lag": t_lag_input.detach(),
        "t_break": params_true[:, 5].detach(),
        "slope": params_true[:, 6].detach(),
    }

    t_grid = _jittered_grid(t_lo, t_hi, n_curve_points, model.training)
    L_curve = _curve_mse(full_params_pred, params_true, t_grid, conc_scale)

    # Parameter MSE for K, r, t0, lam
    growth_names = ["K", "r", "t0", "lam"]
    growth_indices = [0, 1, 2, 3]
    L_param = torch.tensor(0.0, device=x.device)
    for name, idx in zip(growth_names, growth_indices):
        pred_i = growth_params[name]
        true_i = params_true[:, idx]
        if param_stats is not None and name in param_stats:
            mean = param_stats[name]["mean"]
            std = param_stats[name]["std"]
            pred_i = (pred_i - mean) / std
            true_i = (true_i - mean) / std
        L_param = L_param + nn.functional.mse_loss(pred_i, true_i)
    L_param = L_param / len(growth_names)

    total = L_curve + alpha * L_param
    return total, {"curve": L_curve.item(), "param": L_param.item()}


def decline_loss(
    model: nn.Module,
    x: torch.Tensor,
    params_true: torch.Tensor,
    t_lag_input: torch.Tensor,
    growth_params: dict[str, torch.Tensor],
    t_cutoff: torch.Tensor | None = None,
    alpha: float = 0.1,
    gate_weight: float = 0.01,
    slope_weight: float = 0.1,
    param_stats: dict | None = None,
    n_curve_points: int = 32,
    margin_frac: float = 0.1,
    conc_scale: float | None = None,
    gate_threshold: float = 0.01,
    detach_gate: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Loss for DeclineModel: curve on [t_break-margin, T_max] + param MSE + BCE gate.

    Args:
        model: DeclineModel.
        x: Input, shape (B, T, C).
        params_true: Fitted params, shape (B, 7).
        t_lag_input: t_lag from TimingModel (detached), shape (B,).
        growth_params: K, r, t0, lam from GrowthModel (detached), each (B,).
        t_cutoff: Absolute time at end of input window, shape (B,).
            If None, defaults to 0.9 * T_max.
        alpha: Weight for parameter loss.
        gate_weight: Weight for BCE gate loss.
        slope_weight: Weight for masked conditional slope/t_break supervision.
            Only applies to samples where slope_true > gate_threshold.
        param_stats: Unused (kept for API compatibility).
        n_curve_points: Grid points for curve loss.
        margin_frac: Fractional margin before t_break_true.
        conc_scale: Concentration scale.
        gate_threshold: Slope values below this → gate_target = 0.
        detach_gate: If True, detach gate before slope multiplication so
            L_curve gradient flows only to slope/t_break (not gate).

    Returns:
        (total_loss, {"curve": float, "param": float, "gate": float,
                      "slope_cond": float})
    """
    T_max = model.head.T_max
    if t_cutoff is None:
        t_cutoff = torch.full((x.shape[0],), 0.9 * T_max, device=x.device)
    t_cutoff_norm = t_cutoff / T_max

    decline_params = model(x, t_cutoff_norm)

    t_break_true = params_true[:, 5]
    margin = margin_frac * T_max
    t_lo = (t_break_true.detach() - margin).clamp(min=0.0)
    t_hi = torch.full_like(t_lo, T_max)

    # Full params: predicted decline + detached upstream
    gate = decline_params["decline_gate"]
    if detach_gate:
        gate = gate.detach()

    full_params_pred = {
        "K": growth_params["K"].detach(),
        "r": growth_params["r"].detach(),
        "t0": growth_params["t0"].detach(),
        "lam": growth_params["lam"].detach(),
        "t_lag": t_lag_input.detach(),
        "t_break": decline_params["t_break"],
        "slope": decline_params["slope"] * gate,
    }

    t_grid = _jittered_grid(t_lo, t_hi, n_curve_points, model.training)
    L_curve = _curve_mse(full_params_pred, params_true, t_grid, conc_scale)

    # Parameter MSE: t_break only (T_max normalized); slope handled in L_slope_cond via log-space
    tb_pred = decline_params["t_break"] / T_max
    tb_true = params_true[:, 5] / T_max
    L_param = nn.functional.mse_loss(tb_pred, tb_true)

    # BCE gate loss
    gate_target = (params_true[:, 6] > gate_threshold).float()
    L_gate = nn.functional.binary_cross_entropy(
        decline_params["decline_gate"], gate_target,
    )

    # Masked conditional slope + t_break supervision (only declining batches)
    L_slope_cond = torch.tensor(0.0, device=x.device)
    if slope_weight > 0:
        decline_mask = params_true[:, 6] > gate_threshold
        if decline_mask.any():
            n_terms = 0
            # Slope supervision: log-space MSE on declining batches (no dataset stats needed)
            slope_pred_m = torch.log(decline_params["slope"][decline_mask] + 1e-6)
            slope_true_m = torch.log(params_true[:, 6][decline_mask] + 1e-6)
            L_slope_cond = L_slope_cond + nn.functional.mse_loss(slope_pred_m, slope_true_m)
            n_terms += 1
            # t_break supervision (T_max normalization)
            tb_pred_m = decline_params["t_break"][decline_mask] / T_max
            tb_true_m = params_true[:, 5][decline_mask] / T_max
            L_slope_cond = L_slope_cond + nn.functional.mse_loss(tb_pred_m, tb_true_m)
            n_terms += 1
            L_slope_cond = L_slope_cond / n_terms

    total = L_curve + alpha * L_param + gate_weight * L_gate + slope_weight * L_slope_cond
    return total, {
        "curve": L_curve.item(),
        "param": L_param.item(),
        "gate": L_gate.item(),
        "slope_cond": L_slope_cond.item(),
    }


def composite_loss(
    model: nn.Module,
    x: torch.Tensor,
    params_true: torch.Tensor,
    t_cutoff: torch.Tensor,
    alpha: float = 0.1,
    param_stats: dict | None = None,
    n_curve_points: int = 32,
    conc_scale: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Full curve loss for CompositeModel joint fine-tuning.

    Args:
        model: CompositeModel.
        x: Input, shape (B, T, C).
        params_true: Fitted params, shape (B, 7).
        t_cutoff: Per-sample input window end time, shape (B,).
        alpha: Weight for parameter loss.
        param_stats: For z-score normalization.
        n_curve_points: Grid points for curve loss.
        conc_scale: Concentration scale.

    Returns:
        (total_loss, {"curve": float, "param": float})
    """
    if param_stats is None:
        warnings.warn(
            "param_stats is None — parameter loss will use raw (unnormalized) values. "
            "Pass param_stats from create_piecelog_dataloaders() for balanced training.",
            stacklevel=2,
        )

    params_pred = model.get_parameters(x, t_cutoff=t_cutoff, hard_gate=False)

    t_lo = torch.zeros_like(t_cutoff)
    t_grid = _jittered_grid(t_lo, t_cutoff, n_curve_points, model.training)
    L_curve = _curve_mse(params_pred, params_true, t_grid, conc_scale)

    # Full parameter MSE
    L_param = torch.tensor(0.0, device=x.device)
    if alpha > 0:
        for i, name in enumerate(PARAM_NAMES):
            pred_i = params_pred[name]
            true_i = params_true[:, i]
            if param_stats is not None and name in param_stats:
                mean = param_stats[name]["mean"]
                std = param_stats[name]["std"]
                pred_i = (pred_i - mean) / std
                true_i = (true_i - mean) / std
            L_param = L_param + nn.functional.mse_loss(pred_i, true_i)
        L_param = L_param / len(PARAM_NAMES)

    total = L_curve + alpha * L_param
    return total, {"curve": L_curve.item(), "param": L_param.item()}


# ---------------------------------------------------------------------------
# Training loops for individual sub-models
# ---------------------------------------------------------------------------

def train_timing_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 20,
    alpha: float = 0.1,
    param_stats: dict | None = None,
    n_curve_points: int = 32,
    conc_scale: float | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Train TimingModel with early stopping.

    Returns:
        Dict with 'history', 'best_epoch'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": [], "train_curve": [], "train_param": []}

    for epoch in range(n_epochs):
        model.train()
        epoch_loss, epoch_curve, epoch_param, n = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(device)
            params_fitted = batch["params_fitted"].to(device)

            optimizer.zero_grad()
            loss, ld = timing_loss(
                model, x, params_fitted, alpha=alpha,
                param_stats=param_stats, n_curve_points=n_curve_points,
                conc_scale=conc_scale,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_curve += ld["curve"]
            epoch_param += ld["param"]
            n += 1

        history["train_loss"].append(epoch_loss / n)
        history["train_curve"].append(epoch_curve / n)
        history["train_param"].append(epoch_param / n)

        # Validation
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                params_fitted = batch["params_fitted"].to(device)
                loss, _ = timing_loss(
                    model, x, params_fitted, alpha=alpha,
                    param_stats=param_stats, n_curve_points=n_curve_points,
                    conc_scale=conc_scale,
                )
                val_loss_sum += loss.item()
                val_n += 1
        val_loss = val_loss_sum / val_n
        history["val_loss"].append(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"[Timing] Epoch {epoch+1}/{n_epochs} | Loss: {history['train_loss'][-1]:.4f} | Val: {val_loss:.4f}")

        if early_stopping(val_loss, model):
            if verbose:
                print(f"[Timing] Early stopping at epoch {epoch + 1}")
            break

    early_stopping.load_best_state(model)
    model.to(device)
    return {"history": history, "best_epoch": epoch + 1 - early_stopping.counter}


def train_growth_model(
    model: nn.Module,
    timing_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 20,
    alpha: float = 0.1,
    param_stats: dict | None = None,
    n_curve_points: int = 32,
    conc_scale: float | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Train GrowthModel with frozen TimingModel providing t_lag.

    Returns:
        Dict with 'history', 'best_epoch'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    timing_model = timing_model.to(device)
    timing_model.eval()
    for p in timing_model.parameters():
        p.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": [], "train_curve": [], "train_param": []}

    for epoch in range(n_epochs):
        model.train()
        epoch_loss, epoch_curve, epoch_param, n = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(device)
            params_fitted = batch["params_fitted"].to(device)

            with torch.no_grad():
                t_lag_input = timing_model(x)["t_lag"]

            optimizer.zero_grad()
            loss, ld = growth_loss(
                model, x, params_fitted, t_lag_input, alpha=alpha,
                param_stats=param_stats, n_curve_points=n_curve_points,
                conc_scale=conc_scale,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_curve += ld["curve"]
            epoch_param += ld["param"]
            n += 1

        history["train_loss"].append(epoch_loss / n)
        history["train_curve"].append(epoch_curve / n)
        history["train_param"].append(epoch_param / n)

        # Validation
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                params_fitted = batch["params_fitted"].to(device)
                t_lag_input = timing_model(x)["t_lag"]
                loss, _ = growth_loss(
                    model, x, params_fitted, t_lag_input, alpha=alpha,
                    param_stats=param_stats, n_curve_points=n_curve_points,
                    conc_scale=conc_scale,
                )
                val_loss_sum += loss.item()
                val_n += 1
        val_loss = val_loss_sum / val_n
        history["val_loss"].append(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"[Growth] Epoch {epoch+1}/{n_epochs} | Loss: {history['train_loss'][-1]:.4f} | Val: {val_loss:.4f}")

        if early_stopping(val_loss, model):
            if verbose:
                print(f"[Growth] Early stopping at epoch {epoch + 1}")
            break

    early_stopping.load_best_state(model)
    model.to(device)
    return {"history": history, "best_epoch": epoch + 1 - early_stopping.counter}


def _gate_weight_schedule(
    epoch: int,
    n_epochs: int,
    gate_weight: float,
    gate_weight_min: float,
    warmup_frac: float,
) -> float:
    """Cosine-anneal gate_weight from full value down to gate_weight_min.

    During the warmup phase (first ``warmup_frac`` of epochs) the weight stays
    at ``gate_weight``.  After that it follows a cosine decay to
    ``gate_weight_min``.
    """
    warmup_end = int(n_epochs * warmup_frac)
    if epoch < warmup_end:
        return gate_weight
    progress = (epoch - warmup_end) / max(1, n_epochs - warmup_end - 1)
    return gate_weight_min + (gate_weight - gate_weight_min) * 0.5 * (
        1 + math.cos(math.pi * progress)
    )


def train_decline_model(
    model: nn.Module,
    timing_model: nn.Module,
    growth_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 20,
    alpha: float = 0.1,
    gate_weight: float = 0.01,
    gate_weight_min: float | None = None,
    gate_warmup_frac: float = 0.3,
    slope_weight: float = 0.0,
    detach_gate_after_warmup: bool = False,
    param_stats: dict | None = None,
    n_curve_points: int = 32,
    conc_scale: float | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Train DeclineModel with frozen upstream models.

    Args:
        gate_weight_min: Target gate weight after annealing. If *None*
            (default), no annealing is applied and ``gate_weight`` is
            used throughout.
        gate_warmup_frac: Fraction of epochs to keep full ``gate_weight``
            before cosine decay begins.
        slope_weight: Weight for masked conditional slope/t_break supervision.
        detach_gate_after_warmup: If True, detach gate from L_curve after
            the warmup phase so all curve gradient flows to slope/t_break.

    Returns:
        Dict with 'history', 'best_epoch'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    timing_model = timing_model.to(device)
    growth_model = growth_model.to(device)
    timing_model.eval()
    growth_model.eval()
    for p in timing_model.parameters():
        p.requires_grad = False
    for p in growth_model.parameters():
        p.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience)

    use_annealing = gate_weight_min is not None
    warmup_end = int(n_epochs * gate_warmup_frac)

    history = {
        "train_loss": [], "val_loss": [],
        "train_curve": [], "train_param": [], "train_gate": [],
        "train_slope_cond": [], "gate_weight": [],
    }

    for epoch in range(n_epochs):
        gw = (
            _gate_weight_schedule(
                epoch, n_epochs, gate_weight, gate_weight_min, gate_warmup_frac,
            )
            if use_annealing
            else gate_weight
        )
        dg = detach_gate_after_warmup and epoch >= warmup_end

        model.train()
        epoch_loss, epoch_curve, epoch_param, epoch_gate, epoch_slope_cond, n_batch = (
            0.0, 0.0, 0.0, 0.0, 0.0, 0,
        )
        for batch in train_loader:
            x = batch["x"].to(device)
            params_fitted = batch["params_fitted"].to(device)
            t_cutoff = batch["t_cutoff"].to(device)

            with torch.no_grad():
                t_lag_input = timing_model(x)["t_lag"]
                gp = growth_model(x)

            optimizer.zero_grad()
            loss, ld = decline_loss(
                model, x, params_fitted, t_lag_input, gp,
                t_cutoff=t_cutoff, alpha=alpha, gate_weight=gw,
                slope_weight=slope_weight, param_stats=param_stats,
                n_curve_points=n_curve_points, conc_scale=conc_scale, detach_gate=dg,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_curve += ld["curve"]
            epoch_param += ld["param"]
            epoch_gate += ld["gate"]
            epoch_slope_cond += ld["slope_cond"]
            n_batch += 1

        history["train_loss"].append(epoch_loss / n_batch)
        history["train_curve"].append(epoch_curve / n_batch)
        history["train_param"].append(epoch_param / n_batch)
        history["train_gate"].append(epoch_gate / n_batch)
        history["train_slope_cond"].append(epoch_slope_cond / n_batch)
        history["gate_weight"].append(gw)

        # Validation
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                params_fitted = batch["params_fitted"].to(device)
                t_cutoff = batch["t_cutoff"].to(device)
                t_lag_input = timing_model(x)["t_lag"]
                gp = growth_model(x)
                loss, _ = decline_loss(
                    model, x, params_fitted, t_lag_input, gp,
                    t_cutoff=t_cutoff, alpha=alpha, gate_weight=gw,
                    slope_weight=slope_weight, param_stats=param_stats,
                    n_curve_points=n_curve_points, conc_scale=conc_scale,
                )
                val_loss_sum += loss.item()
                val_n += 1
        val_loss = val_loss_sum / val_n
        history["val_loss"].append(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"[Decline] Epoch {epoch+1}/{n_epochs} | Loss: {history['train_loss'][-1]:.4f} | Val: {val_loss:.4f}")

        if early_stopping(val_loss, model):
            if verbose:
                print(f"[Decline] Early stopping at epoch {epoch + 1}")
            break

    early_stopping.load_best_state(model)
    model.to(device)
    return {"history": history, "best_epoch": epoch + 1 - early_stopping.counter}


def train_composite_finetune(
    composite: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 50,
    lr: float = 1e-5,
    weight_decay: float = 1e-4,
    patience: int = 15,
    alpha: float = 0.05,
    param_stats: dict | None = None,
    n_curve_points: int = 32,
    conc_scale: float | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Joint fine-tuning of CompositeModel on full curve loss.

    Returns:
        Dict with 'history', 'best_epoch'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    composite = composite.to(device)
    optimizer = AdamW(composite.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": [], "train_curve": [], "train_param": []}

    for epoch in range(n_epochs):
        composite.train()
        epoch_loss, epoch_curve, epoch_param, n = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(device)
            params_fitted = batch["params_fitted"].to(device)
            t_cutoff = batch["t_cutoff"].to(device)

            optimizer.zero_grad()
            loss, ld = composite_loss(
                composite, x, params_fitted, t_cutoff,
                alpha=alpha, param_stats=param_stats,
                n_curve_points=n_curve_points, conc_scale=conc_scale,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_curve += ld["curve"]
            epoch_param += ld["param"]
            n += 1

        history["train_loss"].append(epoch_loss / n)
        history["train_curve"].append(epoch_curve / n)
        history["train_param"].append(epoch_param / n)

        # Validation
        composite.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                params_fitted = batch["params_fitted"].to(device)
                t_cutoff = batch["t_cutoff"].to(device)
                loss, _ = composite_loss(
                    composite, x, params_fitted, t_cutoff,
                    alpha=alpha, param_stats=param_stats,
                    n_curve_points=n_curve_points, conc_scale=conc_scale,
                )
                val_loss_sum += loss.item()
                val_n += 1
        val_loss = val_loss_sum / val_n
        history["val_loss"].append(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"[Finetune] Epoch {epoch+1}/{n_epochs} | Loss: {history['train_loss'][-1]:.4f} | Val: {val_loss:.4f}")

        if early_stopping(val_loss, composite):
            if verbose:
                print(f"[Finetune] Early stopping at epoch {epoch + 1}")
            break

    early_stopping.load_best_state(composite)
    composite.to(device)
    return {"history": history, "best_epoch": epoch + 1 - early_stopping.counter}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def train_modular_pipeline(
    composite: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs_per_phase: int = 100,
    n_epochs_finetune: int = 50,
    lr: float = 1e-4,
    lr_finetune: float = 1e-5,
    weight_decay: float = 1e-4,
    patience: int = 20,
    alpha: float = 0.1,
    decline_alpha: float | None = None,
    gate_weight_min: float | None = None,
    gate_warmup_frac: float = 0.3,
    slope_weight: float = 0.0,
    detach_gate_after_warmup: bool = False,
    param_stats: dict | None = None,
    n_curve_points: int = 32,
    conc_scale: float | None = None,
    finetune: bool = True,
    device: torch.device | None = None,
    verbose: bool = True,
    decline_train_loader: DataLoader | None = None,
    decline_val_loader: DataLoader | None = None,
) -> dict:
    """Run full modular training pipeline: Phase A (sequential) + Phase B (finetune).

    Args:
        decline_alpha: Override ``alpha`` for decline phase. If *None*,
            uses the same ``alpha`` as the other phases.
        gate_weight_min: Target gate weight after cosine annealing. If
            *None*, no annealing is applied.
        gate_warmup_frac: Fraction of epochs to keep full gate weight
            before cosine decay begins.
        slope_weight: Weight for masked conditional slope/t_break supervision
            in the decline phase.
        detach_gate_after_warmup: If True, detach gate from L_curve after
            the warmup phase during decline training.
        decline_train_loader: Optional DataLoader for Phase A-3 (decline
            training). If None, falls back to ``train_loader``. Pass a
            full-batch loader (``full_batch=True``) so the decline model
            always sees the complete sequence.
        decline_val_loader: Optional DataLoader for Phase A-3 validation.
            If None, falls back to ``val_loader``.

    Returns:
        Dict with results for each phase.
    """
    results = {}

    if verbose:
        print("=" * 60)
        print("Phase A-1: Training TimingModel")
        print("=" * 60)
    results["timing"] = train_timing_model(
        composite.timing_model, train_loader, val_loader,
        n_epochs=n_epochs_per_phase, lr=lr, weight_decay=weight_decay,
        patience=patience, alpha=alpha, param_stats=param_stats,
        n_curve_points=n_curve_points, conc_scale=conc_scale,
        device=device, verbose=verbose,
    )

    if verbose:
        print("=" * 60)
        print("Phase A-2: Training GrowthModel")
        print("=" * 60)
    results["growth"] = train_growth_model(
        composite.growth_model, composite.timing_model,
        train_loader, val_loader,
        n_epochs=n_epochs_per_phase, lr=lr, weight_decay=weight_decay,
        patience=patience, alpha=alpha, param_stats=param_stats,
        n_curve_points=n_curve_points, conc_scale=conc_scale,
        device=device, verbose=verbose,
    )

    if verbose:
        print("=" * 60)
        print("Phase A-3: Training DeclineModel")
        print("=" * 60)
    _decline_train = decline_train_loader if decline_train_loader is not None else train_loader
    _decline_val = decline_val_loader if decline_val_loader is not None else val_loader
    results["decline"] = train_decline_model(
        composite.decline_model, composite.timing_model, composite.growth_model,
        _decline_train, _decline_val,
        n_epochs=n_epochs_per_phase, lr=lr, weight_decay=weight_decay,
        patience=patience,
        alpha=decline_alpha if decline_alpha is not None else alpha,
        gate_weight_min=gate_weight_min, gate_warmup_frac=gate_warmup_frac,
        slope_weight=slope_weight,
        detach_gate_after_warmup=detach_gate_after_warmup,
        param_stats=param_stats,
        n_curve_points=n_curve_points, conc_scale=conc_scale,
        device=device, verbose=verbose,
    )

    if finetune:
        if verbose:
            print("=" * 60)
            print("Phase B: Joint Fine-tuning")
            print("=" * 60)
        results["finetune"] = train_composite_finetune(
            composite, train_loader, val_loader,
            n_epochs=n_epochs_finetune, lr=lr_finetune,
            weight_decay=weight_decay, patience=patience,
            alpha=alpha * 0.5, param_stats=param_stats,
            n_curve_points=n_curve_points, conc_scale=conc_scale,
            device=device, verbose=verbose,
        )

    return results


# ---------------------------------------------------------------------------
# Per-sub-model retrain helpers
# ---------------------------------------------------------------------------

_ENCODER_DEFAULTS: dict[str, int | float] = {
    "patch_len": 16,
    "patch_stride": 8,
    "d_model": 32,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 128,
    "dropout": 0.1,
    "head_hidden": 32,
}


def _sub_model_kwargs(
    encoder,
    config: dict | None,
) -> dict:
    """Build constructor kwargs for a sub-model.

    Data-shape dimensions (``n_features``, ``seq_len``) are extracted from the
    existing *encoder*.  Architecture parameters come from *config*, falling
    back to ``_ENCODER_DEFAULTS``.
    """
    cfg = dict(_ENCODER_DEFAULTS)
    if config is not None:
        cfg.update(config)
    return dict(
        n_features=encoder.n_features,
        seq_len=encoder.seq_len,
        patch_len=cfg["patch_len"],
        patch_stride=cfg["patch_stride"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        head_hidden=cfg["head_hidden"],
    )


def retrain_timing(
    composite: CompositeModel,
    fitted_params_df: pd.DataFrame,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict | None = None,
    **train_kwargs,
) -> dict:
    """Create a fresh TimingModel, swap it into *composite*, and train it.

    Args:
        composite: Existing CompositeModel whose ``timing_model`` will be
            replaced.
        fitted_params_df: Fitted parameters for head-bias initialization.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Architecture overrides (keys from ``_ENCODER_DEFAULTS``).
            If *None*, current defaults are used.
        **train_kwargs: Forwarded to :func:`train_timing_model`.

    Returns:
        Result dict from :func:`train_timing_model`.
    """
    kwargs = _sub_model_kwargs(composite.timing_model.encoder, config)
    T_max = composite.timing_model.head.T_max
    new_model = TimingModel(T_max=T_max, **kwargs)
    initialize_timing_head(new_model, fitted_params_df)
    composite.timing_model = new_model
    return train_timing_model(new_model, train_loader, val_loader, **train_kwargs)


def retrain_growth(
    composite: CompositeModel,
    fitted_params_df: pd.DataFrame,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict | None = None,
    **train_kwargs,
) -> dict:
    """Create a fresh GrowthModel, swap it into *composite*, and train it.

    The existing (frozen) ``composite.timing_model`` provides ``t_lag``.

    Args:
        composite: Existing CompositeModel whose ``growth_model`` will be
            replaced.
        fitted_params_df: Fitted parameters for head-bias initialization.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Architecture overrides.
        **train_kwargs: Forwarded to :func:`train_growth_model`.

    Returns:
        Result dict from :func:`train_growth_model`.
    """
    kwargs = _sub_model_kwargs(composite.growth_model.encoder, config)
    T_max = composite.growth_model.head.T_max
    K_scale = composite.growth_model.head.K_scale
    new_model = GrowthModel(T_max=T_max, K_scale=K_scale, **kwargs)
    initialize_growth_head(new_model, fitted_params_df)
    composite.growth_model = new_model
    return train_growth_model(
        new_model, composite.timing_model, train_loader, val_loader, **train_kwargs,
    )


def retrain_decline(
    composite: CompositeModel,
    fitted_params_df: pd.DataFrame,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict | None = None,
    **train_kwargs,
) -> dict:
    """Create a fresh DeclineModel, swap it into *composite*, and train it.

    The existing (frozen) ``composite.timing_model`` and
    ``composite.growth_model`` provide upstream parameters.

    Args:
        composite: Existing CompositeModel whose ``decline_model`` will be
            replaced.
        fitted_params_df: Fitted parameters for head-bias initialization.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Architecture overrides.
        **train_kwargs: Forwarded to :func:`train_decline_model`.

    Returns:
        Result dict from :func:`train_decline_model`.
    """
    kwargs = _sub_model_kwargs(composite.decline_model.encoder, config)
    T_max = composite.decline_model.head.T_max
    new_model = DeclineModel(T_max=T_max, **kwargs)
    initialize_decline_head(new_model, fitted_params_df)
    composite.decline_model = new_model
    return train_decline_model(
        new_model,
        composite.timing_model,
        composite.growth_model,
        train_loader,
        val_loader,
        **train_kwargs,
    )


def retrain_split_decline(
    composite: CompositeModel,
    fitted_params_df: pd.DataFrame,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict | None = None,
    **train_kwargs,
) -> dict:
    """Create a fresh SplitDeclineModel, swap it into *composite*, and train it.

    Uses step4c's split-head pattern (separate Linear layers per output)
    while keeping the same training loop and loss function as DeclineModel.

    Args:
        composite: Existing CompositeModel whose ``decline_model`` will be
            replaced with a SplitDeclineModel.
        fitted_params_df: Fitted parameters for head-bias initialization.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Architecture overrides.
        **train_kwargs: Forwarded to :func:`train_decline_model`.

    Returns:
        Result dict from :func:`train_decline_model`.
    """
    kwargs = _sub_model_kwargs(composite.decline_model.encoder, config)
    T_max = composite.decline_model.head.T_max
    new_model = SplitDeclineModel(T_max=T_max, **kwargs)
    initialize_split_decline_head(new_model, fitted_params_df)
    composite.decline_model = new_model
    return train_decline_model(
        new_model,
        composite.timing_model,
        composite.growth_model,
        train_loader,
        val_loader,
        **train_kwargs,
    )


# ---------------------------------------------------------------------------
# Ungated decline: loss, training loop, retrain helper
# ---------------------------------------------------------------------------

def ungated_decline_loss(
    model: nn.Module,
    x: torch.Tensor,
    params_true: torch.Tensor,
    t_lag_input: torch.Tensor,
    growth_params: dict[str, torch.Tensor],
    t_cutoff: torch.Tensor | None = None,
    alpha: float = 0.1,
    param_stats: dict | None = None,
    n_curve_points: int = 32,
    margin_frac: float = 0.1,
    conc_scale: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Loss for UngatedDeclineModel: curve MSE + param MSE. No gate loss.

    Slope goes directly into the full params — no gate multiplication.
    L_total = L_curve + alpha * L_param

    Args:
        model: UngatedDeclineModel.
        x: Input, shape (B, T, C).
        params_true: Fitted params, shape (B, 7).
        t_lag_input: t_lag from TimingModel (detached), shape (B,).
        growth_params: K, r, t0, lam from GrowthModel (detached), each (B,).
        t_cutoff: Absolute time at end of input window, shape (B,). If None,
            defaults to 0.9 * T_max.
        alpha: Weight for parameter loss.
        param_stats: Unused (kept for API compatibility).
        n_curve_points: Grid points for curve loss.
        margin_frac: Fractional margin before t_break_true.
        conc_scale: Concentration scale.

    Returns:
        (total_loss, {"curve": float, "param": float})
    """
    T_max = model.head.T_max
    if t_cutoff is None:
        t_cutoff = torch.full((x.shape[0],), 0.9 * T_max, device=x.device)
    t_cutoff_norm = t_cutoff / T_max
    decline_params = model(x, t_cutoff_norm)

    t_break_true = params_true[:, 5]
    margin = margin_frac * T_max
    t_lo = (t_break_true.detach() - margin).clamp(min=0.0)
    t_hi = torch.full_like(t_lo, T_max)

    # Full params: predicted decline (no gate multiplication) + detached upstream
    full_params_pred = {
        "K": growth_params["K"].detach(),
        "r": growth_params["r"].detach(),
        "t0": growth_params["t0"].detach(),
        "lam": growth_params["lam"].detach(),
        "t_lag": t_lag_input.detach(),
        "t_break": decline_params["t_break"],
        "slope": decline_params["slope"],
    }

    t_grid = _jittered_grid(t_lo, t_hi, n_curve_points, model.training)
    L_curve = _curve_mse(full_params_pred, params_true, t_grid, conc_scale)

    # Parameter MSE: t_break (T_max normalized) + slope (log-space, no dataset stats)
    tb_pred = decline_params["t_break"] / T_max
    tb_true = params_true[:, 5] / T_max
    L_param = nn.functional.mse_loss(tb_pred, tb_true)

    slope_pred_log = torch.log(decline_params["slope"] + 1e-6)
    slope_true_log = torch.log(params_true[:, 6] + 1e-6)
    L_param = (L_param + nn.functional.mse_loss(slope_pred_log, slope_true_log)) / 2

    total = L_curve + alpha * L_param
    return total, {"curve": L_curve.item(), "param": L_param.item()}


def train_ungated_decline_model(
    model: nn.Module,
    timing_model: nn.Module,
    growth_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 20,
    alpha: float = 0.1,
    param_stats: dict | None = None,
    n_curve_points: int = 32,
    conc_scale: float | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Train UngatedDeclineModel with frozen upstream models.

    Returns:
        Dict with 'history', 'best_epoch'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    timing_model = timing_model.to(device)
    growth_model = growth_model.to(device)
    timing_model.eval()
    growth_model.eval()
    for p in timing_model.parameters():
        p.requires_grad = False
    for p in growth_model.parameters():
        p.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience)

    history = {
        "train_loss": [], "val_loss": [],
        "train_curve": [], "train_param": [],
    }

    for epoch in range(n_epochs):
        model.train()
        epoch_loss, epoch_curve, epoch_param, n_batch = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(device)
            params_fitted = batch["params_fitted"].to(device)
            t_cutoff = batch["t_cutoff"].to(device)

            with torch.no_grad():
                t_lag_input = timing_model(x)["t_lag"]
                gp = growth_model(x)

            optimizer.zero_grad()
            loss, ld = ungated_decline_loss(
                model, x, params_fitted, t_lag_input, gp,
                t_cutoff=t_cutoff, alpha=alpha, param_stats=param_stats,
                n_curve_points=n_curve_points, conc_scale=conc_scale,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_curve += ld["curve"]
            epoch_param += ld["param"]
            n_batch += 1

        history["train_loss"].append(epoch_loss / n_batch)
        history["train_curve"].append(epoch_curve / n_batch)
        history["train_param"].append(epoch_param / n_batch)

        # Validation
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                params_fitted = batch["params_fitted"].to(device)
                t_cutoff = batch["t_cutoff"].to(device)
                t_lag_input = timing_model(x)["t_lag"]
                gp = growth_model(x)
                loss, _ = ungated_decline_loss(
                    model, x, params_fitted, t_lag_input, gp,
                    t_cutoff=t_cutoff, alpha=alpha, param_stats=param_stats,
                    n_curve_points=n_curve_points, conc_scale=conc_scale,
                )
                val_loss_sum += loss.item()
                val_n += 1
        val_loss = val_loss_sum / val_n
        history["val_loss"].append(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"[UngatedDecline] Epoch {epoch+1}/{n_epochs} | Loss: {history['train_loss'][-1]:.4f} | Val: {val_loss:.4f}")

        if early_stopping(val_loss, model):
            if verbose:
                print(f"[UngatedDecline] Early stopping at epoch {epoch + 1}")
            break

    early_stopping.load_best_state(model)
    model.to(device)
    return {"history": history, "best_epoch": epoch + 1 - early_stopping.counter}


def retrain_ungated_decline(
    composite: CompositeModel,
    fitted_params_df: pd.DataFrame,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict | None = None,
    **train_kwargs,
) -> dict:
    """Create a fresh UngatedDeclineModel, swap it into *composite*, and train.

    The existing (frozen) ``composite.timing_model`` and
    ``composite.growth_model`` provide upstream parameters.

    Args:
        composite: Existing CompositeModel whose ``decline_model`` will be
            replaced with an UngatedDeclineModel.
        fitted_params_df: Fitted parameters for head-bias initialization.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Architecture overrides.
        **train_kwargs: Forwarded to :func:`train_ungated_decline_model`.

    Returns:
        Result dict from :func:`train_ungated_decline_model`.
    """
    kwargs = _sub_model_kwargs(composite.decline_model.encoder, config)
    T_max = composite.decline_model.head.T_max
    new_model = UngatedDeclineModel(T_max=T_max, **kwargs)
    initialize_ungated_decline_head(new_model, fitted_params_df)
    composite.decline_model = new_model
    return train_ungated_decline_model(
        new_model,
        composite.timing_model,
        composite.growth_model,
        train_loader,
        val_loader,
        **train_kwargs,
    )
