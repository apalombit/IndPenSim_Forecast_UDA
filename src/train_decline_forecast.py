"""Loss functions and training loops for DeclineForecastModel.

Multi-task loss: BCE (decline classification) + MSE (delta timing) +
log-space MSE (slope, masked to declining batches only).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from .decline_dataset import T_MAX_NORM
from .train import EarlyStopping


def decline_forecast_loss(
    model: nn.Module,
    x: torch.Tensor,
    T_frac: torch.Tensor,
    decline_target: torch.Tensor,
    delta_target: torch.Tensor,
    slope_target: torch.Tensor,
    w_bce: float = 1.0,
    w_delta: float = 1.0,
    w_slope: float = 0.5,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """Compute multi-task loss for decline forecasting.

    Args:
        model: DeclineForecastModel.
        x: Input signals, shape (B, T, C).
        T_frac: Observation fraction, shape (B,).
        decline_target: Binary decline labels, shape (B,).
        delta_target: Normalized delta targets, shape (B,).
        slope_target: Raw slope targets, shape (B,).
        w_bce: Weight for BCE loss.
        w_delta: Weight for delta MSE loss.
        w_slope: Weight for slope log-MSE loss.
        eps: Epsilon for log stability.

    Returns:
        (total_loss, {"bce": float, "delta_mse": float, "slope_mse": float})
    """
    preds = model(x, T_frac)

    # BCE for decline classification
    L_bce = nn.functional.binary_cross_entropy(
        preds["decline_prob"], decline_target,
    )

    # MSE for delta_t_break (all samples)
    L_delta = nn.functional.mse_loss(preds["delta_t_break"], delta_target)

    # Log-space MSE for slope (declining batches only)
    decline_mask = decline_target > 0.5
    if decline_mask.any():
        slope_pred_log = torch.log(preds["slope"][decline_mask] + eps)
        slope_true_log = torch.log(slope_target[decline_mask] + eps)
        L_slope = nn.functional.mse_loss(slope_pred_log, slope_true_log)
    else:
        L_slope = torch.tensor(0.0, device=x.device)

    total = w_bce * L_bce + w_delta * L_delta + w_slope * L_slope

    return total, {
        "bce": L_bce.item(),
        "delta_mse": L_delta.item(),
        "slope_mse": L_slope.item(),
    }


def _compute_metrics(
    all_probs: np.ndarray,
    all_decline: np.ndarray,
    all_delta_pred: np.ndarray,
    all_delta_true: np.ndarray,
    all_slope_pred: np.ndarray,
    all_slope_true: np.ndarray,
) -> dict:
    """Compute evaluation metrics from accumulated predictions.

    Returns:
        Dict with auc_roc, accuracy, delta_mae_hours, delta_mae_pre,
        delta_mae_post, slope_mae_decline, slope_mae_nodecline.
    """
    metrics = {}

    # AUC-ROC
    if len(np.unique(all_decline)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(all_decline, all_probs))
    else:
        metrics["auc_roc"] = float("nan")

    # Accuracy at threshold 0.5
    preds_binary = (all_probs >= 0.5).astype(float)
    metrics["accuracy"] = float((preds_binary == all_decline).mean())

    # Delta MAE in hours (denormalized)
    delta_error_hours = np.abs(all_delta_pred - all_delta_true) * T_MAX_NORM
    metrics["delta_mae_hours"] = float(delta_error_hours.mean())

    # Delta MAE split by pre/post decline
    # Pre-decline: delta_true > 0 (T < t_break)
    pre_mask = all_delta_true > 0
    if pre_mask.any():
        metrics["delta_mae_pre"] = float(delta_error_hours[pre_mask].mean())
    else:
        metrics["delta_mae_pre"] = float("nan")

    # Post-decline: delta_true <= 0 (T >= t_break)
    post_mask = all_delta_true <= 0
    if post_mask.any():
        metrics["delta_mae_post"] = float(delta_error_hours[post_mask].mean())
    else:
        metrics["delta_mae_post"] = float("nan")

    # Slope MAE on declining batches
    decline_mask = all_decline > 0.5
    if decline_mask.any():
        metrics["slope_mae_decline"] = float(
            np.abs(all_slope_pred[decline_mask] - all_slope_true[decline_mask]).mean()
        )
    else:
        metrics["slope_mae_decline"] = float("nan")

    # Slope MAE on no-decline batches (should be near zero)
    nodecline_mask = ~decline_mask
    if nodecline_mask.any():
        metrics["slope_mae_nodecline"] = float(
            np.abs(all_slope_pred[nodecline_mask]).mean()
        )
    else:
        metrics["slope_mae_nodecline"] = float("nan")

    return metrics


def train_decline_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    w_bce: float = 1.0,
    w_delta: float = 1.0,
    w_slope: float = 0.5,
) -> tuple[float, dict]:
    """Train for one epoch.

    Returns:
        (avg_loss, avg_loss_components)
    """
    model.train()
    total_loss = 0.0
    components = {"bce": 0.0, "delta_mse": 0.0, "slope_mse": 0.0}
    n = 0

    for batch in loader:
        x = batch["x"].to(device)
        T_frac = batch["T_frac"].to(device)
        decline_target = batch["decline_target"].to(device)
        delta_target = batch["delta_target"].to(device)
        slope_target = batch["slope_target"].to(device)

        optimizer.zero_grad()
        loss, ld = decline_forecast_loss(
            model, x, T_frac, decline_target, delta_target, slope_target,
            w_bce=w_bce, w_delta=w_delta, w_slope=w_slope,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k in components:
            components[k] += ld[k]
        n += 1

    return total_loss / n, {k: v / n for k, v in components.items()}


@torch.no_grad()
def eval_decline_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    w_bce: float = 1.0,
    w_delta: float = 1.0,
    w_slope: float = 0.5,
) -> tuple[float, dict, dict]:
    """Evaluate on a dataset.

    Returns:
        (avg_loss, avg_loss_components, metrics)
    """
    model.eval()
    total_loss = 0.0
    components = {"bce": 0.0, "delta_mse": 0.0, "slope_mse": 0.0}
    n = 0

    all_probs, all_decline = [], []
    all_delta_pred, all_delta_true = [], []
    all_slope_pred, all_slope_true = [], []

    for batch in loader:
        x = batch["x"].to(device)
        T_frac = batch["T_frac"].to(device)
        decline_target = batch["decline_target"].to(device)
        delta_target = batch["delta_target"].to(device)
        slope_target = batch["slope_target"].to(device)

        loss, ld = decline_forecast_loss(
            model, x, T_frac, decline_target, delta_target, slope_target,
            w_bce=w_bce, w_delta=w_delta, w_slope=w_slope,
        )

        preds = model(x, T_frac)

        total_loss += loss.item()
        for k in components:
            components[k] += ld[k]
        n += 1

        all_probs.append(preds["decline_prob"].cpu().numpy())
        all_decline.append(decline_target.cpu().numpy())
        all_delta_pred.append(preds["delta_t_break"].cpu().numpy())
        all_delta_true.append(delta_target.cpu().numpy())
        all_slope_pred.append(preds["slope"].cpu().numpy())
        all_slope_true.append(slope_target.cpu().numpy())

    metrics = _compute_metrics(
        np.concatenate(all_probs),
        np.concatenate(all_decline),
        np.concatenate(all_delta_pred),
        np.concatenate(all_delta_true),
        np.concatenate(all_slope_pred),
        np.concatenate(all_slope_true),
    )

    return total_loss / n, {k: v / n for k, v in components.items()}, metrics


def train_and_evaluate_decline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 150,
    lr: float = 1e-3,
    lr_min: float = 0.0,
    weight_decay: float = 1e-4,
    patience: int = 20,
    w_bce: float = 1.0,
    w_delta: float = 1.0,
    w_slope: float = 0.5,
    device: torch.device | None = None,
    verbose: bool = True,
    print_every: int = 10,
) -> dict:
    """Full training loop with early stopping and cosine annealing.

    Args:
        model: DeclineForecastModel.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        n_epochs: Maximum epochs.
        lr: Initial learning rate.
        lr_min: Minimum learning rate for cosine annealing.
        weight_decay: AdamW weight decay.
        patience: Early stopping patience.
        w_bce: BCE loss weight.
        w_delta: Delta MSE loss weight.
        w_slope: Slope log-MSE loss weight.
        device: Compute device.
        verbose: Print progress.
        print_every: Print interval in epochs.

    Returns:
        Dict with 'history', 'best_epoch', 'val_metrics'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr_min)
    early_stopping = EarlyStopping(patience=patience)

    history = {
        "train_loss": [], "val_loss": [],
        "train_bce": [], "train_delta_mse": [], "train_slope_mse": [],
        "val_auc_roc": [], "val_accuracy": [], "val_delta_mae_hours": [],
    }

    for epoch in range(n_epochs):
        train_loss, train_comp = train_decline_epoch(
            model, train_loader, optimizer, device,
            w_bce=w_bce, w_delta=w_delta, w_slope=w_slope,
        )
        scheduler.step()

        val_loss, val_comp, val_metrics = eval_decline_epoch(
            model, val_loader, device,
            w_bce=w_bce, w_delta=w_delta, w_slope=w_slope,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_bce"].append(train_comp["bce"])
        history["train_delta_mse"].append(train_comp["delta_mse"])
        history["train_slope_mse"].append(train_comp["slope_mse"])
        history["val_auc_roc"].append(val_metrics["auc_roc"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_delta_mae_hours"].append(val_metrics["delta_mae_hours"])

        if verbose and (epoch + 1) % print_every == 0:
            print(
                f"Epoch {epoch+1}/{n_epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"AUC: {val_metrics['auc_roc']:.3f} | "
                f"Acc: {val_metrics['accuracy']:.3f} | "
                f"ΔT MAE: {val_metrics['delta_mae_hours']:.1f}h"
            )

        if early_stopping(val_loss, model):
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    early_stopping.load_best_state(model)
    model.to(device)

    # Final evaluation
    _, _, final_metrics = eval_decline_epoch(
        model, val_loader, device,
        w_bce=w_bce, w_delta=w_delta, w_slope=w_slope,
    )

    return {
        "history": history,
        "best_epoch": epoch + 1 - early_stopping.counter,
        "val_metrics": final_metrics,
    }
