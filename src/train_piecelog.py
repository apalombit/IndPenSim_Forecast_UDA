"""Training loop for PieceLog-PatchTST with dual loss."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .piecelog_model import PARAM_NAMES
from .train import EarlyStopping


def make_stepwise_alpha_schedule(
    alpha_max: float,
    n_epochs: int,
    n_steps: int = 5,
) -> list[tuple[int, float]]:
    """Create a stepwise decaying alpha schedule.

    Alpha decreases in equal steps from alpha_max to 0 over n_epochs.

    Example with alpha_max=0.1, n_epochs=100, n_steps=5:
        [(0, 0.1), (20, 0.075), (40, 0.05), (60, 0.025), (80, 0.0)]

    Args:
        alpha_max: Starting alpha value.
        n_epochs: Total training epochs.
        n_steps: Number of constant-alpha phases.

    Returns:
        List of (epoch, alpha) pairs sorted by epoch.
    """
    step_size = n_epochs // n_steps
    schedule = []
    for i in range(n_steps):
        epoch = i * step_size
        alpha = alpha_max * (1.0 - i / (n_steps - 1)) if n_steps > 1 else alpha_max
        schedule.append((epoch, round(alpha, 6)))
    return schedule


def get_alpha_for_epoch(
    epoch: int,
    alpha_schedule: list[tuple[int, float]],
) -> float:
    """Look up alpha value for a given epoch from a stepwise schedule.

    Args:
        epoch: Current epoch (0-indexed).
        alpha_schedule: List of (start_epoch, alpha) pairs, sorted by epoch.

    Returns:
        Alpha value active at the given epoch.
    """
    current_alpha = alpha_schedule[0][1]
    for start_epoch, alpha_val in alpha_schedule:
        if epoch >= start_epoch:
            current_alpha = alpha_val
        else:
            break
    return current_alpha


def piecelog_loss(
    model: nn.Module,
    x: torch.Tensor,
    t_predict: torch.Tensor,
    y_true: torch.Tensor,
    params_true: torch.Tensor,
    criterion: nn.Module,
    alpha: float = 0.1,
    param_stats: dict | None = None,
    conc_scale: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Combined loss: concentration MSE + alpha * parameter MSE.

    Args:
        model: PieceLogPatchTST model.
        x: Input features, shape (B, T, C).
        t_predict: Prediction times, shape (B,).
        y_true: True concentration at t_predict, shape (B,).
        params_true: Fitted piece-log parameters, shape (B, 7).
        criterion: Loss function (MSE).
        alpha: Weight for parameter loss.
        param_stats: Per-parameter {"mean", "std"} for z-score normalization.
            If None, raw MSE is used.
        conc_scale: Std of concentration from training data. If provided,
            both P_pred and y_true are divided by conc_scale before computing
            L_conc, bringing it to ~O(0.1) comparable to normalized L_param.
            Scale-only (no mean shift) since concentrations are non-negative.

    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict has 'conc' and 'param'.
    """
    params_pred = model.get_parameters(x)
    P_pred = model(x, t_predict)

    if conc_scale is not None:
        L_conc = criterion(P_pred / conc_scale, y_true / conc_scale)
    else:
        L_conc = criterion(P_pred, y_true)

    L_param = torch.tensor(0.0, device=x.device)
    if alpha > 0:
        for i, name in enumerate(PARAM_NAMES):
            pred_i = params_pred[name]
            true_i = params_true[:, i]
            if param_stats is not None:
                mean = param_stats[name]["mean"]
                std = param_stats[name]["std"]
                pred_i = (pred_i - mean) / std
                true_i = (true_i - mean) / std
            L_param = L_param + criterion(pred_i, true_i)
        L_param = L_param / len(PARAM_NAMES)

    total = L_conc + alpha * L_param

    return total, {"conc": L_conc.item(), "param": L_param.item()}


def train_epoch_piecelog(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    alpha: float = 0.1,
    param_stats: dict | None = None,
    conc_scale: float | None = None,
) -> dict:
    """Train one epoch with dual loss.

    Args:
        model: PieceLogPatchTST model.
        train_loader: Training DataLoader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device.
        alpha: Parameter loss weight.
        param_stats: Per-parameter z-score stats for normalized param loss.
        conc_scale: Concentration std for normalizing L_conc.

    Returns:
        Dict with total_loss, conc_loss, param_loss averages.
    """
    model.train()
    total_loss = 0.0
    total_conc = 0.0
    total_param = 0.0
    n_batches = 0

    for batch in train_loader:
        x = batch["x"].to(device)
        t_predict = batch["t_predict"].to(device)
        y_conc = batch["y_conc"].to(device)
        params_fitted = batch["params_fitted"].to(device)

        optimizer.zero_grad()
        loss, loss_dict = piecelog_loss(
            model, x, t_predict, y_conc, params_fitted, criterion, alpha,
            param_stats=param_stats, conc_scale=conc_scale,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_conc += loss_dict["conc"]
        total_param += loss_dict["param"]
        n_batches += 1

    return {
        "total_loss": total_loss / n_batches,
        "conc_loss": total_conc / n_batches,
        "param_loss": total_param / n_batches,
    }


def evaluate_piecelog(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    alpha: float = 0.1,
    param_stats: dict | None = None,
    conc_scale: float | None = None,
) -> dict:
    """Evaluate PieceLog-PatchTST on a data split.

    Args:
        model: PieceLogPatchTST model.
        data_loader: DataLoader.
        criterion: Loss function.
        device: Device.
        alpha: Parameter loss weight.
        param_stats: Per-parameter z-score stats for normalized param loss.
        conc_scale: Concentration std for normalizing L_conc.

    Returns:
        Dict with loss, conc_loss, param_loss, mae, rmse, predictions, targets.
    """
    model.eval()
    total_loss = 0.0
    total_conc = 0.0
    total_param = 0.0
    all_preds = []
    all_targets = []
    n_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"].to(device)
            t_predict = batch["t_predict"].to(device)
            y_conc = batch["y_conc"].to(device)
            params_fitted = batch["params_fitted"].to(device)

            loss, loss_dict = piecelog_loss(
                model, x, t_predict, y_conc, params_fitted, criterion, alpha,
                param_stats=param_stats, conc_scale=conc_scale,
            )

            total_loss += loss.item()
            total_conc += loss_dict["conc"]
            total_param += loss_dict["param"]
            n_batches += 1

            P_pred = model(x, t_predict)
            all_preds.append(P_pred.cpu().numpy())
            all_targets.append(y_conc.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))

    return {
        "loss": total_loss / n_batches,
        "conc_loss": total_conc / n_batches,
        "param_loss": total_param / n_batches,
        "mae": mae,
        "rmse": rmse,
        "predictions": all_preds,
        "targets": all_targets,
    }


def train_and_evaluate_piecelog(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    target_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 20,
    alpha: float = 0.1,
    alpha_schedule: list[tuple[int, float]] | None = None,
    freeze_epochs: int = 0,
    device: torch.device | None = None,
    verbose: bool = True,
    param_stats: dict | None = None,
    conc_scale: float | None = None,
) -> dict:
    """Train PieceLog-PatchTST and evaluate on all splits.

    Args:
        model: PieceLogPatchTST model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        target_loader: Target domain DataLoader.
        n_epochs: Maximum epochs.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        patience: Early stopping patience.
        alpha: Fixed parameter loss weight (used when alpha_schedule is None).
        alpha_schedule: Stepwise decay schedule as list of (epoch, alpha) pairs.
            When provided, overrides the fixed alpha. Use make_stepwise_alpha_schedule()
            to create. Example: [(0, 0.1), (20, 0.075), (40, 0.05), (60, 0.025), (80, 0.0)].
        freeze_epochs: Epochs to freeze param_head (0 = no freeze).
        device: Device.
        verbose: Print progress.
        param_stats: Per-parameter z-score stats for normalized param loss.
        conc_scale: Concentration std for normalizing L_conc.

    Returns:
        Dict with history and metrics for all splits.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.MSELoss()

    # Freeze param_head for warm-start phase
    if freeze_epochs > 0:
        for p in model.param_head.parameters():
            p.requires_grad = False

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr / 10)
    early_stopping = EarlyStopping(patience=patience)

    history = {
        "train_loss": [],
        "train_conc_loss": [],
        "train_param_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "lr": [],
        "alpha": [],
        "freeze_epochs": freeze_epochs,
    }

    for epoch in range(n_epochs):
        # Resolve current alpha from schedule or fixed value
        if alpha_schedule is not None:
            current_alpha = get_alpha_for_epoch(epoch, alpha_schedule)
        else:
            current_alpha = alpha

        # Unfreeze param_head and rebuild optimizer/scheduler
        if epoch == freeze_epochs and freeze_epochs > 0:
            for p in model.param_head.parameters():
                p.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=n_epochs - freeze_epochs, eta_min=lr / 10,
            )
            if verbose:
                print(f"Epoch {epoch + 1}: Unfreezing param_head, rebuilding optimizer")

        train_metrics = train_epoch_piecelog(
            model, train_loader, optimizer, criterion, device, current_alpha,
            param_stats=param_stats, conc_scale=conc_scale,
        )

        val_metrics = evaluate_piecelog(
            model, val_loader, criterion, device, current_alpha,
            param_stats=param_stats, conc_scale=conc_scale,
        )

        history["train_loss"].append(train_metrics["total_loss"])
        history["train_conc_loss"].append(train_metrics["conc_loss"])
        history["train_param_loss"].append(train_metrics["param_loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["lr"].append(scheduler.get_last_lr()[0])
        history["alpha"].append(current_alpha)

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{n_epochs} | "
                f"α={current_alpha:.4f} | "
                f"Conc: {train_metrics['conc_loss']:.4f} | "
                f"Param: {train_metrics['param_loss']:.4f} | "
                f"Val MAE: {val_metrics['mae']:.3f}"
            )

        scheduler.step()

        if early_stopping(val_metrics["loss"], model):
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    early_stopping.load_best_state(model)
    model = model.to(device)

    # Final evaluation uses last active alpha
    final_alpha = current_alpha if alpha_schedule is not None else alpha
    final_train = evaluate_piecelog(model, train_loader, criterion, device, final_alpha, param_stats=param_stats, conc_scale=conc_scale)
    final_val = evaluate_piecelog(model, val_loader, criterion, device, final_alpha, param_stats=param_stats, conc_scale=conc_scale)
    target_metrics = evaluate_piecelog(model, target_loader, criterion, device, final_alpha, param_stats=param_stats, conc_scale=conc_scale)

    if verbose:
        print("\n=== Final Results (PieceLog-PatchTST) ===")
        print(f"Train MAE: {final_train['mae']:.3f}")
        print(f"Val MAE:   {final_val['mae']:.3f}")
        print(f"Target MAE: {target_metrics['mae']:.3f}")

    return {
        "history": history,
        "train_metrics": final_train,
        "val_metrics": final_val,
        "target_metrics": target_metrics,
        "best_epoch": epoch + 1 - early_stopping.counter,
        "device": str(device),
    }
