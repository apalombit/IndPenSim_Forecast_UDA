"""Training loop for transformer model."""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .preprocessing import denormalize_target, TARGET_MIN, TARGET_MAX


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss.
            model: Model to save state from.

        Returns:
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def load_best_state(self, model: nn.Module):
        """Load best model state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to train on.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X, y, _ in train_loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stats: dict | None = None,
) -> dict:
    """Evaluate model on data.

    Args:
        model: Model to evaluate.
        data_loader: Data loader.
        criterion: Loss function.
        device: Device.
        stats: Normalization stats with y_min/y_max. If provided, metrics
               are computed in original scale for interpretability.

    Returns:
        Dict with loss, mae, rmse, predictions, and targets.
        If stats provided, predictions/targets are in original scale.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y, _ in data_loader:
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)
            loss = criterion(y_pred, y)

            total_loss += loss.item() * len(y)
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    n_samples = len(all_targets)

    # Normalized loss (for learning)
    mse_norm = total_loss / n_samples

    # Denormalize for interpretable metrics
    if stats is not None:
        y_min = stats.get("y_min", TARGET_MIN)
        y_max = stats.get("y_max", TARGET_MAX)
        all_preds = denormalize_target(all_preds, y_min, y_max)
        all_targets = denormalize_target(all_targets, y_min, y_max)

    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))

    return {
        "loss": mse_norm,
        "mae": mae,
        "rmse": rmse,
        "predictions": all_preds,
        "targets": all_targets,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: torch.device | None = None,
    verbose: bool = True,
    stats: dict | None = None,
) -> dict:
    """Train model with early stopping.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        n_epochs: Maximum number of epochs.
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        patience: Early stopping patience.
        device: Device to train on. Auto-detects if None.
        verbose: Whether to print progress.
        stats: Normalization stats for denormalizing metrics.

    Returns:
        Dict with training history and final metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr / 10)
    early_stopping = EarlyStopping(patience=patience)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "lr": [],
    }

    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate (use normalized loss for early stopping, but log denormalized MAE)
        val_metrics = evaluate(model, val_loader, criterion, device, stats)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["lr"].append(scheduler.get_last_lr()[0])

        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{n_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val MAE: {val_metrics['mae']:.3f}"
            )

        # Learning rate scheduling
        scheduler.step()

        # Early stopping
        if early_stopping(val_metrics["loss"], model):
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    early_stopping.load_best_state(model)
    model = model.to(device)

    # Final evaluation (denormalized for interpretability)
    final_train = evaluate(model, train_loader, criterion, device, stats)
    final_val = evaluate(model, val_loader, criterion, device, stats)

    return {
        "history": history,
        "train_metrics": final_train,
        "val_metrics": final_val,
        "best_epoch": epoch + 1 - early_stopping.counter,
        "device": str(device),
    }


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    target_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: torch.device | None = None,
    verbose: bool = True,
    stats: dict | None = None,
) -> dict:
    """Train model and evaluate on all splits including target domain.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader (source).
        target_loader: Target domain data loader.
        n_epochs: Maximum number of epochs.
        lr: Learning rate.
        weight_decay: Weight decay.
        patience: Early stopping patience.
        device: Device.
        verbose: Print progress.
        stats: Normalization stats for denormalizing metrics.

    Returns:
        Dict with training results and metrics for all splits.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train
    train_results = train_model(
        model, train_loader, val_loader,
        n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
        patience=patience, device=device, verbose=verbose, stats=stats
    )

    # Evaluate on target (denormalized for interpretability)
    criterion = nn.MSELoss()
    target_metrics = evaluate(model, target_loader, criterion, device, stats)

    if verbose:
        print("\n=== Final Results ===")
        print(f"Train MAE: {train_results['train_metrics']['mae']:.3f}")
        print(f"Val MAE:   {train_results['val_metrics']['mae']:.3f}")
        print(f"Target MAE: {target_metrics['mae']:.3f}")

    return {
        **train_results,
        "target_metrics": target_metrics,
    }
