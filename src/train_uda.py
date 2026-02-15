"""UDA training loop with CORAL loss for domain adaptation."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .coral_loss import coral_loss
from .train import EarlyStopping, evaluate


def train_epoch_uda(
    model: nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    coral_lambda: float = 0.5,
) -> dict:
    """Train one epoch with source MSE + CORAL alignment.

    For each source mini-batch, also samples a target mini-batch (cycling
    if target loader is shorter). Source labels drive MSE; target data is
    unlabeled and used only for CORAL feature covariance alignment.

    Args:
        model: Model with get_features() and head attributes.
        source_loader: Source domain DataLoader (labeled).
        target_loader: Target domain DataLoader (unlabeled).
        optimizer: Optimizer.
        criterion: Task loss (MSE).
        device: Device.
        coral_lambda: Weight for CORAL loss term.

    Returns:
        Dict with mse_loss, coral_loss, total_loss averages.
    """
    model.train()
    total_mse = 0.0
    total_coral = 0.0
    n_batches = 0

    target_iter = iter(target_loader)

    for source_X, source_y, _ in source_loader:
        # Get target batch, cycling if exhausted
        try:
            target_X, _, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_X, _, _ = next(target_iter)

        source_X = source_X.to(device)
        source_y = source_y.to(device)
        target_X = target_X.to(device)

        optimizer.zero_grad()

        # Extract encoder features from both domains
        source_features = model.get_features(source_X)
        target_features = model.get_features(target_X)

        # Task loss on source predictions only
        source_pred = model.head(source_features).squeeze(-1)
        mse_loss = criterion(source_pred, source_y)

        # CORAL alignment loss
        c_loss = coral_loss(source_features, target_features)

        # Combined objective
        loss = mse_loss + coral_lambda * c_loss
        loss.backward()
        optimizer.step()

        total_mse += mse_loss.item()
        total_coral += c_loss.item()
        n_batches += 1

    return {
        "mse_loss": total_mse / n_batches,
        "coral_loss": total_coral / n_batches,
        "total_loss": (total_mse + coral_lambda * total_coral) / n_batches,
    }


def train_model_uda(
    model: nn.Module,
    source_train_loader: DataLoader,
    source_val_loader: DataLoader,
    target_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 15,
    coral_lambda: float = 0.5,
    device: torch.device | None = None,
    verbose: bool = True,
    stats: dict | None = None,
) -> dict:
    """Train model with CORAL domain adaptation and early stopping.

    Early stopping monitors source validation MSE loss.

    Args:
        model: PatchTST model.
        source_train_loader: Source training DataLoader.
        source_val_loader: Source validation DataLoader.
        target_loader: Target DataLoader (unlabeled, used for CORAL).
        n_epochs: Maximum epochs.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        patience: Early stopping patience.
        coral_lambda: CORAL loss weight.
        device: Device.
        verbose: Print progress.
        stats: Normalization stats for denormalized metrics.

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
        "train_mse": [],
        "train_coral": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "lr": [],
    }

    for epoch in range(n_epochs):
        train_metrics = train_epoch_uda(
            model, source_train_loader, target_loader,
            optimizer, criterion, device, coral_lambda,
        )

        val_metrics = evaluate(model, source_val_loader, criterion, device, stats)

        history["train_loss"].append(train_metrics["total_loss"])
        history["train_mse"].append(train_metrics["mse_loss"])
        history["train_coral"].append(train_metrics["coral_loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["lr"].append(scheduler.get_last_lr()[0])

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{n_epochs} | "
                f"MSE: {train_metrics['mse_loss']:.4f} | "
                f"CORAL: {train_metrics['coral_loss']:.4f} | "
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

    # Final evaluation
    final_train = evaluate(model, source_train_loader, criterion, device, stats)
    final_val = evaluate(model, source_val_loader, criterion, device, stats)

    return {
        "history": history,
        "train_metrics": final_train,
        "val_metrics": final_val,
        "best_epoch": epoch + 1 - early_stopping.counter,
        "device": str(device),
    }


def train_and_evaluate_uda(
    model: nn.Module,
    source_train_loader: DataLoader,
    source_val_loader: DataLoader,
    target_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 15,
    coral_lambda: float = 0.5,
    device: torch.device | None = None,
    verbose: bool = True,
    stats: dict | None = None,
) -> dict:
    """Train with CORAL and evaluate on all splits including target.

    Args:
        model: PatchTST model.
        source_train_loader, source_val_loader, target_loader: DataLoaders.
        n_epochs, lr, weight_decay, patience, coral_lambda: Training config.
        device: Device.
        verbose: Print progress.
        stats: Normalization stats.

    Returns:
        Dict with training results and metrics for all splits.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = train_model_uda(
        model, source_train_loader, source_val_loader, target_loader,
        n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
        patience=patience, coral_lambda=coral_lambda,
        device=device, verbose=verbose, stats=stats,
    )

    criterion = nn.MSELoss()
    target_metrics = evaluate(model, target_loader, criterion, device, stats)

    if verbose:
        print("\n=== Final Results (CORAL) ===")
        print(f"Train MAE: {results['train_metrics']['mae']:.3f}")
        print(f"Val MAE:   {results['val_metrics']['mae']:.3f}")
        print(f"Target MAE: {target_metrics['mae']:.3f}")

    return {
        **results,
        "target_metrics": target_metrics,
    }
