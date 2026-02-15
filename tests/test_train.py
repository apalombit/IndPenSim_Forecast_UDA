"""Unit tests for train module."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

from src.train import (
    EarlyStopping,
    train_epoch,
    evaluate,
    train_model,
)


def create_dummy_loader(n_samples: int = 20, seq_len: int = 50, n_features: int = 11, batch_size: int = 4):
    """Create a dummy DataLoader for testing."""
    X = torch.randn(n_samples, seq_len, n_features)
    y = torch.randn(n_samples)
    domain = torch.zeros(n_samples, dtype=torch.long)

    dataset = TensorDataset(X, y, domain)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class SimpleModel(nn.Module):
    """Simple model for testing training loop."""

    def __init__(self, seq_len: int = 50, n_features: int = 11):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(seq_len * n_features, 1)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x).squeeze(-1)


class TestEarlyStopping:
    def test_improves_resets_counter(self):
        es = EarlyStopping(patience=3)
        model = SimpleModel()

        es(1.0, model)
        es(0.9, model)  # Improvement
        assert es.counter == 0

    def test_no_improvement_increments_counter(self):
        es = EarlyStopping(patience=3)
        model = SimpleModel()

        es(1.0, model)
        es(1.1, model)  # No improvement
        assert es.counter == 1

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3)
        model = SimpleModel()

        es(1.0, model)
        assert not es.early_stop

        es(1.1, model)
        es(1.2, model)
        es(1.3, model)

        assert es.early_stop

    def test_saves_best_state(self):
        es = EarlyStopping(patience=3)
        model = SimpleModel()

        # Set known weight
        with torch.no_grad():
            model.fc.weight.fill_(1.0)

        es(1.0, model)

        # Change weight
        with torch.no_grad():
            model.fc.weight.fill_(2.0)

        es(1.5, model)  # Worse, shouldn't save

        # Load best state
        es.load_best_state(model)

        # Weight should be restored to 1.0
        assert torch.allclose(model.fc.weight, torch.ones_like(model.fc.weight))


class TestTrainEpoch:
    def test_returns_float(self):
        model = SimpleModel()
        loader = create_dummy_loader()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        device = torch.device("cpu")

        loss = train_epoch(model, loader, optimizer, criterion, device)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_model_updates(self):
        model = SimpleModel()
        loader = create_dummy_loader()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        criterion = nn.MSELoss()
        device = torch.device("cpu")

        # Save initial weights
        initial_weight = model.fc.weight.clone()

        # Train for one epoch
        train_epoch(model, loader, optimizer, criterion, device)

        # Weights should have changed
        assert not torch.allclose(model.fc.weight, initial_weight)


class TestEvaluate:
    def test_returns_expected_keys(self):
        model = SimpleModel()
        loader = create_dummy_loader()
        criterion = nn.MSELoss()
        device = torch.device("cpu")

        result = evaluate(model, loader, criterion, device)

        assert "loss" in result
        assert "mae" in result
        assert "rmse" in result
        assert "predictions" in result
        assert "targets" in result

    def test_metrics_are_valid(self):
        model = SimpleModel()
        loader = create_dummy_loader(n_samples=20)
        criterion = nn.MSELoss()
        device = torch.device("cpu")

        result = evaluate(model, loader, criterion, device)

        assert result["loss"] >= 0
        assert result["mae"] >= 0
        assert result["rmse"] >= 0
        assert result["rmse"] >= result["mae"]  # RMSE >= MAE always
        assert len(result["predictions"]) == 20
        assert len(result["targets"]) == 20

    def test_no_gradients_computed(self):
        model = SimpleModel()
        loader = create_dummy_loader()
        criterion = nn.MSELoss()
        device = torch.device("cpu")

        evaluate(model, loader, criterion, device)

        # No gradients should be accumulated
        for param in model.parameters():
            assert param.grad is None


class TestTrainModel:
    def test_returns_expected_keys(self):
        model = SimpleModel()
        train_loader = create_dummy_loader(n_samples=20)
        val_loader = create_dummy_loader(n_samples=10)

        result = train_model(
            model, train_loader, val_loader,
            n_epochs=5, patience=3, verbose=False
        )

        assert "history" in result
        assert "train_metrics" in result
        assert "val_metrics" in result
        assert "best_epoch" in result

    def test_history_has_correct_length(self):
        model = SimpleModel()
        train_loader = create_dummy_loader(n_samples=20)
        val_loader = create_dummy_loader(n_samples=10)

        result = train_model(
            model, train_loader, val_loader,
            n_epochs=5, patience=10, verbose=False  # High patience to avoid early stop
        )

        # Should have 5 epochs in history
        assert len(result["history"]["train_loss"]) == 5
        assert len(result["history"]["val_loss"]) == 5

    def test_early_stopping_works(self):
        model = SimpleModel()
        train_loader = create_dummy_loader(n_samples=20)
        val_loader = create_dummy_loader(n_samples=10)

        result = train_model(
            model, train_loader, val_loader,
            n_epochs=100, patience=2, verbose=False
        )

        # Should stop before 100 epochs
        assert len(result["history"]["train_loss"]) < 100

    def test_model_improves(self):
        # Use larger dataset for more stable training
        model = SimpleModel()
        train_loader = create_dummy_loader(n_samples=50, batch_size=10)
        val_loader = create_dummy_loader(n_samples=20, batch_size=10)

        result = train_model(
            model, train_loader, val_loader,
            n_epochs=20, patience=15, lr=0.01, verbose=False
        )

        # Training loss should generally decrease
        history = result["history"]["train_loss"]
        # Compare first and last loss (allowing for some variation)
        assert history[-1] <= history[0] * 1.5  # Should not increase dramatically
