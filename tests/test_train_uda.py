"""Tests for UDA training loop."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.train_uda import train_epoch_uda, train_model_uda, train_and_evaluate_uda


class MockModel(nn.Module):
    """Simple model with get_features() and head for testing."""

    def __init__(self, n_features=4, seq_len=16, d_model=16):
        super().__init__()
        self.encoder_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features * seq_len, d_model),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(d_model, 1))

    def forward(self, x):
        features = self.get_features(x)
        return self.head(features).squeeze(-1)

    def get_features(self, x):
        return self.encoder_net(x)


def _make_loaders(n_source=20, n_target=10, n_features=4, seq_len=16, batch_size=8):
    source_X = torch.randn(n_source, seq_len, n_features)
    source_y = torch.randn(n_source)
    source_d = torch.zeros(n_source, dtype=torch.long)
    target_X = torch.randn(n_target, seq_len, n_features)
    target_y = torch.randn(n_target)
    target_d = torch.ones(n_target, dtype=torch.long)

    source_loader = DataLoader(
        TensorDataset(source_X, source_y, source_d),
        batch_size=batch_size, shuffle=True,
    )
    target_loader = DataLoader(
        TensorDataset(target_X, target_y, target_d),
        batch_size=batch_size, shuffle=True,
    )
    return source_loader, target_loader


class TestTrainEpochUDA:

    def test_runs_without_error(self):
        model = MockModel()
        source_loader, target_loader = _make_loaders()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        result = train_epoch_uda(
            model, source_loader, target_loader,
            optimizer, criterion, torch.device("cpu"),
        )
        assert "mse_loss" in result
        assert "coral_loss" in result
        assert "total_loss" in result

    def test_loss_values_non_negative(self):
        model = MockModel()
        source_loader, target_loader = _make_loaders()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        result = train_epoch_uda(
            model, source_loader, target_loader,
            optimizer, criterion, torch.device("cpu"),
        )
        assert result["mse_loss"] >= 0
        assert result["coral_loss"] >= 0

    def test_target_cycling(self):
        """Target loader cycles when smaller than source."""
        model = MockModel()
        source_loader, target_loader = _make_loaders(
            n_source=30, n_target=5, batch_size=4,
        )
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        result = train_epoch_uda(
            model, source_loader, target_loader,
            optimizer, criterion, torch.device("cpu"),
        )
        assert result["mse_loss"] >= 0

    def test_zero_lambda(self):
        """With lambda=0, total_loss equals mse_loss."""
        model = MockModel()
        source_loader, target_loader = _make_loaders()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        result = train_epoch_uda(
            model, source_loader, target_loader,
            optimizer, criterion, torch.device("cpu"),
            coral_lambda=0.0,
        )
        assert result["total_loss"] == pytest.approx(result["mse_loss"], abs=1e-6)

    def test_model_updates(self):
        """Parameters should change after one epoch."""
        model = MockModel()
        params_before = [p.clone() for p in model.parameters()]
        source_loader, target_loader = _make_loaders()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.MSELoss()

        train_epoch_uda(
            model, source_loader, target_loader,
            optimizer, criterion, torch.device("cpu"),
        )
        params_after = list(model.parameters())
        changed = any(
            not torch.equal(b, a) for b, a in zip(params_before, params_after)
        )
        assert changed


class TestTrainModelUDA:

    def test_training_completes(self):
        model = MockModel()
        source_train, target_loader = _make_loaders(n_source=16, n_target=8)
        source_val, _ = _make_loaders(n_source=8, n_target=4)

        result = train_model_uda(
            model, source_train, source_val, target_loader,
            n_epochs=5, patience=3, verbose=False,
        )
        assert "history" in result
        assert "train_metrics" in result
        assert "val_metrics" in result
        assert len(result["history"]["train_loss"]) <= 5

    def test_history_tracks_coral(self):
        model = MockModel()
        source_train, target_loader = _make_loaders(n_source=16, n_target=8)
        source_val, _ = _make_loaders(n_source=8, n_target=4)

        result = train_model_uda(
            model, source_train, source_val, target_loader,
            n_epochs=3, patience=5, verbose=False,
        )
        assert "train_coral" in result["history"]
        assert "train_mse" in result["history"]
        assert len(result["history"]["train_coral"]) == 3

    def test_early_stopping(self):
        model = MockModel()
        source_train, target_loader = _make_loaders(n_source=16, n_target=8)
        source_val, _ = _make_loaders(n_source=8, n_target=4)

        result = train_model_uda(
            model, source_train, source_val, target_loader,
            n_epochs=100, patience=2, verbose=False,
        )
        # Should stop before 100 epochs
        assert len(result["history"]["train_loss"]) < 100


class TestTrainAndEvaluateUDA:

    def test_includes_target_metrics(self):
        model = MockModel()
        source_train, target_loader = _make_loaders(n_source=16, n_target=8)
        source_val, _ = _make_loaders(n_source=8, n_target=4)

        result = train_and_evaluate_uda(
            model, source_train, source_val, target_loader,
            n_epochs=3, patience=5, verbose=False,
        )
        assert "target_metrics" in result
        assert "mae" in result["target_metrics"]
        assert "rmse" in result["target_metrics"]

    def test_different_lambda_values(self):
        """Training should work with various lambda values."""
        for lam in [0.0, 0.1, 1.0, 5.0]:
            model = MockModel()
            source_train, target_loader = _make_loaders(n_source=16, n_target=8)
            source_val, _ = _make_loaders(n_source=8, n_target=4)

            result = train_and_evaluate_uda(
                model, source_train, source_val, target_loader,
                n_epochs=2, patience=5, coral_lambda=lam, verbose=False,
            )
            assert "target_metrics" in result
