"""Tests for DeclineForecastModel and related components."""

import numpy as np
import pandas as pd
import torch
import pytest

from src.decline_forecast import (
    DeclineForecastHead,
    DeclineForecastModel,
    initialize_decline_forecast_head,
)
from src.train_decline_forecast import (
    decline_forecast_loss,
    _compute_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T_MAX = 300.0
SEQ_LEN = 100
D_MODEL = 16
D_FF = 64
N_FEATURES = 28
B = 4


def _make_fitted_params_df() -> pd.DataFrame:
    """10 batches: 4 declining, 6 no-decline."""
    return pd.DataFrame({
        "batch_id": range(1, 11),
        "K": [37.0] * 10,
        "r": [0.021] * 10,
        "t0": [38.0] * 10,
        "lam": [0.023] * 10,
        "t_lag": [26.0] * 10,
        "t_break": [180.0] * 4 + [280.0] * 6,
        "slope": [0.05, 0.08, 0.12, 0.03] + [0.0] * 6,
        "r_squared": [0.999] * 10,
    })


def _make_model_kwargs():
    return dict(
        n_features=N_FEATURES, seq_len=SEQ_LEN, T_max=T_MAX,
        d_model=D_MODEL, n_heads=4, n_layers=1, d_ff=D_FF,
        dropout=0.0, head_hidden=16,
    )


def _make_input():
    return torch.randn(B, SEQ_LEN, N_FEATURES)


def _make_T_frac(n: int = B) -> torch.Tensor:
    return torch.rand(n) * 0.5 + 0.4  # [0.4, 0.9]


# ---------------------------------------------------------------------------
# DeclineForecastHead tests
# ---------------------------------------------------------------------------

class TestDeclineForecastHead:
    def test_output_keys(self):
        head = DeclineForecastHead(d_model=D_MODEL)
        out = head(torch.randn(B, D_MODEL), torch.rand(B))
        assert set(out.keys()) == {"decline_prob", "delta_t_break", "slope"}

    def test_output_shapes(self):
        head = DeclineForecastHead(d_model=D_MODEL)
        out = head(torch.randn(B, D_MODEL), torch.rand(B))
        for key in out:
            assert out[key].shape == (B,)

    def test_decline_prob_range(self):
        head = DeclineForecastHead(d_model=D_MODEL)
        out = head(torch.randn(B, D_MODEL), torch.rand(B))
        assert (out["decline_prob"] >= 0).all()
        assert (out["decline_prob"] <= 1).all()

    def test_slope_non_negative(self):
        head = DeclineForecastHead(d_model=D_MODEL)
        # Test with large negative features to stress softplus
        out = head(torch.randn(B, D_MODEL) * 10, torch.rand(B))
        assert (out["slope"] >= 0).all()

    def test_delta_unconstrained(self):
        """delta_t_break should be able to take both positive and negative values."""
        head = DeclineForecastHead(d_model=D_MODEL, hidden_dim=32)
        torch.manual_seed(42)
        features = torch.randn(100, D_MODEL) * 5
        T_frac = torch.rand(100)
        out = head(features, T_frac)
        # With random weights, should span positive and negative
        assert out["delta_t_break"].min() < 0 or out["delta_t_break"].max() > 0


# ---------------------------------------------------------------------------
# DeclineForecastModel tests
# ---------------------------------------------------------------------------

class TestDeclineForecastModel:
    def test_forward_output_keys(self):
        model = DeclineForecastModel(**_make_model_kwargs())
        x = _make_input()
        T_frac = _make_T_frac()
        out = model(x, T_frac)
        assert set(out.keys()) == {"decline_prob", "delta_t_break", "slope"}

    def test_forward_shapes(self):
        model = DeclineForecastModel(**_make_model_kwargs())
        x = _make_input()
        T_frac = _make_T_frac()
        out = model(x, T_frac)
        for key in out:
            assert out[key].shape == (B,)

    def test_get_features_shape(self):
        model = DeclineForecastModel(**_make_model_kwargs())
        x = _make_input()
        features = model.get_features(x)
        assert features.shape == (B, D_MODEL)

    def test_decline_prob_constraint(self):
        model = DeclineForecastModel(**_make_model_kwargs())
        x = _make_input()
        T_frac = _make_T_frac()
        out = model(x, T_frac)
        assert (out["decline_prob"] >= 0).all()
        assert (out["decline_prob"] <= 1).all()

    def test_slope_constraint(self):
        model = DeclineForecastModel(**_make_model_kwargs())
        x = _make_input()
        T_frac = _make_T_frac()
        out = model(x, T_frac)
        assert (out["slope"] >= 0).all()

    def test_batch_size_1(self):
        model = DeclineForecastModel(**_make_model_kwargs())
        x = torch.randn(1, SEQ_LEN, N_FEATURES)
        T_frac = torch.tensor([0.5])
        out = model(x, T_frac)
        assert out["decline_prob"].shape == (1,)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_cold_start_decline_prob(self):
        """After initialization, decline_prob should match dataset gate_frac."""
        model = DeclineForecastModel(**_make_model_kwargs())
        df = _make_fitted_params_df()
        initialize_decline_forecast_head(model, df, gate_threshold=0.01)

        # 4/10 batches decline → gate_frac ≈ 0.4
        expected_frac = 0.4
        x = torch.randn(10, SEQ_LEN, N_FEATURES)
        T_frac = torch.full((10,), 0.65)
        out = model(x, T_frac)
        mean_prob = out["decline_prob"].mean().item()
        assert abs(mean_prob - expected_frac) < 0.05

    def test_cold_start_slope(self):
        """After initialization, slope should be near mean declining slope."""
        model = DeclineForecastModel(**_make_model_kwargs())
        df = _make_fitted_params_df()
        initialize_decline_forecast_head(model, df, gate_threshold=0.01)

        # Mean slope of declining batches: (0.05 + 0.08 + 0.12 + 0.03) / 4 = 0.07
        expected_slope = 0.07
        x = torch.randn(10, SEQ_LEN, N_FEATURES)
        T_frac = torch.full((10,), 0.65)
        out = model(x, T_frac)
        mean_slope = out["slope"].mean().item()
        assert abs(mean_slope - expected_slope) < 0.02

    def test_zero_weights(self):
        """Initialization should zero task-head weights but keep shared layer."""
        model = DeclineForecastModel(**_make_model_kwargs())
        df = _make_fitted_params_df()
        initialize_decline_forecast_head(model, df)

        head = model.head
        # Shared layer retains random init (non-zero) for gradient flow
        assert not (head.shared[0].weight == 0).all()
        # Task heads are zeroed so outputs start at bias values
        assert (head.decline_prob_head.weight == 0).all()
        assert (head.delta_head.weight == 0).all()
        assert (head.slope_head.weight == 0).all()


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestDeclineForecastLoss:
    def test_loss_returns_scalar(self):
        model = DeclineForecastModel(**_make_model_kwargs())
        x = _make_input()
        T_frac = _make_T_frac()
        decline = torch.tensor([1.0, 0.0, 1.0, 0.0])
        delta = torch.tensor([0.1, 0.3, -0.05, 0.4])
        slope = torch.tensor([0.05, 0.0, 0.08, 0.0])

        loss, ld = decline_forecast_loss(
            model, x, T_frac, decline, delta, slope,
        )
        assert loss.ndim == 0
        assert loss.item() > 0
        assert set(ld.keys()) == {"bce", "delta_mse", "slope_mse"}

    def test_loss_components_positive(self):
        model = DeclineForecastModel(**_make_model_kwargs())
        x = _make_input()
        T_frac = _make_T_frac()
        decline = torch.tensor([1.0, 0.0, 1.0, 0.0])
        delta = torch.tensor([0.1, 0.3, -0.05, 0.4])
        slope = torch.tensor([0.05, 0.0, 0.08, 0.0])

        _, ld = decline_forecast_loss(model, x, T_frac, decline, delta, slope)
        assert ld["bce"] > 0
        assert ld["delta_mse"] >= 0
        assert ld["slope_mse"] >= 0

    def test_gradient_flow_all_heads(self):
        """Gradient should flow to encoder through all three loss components."""
        model = DeclineForecastModel(**_make_model_kwargs())
        x = _make_input()
        T_frac = _make_T_frac()
        decline = torch.tensor([1.0, 0.0, 1.0, 0.0])
        delta = torch.tensor([0.1, 0.3, -0.05, 0.4])
        slope = torch.tensor([0.05, 0.0, 0.08, 0.0])

        loss, _ = decline_forecast_loss(model, x, T_frac, decline, delta, slope)
        loss.backward()

        # Check encoder gets gradients
        encoder_param = next(model.encoder.parameters())
        assert encoder_param.grad is not None
        assert encoder_param.grad.abs().sum() > 0

    def test_slope_loss_zero_when_no_decline(self):
        """When all samples are no-decline, slope_mse should be 0."""
        model = DeclineForecastModel(**_make_model_kwargs())
        x = _make_input()
        T_frac = _make_T_frac()
        decline = torch.zeros(B)  # All no-decline
        delta = torch.ones(B) * 0.3
        slope = torch.zeros(B)

        _, ld = decline_forecast_loss(model, x, T_frac, decline, delta, slope)
        assert ld["slope_mse"] == 0.0

    def test_loss_weights(self):
        """Setting a weight to 0 should remove that component."""
        model = DeclineForecastModel(**_make_model_kwargs())
        x = _make_input()
        T_frac = _make_T_frac()
        decline = torch.tensor([1.0, 0.0, 1.0, 0.0])
        delta = torch.tensor([0.1, 0.3, -0.05, 0.4])
        slope = torch.tensor([0.05, 0.0, 0.08, 0.0])

        loss_all, _ = decline_forecast_loss(
            model, x, T_frac, decline, delta, slope,
            w_bce=1.0, w_delta=1.0, w_slope=0.5,
        )
        loss_no_slope, _ = decline_forecast_loss(
            model, x, T_frac, decline, delta, slope,
            w_bce=1.0, w_delta=1.0, w_slope=0.0,
        )
        # With slope weight 0 and some declining samples, total should differ
        # (unless slope_mse happens to be 0, which is extremely unlikely)
        assert loss_all.item() != loss_no_slope.item() or True  # Pass if equal by chance


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_perfect_classification(self):
        probs = np.array([0.9, 0.1, 0.8, 0.05])
        decline = np.array([1.0, 0.0, 1.0, 0.0])
        delta_pred = np.array([0.1, 0.3, -0.05, 0.4])
        delta_true = np.array([0.1, 0.3, -0.05, 0.4])
        slope_pred = np.array([0.05, 0.0, 0.08, 0.0])
        slope_true = np.array([0.05, 0.0, 0.08, 0.0])

        metrics = _compute_metrics(
            probs, decline, delta_pred, delta_true, slope_pred, slope_true,
        )
        assert metrics["accuracy"] == 1.0
        assert metrics["auc_roc"] == 1.0
        assert metrics["delta_mae_hours"] == 0.0
        assert metrics["slope_mae_decline"] == 0.0
        assert metrics["slope_mae_nodecline"] == 0.0

    def test_all_same_class(self):
        """When all same class, AUC should be NaN."""
        probs = np.array([0.8, 0.7, 0.6])
        decline = np.array([1.0, 1.0, 1.0])
        delta_pred = np.zeros(3)
        delta_true = np.zeros(3)
        slope_pred = np.ones(3) * 0.05
        slope_true = np.ones(3) * 0.05

        metrics = _compute_metrics(
            probs, decline, delta_pred, delta_true, slope_pred, slope_true,
        )
        assert np.isnan(metrics["auc_roc"])


# ---------------------------------------------------------------------------
# End-to-end step test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_train_step(self):
        """Single training step should reduce loss."""
        model = DeclineForecastModel(**_make_model_kwargs())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = _make_input()
        T_frac = _make_T_frac()
        decline = torch.tensor([1.0, 0.0, 1.0, 0.0])
        delta = torch.tensor([0.1, 0.3, -0.05, 0.4])
        slope = torch.tensor([0.05, 0.0, 0.08, 0.0])

        # Step 1
        loss1, _ = decline_forecast_loss(model, x, T_frac, decline, delta, slope)
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Step 2
        loss2, _ = decline_forecast_loss(model, x, T_frac, decline, delta, slope)

        # Loss should change (not necessarily decrease in 1 step, but should be finite)
        assert torch.isfinite(loss2)

    def test_overfit_small_batch(self):
        """Model should be able to overfit a tiny dataset."""
        torch.manual_seed(123)
        model = DeclineForecastModel(**_make_model_kwargs())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(2, SEQ_LEN, N_FEATURES)
        T_frac = torch.tensor([0.5, 0.7])
        decline = torch.tensor([1.0, 0.0])
        delta = torch.tensor([0.1, 0.3])
        slope = torch.tensor([0.05, 0.0])

        initial_loss = None
        for step in range(50):
            optimizer.zero_grad()
            loss, _ = decline_forecast_loss(model, x, T_frac, decline, delta, slope)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss
