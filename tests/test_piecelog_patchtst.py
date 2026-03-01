"""Tests for PieceLog-PatchTST model."""

import numpy as np
import pandas as pd
import torch
import pytest

from src.piecelog_model import PARAM_NAMES
from src.piecelog_patchtst import (
    ConstrainedParameterHead,
    PieceLogPatchTST,
    create_piecelog_model,
    initialize_param_head,
)


class TestConstrainedParameterHead:
    def test_output_keys(self):
        head = ConstrainedParameterHead(d_model=32)
        features = torch.randn(4, 32)
        params = head(features)
        expected_keys = {"K", "r", "t0", "lam", "t_lag", "t_break", "slope"}
        assert set(params.keys()) == expected_keys

    def test_output_shapes(self):
        head = ConstrainedParameterHead(d_model=32)
        features = torch.randn(4, 32)
        params = head(features)
        for key, val in params.items():
            assert val.shape == (4,), f"{key} has wrong shape"

    def test_K_positive(self):
        head = ConstrainedParameterHead(d_model=32, K_scale=60.0)
        features = torch.randn(100, 32)
        params = head(features)
        assert (params["K"] > 0).all()

    def test_r_positive(self):
        head = ConstrainedParameterHead(d_model=32)
        features = torch.randn(100, 32)
        params = head(features)
        assert (params["r"] > 0).all()

    def test_lam_positive(self):
        head = ConstrainedParameterHead(d_model=32)
        features = torch.randn(100, 32)
        params = head(features)
        assert (params["lam"] > 0).all()

    def test_slope_positive(self):
        head = ConstrainedParameterHead(d_model=32)
        features = torch.randn(100, 32)
        params = head(features)
        assert (params["slope"] > 0).all()

    def test_t0_in_range(self):
        T_max = 400.0
        head = ConstrainedParameterHead(d_model=32, T_max=T_max)
        features = torch.randn(100, 32)
        params = head(features)
        assert (params["t0"] >= 0).all()
        assert (params["t0"] <= T_max).all()

    def test_t_lag_in_range(self):
        T_max = 400.0
        head = ConstrainedParameterHead(d_model=32, T_max=T_max)
        features = torch.randn(100, 32)
        params = head(features)
        assert (params["t_lag"] >= 0).all()
        assert (params["t_lag"] <= 0.3 * T_max + 1e-4).all()

    def test_t_break_in_range(self):
        T_max = 400.0
        head = ConstrainedParameterHead(d_model=32, T_max=T_max)
        features = torch.randn(100, 32)
        params = head(features)
        assert (params["t_break"] >= 0).all()
        assert (params["t_break"] <= T_max + 1e-4).all()

    def test_gradient_flow(self):
        head = ConstrainedParameterHead(d_model=32)
        features = torch.randn(4, 32, requires_grad=True)
        params = head(features)
        loss = sum(v.sum() for v in params.values())
        loss.backward()
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()


class TestPieceLogPatchTST:
    def test_forward_output_shape(self):
        model = PieceLogPatchTST(n_features=28, seq_len=200, d_model=32)
        x = torch.randn(4, 200, 28)
        t_predict = torch.tensor([100.0, 200.0, 300.0, 350.0])
        out = model(x, t_predict)
        assert out.shape == (4,)

    def test_get_parameters_output(self):
        model = PieceLogPatchTST(n_features=28, seq_len=200, d_model=32)
        x = torch.randn(4, 200, 28)
        params = model.get_parameters(x)
        assert len(params) == 7
        for key, val in params.items():
            assert val.shape == (4,)

    def test_get_features_output(self):
        model = PieceLogPatchTST(n_features=28, seq_len=200, d_model=32)
        x = torch.randn(4, 200, 28)
        features = model.get_features(x)
        assert features.shape == (4, 32)

    def test_output_non_negative(self):
        model = PieceLogPatchTST(n_features=28, seq_len=200, d_model=32)
        x = torch.randn(8, 200, 28)
        t_predict = torch.rand(8) * 400
        out = model(x, t_predict)
        assert (out >= 0).all()

    def test_gradient_flow_from_loss_to_encoder(self):
        """Concentration loss should propagate gradients to encoder weights."""
        model = PieceLogPatchTST(n_features=28, seq_len=200, d_model=32)
        x = torch.randn(4, 200, 28)
        t_predict = torch.tensor([100.0, 200.0, 300.0, 350.0])
        y_true = torch.tensor([10.0, 25.0, 30.0, 28.0])

        out = model(x, t_predict)
        loss = torch.nn.functional.mse_loss(out, y_true)
        loss.backward()

        # Check encoder weights received gradients
        encoder_param = next(model.encoder.parameters())
        assert encoder_param.grad is not None
        assert not torch.isnan(encoder_param.grad).any()

    def test_gradient_flow_param_head(self):
        """Parameter head should receive gradients."""
        model = PieceLogPatchTST(n_features=28, seq_len=200, d_model=32)
        x = torch.randn(4, 200, 28)
        t_predict = torch.tensor([150.0, 200.0, 250.0, 300.0])

        out = model(x, t_predict)
        loss = out.sum()
        loss.backward()

        head_param = next(model.param_head.parameters())
        assert head_param.grad is not None

    def test_different_t_predict_different_output(self):
        """Different prediction times should produce different outputs."""
        torch.manual_seed(0)
        model = PieceLogPatchTST(n_features=28, seq_len=200, d_model=32, T_max=400.0)
        # Initialize to mean-like params so outputs are non-trivially different
        mock_df = pd.DataFrame([{
            "K": 37.0, "r": 0.021, "t0": 50.0, "lam": 0.023,
            "t_lag": 25.8, "t_break": 200.0, "slope": 0.048,
        }])
        initialize_param_head(model, mock_df)
        model.eval()
        x = torch.randn(1, 200, 28)

        with torch.no_grad():
            out1 = model(x, torch.tensor([100.0]))
            out2 = model(x, torch.tensor([300.0]))

        assert not torch.allclose(out1, out2)


class TestCreatePiecelogModel:
    def test_default_config(self):
        model = create_piecelog_model(n_features=28, seq_len=200)
        assert isinstance(model, PieceLogPatchTST)
        assert model.n_features == 28
        assert model.d_model == 32

    def test_custom_config(self):
        config = {"d_model": 64, "n_layers": 3, "head_hidden": 64}
        model = create_piecelog_model(n_features=28, seq_len=200, config=config)
        assert model.d_model == 64

    def test_custom_T_max(self):
        model = create_piecelog_model(n_features=28, seq_len=200, T_max=500.0)
        assert model.T_max == 500.0


class TestEndToEnd:
    def test_single_training_step(self):
        """Single forward-backward pass should work."""
        model = PieceLogPatchTST(n_features=28, seq_len=100, d_model=16, d_ff=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        x = torch.randn(4, 100, 28)
        t_predict = torch.tensor([100.0, 200.0, 300.0, 350.0])
        y_true = torch.tensor([10.0, 25.0, 30.0, 28.0])

        model.train()
        out = model(x, t_predict)
        loss = criterion(out, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)

    def test_dual_loss_training_step(self):
        """Dual loss (concentration + parameters) should backprop."""
        model = PieceLogPatchTST(n_features=28, seq_len=100, d_model=16, d_ff=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        x = torch.randn(4, 100, 28)
        t_predict = torch.tensor([100.0, 200.0, 300.0, 350.0])
        y_true = torch.tensor([10.0, 25.0, 30.0, 28.0])
        params_true = torch.rand(4, 7) * 50  # dummy params

        model.train()
        params_pred = model.get_parameters(x)
        out = model(x, t_predict)

        L_conc = criterion(out, y_true)
        L_param = sum(
            criterion(params_pred[name], params_true[:, i])
            for i, name in enumerate(PARAM_NAMES)
        ) / 7
        loss = L_conc + 0.1 * L_param

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)

    def test_freeze_then_unfreeze_training(self):
        """Frozen epochs should not update param_head; unfrozen should."""
        model = PieceLogPatchTST(n_features=28, seq_len=100, d_model=16, d_ff=64)
        criterion = torch.nn.MSELoss()

        x = torch.randn(4, 100, 28)
        t_predict = torch.tensor([100.0, 200.0, 300.0, 350.0])
        y_true = torch.tensor([10.0, 25.0, 30.0, 28.0])

        # Freeze param_head
        for p in model.param_head.parameters():
            p.requires_grad = False

        head_weight_before = model.param_head.mlp[2].weight.clone()

        # Train one step (frozen)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        model.train()
        out = model(x, t_predict)
        loss = criterion(out, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        head_weight_after_frozen = model.param_head.mlp[2].weight.clone()
        assert torch.allclose(head_weight_before, head_weight_after_frozen), \
            "param_head weights should not change while frozen"

        # Unfreeze
        for p in model.param_head.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train one step (unfrozen)
        out = model(x, t_predict)
        loss = criterion(out, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        head_weight_after_unfrozen = model.param_head.mlp[2].weight.clone()
        assert not torch.allclose(head_weight_after_frozen, head_weight_after_unfrozen), \
            "param_head weights should change after unfreezing"


def _make_fitted_params_df(T_max: float = 400.0) -> pd.DataFrame:
    """Create a synthetic fitted_params_df for testing.

    Values must be within constraint bounds for the given T_max:
    - t0 in (0, T_max), t_lag in (0, 0.3*T_max), t_break in (0.5*T_max, T_max)
    """
    return pd.DataFrame({
        "batch_id": range(1, 11),
        "K": [37.0] * 10,
        "r": [0.021] * 10,
        "t0": [38.0] * 10,
        "lam": [0.023] * 10,
        "t_lag": [26.0] * 10,
        "t_break": [0.6 * T_max] * 10,  # safely in [0.5*T_max, T_max]
        "slope": [0.048] * 10,
        "r_squared": [0.999] * 10,
    })


class TestInitializeParamHead:
    def test_output_matches_mean_params(self):
        """After initialization, model output should match mean fitted params."""
        T_max = 400.0
        model = PieceLogPatchTST(
            n_features=28, seq_len=100, d_model=16, d_ff=64, T_max=T_max
        )
        df = _make_fitted_params_df(T_max)
        initialize_param_head(model, df)

        model.eval()
        x = torch.randn(8, 100, 28)
        with torch.no_grad():
            params = model.get_parameters(x)

        for name in PARAM_NAMES:
            expected = df[name].mean()
            predicted = params[name].numpy()
            np.testing.assert_allclose(
                predicted, expected, rtol=0.01,
                err_msg=f"{name}: expected {expected}, got {predicted.mean()}",
            )

    def test_output_constant_across_inputs(self):
        """With zeroed weights, output should be the same for any input."""
        model = PieceLogPatchTST(
            n_features=28, seq_len=100, d_model=16, d_ff=64, T_max=400.0
        )
        df = _make_fitted_params_df()
        initialize_param_head(model, df)

        model.eval()
        with torch.no_grad():
            params1 = model.get_parameters(torch.randn(2, 100, 28))
            params2 = model.get_parameters(torch.randn(2, 100, 28))

        for name in PARAM_NAMES:
            torch.testing.assert_close(params1[name], params2[name])

    def test_output_layer_weights_zeroed(self):
        """Output layer weights should be zero after initialization."""
        model = PieceLogPatchTST(
            n_features=28, seq_len=100, d_model=16, d_ff=64, T_max=400.0
        )
        df = _make_fitted_params_df()
        initialize_param_head(model, df)

        output_layer = model.param_head.mlp[2]
        assert torch.all(output_layer.weight == 0)

    def test_gradient_flow_after_init(self):
        """Gradients should still flow through the initialized head."""
        model = PieceLogPatchTST(
            n_features=28, seq_len=100, d_model=16, d_ff=64, T_max=400.0
        )
        df = _make_fitted_params_df()
        initialize_param_head(model, df)

        x = torch.randn(4, 100, 28)
        t_predict = torch.tensor([100.0, 200.0, 300.0, 350.0])
        out = model(x, t_predict)
        loss = out.sum()
        loss.backward()

        encoder_param = next(model.encoder.parameters())
        assert encoder_param.grad is not None
