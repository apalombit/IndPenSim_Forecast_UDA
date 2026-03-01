"""Tests for modular PieceLog sub-models."""

import numpy as np
import pandas as pd
import torch
import pytest

from src.piecelog_model import PARAM_NAMES
from src.modular_piecelog import (
    TimingHead,
    GrowthHead,
    DeclineHead,
    SplitDeclineHead,
    UngatedDeclineHead,
    TimingModel,
    GrowthModel,
    DeclineModel,
    SplitDeclineModel,
    UngatedDeclineModel,
    CompositeModel,
    create_modular_piecelog,
    initialize_timing_head,
    initialize_growth_head,
    initialize_decline_head,
    initialize_split_decline_head,
    initialize_ungated_decline_head,
)
from src.train_modular_piecelog import (
    timing_loss,
    growth_loss,
    decline_loss,
    ungated_decline_loss,
    composite_loss,
    _gate_weight_schedule,
    retrain_timing,
    retrain_growth,
    retrain_decline,
    retrain_split_decline,
    retrain_ungated_decline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T_MAX = 400.0
SEQ_LEN = 100
D_MODEL = 16
D_FF = 64
N_FEATURES = 28
B = 4


def _make_fitted_params_df() -> pd.DataFrame:
    return pd.DataFrame({
        "batch_id": range(1, 11),
        "K": [37.0] * 10,
        "r": [0.021] * 10,
        "t0": [38.0] * 10,
        "lam": [0.023] * 10,
        "t_lag": [26.0] * 10,
        "t_break": [0.6 * T_MAX] * 10,
        "slope": [0.048] * 10,
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


def _make_t_cutoff_norm(n: int = B, frac: float = 0.9) -> torch.Tensor:
    """Normalized t_cutoff for DeclineHead / DeclineModel tests."""
    return torch.full((n,), frac)


def _make_params_true():
    return torch.tensor([
        [37.0, 0.021, 38.0, 0.023, 26.0, 240.0, 0.048],
    ]).expand(B, 7).clone()


# ---------------------------------------------------------------------------
# TimingHead tests
# ---------------------------------------------------------------------------

class TestTimingHead:
    def test_output_keys(self):
        head = TimingHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(B, D_MODEL))
        assert set(out.keys()) == {"t_lag"}

    def test_output_shape(self):
        head = TimingHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(B, D_MODEL))
        assert out["t_lag"].shape == (B,)

    def test_range(self):
        head = TimingHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(100, D_MODEL))
        assert (out["t_lag"] >= 0).all()
        assert (out["t_lag"] <= 0.3 * T_MAX + 1e-4).all()

    def test_gradient_flow(self):
        head = TimingHead(d_model=D_MODEL, T_max=T_MAX)
        features = torch.randn(B, D_MODEL, requires_grad=True)
        out = head(features)
        out["t_lag"].sum().backward()
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()

    def test_different_T_max(self):
        head = TimingHead(d_model=D_MODEL, T_max=200.0)
        out = head(torch.randn(100, D_MODEL))
        assert (out["t_lag"] <= 0.3 * 200.0 + 1e-4).all()


# ---------------------------------------------------------------------------
# GrowthHead tests
# ---------------------------------------------------------------------------

class TestGrowthHead:
    def test_output_keys(self):
        head = GrowthHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(B, D_MODEL))
        assert set(out.keys()) == {"K", "r", "t0", "lam"}

    def test_output_shapes(self):
        head = GrowthHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(B, D_MODEL))
        for key in ["K", "r", "t0", "lam"]:
            assert out[key].shape == (B,)

    def test_K_r_positive(self):
        head = GrowthHead(d_model=D_MODEL, T_max=T_MAX, K_scale=60.0)
        out = head(torch.randn(100, D_MODEL))
        assert (out["K"] > 0).all()
        assert (out["r"] > 0).all()

    def test_t0_in_range(self):
        head = GrowthHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(100, D_MODEL))
        assert (out["t0"] >= 0).all()
        assert (out["t0"] <= T_MAX).all()

    def test_gradient_flow(self):
        head = GrowthHead(d_model=D_MODEL, T_max=T_MAX)
        features = torch.randn(B, D_MODEL, requires_grad=True)
        out = head(features)
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        assert features.grad is not None


# ---------------------------------------------------------------------------
# DeclineHead tests
# ---------------------------------------------------------------------------

class TestDeclineHead:
    def test_output_keys(self):
        head = DeclineHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(B, D_MODEL), _make_t_cutoff_norm())
        assert set(out.keys()) == {"t_break", "slope", "decline_gate"}

    def test_t_break_range(self):
        head = DeclineHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(100, D_MODEL), _make_t_cutoff_norm(100))
        assert (out["t_break"] >= 0).all()
        assert (out["t_break"] <= T_MAX + 1e-4).all()

    def test_slope_positive(self):
        head = DeclineHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(100, D_MODEL), _make_t_cutoff_norm(100))
        assert (out["slope"] > 0).all()

    def test_gate_in_range(self):
        head = DeclineHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(100, D_MODEL), _make_t_cutoff_norm(100))
        assert (out["decline_gate"] >= 0).all()
        assert (out["decline_gate"] <= 1).all()

    def test_gradient_flow(self):
        head = DeclineHead(d_model=D_MODEL, T_max=T_MAX)
        features = torch.randn(B, D_MODEL, requires_grad=True)
        out = head(features, _make_t_cutoff_norm())
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        assert features.grad is not None


# ---------------------------------------------------------------------------
# TimingModel tests
# ---------------------------------------------------------------------------

class TestTimingModel:
    def test_forward(self):
        model = TimingModel(**_make_model_kwargs())
        out = model(_make_input())
        assert "t_lag" in out
        assert out["t_lag"].shape == (B,)

    def test_get_features(self):
        model = TimingModel(**_make_model_kwargs())
        feat = model.get_features(_make_input())
        assert feat.shape == (B, D_MODEL)

    def test_gradient_flow_through_encoder(self):
        model = TimingModel(**_make_model_kwargs())
        x = _make_input()
        out = model(x)
        out["t_lag"].sum().backward()
        encoder_param = next(model.encoder.parameters())
        assert encoder_param.grad is not None

    def test_range_constraint(self):
        model = TimingModel(**_make_model_kwargs())
        out = model(torch.randn(32, SEQ_LEN, N_FEATURES))
        assert (out["t_lag"] >= 0).all()
        assert (out["t_lag"] <= 0.3 * T_MAX + 1e-4).all()


# ---------------------------------------------------------------------------
# GrowthModel tests
# ---------------------------------------------------------------------------

class TestGrowthModel:
    def test_forward(self):
        model = GrowthModel(**_make_model_kwargs())
        out = model(_make_input())
        assert set(out.keys()) == {"K", "r", "t0", "lam"}

    def test_gradient_flow(self):
        model = GrowthModel(**_make_model_kwargs())
        x = _make_input()
        out = model(x)
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        encoder_param = next(model.encoder.parameters())
        assert encoder_param.grad is not None

    def test_constraints(self):
        model = GrowthModel(**_make_model_kwargs())
        out = model(torch.randn(32, SEQ_LEN, N_FEATURES))
        assert (out["K"] > 0).all()
        assert (out["r"] > 0).all()
        assert (out["lam"] > 0).all()
        assert (out["t0"] >= 0).all()
        assert (out["t0"] <= T_MAX).all()


# ---------------------------------------------------------------------------
# DeclineModel tests
# ---------------------------------------------------------------------------

class TestDeclineModel:
    def test_forward(self):
        model = DeclineModel(**_make_model_kwargs())
        out = model(_make_input(), _make_t_cutoff_norm())
        assert set(out.keys()) == {"t_break", "slope", "decline_gate"}

    def test_gradient_flow(self):
        model = DeclineModel(**_make_model_kwargs())
        x = _make_input()
        out = model(x, _make_t_cutoff_norm())
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        encoder_param = next(model.encoder.parameters())
        assert encoder_param.grad is not None

    def test_constraints(self):
        model = DeclineModel(**_make_model_kwargs())
        out = model(torch.randn(32, SEQ_LEN, N_FEATURES), _make_t_cutoff_norm(32))
        assert (out["t_break"] >= 0).all()
        assert (out["t_break"] <= T_MAX + 1e-4).all()
        assert (out["slope"] > 0).all()
        assert (out["decline_gate"] >= 0).all()
        assert (out["decline_gate"] <= 1).all()


# ---------------------------------------------------------------------------
# CompositeModel tests
# ---------------------------------------------------------------------------

class TestCompositeModel:
    def _make_composite(self):
        kwargs = _make_model_kwargs()
        return CompositeModel(
            TimingModel(**kwargs),
            GrowthModel(**kwargs),
            DeclineModel(**kwargs),
        )

    def test_all_7_keys(self):
        model = self._make_composite()
        params = model.get_parameters(_make_input())
        assert set(params.keys()) == set(PARAM_NAMES)

    def test_all_shapes(self):
        model = self._make_composite()
        params = model.get_parameters(_make_input())
        for name in PARAM_NAMES:
            assert params[name].shape == (B,), f"{name} wrong shape"

    def test_P_hat_shape(self):
        model = self._make_composite()
        x = _make_input()
        t_predict = torch.tensor([100.0, 200.0, 300.0, 350.0])
        P_hat = model(x, t_predict)
        assert P_hat.shape == (B,)

    def test_P_hat_non_negative(self):
        model = self._make_composite()
        x = torch.randn(8, SEQ_LEN, N_FEATURES)
        t_predict = torch.rand(8) * T_MAX
        P_hat = model(x, t_predict)
        assert (P_hat >= 0).all()

    def test_hard_gate_zeros_slope(self):
        model = self._make_composite()
        x = _make_input()

        # Force gate output < 0.5 by setting bias very negative
        with torch.no_grad():
            model.decline_model.head.mlp[2].bias[2] = -10.0
        params = model.get_parameters(x, hard_gate=True)
        assert (params["slope"] == 0).all()

    def test_soft_gate_preserves_gradients(self):
        model = self._make_composite()
        x = _make_input()
        params = model.get_parameters(x, hard_gate=False)
        params["slope"].sum().backward()
        # Gradient should flow to decline encoder
        decline_param = next(model.decline_model.encoder.parameters())
        assert decline_param.grad is not None

    def test_gradient_flow_to_all_encoders(self):
        model = self._make_composite()
        x = _make_input()
        t_predict = torch.tensor([100.0, 200.0, 300.0, 350.0])
        P_hat = model(x, t_predict)
        P_hat.sum().backward()

        for name, sub in [("timing", model.timing_model), ("growth", model.growth_model), ("decline", model.decline_model)]:
            param = next(sub.encoder.parameters())
            assert param.grad is not None, f"No gradient for {name} encoder"

    def test_get_features(self):
        model = self._make_composite()
        feats = model.get_features(_make_input())
        assert set(feats.keys()) == {"timing", "growth", "decline"}
        for key, val in feats.items():
            assert val.shape == (B, D_MODEL), f"{key} features wrong shape"


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_timing_init_matches_mean(self):
        model = TimingModel(**_make_model_kwargs())
        df = _make_fitted_params_df()
        initialize_timing_head(model, df)

        model.eval()
        with torch.no_grad():
            out = model(_make_input())
        np.testing.assert_allclose(
            out["t_lag"].numpy(), 26.0, rtol=0.01,
        )

    def test_growth_init_matches_mean(self):
        model = GrowthModel(**_make_model_kwargs())
        df = _make_fitted_params_df()
        initialize_growth_head(model, df)

        model.eval()
        with torch.no_grad():
            out = model(_make_input())
        for name, expected in [("K", 37.0), ("r", 0.021), ("t0", 38.0), ("lam", 0.023)]:
            np.testing.assert_allclose(
                out[name].numpy(), expected, rtol=0.02,
                err_msg=f"{name}: expected {expected}, got {out[name].mean().item():.4f}",
            )

    def test_decline_init_matches_mean(self):
        model = DeclineModel(**_make_model_kwargs())
        df = _make_fitted_params_df()
        initialize_decline_head(model, df)

        # At init: bias[0] = logit(mean_t_break / T_max) → t_break ≈ mean_t_break
        t_cutoff_norm = _make_t_cutoff_norm(B, 0.9)
        model.eval()
        with torch.no_grad():
            out = model(_make_input(), t_cutoff_norm)
        mean_t_break = df["t_break"].mean()  # 0.6 * T_MAX = 240.0
        np.testing.assert_allclose(
            out["t_break"].numpy(), mean_t_break, rtol=0.02,
        )
        np.testing.assert_allclose(
            out["slope"].numpy(), 0.048, rtol=0.02,
        )

    def test_gradient_flow_after_init(self):
        model = TimingModel(**_make_model_kwargs())
        df = _make_fitted_params_df()
        initialize_timing_head(model, df)

        x = _make_input()
        out = model(x)
        out["t_lag"].sum().backward()
        encoder_param = next(model.encoder.parameters())
        assert encoder_param.grad is not None


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestLossFunctions:
    def test_timing_loss_smoke(self):
        model = TimingModel(**_make_model_kwargs())
        x = _make_input()
        params_true = _make_params_true()
        loss, ld = timing_loss(model, x, params_true)
        assert torch.isfinite(loss)
        assert "curve" in ld and "param" in ld

    def test_timing_loss_backward(self):
        model = TimingModel(**_make_model_kwargs())
        x = _make_input()
        params_true = _make_params_true()
        loss, _ = timing_loss(model, x, params_true)
        loss.backward()
        param = next(model.parameters())
        assert param.grad is not None

    def test_growth_loss_smoke(self):
        model = GrowthModel(**_make_model_kwargs())
        x = _make_input()
        params_true = _make_params_true()
        t_lag_input = torch.full((B,), 26.0)
        loss, ld = growth_loss(model, x, params_true, t_lag_input)
        assert torch.isfinite(loss)
        assert "curve" in ld and "param" in ld

    def test_growth_loss_backward(self):
        model = GrowthModel(**_make_model_kwargs())
        x = _make_input()
        params_true = _make_params_true()
        t_lag_input = torch.full((B,), 26.0)
        loss, _ = growth_loss(model, x, params_true, t_lag_input)
        loss.backward()
        param = next(model.parameters())
        assert param.grad is not None

    def test_decline_loss_smoke(self):
        model = DeclineModel(**_make_model_kwargs())
        x = _make_input()
        params_true = _make_params_true()
        t_lag_input = torch.full((B,), 26.0)
        gp = {"K": params_true[:, 0], "r": params_true[:, 1],
              "t0": params_true[:, 2], "lam": params_true[:, 3]}
        loss, ld = decline_loss(model, x, params_true, t_lag_input, gp)
        assert torch.isfinite(loss)
        assert "curve" in ld and "param" in ld and "gate" in ld
        assert "slope_cond" in ld

    def test_decline_loss_backward(self):
        model = DeclineModel(**_make_model_kwargs())
        x = _make_input()
        params_true = _make_params_true()
        t_lag_input = torch.full((B,), 26.0)
        gp = {"K": params_true[:, 0], "r": params_true[:, 1],
              "t0": params_true[:, 2], "lam": params_true[:, 3]}
        loss, _ = decline_loss(model, x, params_true, t_lag_input, gp)
        loss.backward()
        param = next(model.parameters())
        assert param.grad is not None

    def test_decline_loss_slope_weight(self):
        """slope_weight > 0 should add masked conditional supervision."""
        model = DeclineModel(**_make_model_kwargs())
        x = _make_input()
        # All samples have slope=0.048 > 0.01 threshold → decline_mask is all True
        params_true = _make_params_true()
        t_lag_input = torch.full((B,), 26.0)
        gp = {"K": params_true[:, 0], "r": params_true[:, 1],
              "t0": params_true[:, 2], "lam": params_true[:, 3]}
        _, ld = decline_loss(model, x, params_true, t_lag_input, gp,
                             slope_weight=0.1)
        assert ld["slope_cond"] > 0.0

    def test_decline_loss_slope_weight_zero_no_cond(self):
        """slope_weight=0 should skip conditional supervision."""
        model = DeclineModel(**_make_model_kwargs())
        x = _make_input()
        params_true = _make_params_true()
        t_lag_input = torch.full((B,), 26.0)
        gp = {"K": params_true[:, 0], "r": params_true[:, 1],
              "t0": params_true[:, 2], "lam": params_true[:, 3]}
        _, ld = decline_loss(model, x, params_true, t_lag_input, gp,
                             slope_weight=0.0)
        assert ld["slope_cond"] == 0.0

    def test_decline_loss_no_declining_samples(self):
        """When no samples exceed gate_threshold, slope_cond should be 0."""
        model = DeclineModel(**_make_model_kwargs())
        x = _make_input()
        # Set slope to 0.0 (below threshold) for all samples
        params_true = _make_params_true()
        params_true[:, 6] = 0.0
        t_lag_input = torch.full((B,), 26.0)
        gp = {"K": params_true[:, 0], "r": params_true[:, 1],
              "t0": params_true[:, 2], "lam": params_true[:, 3]}
        _, ld = decline_loss(model, x, params_true, t_lag_input, gp,
                             slope_weight=0.1)
        assert ld["slope_cond"] == 0.0

    def test_decline_loss_detach_gate(self):
        """detach_gate=True should block gradient to gate from L_curve."""
        model = DeclineModel(**_make_model_kwargs())
        x = _make_input()
        params_true = _make_params_true()
        t_lag_input = torch.full((B,), 26.0)
        gp = {"K": params_true[:, 0], "r": params_true[:, 1],
              "t0": params_true[:, 2], "lam": params_true[:, 3]}
        # With detach_gate, loss should still be finite and backprop should work
        loss, ld = decline_loss(model, x, params_true, t_lag_input, gp,
                                detach_gate=True, gate_weight=0.0)
        loss.backward()
        param = next(model.parameters())
        assert param.grad is not None

    def test_composite_loss_smoke(self):
        kwargs = _make_model_kwargs()
        composite = CompositeModel(
            TimingModel(**kwargs), GrowthModel(**kwargs), DeclineModel(**kwargs),
        )
        x = _make_input()
        params_true = _make_params_true()
        t_cutoff = torch.full((B,), 200.0)
        loss, ld = composite_loss(composite, x, params_true, t_cutoff)
        assert torch.isfinite(loss)
        assert "curve" in ld and "param" in ld

    def test_composite_loss_backward(self):
        kwargs = _make_model_kwargs()
        composite = CompositeModel(
            TimingModel(**kwargs), GrowthModel(**kwargs), DeclineModel(**kwargs),
        )
        x = _make_input()
        params_true = _make_params_true()
        t_cutoff = torch.full((B,), 200.0)
        loss, _ = composite_loss(composite, x, params_true, t_cutoff)
        loss.backward()
        # Check all encoders got gradients
        for name, sub in [("timing", composite.timing_model), ("growth", composite.growth_model), ("decline", composite.decline_model)]:
            param = next(sub.parameters())
            assert param.grad is not None, f"No gradient for {name}"

    def test_loss_warns_without_param_stats(self):
        """All loss functions should warn when param_stats is None."""
        model = TimingModel(**_make_model_kwargs())
        x = _make_input()
        params_true = _make_params_true()
        with pytest.warns(UserWarning, match="param_stats is None"):
            timing_loss(model, x, params_true, param_stats=None)


# ---------------------------------------------------------------------------
# Gate-weight annealing tests
# ---------------------------------------------------------------------------

class TestGateWeightSchedule:
    def test_warmup_returns_full_weight(self):
        """During warmup phase, gate_weight should be unchanged."""
        gw = 1.0
        for epoch in range(30):  # warmup_frac=0.3, n_epochs=100 → warmup_end=30
            w = _gate_weight_schedule(epoch, 100, gw, 0.01, 0.3)
            assert w == gw, f"Epoch {epoch}: expected {gw}, got {w}"

    def test_last_epoch_returns_min(self):
        """At the final epoch, gate_weight should equal gate_weight_min."""
        w = _gate_weight_schedule(99, 100, 1.0, 0.01, 0.3)
        assert abs(w - 0.01) < 1e-6

    def test_monotonic_decrease(self):
        """After warmup, gate_weight should decrease monotonically."""
        weights = [
            _gate_weight_schedule(e, 100, 1.0, 0.01, 0.3)
            for e in range(100)
        ]
        # After warmup (epoch 30+), should be non-increasing
        post_warmup = weights[30:]
        for i in range(1, len(post_warmup)):
            assert post_warmup[i] <= post_warmup[i - 1] + 1e-9

    def test_no_annealing_when_min_equals_max(self):
        """If gate_weight_min == gate_weight, no annealing occurs."""
        for epoch in range(100):
            w = _gate_weight_schedule(epoch, 100, 0.5, 0.5, 0.3)
            assert abs(w - 0.5) < 1e-9

    def test_zero_warmup(self):
        """With warmup_frac=0, annealing starts immediately."""
        w0 = _gate_weight_schedule(0, 100, 1.0, 0.01, 0.0)
        assert w0 == 1.0  # First epoch = start of cosine
        w_last = _gate_weight_schedule(99, 100, 1.0, 0.01, 0.0)
        assert abs(w_last - 0.01) < 1e-6


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_single_timing_training_step(self):
        model = TimingModel(**_make_model_kwargs())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = _make_input()
        params_true = _make_params_true()

        model.train()
        loss, _ = timing_loss(model, x, params_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)

    def test_single_growth_training_step(self):
        model = GrowthModel(**_make_model_kwargs())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = _make_input()
        params_true = _make_params_true()
        t_lag_input = torch.full((B,), 26.0)

        model.train()
        loss, _ = growth_loss(model, x, params_true, t_lag_input)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)

    def test_frozen_upstream_unchanged(self):
        """Upstream models' weights should not change during downstream training."""
        kwargs = _make_model_kwargs()
        timing = TimingModel(**kwargs)
        growth = GrowthModel(**kwargs)

        # Snapshot timing weights
        timing_w_before = next(timing.encoder.parameters()).clone()

        # Freeze timing
        timing.eval()
        for p in timing.parameters():
            p.requires_grad = False

        # One growth step
        optimizer = torch.optim.Adam(growth.parameters(), lr=1e-3)
        x = _make_input()
        params_true = _make_params_true()

        with torch.no_grad():
            t_lag_input = timing(x)["t_lag"]

        growth.train()
        loss, _ = growth_loss(growth, x, params_true, t_lag_input)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        timing_w_after = next(timing.encoder.parameters())
        assert torch.allclose(timing_w_before, timing_w_after), \
            "Frozen timing weights changed during growth training"


# ---------------------------------------------------------------------------
# Factory test
# ---------------------------------------------------------------------------

class TestFactory:
    def test_create_modular_piecelog(self):
        model = create_modular_piecelog(n_features=28, seq_len=100, T_max=400.0)
        assert isinstance(model, CompositeModel)
        assert isinstance(model.timing_model, TimingModel)
        assert isinstance(model.growth_model, GrowthModel)
        assert isinstance(model.decline_model, DeclineModel)

    def test_create_with_config(self):
        model = create_modular_piecelog(
            n_features=28, seq_len=100, T_max=400.0,
            config={"d_model": 64, "n_layers": 3},
        )
        # Verify d_model propagated
        feat = model.timing_model.get_features(torch.randn(2, 100, 28))
        assert feat.shape == (2, 64)

    def test_create_with_per_model_config(self):
        model = create_modular_piecelog(
            n_features=28, seq_len=100, T_max=400.0,
            decline_config={"d_model": 64, "n_layers": 3},
        )
        # Decline encoder should have d_model=64
        x = torch.randn(2, 100, 28)
        decline_feat = model.decline_model.get_features(x)
        assert decline_feat.shape == (2, 64)
        # Timing/growth should still use default d_model=32
        timing_feat = model.timing_model.get_features(x)
        assert timing_feat.shape == (2, 32)

    def test_per_model_overrides_shared(self):
        model = create_modular_piecelog(
            n_features=28, seq_len=100, T_max=400.0,
            config={"d_model": 16, "n_heads": 4, "d_ff": 64},
            decline_config={"d_model": 32, "d_ff": 128},
        )
        x = torch.randn(2, 100, 28)
        # Timing gets shared d_model=16
        timing_feat = model.timing_model.get_features(x)
        assert timing_feat.shape == (2, 16)
        # Decline gets overridden d_model=32
        decline_feat = model.decline_model.get_features(x)
        assert decline_feat.shape == (2, 32)

    def test_forward_with_mixed_configs(self):
        model = create_modular_piecelog(
            n_features=28, seq_len=100, T_max=400.0,
            config={"d_model": 16, "n_heads": 4, "d_ff": 64},
            decline_config={"d_model": 32, "d_ff": 128},
        )
        x = torch.randn(4, 100, 28)
        t_predict = torch.tensor([100.0, 200.0, 300.0, 350.0])
        P_hat = model(x, t_predict)
        assert P_hat.shape == (4,)
        assert (P_hat >= 0).all()

    def test_forward_works(self):
        model = create_modular_piecelog(n_features=28, seq_len=100, T_max=400.0)
        x = torch.randn(4, 100, 28)
        t_predict = torch.tensor([100.0, 200.0, 300.0, 350.0])
        P_hat = model(x, t_predict)
        assert P_hat.shape == (4,)
        assert (P_hat >= 0).all()


# ---------------------------------------------------------------------------
# Retrain tests
# ---------------------------------------------------------------------------

from torch.utils.data import DataLoader, TensorDataset


def _make_fake_loader(n_samples: int = 8, batch_size: int = 4) -> DataLoader:
    """Build a minimal DataLoader with the keys expected by the training loops."""
    x = torch.randn(n_samples, SEQ_LEN, N_FEATURES)
    params_fitted = _make_params_true()[:1].expand(n_samples, 7).clone()
    t_predict = torch.full((n_samples,), 200.0)
    t_cutoff = torch.full((n_samples,), 180.0)
    y_conc = torch.full((n_samples,), 20.0)
    domain_label = torch.zeros(n_samples)

    class _DictDataset(torch.utils.data.Dataset):
        def __init__(self, x, params, t_pred, t_cut, y, dl):
            self.x = x
            self.params = params
            self.t_pred = t_pred
            self.t_cut = t_cut
            self.y = y
            self.dl = dl

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, idx):
            return {
                "x": self.x[idx],
                "params_fitted": self.params[idx],
                "t_predict": self.t_pred[idx],
                "t_cutoff": self.t_cut[idx],
                "y_conc": self.y[idx],
                "domain_label": self.dl[idx],
            }

    ds = _DictDataset(x, params_fitted, t_predict, t_cutoff, y_conc, domain_label)
    return DataLoader(ds, batch_size=batch_size)


class TestRetrain:
    def _make_composite_and_loader(self):
        composite = create_modular_piecelog(
            n_features=N_FEATURES, seq_len=SEQ_LEN, T_max=T_MAX,
            config={"d_model": D_MODEL, "n_heads": 4, "n_layers": 1,
                    "d_ff": D_FF, "head_hidden": 16, "dropout": 0.0},
        )
        df = _make_fitted_params_df()
        initialize_timing_head(composite.timing_model, df)
        initialize_growth_head(composite.growth_model, df)
        initialize_decline_head(composite.decline_model, df)
        loader = _make_fake_loader()
        return composite, df, loader

    def test_retrain_decline_runs(self):
        composite, df, loader = self._make_composite_and_loader()
        result = retrain_decline(
            composite, df, train_loader=loader, val_loader=loader,
            config={"d_model": 8, "n_heads": 4, "n_layers": 1,
                    "d_ff": 32, "head_hidden": 8},
            n_epochs=3, patience=5, verbose=False,
        )
        assert "history" in result
        assert "best_epoch" in result
        # Verify the composite's decline model was replaced
        feat = composite.decline_model.get_features(torch.randn(2, SEQ_LEN, N_FEATURES))
        assert feat.shape == (2, 8)

    def test_retrain_decline_preserves_upstream(self):
        composite, df, loader = self._make_composite_and_loader()
        # Snapshot upstream weights
        timing_w = next(composite.timing_model.encoder.parameters()).clone()
        growth_w = next(composite.growth_model.encoder.parameters()).clone()

        retrain_decline(
            composite, df, train_loader=loader, val_loader=loader,
            n_epochs=3, patience=5, verbose=False,
        )

        timing_w_after = next(composite.timing_model.encoder.parameters())
        growth_w_after = next(composite.growth_model.encoder.parameters())
        assert torch.allclose(timing_w, timing_w_after), \
            "Timing weights changed during decline retrain"
        assert torch.allclose(growth_w, growth_w_after), \
            "Growth weights changed during decline retrain"

    def test_retrain_timing_runs(self):
        composite, df, loader = self._make_composite_and_loader()
        result = retrain_timing(
            composite, df, train_loader=loader, val_loader=loader,
            config={"d_model": 8, "n_heads": 4, "n_layers": 1,
                    "d_ff": 32, "head_hidden": 8},
            n_epochs=3, patience=5, verbose=False,
        )
        assert "history" in result
        assert "best_epoch" in result
        feat = composite.timing_model.get_features(torch.randn(2, SEQ_LEN, N_FEATURES))
        assert feat.shape == (2, 8)

    def test_retrain_growth_runs(self):
        composite, df, loader = self._make_composite_and_loader()
        result = retrain_growth(
            composite, df, train_loader=loader, val_loader=loader,
            config={"d_model": 8, "n_heads": 4, "n_layers": 1,
                    "d_ff": 32, "head_hidden": 8},
            n_epochs=3, patience=5, verbose=False,
        )
        assert "history" in result
        assert "best_epoch" in result
        feat = composite.growth_model.get_features(torch.randn(2, SEQ_LEN, N_FEATURES))
        assert feat.shape == (2, 8)

    def test_retrain_ungated_decline_runs(self):
        composite, df, loader = self._make_composite_and_loader()
        result = retrain_ungated_decline(
            composite, df, train_loader=loader, val_loader=loader,
            config={"d_model": 8, "n_heads": 4, "n_layers": 1,
                    "d_ff": 32, "head_hidden": 8},
            n_epochs=3, patience=5, verbose=False,
        )
        assert "history" in result
        assert "best_epoch" in result
        # Verify the composite's decline model was replaced with ungated
        assert isinstance(composite.decline_model, UngatedDeclineModel)
        feat = composite.decline_model.get_features(torch.randn(2, SEQ_LEN, N_FEATURES))
        assert feat.shape == (2, 8)
        # Verify composite still works end-to-end
        x = torch.randn(2, SEQ_LEN, N_FEATURES)
        t_predict = torch.tensor([100.0, 200.0])
        P_hat = composite(x, t_predict, hard_gate=True)
        assert P_hat.shape == (2,)


# ---------------------------------------------------------------------------
# UngatedDeclineHead tests
# ---------------------------------------------------------------------------

class TestUngatedDeclineHead:
    def test_output_keys(self):
        head = UngatedDeclineHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(B, D_MODEL), _make_t_cutoff_norm())
        assert set(out.keys()) == {"t_break", "slope", "decline_gate"}

    def test_output_shapes(self):
        head = UngatedDeclineHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(B, D_MODEL), _make_t_cutoff_norm())
        assert out["t_break"].shape == (B,)
        assert out["slope"].shape == (B,)
        assert out["decline_gate"].shape == (B,)

    def test_t_break_range(self):
        head = UngatedDeclineHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(100, D_MODEL), _make_t_cutoff_norm(100))
        assert (out["t_break"] >= 0).all()
        assert (out["t_break"] <= T_MAX + 1e-4).all()

    def test_slope_positive(self):
        head = UngatedDeclineHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(100, D_MODEL), _make_t_cutoff_norm(100))
        assert (out["slope"] > 0).all()

    def test_gate_always_one(self):
        head = UngatedDeclineHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(100, D_MODEL), _make_t_cutoff_norm(100))
        assert (out["decline_gate"] == 1.0).all()

    def test_gradient_flow(self):
        head = UngatedDeclineHead(d_model=D_MODEL, T_max=T_MAX)
        features = torch.randn(B, D_MODEL, requires_grad=True)
        out = head(features, _make_t_cutoff_norm())
        loss = out["t_break"].sum() + out["slope"].sum()
        loss.backward()
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()


# ---------------------------------------------------------------------------
# UngatedDeclineModel tests
# ---------------------------------------------------------------------------

class TestUngatedDeclineModel:
    def test_forward(self):
        model = UngatedDeclineModel(**_make_model_kwargs())
        out = model(_make_input(), _make_t_cutoff_norm())
        assert set(out.keys()) == {"t_break", "slope", "decline_gate"}

    def test_gradient_flow(self):
        model = UngatedDeclineModel(**_make_model_kwargs())
        x = _make_input()
        out = model(x, _make_t_cutoff_norm())
        loss = out["t_break"].sum() + out["slope"].sum()
        loss.backward()
        encoder_param = next(model.encoder.parameters())
        assert encoder_param.grad is not None

    def test_constraints(self):
        model = UngatedDeclineModel(**_make_model_kwargs())
        out = model(torch.randn(32, SEQ_LEN, N_FEATURES), _make_t_cutoff_norm(32))
        assert (out["t_break"] >= 0).all()
        assert (out["t_break"] <= T_MAX + 1e-4).all()
        assert (out["slope"] > 0).all()
        assert (out["decline_gate"] == 1.0).all()

    def test_composite_compatibility(self):
        """UngatedDeclineModel should work as drop-in for CompositeModel."""
        kwargs = _make_model_kwargs()
        composite = CompositeModel(
            TimingModel(**kwargs),
            GrowthModel(**kwargs),
            UngatedDeclineModel(**kwargs),
        )
        x = _make_input()
        t_cutoff = torch.full((B,), 0.9 * T_MAX)
        params = composite.get_parameters(x, t_cutoff=t_cutoff, hard_gate=True)
        assert set(params.keys()) == set(PARAM_NAMES)
        # With gate=1.0, hard_gate should pass slope through (1.0 >= 0.5)
        t_cutoff_norm = t_cutoff / composite.decline_model.head.T_max
        raw_slope = composite.decline_model(x, t_cutoff_norm)["slope"]
        assert torch.allclose(params["slope"], raw_slope)

    def test_composite_forward(self):
        kwargs = _make_model_kwargs()
        composite = CompositeModel(
            TimingModel(**kwargs),
            GrowthModel(**kwargs),
            UngatedDeclineModel(**kwargs),
        )
        x = _make_input()
        t_predict = torch.tensor([100.0, 200.0, 300.0, 350.0])
        P_hat = composite(x, t_predict)
        assert P_hat.shape == (B,)
        assert (P_hat >= 0).all()


# ---------------------------------------------------------------------------
# UngatedDeclineHead initialization tests
# ---------------------------------------------------------------------------

class TestUngatedInitialization:
    def test_init_matches_mean(self):
        model = UngatedDeclineModel(**_make_model_kwargs())
        df = _make_fitted_params_df()
        initialize_ungated_decline_head(model, df)

        # At init: bias[0] = logit(mean_t_break / T_max) → t_break ≈ mean_t_break
        t_cutoff_norm = _make_t_cutoff_norm(B, 0.9)
        model.eval()
        with torch.no_grad():
            out = model(_make_input(), t_cutoff_norm)
        mean_t_break = df["t_break"].mean()  # 0.6 * T_MAX = 240.0
        np.testing.assert_allclose(
            out["t_break"].numpy(), mean_t_break, rtol=0.02,
        )
        np.testing.assert_allclose(
            out["slope"].numpy(), 0.048, rtol=0.02,
        )
        assert (out["decline_gate"] == 1.0).all()


# ---------------------------------------------------------------------------
# Ungated decline loss tests
# ---------------------------------------------------------------------------

class TestUngatedDeclineLoss:
    def test_smoke(self):
        model = UngatedDeclineModel(**_make_model_kwargs())
        x = _make_input()
        params_true = _make_params_true()
        t_lag_input = torch.full((B,), 26.0)
        t_cutoff = torch.full((B,), 0.9 * T_MAX)
        gp = {"K": params_true[:, 0], "r": params_true[:, 1],
              "t0": params_true[:, 2], "lam": params_true[:, 3]}
        loss, ld = ungated_decline_loss(model, x, params_true, t_lag_input, gp,
                                        t_cutoff=t_cutoff)
        assert torch.isfinite(loss)
        assert "curve" in ld and "param" in ld
        assert "gate" not in ld  # no gate loss in ungated version

    def test_backward(self):
        model = UngatedDeclineModel(**_make_model_kwargs())
        x = _make_input()
        params_true = _make_params_true()
        t_lag_input = torch.full((B,), 26.0)
        t_cutoff = torch.full((B,), 0.9 * T_MAX)
        gp = {"K": params_true[:, 0], "r": params_true[:, 1],
              "t0": params_true[:, 2], "lam": params_true[:, 3]}
        loss, _ = ungated_decline_loss(model, x, params_true, t_lag_input, gp,
                                       t_cutoff=t_cutoff)
        loss.backward()
        param = next(model.parameters())
        assert param.grad is not None

    def test_zero_slope_batches(self):
        """Ungated loss should handle batches with slope=0 gracefully."""
        model = UngatedDeclineModel(**_make_model_kwargs())
        x = _make_input()
        params_true = _make_params_true()
        params_true[:, 6] = 0.0  # all slopes zero
        t_lag_input = torch.full((B,), 26.0)
        t_cutoff = torch.full((B,), 0.9 * T_MAX)
        gp = {"K": params_true[:, 0], "r": params_true[:, 1],
              "t0": params_true[:, 2], "lam": params_true[:, 3]}
        loss, ld = ungated_decline_loss(model, x, params_true, t_lag_input, gp,
                                        t_cutoff=t_cutoff)
        assert torch.isfinite(loss)
        loss.backward()


# ---------------------------------------------------------------------------
# SplitDeclineHead tests
# ---------------------------------------------------------------------------

class TestSplitDeclineHead:
    def test_output_keys(self):
        head = SplitDeclineHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(B, D_MODEL), _make_t_cutoff_norm())
        assert set(out.keys()) == {"t_break", "slope", "decline_gate"}

    def test_output_shapes(self):
        head = SplitDeclineHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(B, D_MODEL), _make_t_cutoff_norm())
        assert out["t_break"].shape == (B,)
        assert out["slope"].shape == (B,)
        assert out["decline_gate"].shape == (B,)

    def test_t_break_range(self):
        head = SplitDeclineHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(100, D_MODEL), _make_t_cutoff_norm(100))
        assert (out["t_break"] >= 0).all()
        assert (out["t_break"] <= T_MAX + 1e-4).all()

    def test_slope_positive(self):
        head = SplitDeclineHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(100, D_MODEL), _make_t_cutoff_norm(100))
        assert (out["slope"] > 0).all()

    def test_gate_in_range(self):
        head = SplitDeclineHead(d_model=D_MODEL, T_max=T_MAX)
        out = head(torch.randn(100, D_MODEL), _make_t_cutoff_norm(100))
        assert (out["decline_gate"] >= 0).all()
        assert (out["decline_gate"] <= 1).all()

    def test_gradient_flow(self):
        head = SplitDeclineHead(d_model=D_MODEL, T_max=T_MAX)
        features = torch.randn(B, D_MODEL, requires_grad=True)
        out = head(features, _make_t_cutoff_norm())
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        assert features.grad is not None

    def test_separate_heads_exist(self):
        """Verify split head has 3 independent Linear layers (not shared)."""
        head = SplitDeclineHead(d_model=D_MODEL, hidden_dim=32, T_max=T_MAX)
        assert isinstance(head.t_break_head, torch.nn.Linear)
        assert isinstance(head.slope_head, torch.nn.Linear)
        assert isinstance(head.gate_head, torch.nn.Linear)
        # Each head is independent (different id)
        assert head.t_break_head is not head.slope_head
        assert head.slope_head is not head.gate_head


# ---------------------------------------------------------------------------
# SplitDeclineModel tests
# ---------------------------------------------------------------------------

class TestSplitDeclineModel:
    def test_forward(self):
        model = SplitDeclineModel(**_make_model_kwargs())
        out = model(_make_input(), _make_t_cutoff_norm())
        assert set(out.keys()) == {"t_break", "slope", "decline_gate"}

    def test_gradient_flow(self):
        model = SplitDeclineModel(**_make_model_kwargs())
        x = _make_input()
        out = model(x, _make_t_cutoff_norm())
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        encoder_param = next(model.encoder.parameters())
        assert encoder_param.grad is not None

    def test_constraints(self):
        model = SplitDeclineModel(**_make_model_kwargs())
        out = model(torch.randn(32, SEQ_LEN, N_FEATURES), _make_t_cutoff_norm(32))
        assert (out["t_break"] >= 0).all()
        assert (out["t_break"] <= T_MAX + 1e-4).all()
        assert (out["slope"] > 0).all()
        assert (out["decline_gate"] >= 0).all()
        assert (out["decline_gate"] <= 1).all()

    def test_composite_compatibility(self):
        """SplitDeclineModel should work as drop-in for CompositeModel."""
        kwargs = _make_model_kwargs()
        composite = CompositeModel(
            TimingModel(**kwargs),
            GrowthModel(**kwargs),
            SplitDeclineModel(**kwargs),
        )
        x = _make_input()
        t_cutoff = torch.full((B,), 0.9 * T_MAX)
        params = composite.get_parameters(x, t_cutoff=t_cutoff, hard_gate=True)
        assert set(params.keys()) == set(PARAM_NAMES)

    def test_composite_forward(self):
        kwargs = _make_model_kwargs()
        composite = CompositeModel(
            TimingModel(**kwargs),
            GrowthModel(**kwargs),
            SplitDeclineModel(**kwargs),
        )
        x = _make_input()
        t_predict = torch.tensor([100.0, 200.0, 300.0, 350.0])
        P_hat = composite(x, t_predict)
        assert P_hat.shape == (B,)
        assert (P_hat >= 0).all()

    def test_get_features(self):
        model = SplitDeclineModel(**_make_model_kwargs())
        feat = model.get_features(_make_input())
        assert feat.shape == (B, D_MODEL)


# ---------------------------------------------------------------------------
# SplitDeclineHead initialization tests
# ---------------------------------------------------------------------------

class TestSplitDeclineInitialization:
    def test_init_matches_mean(self):
        model = SplitDeclineModel(**_make_model_kwargs())
        df = _make_fitted_params_df()
        initialize_split_decline_head(model, df)

        t_cutoff_norm = _make_t_cutoff_norm(B, 0.9)
        model.eval()
        with torch.no_grad():
            out = model(_make_input(), t_cutoff_norm)
        mean_t_break = df["t_break"].mean()  # 0.6 * T_MAX = 240.0
        np.testing.assert_allclose(
            out["t_break"].numpy(), mean_t_break, rtol=0.02,
        )
        np.testing.assert_allclose(
            out["slope"].numpy(), 0.048, rtol=0.02,
        )

    def test_gate_init_matches_fraction(self):
        model = SplitDeclineModel(**_make_model_kwargs())
        df = _make_fitted_params_df()
        initialize_split_decline_head(model, df)

        model.eval()
        with torch.no_grad():
            out = model(_make_input(), _make_t_cutoff_norm())
        # All 10 samples in df have slope=0.048 > 0.01 threshold → gate_frac=1.0
        # sigmoid(logit(1-eps)) ≈ 1.0
        assert (out["decline_gate"] > 0.9).all()

    def test_shared_layer_not_zeroed(self):
        """Shared layer should keep its Xavier init (not zeroed)."""
        model = SplitDeclineModel(**_make_model_kwargs())
        df = _make_fitted_params_df()
        initialize_split_decline_head(model, df)

        shared_weight = model.head.shared[0].weight
        assert not torch.allclose(shared_weight, torch.zeros_like(shared_weight))


# ---------------------------------------------------------------------------
# SplitDeclineModel decline_loss compatibility tests
# ---------------------------------------------------------------------------

class TestSplitDeclineLoss:
    def test_decline_loss_smoke(self):
        """decline_loss should work with SplitDeclineModel."""
        model = SplitDeclineModel(**_make_model_kwargs())
        x = _make_input()
        params_true = _make_params_true()
        t_lag_input = torch.full((B,), 26.0)
        gp = {"K": params_true[:, 0], "r": params_true[:, 1],
              "t0": params_true[:, 2], "lam": params_true[:, 3]}
        loss, ld = decline_loss(model, x, params_true, t_lag_input, gp)
        assert torch.isfinite(loss)
        assert "curve" in ld and "param" in ld and "gate" in ld

    def test_decline_loss_backward(self):
        model = SplitDeclineModel(**_make_model_kwargs())
        x = _make_input()
        params_true = _make_params_true()
        t_lag_input = torch.full((B,), 26.0)
        gp = {"K": params_true[:, 0], "r": params_true[:, 1],
              "t0": params_true[:, 2], "lam": params_true[:, 3]}
        loss, _ = decline_loss(model, x, params_true, t_lag_input, gp)
        loss.backward()
        param = next(model.parameters())
        assert param.grad is not None


# ---------------------------------------------------------------------------
# retrain_split_decline tests
# ---------------------------------------------------------------------------

class TestRetrainSplitDecline:
    def _make_composite_and_loader(self):
        composite = create_modular_piecelog(
            n_features=N_FEATURES, seq_len=SEQ_LEN, T_max=T_MAX,
            config={"d_model": D_MODEL, "n_heads": 4, "n_layers": 1,
                    "d_ff": D_FF, "head_hidden": 16, "dropout": 0.0},
        )
        df = _make_fitted_params_df()
        initialize_timing_head(composite.timing_model, df)
        initialize_growth_head(composite.growth_model, df)
        initialize_decline_head(composite.decline_model, df)
        loader = _make_fake_loader()
        return composite, df, loader

    def test_retrain_runs(self):
        composite, df, loader = self._make_composite_and_loader()
        result = retrain_split_decline(
            composite, df, train_loader=loader, val_loader=loader,
            config={"d_model": 8, "n_heads": 4, "n_layers": 1,
                    "d_ff": 32, "head_hidden": 8},
            n_epochs=3, patience=5, verbose=False,
        )
        assert "history" in result
        assert "best_epoch" in result
        # Verify the composite's decline model was replaced with SplitDeclineModel
        assert isinstance(composite.decline_model, SplitDeclineModel)
        feat = composite.decline_model.get_features(torch.randn(2, SEQ_LEN, N_FEATURES))
        assert feat.shape == (2, 8)

    def test_preserves_upstream(self):
        composite, df, loader = self._make_composite_and_loader()
        timing_w = next(composite.timing_model.encoder.parameters()).clone()
        growth_w = next(composite.growth_model.encoder.parameters()).clone()

        retrain_split_decline(
            composite, df, train_loader=loader, val_loader=loader,
            n_epochs=3, patience=5, verbose=False,
        )

        timing_w_after = next(composite.timing_model.encoder.parameters())
        growth_w_after = next(composite.growth_model.encoder.parameters())
        assert torch.allclose(timing_w, timing_w_after), \
            "Timing weights changed during split decline retrain"
        assert torch.allclose(growth_w, growth_w_after), \
            "Growth weights changed during split decline retrain"

    def test_composite_forward_after_retrain(self):
        composite, df, loader = self._make_composite_and_loader()
        retrain_split_decline(
            composite, df, train_loader=loader, val_loader=loader,
            n_epochs=3, patience=5, verbose=False,
        )
        x = torch.randn(2, SEQ_LEN, N_FEATURES)
        t_predict = torch.tensor([100.0, 200.0])
        P_hat = composite(x, t_predict, hard_gate=True)
        assert P_hat.shape == (2,)
