"""Tests for PieceLog-PatchTST training with param z-score normalization."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytest

from src.piecelog_model import PARAM_NAMES
from src.piecelog_patchtst import create_piecelog_model
from src.piecelog_dataset import compute_param_stats
from src.train_piecelog import (
    piecelog_loss,
    make_stepwise_alpha_schedule,
    get_alpha_for_epoch,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    """Small PieceLog-PatchTST for testing."""
    return create_piecelog_model(
        n_features=28, seq_len=32, T_max=300.0,
        config={"d_model": 16, "n_heads": 2, "n_layers": 1,
                "d_ff": 32, "patch_len": 8, "patch_stride": 4,
                "head_hidden": 16},
    )


@pytest.fixture
def batch(model):
    """Synthetic batch with realistic param scales."""
    B = 4
    x = torch.randn(B, 32, 28)
    t_predict = torch.tensor([100.0, 150.0, 200.0, 250.0])
    y_true = torch.tensor([10.0, 20.0, 30.0, 25.0])
    params_true = torch.tensor([
        [37.0, 0.021, 38.0, 0.023, 25.8, 196.0, 0.048],
        [40.0, 0.015, 50.0, 0.010, 24.0, 210.0, 0.100],
        [30.0, 0.030, 10.0, 0.050, 27.0, 180.0, 0.000],
        [45.0, 0.020, 70.0, 0.012, 26.0, 200.0, 0.060],
    ])
    return x, t_predict, y_true, params_true


@pytest.fixture
def param_stats():
    """Realistic param_stats matching the fitted-param table in CLAUDE.md."""
    return {
        "K":       {"mean": 37.06, "std": 9.02},
        "r":       {"mean": 0.021, "std": 0.009},
        "t0":      {"mean": 37.89, "std": 40.99},
        "lam":     {"mean": 0.023, "std": 0.023},
        "t_lag":   {"mean": 25.83, "std": 1.69},
        "t_break": {"mean": 196.48, "std": 30.28},
        "slope":   {"mean": 0.048, "std": 0.064},
    }


# ---------------------------------------------------------------------------
# Tests for compute_param_stats
# ---------------------------------------------------------------------------

class TestComputeParamStats:
    def _make_df(self):
        return pd.DataFrame([
            {"batch_id": 1, "K": 30.0, "r": 0.01, "t0": 10.0, "lam": 0.01,
             "t_lag": 24.0, "t_break": 180.0, "slope": 0.05},
            {"batch_id": 2, "K": 40.0, "r": 0.03, "t0": 60.0, "lam": 0.03,
             "t_lag": 26.0, "t_break": 220.0, "slope": 0.10},
            {"batch_id": 3, "K": 50.0, "r": 0.02, "t0": 40.0, "lam": 0.02,
             "t_lag": 28.0, "t_break": 190.0, "slope": 0.00},
        ])

    def test_returns_all_params(self):
        stats = compute_param_stats(self._make_df())
        assert set(stats.keys()) == set(PARAM_NAMES)

    def test_has_mean_and_std(self):
        stats = compute_param_stats(self._make_df())
        for name in PARAM_NAMES:
            assert "mean" in stats[name]
            assert "std" in stats[name]

    def test_mean_correct(self):
        stats = compute_param_stats(self._make_df())
        assert abs(stats["K"]["mean"] - 40.0) < 1e-6

    def test_filter_by_batch_ids(self):
        stats = compute_param_stats(self._make_df(), batch_ids=[1, 2])
        assert abs(stats["K"]["mean"] - 35.0) < 1e-6

    def test_std_clamped_to_min(self):
        # All identical values → std would be 0, should be clamped to 1e-6
        df = pd.DataFrame([
            {"batch_id": 1, "K": 30.0, "r": 0.01, "t0": 10.0, "lam": 0.01,
             "t_lag": 24.0, "t_break": 180.0, "slope": 0.05},
            {"batch_id": 2, "K": 30.0, "r": 0.01, "t0": 10.0, "lam": 0.01,
             "t_lag": 24.0, "t_break": 180.0, "slope": 0.05},
        ])
        stats = compute_param_stats(df)
        for name in PARAM_NAMES:
            assert stats[name]["std"] >= 1e-6


# ---------------------------------------------------------------------------
# Tests for piecelog_loss with param_stats
# ---------------------------------------------------------------------------

class TestPiecelogLossNormalization:
    def test_none_preserves_old_behavior(self, model, batch):
        """param_stats=None should give same result as before."""
        x, t_predict, y_true, params_true = batch
        criterion = nn.MSELoss()

        _, d1 = piecelog_loss(model, x, t_predict, y_true, params_true,
                              criterion, alpha=0.5, param_stats=None)
        # Without param_stats the raw MSE is computed — just verify it runs
        assert "conc" in d1
        assert "param" in d1

    def test_normalized_param_loss_differs(self, model, batch, param_stats):
        """With param_stats, L_param should differ from raw MSE."""
        x, t_predict, y_true, params_true = batch
        criterion = nn.MSELoss()

        _, raw = piecelog_loss(model, x, t_predict, y_true, params_true,
                               criterion, alpha=0.5, param_stats=None)
        _, norm = piecelog_loss(model, x, t_predict, y_true, params_true,
                                criterion, alpha=0.5, param_stats=param_stats)

        assert raw["param"] != pytest.approx(norm["param"], rel=0.01)

    def test_normalized_param_loss_smaller(self, model, batch, param_stats):
        """Z-score normalization should reduce param loss magnitude vs raw."""
        x, t_predict, y_true, params_true = batch
        criterion = nn.MSELoss()

        _, raw = piecelog_loss(model, x, t_predict, y_true, params_true,
                               criterion, alpha=0.5, param_stats=None)
        _, norm = piecelog_loss(model, x, t_predict, y_true, params_true,
                                criterion, alpha=0.5, param_stats=param_stats)

        # Raw param MSE is dominated by K, t0, t_break (~hundreds);
        # normalized should be ~O(1)
        assert norm["param"] < raw["param"]

    def test_backward_works_with_normalization(self, model, batch, param_stats):
        """Gradients should flow through normalized param loss."""
        x, t_predict, y_true, params_true = batch
        criterion = nn.MSELoss()

        loss, _ = piecelog_loss(model, x, t_predict, y_true, params_true,
                                criterion, alpha=0.5, param_stats=param_stats)
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_alpha_zero_skips_param(self, model, batch, param_stats):
        """When alpha=0, param loss should be 0 regardless of param_stats."""
        x, t_predict, y_true, params_true = batch
        criterion = nn.MSELoss()

        _, d = piecelog_loss(model, x, t_predict, y_true, params_true,
                             criterion, alpha=0.0, param_stats=param_stats)
        assert d["param"] == 0.0

    def test_conc_loss_unchanged(self, model, batch, param_stats):
        """Concentration loss should be identical with or without param_stats."""
        x, t_predict, y_true, params_true = batch
        criterion = nn.MSELoss()
        model.eval()  # disable dropout for deterministic comparison

        _, raw = piecelog_loss(model, x, t_predict, y_true, params_true,
                               criterion, alpha=0.5, param_stats=None)
        _, norm = piecelog_loss(model, x, t_predict, y_true, params_true,
                                criterion, alpha=0.5, param_stats=param_stats)

        assert raw["conc"] == pytest.approx(norm["conc"], rel=1e-5)


# ---------------------------------------------------------------------------
# Tests for piecelog_loss with conc_scale
# ---------------------------------------------------------------------------

class TestPiecelogLossConcScale:
    def test_conc_scale_reduces_conc_loss(self, model, batch):
        """With conc_scale, L_conc magnitude should be smaller than raw."""
        x, t_predict, y_true, params_true = batch
        criterion = nn.MSELoss()
        model.eval()

        _, raw = piecelog_loss(model, x, t_predict, y_true, params_true,
                               criterion, alpha=0.0, conc_scale=None)
        _, scaled = piecelog_loss(model, x, t_predict, y_true, params_true,
                                  criterion, alpha=0.0, conc_scale=12.0)

        # Dividing by 12.0 reduces MSE by factor of 144
        assert scaled["conc"] < raw["conc"]

    def test_conc_scale_none_preserves_behavior(self, model, batch):
        """conc_scale=None should give identical result to omitting it."""
        x, t_predict, y_true, params_true = batch
        criterion = nn.MSELoss()
        model.eval()

        _, d1 = piecelog_loss(model, x, t_predict, y_true, params_true,
                              criterion, alpha=0.5)
        _, d2 = piecelog_loss(model, x, t_predict, y_true, params_true,
                              criterion, alpha=0.5, conc_scale=None)

        assert d1["conc"] == pytest.approx(d2["conc"], rel=1e-6)
        assert d1["param"] == pytest.approx(d2["param"], rel=1e-6)

    def test_conc_scale_backward_works(self, model, batch):
        """Gradients should flow correctly with normalized conc loss."""
        x, t_predict, y_true, params_true = batch
        criterion = nn.MSELoss()

        loss, _ = piecelog_loss(model, x, t_predict, y_true, params_true,
                                criterion, alpha=0.5, conc_scale=12.0)
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_conc_scale_exact_factor(self, model, batch):
        """Normalized L_conc should equal raw L_conc / conc_scale^2."""
        x, t_predict, y_true, params_true = batch
        criterion = nn.MSELoss()
        model.eval()

        scale = 12.0
        _, raw = piecelog_loss(model, x, t_predict, y_true, params_true,
                               criterion, alpha=0.0, conc_scale=None)
        _, scaled = piecelog_loss(model, x, t_predict, y_true, params_true,
                                  criterion, alpha=0.0, conc_scale=scale)

        assert scaled["conc"] == pytest.approx(raw["conc"] / scale**2, rel=1e-5)


# ---------------------------------------------------------------------------
# Tests for alpha schedule
# ---------------------------------------------------------------------------

class TestMakeStepwiseAlphaSchedule:
    def test_default_5_steps(self):
        schedule = make_stepwise_alpha_schedule(0.1, n_epochs=100, n_steps=5)
        assert len(schedule) == 5
        assert schedule[0] == (0, 0.1)
        assert schedule[-1] == (80, 0.0)

    def test_epochs_evenly_spaced(self):
        schedule = make_stepwise_alpha_schedule(0.1, n_epochs=100, n_steps=5)
        epochs = [s[0] for s in schedule]
        assert epochs == [0, 20, 40, 60, 80]

    def test_alpha_values_decrease(self):
        schedule = make_stepwise_alpha_schedule(0.1, n_epochs=100, n_steps=5)
        alphas = [s[1] for s in schedule]
        for i in range(len(alphas) - 1):
            assert alphas[i] >= alphas[i + 1]

    def test_alpha_values_match_expected(self):
        schedule = make_stepwise_alpha_schedule(0.1, n_epochs=100, n_steps=5)
        alphas = [s[1] for s in schedule]
        expected = [0.1, 0.075, 0.05, 0.025, 0.0]
        for a, e in zip(alphas, expected):
            assert a == pytest.approx(e, abs=1e-6)

    def test_single_step(self):
        schedule = make_stepwise_alpha_schedule(0.1, n_epochs=100, n_steps=1)
        assert len(schedule) == 1
        assert schedule[0] == (0, 0.1)

    def test_two_steps(self):
        schedule = make_stepwise_alpha_schedule(0.2, n_epochs=50, n_steps=2)
        assert schedule == [(0, 0.2), (25, 0.0)]


class TestGetAlphaForEpoch:
    def test_first_phase(self):
        schedule = [(0, 0.1), (20, 0.075), (40, 0.05), (60, 0.025), (80, 0.0)]
        assert get_alpha_for_epoch(0, schedule) == 0.1
        assert get_alpha_for_epoch(19, schedule) == 0.1

    def test_middle_phase(self):
        schedule = [(0, 0.1), (20, 0.075), (40, 0.05), (60, 0.025), (80, 0.0)]
        assert get_alpha_for_epoch(20, schedule) == 0.075
        assert get_alpha_for_epoch(39, schedule) == 0.075

    def test_last_phase(self):
        schedule = [(0, 0.1), (20, 0.075), (40, 0.05), (60, 0.025), (80, 0.0)]
        assert get_alpha_for_epoch(80, schedule) == 0.0
        assert get_alpha_for_epoch(99, schedule) == 0.0

    def test_boundary_exact(self):
        schedule = [(0, 0.1), (50, 0.0)]
        assert get_alpha_for_epoch(49, schedule) == 0.1
        assert get_alpha_for_epoch(50, schedule) == 0.0
