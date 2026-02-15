"""Tests for piece-log model functions and fitting."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.piecelog_model import (
    PARAM_NAMES,
    piecelog_numpy,
    piecelog_torch,
    fit_piecelog,
    fit_all_batches,
)


# Reference parameters for testing
REF_PARAMS = {
    "K": 40.0,
    "r": 0.1,
    "t0": 80.0,
    "lam": 0.05,
    "t_lag": 20.0,
    "t_break": 300.0,
    "slope": 0.02,
}


class TestPiecelogNumpy:
    def test_delay_phase_is_zero(self):
        """P(t) = 0 for t < t_lag."""
        t = np.array([0.0, 5.0, 10.0, 19.9])
        result = piecelog_numpy(t, **REF_PARAMS)
        np.testing.assert_array_equal(result, 0.0)

    def test_growth_phase_positive(self):
        """P(t) > 0 for t_lag < t < t_break (well into growth)."""
        t = np.array([50.0, 100.0, 150.0, 200.0])
        result = piecelog_numpy(t, **REF_PARAMS)
        assert (result > 0).all()

    def test_decline_phase(self):
        """P decreases after t_break."""
        t = np.array([310.0, 320.0, 330.0])
        result = piecelog_numpy(t, **REF_PARAMS)
        assert result[0] > result[1] > result[2]

    def test_continuity_at_t_break(self):
        """P is continuous at t_break."""
        eps = 0.001
        t = np.array([REF_PARAMS["t_break"] - eps, REF_PARAMS["t_break"] + eps])
        result = piecelog_numpy(t, **REF_PARAMS)
        np.testing.assert_allclose(result[0], result[1], atol=0.01)

    def test_floor_at_zero(self):
        """P never goes negative in decline."""
        t = np.array([10000.0])  # very far into decline
        result = piecelog_numpy(t, **REF_PARAMS)
        assert result[0] >= 0.0

    def test_output_shape(self):
        t = np.linspace(0, 400, 1000)
        result = piecelog_numpy(t, **REF_PARAMS)
        assert result.shape == (1000,)

    def test_monotonic_growth(self):
        """Growth phase should be monotonically increasing (well past lag)."""
        t = np.linspace(50, 250, 100)
        result = piecelog_numpy(t, **REF_PARAMS)
        diffs = np.diff(result)
        assert (diffs >= -1e-10).all()  # Monotonically non-decreasing


class TestPiecelogTorch:
    def test_matches_numpy(self):
        """Torch and numpy implementations should agree."""
        t_np = np.linspace(0, 400, 200)
        result_np = piecelog_numpy(t_np, **REF_PARAMS)

        t_torch = torch.tensor(t_np, dtype=torch.float32)
        params_torch = {k: torch.tensor(v, dtype=torch.float32) for k, v in REF_PARAMS.items()}
        result_torch = piecelog_torch(t_torch, **params_torch)

        np.testing.assert_allclose(
            result_torch.numpy(), result_np, atol=1e-4, rtol=1e-4
        )

    def test_delay_phase_is_zero(self):
        t = torch.tensor([0.0, 5.0, 10.0, 19.0])
        params = {k: torch.tensor(v) for k, v in REF_PARAMS.items()}
        result = piecelog_torch(t, **params)
        assert torch.allclose(result, torch.zeros(4), atol=1e-6)

    def test_gradient_flow(self):
        """Gradients should flow through the torch implementation."""
        t = torch.tensor([50.0, 150.0, 250.0, 350.0], requires_grad=False)
        K = torch.tensor(40.0, requires_grad=True)
        r = torch.tensor(0.1, requires_grad=True)
        t0 = torch.tensor(80.0, requires_grad=True)
        lam = torch.tensor(0.05, requires_grad=True)
        t_lag = torch.tensor(20.0, requires_grad=True)
        t_break = torch.tensor(300.0, requires_grad=True)
        slope = torch.tensor(0.02, requires_grad=True)

        result = piecelog_torch(t, K, r, t0, lam, t_lag, t_break, slope)
        loss = result.sum()
        loss.backward()

        assert K.grad is not None
        assert r.grad is not None
        assert not torch.isnan(K.grad)

    def test_batch_dimension(self):
        """Should handle batch of different times."""
        B = 8
        t = torch.rand(B) * 400
        params = {k: torch.full((B,), v) for k, v in REF_PARAMS.items()}
        result = piecelog_torch(t, **params)
        assert result.shape == (B,)

    def test_non_negative(self):
        """Output should always be non-negative."""
        t = torch.linspace(0, 500, 200)
        params = {k: torch.full((200,), v) for k, v in REF_PARAMS.items()}
        result = piecelog_torch(t, **params)
        assert (result >= 0).all()


class TestFitPiecelog:
    def test_fits_synthetic_data(self):
        """Should recover known parameters from synthetic data."""
        t = np.linspace(0, 400, 2000)
        y = piecelog_numpy(t, **REF_PARAMS)
        # Add small noise
        rng = np.random.default_rng(42)
        y_noisy = y + rng.normal(0, 0.1, size=len(y))
        y_noisy = np.maximum(y_noisy, 0)

        result = fit_piecelog(t, y_noisy)
        assert result["success"]
        assert result["r_squared"] > 0.99

        # Check parameter recovery (allowing tolerance)
        # t_lag and t0 can trade off, so use looser tolerance for those
        for name in PARAM_NAMES:
            rtol = 0.4 if name in ("t_lag", "t0") else 0.2
            np.testing.assert_allclose(
                result["params"][name], REF_PARAMS[name],
                rtol=rtol,
                err_msg=f"Parameter {name} not recovered",
            )

    def test_handles_nans(self):
        """Should handle NaN values in data."""
        t = np.linspace(0, 400, 500)
        y = piecelog_numpy(t, **REF_PARAMS)
        y[::3] = np.nan  # Every 3rd value is NaN

        result = fit_piecelog(t, y)
        assert result["success"]
        assert result["r_squared"] > 0.95

    def test_returns_failure_for_too_short(self):
        """Should fail gracefully with < 10 points."""
        t = np.array([0, 1, 2])
        y = np.array([0, 0.1, 0.2])
        result = fit_piecelog(t, y)
        assert not result["success"]
        assert result["params"] is None

    def test_result_structure(self):
        t = np.linspace(0, 400, 500)
        y = piecelog_numpy(t, **REF_PARAMS)
        result = fit_piecelog(t, y)
        assert "params" in result
        assert "r_squared" in result
        assert "success" in result
        for name in PARAM_NAMES:
            assert name in result["params"]


class TestFitAllBatches:
    def test_returns_dataframe(self):
        """fit_all_batches should return DataFrame with correct columns."""
        # Create synthetic batches
        batches = {}
        for i in range(1, 4):
            t = np.linspace(0, 400, 500)
            y = piecelog_numpy(t, **REF_PARAMS) + np.random.default_rng(i).normal(0, 0.1, 500)
            y = np.maximum(y, 0)
            batches[i] = pd.DataFrame({"time": t, "P": y})

        result = fit_all_batches(batches, exclude_faults=False)
        assert isinstance(result, pd.DataFrame)
        assert "batch_id" in result.columns
        assert "r_squared" in result.columns
        for name in PARAM_NAMES:
            assert name in result.columns
        assert len(result) == 3

    def test_excludes_faults(self):
        """Batches > 90 should be excluded when exclude_faults=True."""
        t = np.linspace(0, 400, 200)
        y = piecelog_numpy(t, **REF_PARAMS)
        batches = {
            1: pd.DataFrame({"time": t, "P": y}),
            91: pd.DataFrame({"time": t, "P": y}),
        }
        result = fit_all_batches(batches, exclude_faults=True)
        assert len(result) == 1
        assert result.iloc[0]["batch_id"] == 1

    def test_all_r_squared_high(self):
        """All fits on clean data should have high R²."""
        batches = {}
        for i in range(1, 6):
            t = np.linspace(0, 400, 500)
            y = piecelog_numpy(t, **REF_PARAMS)
            batches[i] = pd.DataFrame({"time": t, "P": y})

        result = fit_all_batches(batches, exclude_faults=False)
        assert (result["r_squared"] > 0.99).all()
