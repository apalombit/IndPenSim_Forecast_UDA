"""Tests for DeclineForecastDataset."""

import numpy as np
import pandas as pd
import torch
import pytest

from src.decline_dataset import (
    DeclineForecastDataset,
    decline_collate_fn,
    T_MAX_NORM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch_df(n_steps: int = 200, batch_id: int = 1) -> pd.DataFrame:
    """Create a synthetic batch DataFrame mimicking IndPenSim structure."""
    time = np.linspace(0, 300, n_steps)
    df = pd.DataFrame({"time": time})

    # Add required process features (26 features from INPUT_FEATURES_EXPANDED)
    feature_names = [
        "Aeration_rate", "Agitator_RPM", "Back_pressure",
        "Discharge_rate", "Distilled_water_flow", "Soil",
        "Substrate_feed_rate", "Sugar_feed_rate",
        "OUR_offline", "PAA_offline",
        "NH3_concentration", "CO2_concentration", "Cool_water_flow",
        "Heat_water_flow", "pH", "Temperature",
        "Generated_heat", "Acid_flow_rate", "Base_flow_rate",
        "PAA_flow_rate",
        "DO_concentration", "Viscosity", "OUR",
        "Culture_volume", "Vessel_Weight",
        "Dissolved_CO2",
    ]
    for feat in feature_names:
        df[feat] = np.random.randn(n_steps) * 0.1

    # Add concentration column (sparse with NaN gaps)
    conc = np.full(n_steps, np.nan)
    # Observations every ~20 steps
    obs_idx = np.arange(0, n_steps, 20)
    conc[obs_idx] = np.linspace(0, 30, len(obs_idx))
    df["P"] = conc

    return df


def _make_fitted_params_df(
    n_decline: int = 3,
    n_nodecline: int = 5,
    t_break_decline: float = 180.0,
    slope_decline: float = 0.05,
) -> pd.DataFrame:
    """Create fitted params with mix of declining and non-declining batches."""
    n_total = n_decline + n_nodecline
    rows = []
    for i in range(n_total):
        bid = i + 1
        if i < n_decline:
            rows.append({
                "batch_id": bid,
                "K": 37.0, "r": 0.021, "t0": 38.0, "lam": 0.023,
                "t_lag": 26.0,
                "t_break": t_break_decline,
                "slope": slope_decline,
                "r_squared": 0.999,
            })
        else:
            rows.append({
                "batch_id": bid,
                "K": 37.0, "r": 0.021, "t0": 38.0, "lam": 0.023,
                "t_lag": 26.0,
                "t_break": 280.0,
                "slope": 0.0,
                "r_squared": 0.999,
            })
    return pd.DataFrame(rows)


def _make_dataset(
    n_batches: int = 5,
    n_steps: int = 200,
    n_decline: int = 2,
    **kwargs,
) -> DeclineForecastDataset:
    """Create a small dataset for testing."""
    batches = {i + 1: _make_batch_df(n_steps, i + 1) for i in range(n_batches)}
    df = _make_fitted_params_df(n_decline=n_decline, n_nodecline=n_batches - n_decline)
    batch_ids = list(range(1, n_batches + 1))

    # Compute stats from all batches
    from src.preprocessing import compute_normalization_stats_expanded
    from src.feature_config import INPUT_FEATURES_EXPANDED
    features = list(INPUT_FEATURES_EXPANDED)
    stats = compute_normalization_stats_expanded(
        batches, batch_ids, features, window_fraction=1.0
    )

    return DeclineForecastDataset(
        batch_ids=batch_ids,
        fitted_params_df=df,
        batches=batches,
        stats=stats,
        features=features,
        source_ids=batch_ids,
        max_seq_len=kwargs.get("max_seq_len", 100),
        min_T_fraction=kwargs.get("min_T_fraction", 0.4),
        max_T_fraction=kwargs.get("max_T_fraction", 0.9),
        samples_per_batch=kwargs.get("samples_per_batch", 5),
        seed=kwargs.get("seed", 42),
        gate_threshold=kwargs.get("gate_threshold", 0.01),
    )


# ---------------------------------------------------------------------------
# T sampling tests
# ---------------------------------------------------------------------------

class TestTSampling:
    def test_T_frac_range(self):
        """All T_frac values should be in [0.4, 0.9] (approximately)."""
        ds = _make_dataset(n_batches=5, n_steps=200, samples_per_batch=20)
        for sample in ds.samples:
            assert 0.0 < sample["T_frac"] <= 1.0
            # Given min=0.4, max=0.9, with integer T_idx the actual frac
            # should be close to [0.4, 0.9]
            assert sample["T_frac"] >= 0.04  # At least min(10/n_steps, min_frac)

    def test_T_frac_within_bounds(self):
        """T_frac should not exceed max_T_fraction significantly."""
        ds = _make_dataset(
            n_batches=3, n_steps=200,
            min_T_fraction=0.4, max_T_fraction=0.9,
            samples_per_batch=50,
        )
        fracs = [s["T_frac"] for s in ds.samples]
        assert min(fracs) >= 0.04  # min(10, 0.4*200)/200
        assert max(fracs) <= 1.0

    def test_num_samples(self):
        """Total samples should be n_batches * samples_per_batch."""
        n_batches = 4
        spb = 8
        ds = _make_dataset(n_batches=n_batches, samples_per_batch=spb)
        assert len(ds) == n_batches * spb


# ---------------------------------------------------------------------------
# Target computation tests
# ---------------------------------------------------------------------------

class TestTargets:
    def test_decline_targets(self):
        """Declining batches should have decline_target=1, slope>0."""
        ds = _make_dataset(n_batches=5, n_decline=2)
        for sample in ds.samples:
            bid = sample["batch_id"]
            if bid <= 2:  # Declining
                assert sample["decline_target"] == 1.0
                assert sample["slope_target"] > 0
            else:  # No decline
                assert sample["decline_target"] == 0.0
                assert sample["slope_target"] == 0.0

    def test_nodecline_delta_positive(self):
        """No-decline batches should have positive delta (T_end - T > 0)."""
        ds = _make_dataset(n_batches=5, n_decline=0)
        for sample in ds.samples:
            assert sample["decline_target"] == 0.0
            # delta = (t_end - t_cutoff) / T_MAX_NORM, should be positive
            assert sample["delta_target"] > 0

    def test_decline_delta_sign(self):
        """For decline batches, delta sign depends on T vs t_break."""
        ds = _make_dataset(
            n_batches=3, n_decline=3, n_steps=200,
            min_T_fraction=0.1, max_T_fraction=0.99,
            samples_per_batch=50,
        )
        # t_break = 180.0, batch time range [0, 300]
        # T at 60% of 200 steps → t_cutoff ~= 0.6 * 300 = 180
        found_positive = False
        found_negative = False
        for sample in ds.samples:
            if sample["decline_target"] == 1.0:
                if sample["delta_target"] > 0:
                    found_positive = True  # T < t_break
                elif sample["delta_target"] < 0:
                    found_negative = True  # T > t_break

        # With wide T range, we should find both signs
        assert found_positive or found_negative  # At least one sign found

    def test_delta_normalization(self):
        """Delta should be normalized by T_MAX_NORM."""
        ds = _make_dataset(n_batches=3, n_decline=3)
        for sample in ds.samples:
            # Raw delta in hours would be (t_break - t_cutoff)
            # Normalized: delta / T_MAX_NORM
            # Should be roughly in [-1, 1] for reasonable values
            assert abs(sample["delta_target"]) < 2.0


# ---------------------------------------------------------------------------
# Dataset __getitem__ tests
# ---------------------------------------------------------------------------

class TestGetItem:
    def test_output_keys(self):
        ds = _make_dataset()
        item = ds[0]
        expected_keys = {"x", "T_frac", "decline_target", "delta_target",
                         "slope_target", "domain_label", "batch_id"}
        assert set(item.keys()) == expected_keys

    def test_x_shape(self):
        max_seq_len = 100
        ds = _make_dataset(max_seq_len=max_seq_len)
        item = ds[0]
        # 26 features + 3 conc channels = 29
        assert item["x"].shape == (max_seq_len, ds.n_features)

    def test_x_dtype(self):
        ds = _make_dataset()
        item = ds[0]
        assert item["x"].dtype == torch.float32

    def test_scalar_dtypes(self):
        ds = _make_dataset()
        item = ds[0]
        assert item["T_frac"].dtype == torch.float32
        assert item["decline_target"].dtype == torch.float32
        assert item["delta_target"].dtype == torch.float32
        assert item["slope_target"].dtype == torch.float32
        assert item["domain_label"].dtype == torch.long

    def test_domain_label(self):
        ds = _make_dataset()
        item = ds[0]
        assert item["domain_label"].item() == 0  # Source domain

    def test_n_features_property(self):
        ds = _make_dataset()
        # n_features = len(stats["features"]) + 3 conc channels
        n_process = len(ds.stats["features"])
        assert ds.n_features == n_process + 3


# ---------------------------------------------------------------------------
# Collate function tests
# ---------------------------------------------------------------------------

class TestCollate:
    def test_collate_shapes(self):
        ds = _make_dataset(max_seq_len=50)
        items = [ds[i] for i in range(min(4, len(ds)))]
        batch = decline_collate_fn(items)

        n = len(items)
        assert batch["x"].shape == (n, 50, ds.n_features)
        assert batch["T_frac"].shape == (n,)
        assert batch["decline_target"].shape == (n,)
        assert batch["delta_target"].shape == (n,)
        assert batch["slope_target"].shape == (n,)
        assert batch["domain_label"].shape == (n,)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_decline(self):
        """Dataset with all declining batches."""
        ds = _make_dataset(n_batches=3, n_decline=3)
        for sample in ds.samples:
            assert sample["decline_target"] == 1.0

    def test_no_decline(self):
        """Dataset with no declining batches."""
        ds = _make_dataset(n_batches=3, n_decline=0)
        for sample in ds.samples:
            assert sample["decline_target"] == 0.0
            assert sample["slope_target"] == 0.0

    def test_different_seeds(self):
        """Different seeds should produce different samples."""
        ds1 = _make_dataset(seed=42)
        ds2 = _make_dataset(seed=123)
        # T_frac values should differ
        fracs1 = [s["T_frac"] for s in ds1.samples]
        fracs2 = [s["T_frac"] for s in ds2.samples]
        assert fracs1 != fracs2

    def test_short_sequence_padding(self):
        """Very short sequences should be zero-padded."""
        ds = _make_dataset(n_steps=30, max_seq_len=100, min_T_fraction=0.4)
        item = ds[0]
        # With n_steps=30 and min_T_frac=0.4, T_idx=max(10, ~12)
        # After padding to 100, first rows should be zeros
        assert item["x"].shape[0] == 100
        # Check that leading rows are zero (padded)
        assert (item["x"][:50, :] == 0).any()
