"""Unit tests for preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    INPUT_FEATURES,
    apply_savgol_smoothing,
    compute_normalization_stats,
    compute_normalization_stats_expanded,
    compute_target_length,
    extract_early_window,
    get_final_target,
    interpolate_sparse_feature,
    normalize_features,
    normalize_features_expanded,
    pad_or_truncate,
    prepare_batch,
    prepare_batch_expanded,
    preprocess_expanded_features,
    robust_scale_stats,
    select_features,
)
from src.feature_config import (
    INPUT_FEATURES_EXPANDED,
    SPARSE_OFFLINE_FEATURES,
    SPIKY_SPARSE_FEATURES,
)
from src.data_loader import load_batches


class TestExtractEarlyWindow:
    def test_extracts_correct_fraction(self):
        df = pd.DataFrame({"time": np.arange(0, 100, 1)})
        early = extract_early_window(df, fraction=0.25)
        assert early["time"].max() <= 25

    def test_preserves_columns(self):
        df = pd.DataFrame({"time": [0, 10, 20], "value": [1, 2, 3]})
        early = extract_early_window(df, fraction=0.5)
        assert "value" in early.columns


class TestGetFinalTarget:
    def test_returns_last_value(self):
        df = pd.DataFrame({"P": [1, 2, 3, 4, 5]})
        assert get_final_target(df, "P") == 5


class TestSelectFeatures:
    def test_selects_available_features(self):
        df = pd.DataFrame({"DO2": [1], "Fs": [2], "other": [3]})
        selected = select_features(df, ["DO2", "Fs", "missing"])
        assert list(selected.columns) == ["DO2", "Fs"]


class TestPadOrTruncate:
    def test_truncates_longer(self):
        arr = np.ones((10, 3))
        result = pad_or_truncate(arr, 5)
        assert result.shape == (5, 3)

    def test_pads_shorter(self):
        arr = np.ones((5, 3))
        result = pad_or_truncate(arr, 10, pad_value=0)
        assert result.shape == (10, 3)
        assert np.all(result[5:] == 0)

    def test_unchanged_if_equal(self):
        arr = np.ones((5, 3))
        result = pad_or_truncate(arr, 5)
        assert result.shape == (5, 3)


class TestNormalizeFeatures:
    def test_normalizes_correctly(self):
        df = pd.DataFrame({"a": [0, 10, 20], "b": [100, 200, 300]})
        stats = {
            "mean": np.array([10, 200]),
            "std": np.array([10, 100]),
            "features": ["a", "b"],
        }
        result = normalize_features(df, stats)
        expected = np.array([[-1, -1], [0, 0], [1, 1]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_handles_zero_std(self):
        df = pd.DataFrame({"a": [5, 5, 5]})
        stats = {"mean": np.array([5]), "std": np.array([0]), "features": ["a"]}
        result = normalize_features(df, stats)
        assert not np.any(np.isnan(result))


@pytest.fixture(scope="module")
def batches():
    try:
        return load_batches()
    except FileNotFoundError:
        pytest.skip("Data file not found")


class TestComputeNormalizationStats:
    def test_returns_expected_keys(self, batches):
        stats = compute_normalization_stats(batches, [1, 2, 3], ["DO2", "Fs"])
        assert "mean" in stats
        assert "std" in stats
        assert "features" in stats

    def test_stats_have_correct_length(self, batches):
        features = ["DO2", "Fs", "T"]
        stats = compute_normalization_stats(batches, [1, 2], features)
        assert len(stats["mean"]) == len(features)
        assert len(stats["std"]) == len(features)


class TestComputeTargetLength:
    def test_returns_positive_int(self, batches):
        length = compute_target_length(batches, [1, 2, 3])
        assert isinstance(length, int)
        assert length > 0


class TestPrepareBatch:
    def test_returns_correct_shapes(self, batches):
        stats = compute_normalization_stats(batches, [1, 2], ["DO2", "Fs"])
        X, y = prepare_batch(batches[1], stats, target_len=100)
        assert X.shape == (100, 2)
        assert isinstance(y, float)


# =============================================================================
# Tests for Expanded Feature Preprocessing
# =============================================================================


class TestInterpolateSparseFeature:
    def test_handles_interior_nans(self):
        series = pd.Series([1.0, np.nan, np.nan, 4.0, 5.0])
        result = interpolate_sparse_feature(series)
        assert not result.isna().any()
        assert result.iloc[1] == 2.0  # Linear interpolation
        assert result.iloc[2] == 3.0

    def test_handles_leading_nans(self):
        series = pd.Series([np.nan, np.nan, 3.0, 4.0, 5.0])
        result = interpolate_sparse_feature(series)
        assert not result.isna().any()
        assert result.iloc[0] == 3.0  # Backward-filled
        assert result.iloc[1] == 3.0

    def test_handles_trailing_nans(self):
        series = pd.Series([1.0, 2.0, 3.0, np.nan, np.nan])
        result = interpolate_sparse_feature(series)
        assert not result.isna().any()
        assert result.iloc[3] == 3.0  # Forward-filled
        assert result.iloc[4] == 3.0

    def test_handles_all_nans(self):
        series = pd.Series([np.nan, np.nan, np.nan])
        result = interpolate_sparse_feature(series)
        # All NaN case: returns NaN (can't interpolate)
        assert result.isna().all()

    def test_handles_single_value(self):
        series = pd.Series([np.nan, 5.0, np.nan, np.nan])
        result = interpolate_sparse_feature(series)
        assert not result.isna().any()
        assert all(result == 5.0)

    def test_preserves_non_nan_values(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = interpolate_sparse_feature(series)
        pd.testing.assert_series_equal(result, series)


class TestApplySavgolSmoothing:
    def test_smooths_noisy_signal(self):
        # Create noisy signal
        np.random.seed(42)
        values = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.randn(100) * 0.1
        result = apply_savgol_smoothing(values, window=15, order=2)
        # Smoothed should have smaller variance
        assert result.std() < values.std()
        assert len(result) == len(values)

    def test_handles_short_sequence(self):
        values = np.array([1.0, 2.0, 3.0])
        result = apply_savgol_smoothing(values, window=15, order=2)
        assert len(result) == 3
        # Should return unchanged when too short
        np.testing.assert_array_equal(result, values)

    def test_handles_minimum_length(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = apply_savgol_smoothing(values, window=7, order=2)
        assert len(result) == len(values)

    def test_handles_even_length_adjustment(self):
        # Even-length array shorter than window
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = apply_savgol_smoothing(values, window=15, order=2)
        assert len(result) == len(values)


class TestRobustScaleStats:
    def test_computes_median_iqr(self):
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        median, iqr = robust_scale_stats(values)
        assert median == 5.0
        assert iqr == 4.0  # Q3=7, Q1=3

    def test_handles_zero_inflated(self):
        # Many zeros with some spikes (common for sparse signals)
        values = np.array([0, 0, 0, 0, 0, 0, 0, 10, 0, 0])
        median, iqr = robust_scale_stats(values)
        assert median == 0.0
        # IQR should default to 1.0 when zero
        assert iqr == 1.0

    def test_handles_constant_values(self):
        values = np.array([5, 5, 5, 5, 5])
        median, iqr = robust_scale_stats(values)
        assert median == 5.0
        assert iqr == 1.0  # Default when IQR is zero


class TestPreprocessExpandedFeatures:
    def test_interpolates_sparse_features(self):
        df = pd.DataFrame({
            "time": [0, 1, 2, 3, 4],
            "PAA_offline": [1.0, np.nan, np.nan, 4.0, np.nan],
            "DO2": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = preprocess_expanded_features(df, ["PAA_offline", "DO2"])
        assert not result["PAA_offline"].isna().any()
        # DO2 should be unchanged (not sparse)
        pd.testing.assert_series_equal(result["DO2"], df["DO2"])

    def test_applies_smoothing_when_enabled(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "time": np.arange(100),
            "Fb": np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.randn(100) * 0.3,
        })
        result_smoothed = preprocess_expanded_features(df, ["Fb"], apply_smoothing=True)
        result_raw = preprocess_expanded_features(df, ["Fb"], apply_smoothing=False)
        # Smoothed should have smaller variance
        assert result_smoothed["Fb"].std() < result_raw["Fb"].std()

    def test_preserves_step_wise_features(self):
        df = pd.DataFrame({
            "time": [0, 1, 2, 3, 4],
            "Fg": [1.0, 1.0, 5.0, 5.0, 5.0],  # Step change
        })
        result = preprocess_expanded_features(df, ["Fg"], apply_smoothing=True)
        # Step-wise features should NOT be smoothed
        pd.testing.assert_series_equal(result["Fg"], df["Fg"])

    def test_handles_missing_features(self):
        df = pd.DataFrame({
            "time": [0, 1, 2],
            "DO2": [1.0, 2.0, 3.0],
        })
        # Request a feature that doesn't exist
        result = preprocess_expanded_features(df, ["DO2", "missing_feature"])
        assert "DO2" in result.columns
        assert "missing_feature" not in result.columns

    def test_includes_time_column(self):
        df = pd.DataFrame({
            "time": [0, 1, 2],
            "DO2": [1.0, 2.0, 3.0],
        })
        result = preprocess_expanded_features(df, ["DO2"])
        assert "time" in result.columns


class TestComputeNormalizationStatsExpanded:
    def test_returns_expected_keys(self, batches):
        stats = compute_normalization_stats_expanded(
            batches, [1, 2, 3], ["DO2", "Fs", "Fa"]
        )
        assert "features" in stats
        assert "signal_types" in stats
        assert "scaling" in stats
        assert "y_min" in stats
        assert "y_max" in stats

    def test_uses_robust_scaling_for_spiky_features(self, batches):
        stats = compute_normalization_stats_expanded(
            batches, [1, 2], ["Fa"]  # Spiky sparse feature
        )
        assert stats["scaling"]["Fa"]["method"] == "robust"
        assert "median" in stats["scaling"]["Fa"]
        assert "iqr" in stats["scaling"]["Fa"]

    def test_uses_zscore_for_trend_features(self, batches):
        stats = compute_normalization_stats_expanded(
            batches, [1, 2], ["DO2"]  # Trend with jumps feature
        )
        assert stats["scaling"]["DO2"]["method"] == "zscore"
        assert "mean" in stats["scaling"]["DO2"]
        assert "std" in stats["scaling"]["DO2"]


class TestNormalizeFeaturesExpanded:
    def test_applies_zscore_scaling(self):
        df = pd.DataFrame({"DO2": [0, 10, 20]})
        stats = {
            "features": ["DO2"],
            "scaling": {"DO2": {"method": "zscore", "mean": 10.0, "std": 10.0}},
        }
        result = normalize_features_expanded(df, stats)
        expected = np.array([[-1], [0], [1]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_applies_robust_scaling(self):
        df = pd.DataFrame({"Fa": [0, 5, 10]})
        stats = {
            "features": ["Fa"],
            "scaling": {"Fa": {"method": "robust", "median": 5.0, "iqr": 5.0}},
        }
        result = normalize_features_expanded(df, stats)
        expected = np.array([[-1], [0], [1]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_handles_mixed_scaling(self):
        df = pd.DataFrame({"DO2": [0, 10, 20], "Fa": [0, 5, 10]})
        stats = {
            "features": ["DO2", "Fa"],
            "scaling": {
                "DO2": {"method": "zscore", "mean": 10.0, "std": 10.0},
                "Fa": {"method": "robust", "median": 5.0, "iqr": 5.0},
            },
        }
        result = normalize_features_expanded(df, stats)
        assert result.shape == (3, 2)
        # Both should be normalized to [-1, 0, 1]
        np.testing.assert_array_almost_equal(result[:, 0], [-1, 0, 1])
        np.testing.assert_array_almost_equal(result[:, 1], [-1, 0, 1])


class TestPrepareBatchExpanded:
    def test_returns_correct_shapes(self, batches):
        stats = compute_normalization_stats_expanded(batches, [1, 2], ["DO2", "Fs"])
        X, y = prepare_batch_expanded(batches[1], stats, target_len=100)
        assert X.shape == (100, 2)
        assert isinstance(y, float)

    def test_no_nans_in_output(self, batches):
        # Use features that include sparse offline
        features = ["DO2", "Fs", "Viscosity_offline"]
        available = [f for f in features if f in batches[1].columns]
        if len(available) >= 2:
            stats = compute_normalization_stats_expanded(batches, [1, 2], available)
            X, y = prepare_batch_expanded(batches[1], stats, target_len=100)
            assert not np.isnan(X).any()
            assert not np.isnan(y)
