"""Unit tests for feature_config module."""

import pytest

from src.feature_config import (
    INPUT_FEATURES_EXPANDED,
    FEATURE_SIGNAL_TYPE,
    SIGNAL_STEP_WISE,
    SIGNAL_SPIKY_SPARSE,
    SIGNAL_NOISY_TREND,
    SIGNAL_NOISY_SHORT_TREND,
    SIGNAL_TREND_JUMP,
    SIGNAL_SPARSE_OFFLINE,
    STEP_WISE_FEATURES,
    SPIKY_SPARSE_FEATURES,
    NOISY_TREND_FEATURES,
    NOISY_SHORT_TREND_FEATURES,
    TREND_JUMP_FEATURES,
    SPARSE_OFFLINE_FEATURES,
    SMOOTHING_PARAMS,
    get_signal_type,
    should_smooth,
    get_smoothing_params,
)


class TestFeatureLists:
    def test_expanded_features_count(self):
        assert len(INPUT_FEATURES_EXPANDED) == 25

    def test_step_wise_features_count(self):
        assert len(STEP_WISE_FEATURES) == 6

    def test_spiky_sparse_features_count(self):
        assert len(SPIKY_SPARSE_FEATURES) == 3

    def test_noisy_trend_features_count(self):
        assert len(NOISY_TREND_FEATURES) == 5

    def test_noisy_short_trend_features_count(self):
        assert len(NOISY_SHORT_TREND_FEATURES) == 2

    def test_trend_jump_features_count(self):
        assert len(TREND_JUMP_FEATURES) == 5

    def test_sparse_offline_features_count(self):
        assert len(SPARSE_OFFLINE_FEATURES) == 4

    def test_no_duplicate_features(self):
        assert len(INPUT_FEATURES_EXPANDED) == len(set(INPUT_FEATURES_EXPANDED))

    def test_all_features_have_signal_type(self):
        for feat in INPUT_FEATURES_EXPANDED:
            assert feat in FEATURE_SIGNAL_TYPE


class TestGetSignalType:
    def test_step_wise_feature(self):
        assert get_signal_type("Fg") == SIGNAL_STEP_WISE

    def test_spiky_sparse_feature(self):
        assert get_signal_type("Fa") == SIGNAL_SPIKY_SPARSE

    def test_noisy_trend_feature(self):
        assert get_signal_type("Fb") == SIGNAL_NOISY_TREND

    def test_noisy_short_trend_feature(self):
        assert get_signal_type("pH") == SIGNAL_NOISY_SHORT_TREND

    def test_trend_jump_feature(self):
        assert get_signal_type("DO2") == SIGNAL_TREND_JUMP

    def test_sparse_offline_feature(self):
        assert get_signal_type("Viscosity_offline") == SIGNAL_SPARSE_OFFLINE

    def test_unknown_feature(self):
        assert get_signal_type("unknown_feature") == "unknown"


class TestShouldSmooth:
    def test_noisy_trend_should_smooth(self):
        assert should_smooth("Fb") is True
        assert should_smooth("OUR") is True

    def test_noisy_short_trend_should_smooth(self):
        assert should_smooth("pH") is True
        assert should_smooth("T") is True

    def test_step_wise_should_not_smooth(self):
        assert should_smooth("Fg") is False

    def test_trend_jump_should_not_smooth(self):
        assert should_smooth("DO2") is False

    def test_sparse_offline_should_not_smooth(self):
        assert should_smooth("Viscosity_offline") is False


class TestGetSmoothingParams:
    def test_noisy_trend_params(self):
        params = get_smoothing_params("Fb")
        assert params is not None
        assert params["window"] == 15
        assert params["order"] == 2

    def test_noisy_short_trend_params(self):
        params = get_smoothing_params("pH")
        assert params is not None
        assert params["window"] == 7
        assert params["order"] == 2

    def test_non_smoothable_returns_none(self):
        assert get_smoothing_params("Fg") is None
        assert get_smoothing_params("DO2") is None


class TestSmoothingParams:
    def test_smoothing_params_have_required_keys(self):
        for signal_type, params in SMOOTHING_PARAMS.items():
            assert "window" in params
            assert "order" in params
            assert params["window"] > params["order"]
