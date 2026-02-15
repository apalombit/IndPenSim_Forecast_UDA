"""Unit tests for baseline_model module."""

import numpy as np
import pandas as pd
import pytest

from src.baseline_model import (
    RidgeBaseline,
    extract_handcrafted_features,
    prepare_baseline_data,
    train_and_evaluate_baseline,
)
from src.data_loader import load_batches

# Features without NaN values for reliable testing
TEST_FEATURES = ["DO2", "Fs", "Fa", "Fb", "Fg", "T", "pH"]


class TestExtractHandcraftedFeatures:
    def test_returns_expected_keys(self):
        df = pd.DataFrame({
            "time": [0, 1, 2, 3, 4],
            "DO2": [10, 12, 11, 13, 14],
            "Fs": [5, 6, 7, 8, 9],
        })
        feats = extract_handcrafted_features(df, features=["DO2", "Fs"], window_fraction=1.0)

        expected_suffixes = ["_mean", "_std", "_min", "_max", "_slope"]
        for feat in ["DO2", "Fs"]:
            for suffix in expected_suffixes:
                assert f"{feat}{suffix}" in feats

    def test_computes_correct_values(self):
        df = pd.DataFrame({
            "time": [0, 1, 2, 3, 4],
            "DO2": [10, 10, 10, 10, 10],  # constant
            "Fs": [0, 1, 2, 3, 4],  # linear increase
        })
        feats = extract_handcrafted_features(df, features=["DO2", "Fs"], window_fraction=1.0)

        assert feats["DO2_mean"] == 10.0
        assert feats["DO2_std"] == 0.0
        assert feats["DO2_slope"] == 0.0
        assert feats["Fs_slope"] == pytest.approx(1.0)


@pytest.fixture(scope="module")
def batches():
    try:
        return load_batches()
    except FileNotFoundError:
        pytest.skip("Data file not found")


class TestPrepareBaselineData:
    def test_returns_correct_shapes(self, batches):
        X, y = prepare_baseline_data(batches, [1, 2, 3], features=["DO2", "Fs"])
        assert X.shape[0] == 3  # 3 batches
        assert X.shape[1] == 10  # 5 stats * 2 features
        assert y.shape == (3,)


class TestRidgeBaseline:
    def test_fit_and_predict(self, batches):
        model = RidgeBaseline(alpha=1.0)
        train_ids = [1, 2, 3, 4, 5]
        model.fit(batches, train_ids, features=["DO2", "Fs"])

        preds = model.predict(batches, [6, 7], features=["DO2", "Fs"])
        assert preds.shape == (2,)
        assert all(p > 0 for p in preds)  # Penicillin should be positive

    def test_evaluate_returns_metrics(self, batches):
        model = RidgeBaseline(alpha=1.0)
        train_ids = list(range(1, 11))
        model.fit(batches, train_ids, features=TEST_FEATURES)

        result = model.evaluate(batches, [11, 12], features=TEST_FEATURES)
        assert "mae" in result
        assert "rmse" in result
        assert "y_true" in result
        assert "y_pred" in result
        assert result["mae"] >= 0
        assert result["rmse"] >= 0


class TestTrainAndEvaluateBaseline:
    def test_returns_expected_keys(self, batches):
        result = train_and_evaluate_baseline(
            batches,
            train_ids=list(range(1, 21)),
            val_ids=list(range(21, 26)),
            target_ids=list(range(61, 71)),
            features=TEST_FEATURES,
        )
        assert "model" in result
        assert "train" in result
        assert "val" in result
        assert "target" in result

    def test_metrics_are_valid(self, batches):
        result = train_and_evaluate_baseline(
            batches,
            train_ids=list(range(1, 21)),
            val_ids=list(range(21, 26)),
            target_ids=list(range(61, 71)),
            features=TEST_FEATURES,
        )
        for split in ["train", "val", "target"]:
            assert result[split]["mae"] >= 0
            assert result[split]["rmse"] >= 0
            assert result[split]["rmse"] >= result[split]["mae"]  # RMSE >= MAE always
