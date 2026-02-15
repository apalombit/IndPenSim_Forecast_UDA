"""Unit tests for rf_baseline module."""

import numpy as np
import pandas as pd
import pytest

from src.rf_baseline import (
    RandomForestBaseline,
    compute_permutation_importance,
    get_feature_importance_ranking,
    train_and_evaluate_rf_baseline,
)
from src.data_loader import load_batches


@pytest.fixture(scope="module")
def batches():
    try:
        return load_batches()
    except FileNotFoundError:
        pytest.skip("Data file not found")


class TestRandomForestBaseline:
    def test_fit_and_predict(self, batches):
        model = RandomForestBaseline(n_estimators=10, max_depth=3)
        train_ids = [1, 2, 3, 4, 5]
        model.fit(batches, train_ids, features=["DO2", "Fs"])

        preds = model.predict(batches, [6, 7], features=["DO2", "Fs"])
        assert preds.shape == (2,)
        assert all(p > 0 for p in preds)  # Penicillin should be positive

    def test_evaluate_returns_metrics(self, batches):
        model = RandomForestBaseline(n_estimators=10, max_depth=3)
        train_ids = list(range(1, 11))
        model.fit(batches, train_ids)

        result = model.evaluate(batches, [11, 12])
        assert "mae" in result
        assert "rmse" in result
        assert "y_true" in result
        assert "y_pred" in result
        assert result["mae"] >= 0
        assert result["rmse"] >= 0

    def test_feature_names_stored(self, batches):
        model = RandomForestBaseline(n_estimators=10, max_depth=3)
        train_ids = [1, 2, 3]
        model.fit(batches, train_ids, features=["DO2", "Fs"])

        assert model.feature_names is not None
        assert len(model.feature_names) == 10  # 5 stats * 2 features


class TestPermutationImportance:
    def test_returns_correct_shape(self, batches):
        model = RandomForestBaseline(n_estimators=10, max_depth=3)
        train_ids = list(range(1, 11))
        model.fit(batches, train_ids, features=["DO2", "Fs"])

        importance_df = compute_permutation_importance(
            model, batches, train_ids, features=["DO2", "Fs"], n_repeats=5
        )

        assert len(importance_df) == 10  # 5 stats * 2 features
        assert "feature" in importance_df.columns
        assert "importance_mean" in importance_df.columns
        assert "importance_std" in importance_df.columns

    def test_importance_values_are_numeric(self, batches):
        model = RandomForestBaseline(n_estimators=10, max_depth=3)
        train_ids = list(range(1, 11))
        model.fit(batches, train_ids, features=["DO2", "Fs"])

        importance_df = compute_permutation_importance(
            model, batches, train_ids, features=["DO2", "Fs"], n_repeats=5
        )

        assert importance_df["importance_mean"].dtype == np.float64
        assert importance_df["importance_std"].dtype == np.float64


class TestFeatureImportanceRanking:
    def test_sorts_by_importance(self):
        importance_df = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "importance_mean": [0.1, 0.5, 0.3],
            "importance_std": [0.01, 0.02, 0.015],
        })

        ranked = get_feature_importance_ranking(importance_df)

        assert ranked.iloc[0]["feature"] == "b"
        assert ranked.iloc[1]["feature"] == "c"
        assert ranked.iloc[2]["feature"] == "a"

    def test_adds_rank_column(self):
        importance_df = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "importance_mean": [0.1, 0.5, 0.3],
            "importance_std": [0.01, 0.02, 0.015],
        })

        ranked = get_feature_importance_ranking(importance_df)

        assert "rank" in ranked.columns
        assert list(ranked["rank"]) == [1, 2, 3]


class TestTrainAndEvaluateRFBaseline:
    def test_returns_expected_keys(self, batches):
        result = train_and_evaluate_rf_baseline(
            batches,
            train_ids=list(range(1, 21)),
            val_ids=list(range(21, 26)),
            target_ids=list(range(61, 71)),
            n_estimators=10,
            max_depth=3,
        )
        assert "model" in result
        assert "train" in result
        assert "val" in result
        assert "target" in result

    def test_metrics_are_valid(self, batches):
        result = train_and_evaluate_rf_baseline(
            batches,
            train_ids=list(range(1, 21)),
            val_ids=list(range(21, 26)),
            target_ids=list(range(61, 71)),
            n_estimators=10,
            max_depth=3,
        )
        for split in ["train", "val", "target"]:
            assert result[split]["mae"] >= 0
            assert result[split]["rmse"] >= 0
            assert result[split]["rmse"] >= result[split]["mae"]  # RMSE >= MAE always
