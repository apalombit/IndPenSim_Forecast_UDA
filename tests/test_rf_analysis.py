"""Unit tests for rf_analysis module."""

import numpy as np
import pandas as pd
import pytest

from src.rf_analysis import (
    compute_partial_dependence,
    extract_proximity_matrix,
    compute_prediction_uncertainty,
    cluster_from_proximity,
    compare_cluster_assignments,
    get_top_features_by_importance,
)
from src.rf_baseline import RandomForestBaseline
from src.data_loader import load_batches


@pytest.fixture(scope="module")
def batches():
    try:
        return load_batches()
    except FileNotFoundError:
        pytest.skip("Data file not found")


@pytest.fixture(scope="module")
def fitted_model(batches):
    model = RandomForestBaseline(n_estimators=20, max_depth=5)
    train_ids = list(range(1, 21))
    model.fit(batches, train_ids, features=["DO2", "Fs"])
    return model


class TestProximityMatrix:
    def test_shape_is_square(self, batches, fitted_model):
        batch_ids = [1, 2, 3, 4, 5]
        proximity, ids = extract_proximity_matrix(
            fitted_model, batches, batch_ids, features=["DO2", "Fs"]
        )
        assert proximity.shape == (5, 5)
        assert ids == batch_ids

    def test_is_symmetric(self, batches, fitted_model):
        batch_ids = list(range(1, 11))
        proximity, _ = extract_proximity_matrix(
            fitted_model, batches, batch_ids, features=["DO2", "Fs"]
        )
        np.testing.assert_array_almost_equal(proximity, proximity.T)

    def test_diagonal_is_one(self, batches, fitted_model):
        batch_ids = list(range(1, 11))
        proximity, _ = extract_proximity_matrix(
            fitted_model, batches, batch_ids, features=["DO2", "Fs"]
        )
        np.testing.assert_array_almost_equal(np.diag(proximity), np.ones(10))

    def test_values_between_zero_and_one(self, batches, fitted_model):
        batch_ids = list(range(1, 11))
        proximity, _ = extract_proximity_matrix(
            fitted_model, batches, batch_ids, features=["DO2", "Fs"]
        )
        assert np.all(proximity >= 0)
        assert np.all(proximity <= 1)


class TestPredictionUncertainty:
    def test_returns_correct_shapes(self, batches, fitted_model):
        batch_ids = [1, 2, 3, 4, 5]
        std, mean, ids = compute_prediction_uncertainty(
            fitted_model, batches, batch_ids, features=["DO2", "Fs"]
        )
        assert std.shape == (5,)
        assert mean.shape == (5,)
        assert ids == batch_ids

    def test_uncertainty_is_non_negative(self, batches, fitted_model):
        batch_ids = list(range(1, 11))
        std, _, _ = compute_prediction_uncertainty(
            fitted_model, batches, batch_ids, features=["DO2", "Fs"]
        )
        assert np.all(std >= 0)

    def test_mean_matches_model_predict(self, batches, fitted_model):
        batch_ids = [1, 2, 3, 4, 5]
        _, mean, _ = compute_prediction_uncertainty(
            fitted_model, batches, batch_ids, features=["DO2", "Fs"]
        )
        preds = fitted_model.predict(batches, batch_ids, features=["DO2", "Fs"])
        np.testing.assert_array_almost_equal(mean, preds)


class TestClusterFromProximity:
    def test_returns_correct_length(self):
        proximity = np.array([
            [1.0, 0.8, 0.2, 0.1],
            [0.8, 1.0, 0.3, 0.2],
            [0.2, 0.3, 1.0, 0.9],
            [0.1, 0.2, 0.9, 1.0],
        ])
        labels = cluster_from_proximity(proximity, n_clusters=2)
        assert len(labels) == 4

    def test_assigns_correct_number_of_clusters(self):
        proximity = np.array([
            [1.0, 0.8, 0.2, 0.1],
            [0.8, 1.0, 0.3, 0.2],
            [0.2, 0.3, 1.0, 0.9],
            [0.1, 0.2, 0.9, 1.0],
        ])
        labels = cluster_from_proximity(proximity, n_clusters=2)
        assert len(np.unique(labels)) == 2


class TestCompareClusterAssignments:
    def test_identical_clusterings_return_one(self):
        labels1 = np.array([0, 0, 1, 1, 0, 1])
        labels2 = np.array([0, 0, 1, 1, 0, 1])
        ari = compare_cluster_assignments(labels1, labels2)
        assert ari == 1.0

    def test_permuted_labels_return_one(self):
        labels1 = np.array([0, 0, 1, 1, 0, 1])
        labels2 = np.array([1, 1, 0, 0, 1, 0])  # Same clustering, different labels
        ari = compare_cluster_assignments(labels1, labels2)
        assert ari == 1.0

    def test_returns_valid_ari_range(self):
        labels1 = np.array([0, 0, 0, 1, 1, 1])
        labels2 = np.array([0, 1, 0, 1, 0, 1])
        ari = compare_cluster_assignments(labels1, labels2)
        # ARI is in [-0.5, 1] for random to identical
        assert -0.5 <= ari <= 1.0


class TestPartialDependence:
    def test_returns_results_for_each_feature(self, batches, fitted_model):
        batch_ids = list(range(1, 11))
        feature_indices = [0, 1]  # First two features
        results = compute_partial_dependence(
            fitted_model, batches, batch_ids, feature_indices,
            features=["DO2", "Fs"], grid_resolution=20
        )
        assert len(results) == 2
        for feat_name, data in results.items():
            assert "grid" in data
            assert "pd_values" in data
            assert len(data["grid"]) == 20
            assert len(data["pd_values"]) == 20


class TestGetTopFeaturesByImportance:
    def test_returns_correct_number(self):
        importance_df = pd.DataFrame({
            "feature": ["a", "b", "c", "d", "e"],
            "importance_mean": [0.1, 0.5, 0.3, 0.2, 0.4],
        })
        indices = get_top_features_by_importance(importance_df, n_top=3)
        assert len(indices) == 3

    def test_returns_highest_importance_features(self):
        importance_df = pd.DataFrame({
            "feature": ["a", "b", "c", "d", "e"],
            "importance_mean": [0.1, 0.5, 0.3, 0.2, 0.4],
        })
        indices = get_top_features_by_importance(importance_df, n_top=3)
        # b (0.5), e (0.4), c (0.3) should be top 3
        assert indices == [1, 4, 2]
