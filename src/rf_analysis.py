"""Extended Random Forest analysis: partial dependence, proximity, uncertainty."""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

from .baseline_model import prepare_baseline_data
from .preprocessing import INPUT_FEATURES
from .rf_baseline import RandomForestBaseline


def compute_partial_dependence(
    model: RandomForestBaseline,
    batches: dict[int, pd.DataFrame],
    batch_ids: list[int],
    feature_indices: list[int],
    features: list[str] = INPUT_FEATURES,
    window_fraction: float = 0.25,
    grid_resolution: int = 50,
) -> dict:
    """Compute partial dependence for specified features.

    Args:
        model: Fitted RandomForestBaseline model.
        batches: Dict of batch DataFrames.
        batch_ids: Batch IDs to use for computing PD.
        feature_indices: Indices of features to compute PD for.
        features: Input feature columns.
        window_fraction: Fraction of batch for early window.
        grid_resolution: Number of points in the grid.

    Returns:
        Dict with 'feature_names', 'grid_values', 'pd_values' for each feature.
    """
    X, _ = prepare_baseline_data(batches, batch_ids, features, window_fraction)
    X_scaled = model.scaler.transform(X)

    results = {}
    for idx in feature_indices:
        feat_name = model.feature_names[idx]

        # Create grid over the feature range
        feat_min = X_scaled[:, idx].min()
        feat_max = X_scaled[:, idx].max()
        grid = np.linspace(feat_min, feat_max, grid_resolution)

        # Compute partial dependence
        pd_values = []
        for val in grid:
            X_temp = X_scaled.copy()
            X_temp[:, idx] = val
            preds = model.model.predict(X_temp)
            pd_values.append(preds.mean())

        results[feat_name] = {
            "grid": grid,
            "pd_values": np.array(pd_values),
            "feature_index": idx,
        }

    return results


def extract_proximity_matrix(
    model: RandomForestBaseline,
    batches: dict[int, pd.DataFrame],
    batch_ids: list[int],
    features: list[str] = INPUT_FEATURES,
    window_fraction: float = 0.25,
) -> tuple[np.ndarray, list[int]]:
    """Extract batch-to-batch proximity from RF leaf co-occurrence.

    Proximity[i,j] = (number of trees where i and j end up in same leaf) / n_estimators

    Args:
        model: Fitted RandomForestBaseline model.
        batches: Dict of batch DataFrames.
        batch_ids: Batch IDs to compute proximity for.
        features: Input feature columns.
        window_fraction: Fraction of batch for early window.

    Returns:
        Tuple of (proximity_matrix, batch_ids).
    """
    X, _ = prepare_baseline_data(batches, batch_ids, features, window_fraction)
    X_scaled = model.scaler.transform(X)

    # Get leaf indices for each sample in each tree
    leaf_indices = model.model.apply(X_scaled)  # (n_samples, n_estimators)

    n_samples = X_scaled.shape[0]
    n_estimators = model.model.n_estimators

    # Compute proximity matrix
    proximity = np.zeros((n_samples, n_samples))

    for tree_idx in range(n_estimators):
        leaves = leaf_indices[:, tree_idx]
        for i in range(n_samples):
            for j in range(i, n_samples):
                if leaves[i] == leaves[j]:
                    proximity[i, j] += 1
                    if i != j:
                        proximity[j, i] += 1

    proximity /= n_estimators

    return proximity, batch_ids


def compute_prediction_uncertainty(
    model: RandomForestBaseline,
    batches: dict[int, pd.DataFrame],
    batch_ids: list[int],
    features: list[str] = INPUT_FEATURES,
    window_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Compute per-batch prediction std across trees.

    Args:
        model: Fitted RandomForestBaseline model.
        batches: Dict of batch DataFrames.
        batch_ids: Batch IDs to compute uncertainty for.
        features: Input feature columns.
        window_fraction: Fraction of batch for early window.

    Returns:
        Tuple of (uncertainty_std, prediction_mean, batch_ids).
    """
    X, _ = prepare_baseline_data(batches, batch_ids, features, window_fraction)
    X_scaled = model.scaler.transform(X)

    # Get predictions from each tree
    tree_preds = np.array([
        tree.predict(X_scaled) for tree in model.model.estimators_
    ])  # (n_estimators, n_samples)

    pred_std = tree_preds.std(axis=0)
    pred_mean = tree_preds.mean(axis=0)

    return pred_std, pred_mean, batch_ids


def cluster_from_proximity(
    proximity_matrix: np.ndarray,
    n_clusters: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """Apply k-means to proximity matrix for RF-derived clusters.

    Args:
        proximity_matrix: Square proximity matrix from extract_proximity_matrix.
        n_clusters: Number of clusters.
        random_state: Random seed for k-means.

    Returns:
        Array of cluster labels.
    """
    # Use proximity as a similarity measure, convert to distance for clustering
    # Distance = 1 - proximity
    distance_matrix = 1 - proximity_matrix

    # Use k-means on the distance matrix rows as features
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(distance_matrix)

    return labels


def compare_cluster_assignments(
    labels1: np.ndarray,
    labels2: np.ndarray,
) -> float:
    """Compute adjusted Rand index between two clusterings.

    Args:
        labels1: First clustering labels.
        labels2: Second clustering labels.

    Returns:
        Adjusted Rand index score.
    """
    return adjusted_rand_score(labels1, labels2)


def get_top_features_by_importance(
    importance_df: pd.DataFrame,
    n_top: int = 5,
) -> list[int]:
    """Get indices of top-N features by importance.

    Args:
        importance_df: DataFrame with 'feature' and 'importance_mean' columns.
        n_top: Number of top features to return.

    Returns:
        List of feature indices.
    """
    sorted_df = importance_df.sort_values("importance_mean", ascending=False)
    top_features = sorted_df.head(n_top)["feature"].tolist()

    # Get indices from full feature list
    all_features = importance_df["feature"].tolist()
    indices = [all_features.index(f) for f in top_features]

    return indices
