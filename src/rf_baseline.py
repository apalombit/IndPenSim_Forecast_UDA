"""Random Forest baseline with permutation-based feature importance."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

from .baseline_model import extract_handcrafted_features, prepare_baseline_data
from .feature_config import INPUT_FEATURES_EXPANDED
from .preprocessing import (
    denormalize_target,
    TARGET_MIN,
    TARGET_MAX,
)


class RandomForestBaseline:
    """Random Forest regression baseline with standardization."""

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 3,
        min_samples_leaf: int = 3,
        random_state: int = 42,
        normalize_y: bool = True,
        y_min: float = TARGET_MIN,
        y_max: float = TARGET_MAX,
    ):
        """Initialize Random Forest baseline model.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of each tree.
            random_state: Random seed for reproducibility.
            normalize_y: If True, normalize targets during training.
            y_min: Min value for target normalization.
            y_max: Max value for target normalization.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.normalize_y = normalize_y
        self.y_min = y_min
        self.y_max = y_max
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            max_features='sqrt'
        )
        self.feature_names = None

    def fit(
        self,
        batches: dict[int, pd.DataFrame],
        train_ids: list[int],
        features: list[str] = INPUT_FEATURES_EXPANDED,
        window_fraction: float = 0.25,
    ) -> "RandomForestBaseline":
        """Fit Random Forest model on training data.

        Args:
            batches: Dict of batch DataFrames.
            train_ids: Training batch IDs.
            features: Input feature columns.
            window_fraction: Fraction of batch for early window.

        Returns:
            self
        """
        X, y = prepare_baseline_data(
            batches, train_ids, features, window_fraction,
            normalize_y=self.normalize_y, y_min=self.y_min, y_max=self.y_max
        )

        # Store feature names
        sample_feats = extract_handcrafted_features(
            batches[train_ids[0]], features, window_fraction
        )
        self.feature_names = list(sample_feats.keys())

        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        return self

    def predict(
        self,
        batches: dict[int, pd.DataFrame],
        batch_ids: list[int],
        features: list[str] = INPUT_FEATURES_EXPANDED,
        window_fraction: float = 0.25,
    ) -> np.ndarray:
        """Predict final penicillin concentration.

        Args:
            batches: Dict of batch DataFrames.
            batch_ids: Batch IDs to predict.
            features: Input feature columns.
            window_fraction: Fraction of batch for early window.

        Returns:
            Array of predictions.
        """
        X, _ = prepare_baseline_data(batches, batch_ids, features, window_fraction)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(
        self,
        batches: dict[int, pd.DataFrame],
        batch_ids: list[int],
        features: list[str] = INPUT_FEATURES_EXPANDED,
        window_fraction: float = 0.25,
    ) -> dict:
        """Evaluate model on given batches.

        Args:
            batches: Dict of batch DataFrames.
            batch_ids: Batch IDs to evaluate.
            features: Input feature columns.
            window_fraction: Fraction of batch for early window.

        Returns:
            Dict with MAE, RMSE, and predictions (in original scale).
        """
        X, y_norm = prepare_baseline_data(
            batches, batch_ids, features, window_fraction,
            normalize_y=self.normalize_y, y_min=self.y_min, y_max=self.y_max
        )
        X_scaled = self.scaler.transform(X)
        y_pred_norm = self.model.predict(X_scaled)

        # Denormalize for interpretable metrics
        if self.normalize_y:
            y_true = denormalize_target(y_norm, self.y_min, self.y_max)
            y_pred = denormalize_target(y_pred_norm, self.y_min, self.y_max)
        else:
            y_true = y_norm
            y_pred = y_pred_norm

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        return {
            "mae": mae,
            "rmse": rmse,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def fit_cv(
        self,
        batches: dict[int, pd.DataFrame],
        train_ids: list[int],
        features: list[str] = INPUT_FEATURES_EXPANDED,
        window_fraction: float = 0.25,
        n_folds: int = 5,
        random_state: int | None = None,
    ) -> dict:
        """Fit model with cross-validation.

        Uses K-Fold CV to get out-of-fold predictions and per-fold metrics.

        Args:
            batches: Dict of batch DataFrames.
            train_ids: Training batch IDs.
            features: Input feature columns.
            window_fraction: Fraction of batch for early window.
            n_folds: Number of cross-validation folds.
            random_state: Random seed for fold splitting. If None, uses self.random_state.

        Returns:
            Dict with CV results including per-fold metrics and OOF predictions.
        """
        if random_state is None:
            random_state = self.random_state

        X, y = prepare_baseline_data(
            batches, train_ids, features, window_fraction,
            normalize_y=self.normalize_y, y_min=self.y_min, y_max=self.y_max
        )

        # Store feature names
        sample_feats = extract_handcrafted_features(
            batches[train_ids[0]], features, window_fraction
        )
        self.feature_names = list(sample_feats.keys())

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        fold_metrics = []
        oof_predictions = np.zeros(len(y))
        feature_importances = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit scaler on this fold's training data
            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train)
            X_val_scaled = fold_scaler.transform(X_val)

            # Train RF on this fold
            fold_model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            )
            fold_model.fit(X_train_scaled, y_train)
            y_pred = fold_model.predict(X_val_scaled)

            oof_predictions[val_idx] = y_pred
            feature_importances.append(fold_model.feature_importances_)

            # Denormalize for interpretable metrics
            if self.normalize_y:
                y_val_orig = denormalize_target(y_val, self.y_min, self.y_max)
                y_pred_orig = denormalize_target(y_pred, self.y_min, self.y_max)
            else:
                y_val_orig = y_val
                y_pred_orig = y_pred

            fold_mae = mean_absolute_error(y_val_orig, y_pred_orig)
            fold_rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))

            fold_metrics.append({
                "fold": fold_idx + 1,
                "mae": fold_mae,
                "rmse": fold_rmse,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
            })

        # Compute overall CV metrics (denormalized)
        if self.normalize_y:
            y_orig = denormalize_target(y, self.y_min, self.y_max)
            oof_orig = denormalize_target(oof_predictions, self.y_min, self.y_max)
        else:
            y_orig = y
            oof_orig = oof_predictions

        cv_mae = mean_absolute_error(y_orig, oof_orig)
        cv_rmse = np.sqrt(mean_squared_error(y_orig, oof_orig))

        # Average feature importances across folds
        avg_importances = np.mean(feature_importances, axis=0)
        std_importances = np.std(feature_importances, axis=0)

        # Fit final model on all data
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)

        return {
            "cv_mae": cv_mae,
            "cv_rmse": cv_rmse,
            "fold_metrics": fold_metrics,
            "oof_predictions": oof_orig,
            "y_true": y_orig,
            "batch_ids": train_ids,
            "n_folds": n_folds,
            "feature_importances_mean": avg_importances,
            "feature_importances_std": std_importances,
        }


def compute_permutation_importance(
    model: RandomForestBaseline,
    batches: dict[int, pd.DataFrame],
    batch_ids: list[int],
    features: list[str] = INPUT_FEATURES_EXPANDED,
    window_fraction: float = 0.25,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation-based feature importance.

    Args:
        model: Fitted RandomForestBaseline model.
        batches: Dict of batch DataFrames.
        batch_ids: Batch IDs for importance computation.
        features: Input feature columns.
        window_fraction: Fraction of batch for early window.
        n_repeats: Number of times to permute each feature.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with columns: feature, importance_mean, importance_std.
    """
    X, y = prepare_baseline_data(
        batches, batch_ids, features, window_fraction,
        normalize_y=model.normalize_y, y_min=model.y_min, y_max=model.y_max
    )
    X_scaled = model.scaler.transform(X)

    result = permutation_importance(
        model.model,
        X_scaled,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": model.feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    return importance_df


def get_feature_importance_ranking(importance_df: pd.DataFrame) -> pd.DataFrame:
    """Sort feature importance DataFrame by mean importance (descending).

    Args:
        importance_df: DataFrame from compute_permutation_importance.

    Returns:
        Sorted DataFrame with rank column added.
    """
    sorted_df = importance_df.sort_values("importance_mean", ascending=False).copy()
    sorted_df["rank"] = range(1, len(sorted_df) + 1)
    sorted_df = sorted_df.reset_index(drop=True)
    return sorted_df


def train_and_evaluate_rf_baseline(
    batches: dict[int, pd.DataFrame],
    train_ids: list[int],
    val_ids: list[int],
    target_ids: list[int],
    n_estimators: int = 100,
    max_depth: int = 5,
    window_fraction: float = 0.25,
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> dict:
    """Train Random Forest baseline and evaluate on all splits.

    Args:
        batches: Dict of batch DataFrames.
        train_ids: Training batch IDs (source).
        val_ids: Validation batch IDs (source).
        target_ids: Target domain batch IDs.
        n_estimators: Number of trees.
        min_samples_leaf: Min number of samples in each leaf node.
        max_depth: Maximum tree depth.
        window_fraction: Fraction of batch for early window.
        random_state: Random seed.

    Returns:
        Dict with model and metrics for each split.
    """
    model = RandomForestBaseline(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(batches, train_ids, window_fraction=window_fraction)

    results = {
        "model": model,
        "train": model.evaluate(batches, train_ids, window_fraction=window_fraction),
        "val": model.evaluate(batches, val_ids, window_fraction=window_fraction),
        "target": model.evaluate(batches, target_ids, window_fraction=window_fraction),
    }

    return results
