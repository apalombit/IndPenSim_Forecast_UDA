"""Baseline model using hand-crafted features and Ridge regression."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

from .feature_config import (
    INPUT_FEATURES_EXPANDED,
    SIGNAL_SPARSE_OFFLINE,
    get_signal_type,
)
from .preprocessing import (
    extract_early_window,
    get_final_target,
    interpolate_sparse_feature,
    normalize_target,
    denormalize_target,
    TARGET_MIN,
    TARGET_MAX,
)


def extract_handcrafted_features(
    df: pd.DataFrame,
    features: list[str] = INPUT_FEATURES_EXPANDED,
    window_fraction: float = 0.25,
) -> dict:
    """Extract hand-crafted statistical features from early window.

    For each input variable, computes: mean, std, min, max, slope.

    Args:
        df: Full batch DataFrame.
        features: Input feature columns.
        window_fraction: Fraction of batch for early window.

    Returns:
        Dict of feature name -> value.
    """
    early = extract_early_window(df, window_fraction)
    result = {}

    for feat in features:
        if feat not in early.columns:
            continue

        values = early[feat]

        # Interpolate sparse offline features to fill NaNs
        if get_signal_type(feat) == SIGNAL_SPARSE_OFFLINE:
            values = interpolate_sparse_feature(values)

        values = values.values
        time = early["time"].values

        # Basic statistics
        result[f"{feat}_mean"] = np.mean(values)
        result[f"{feat}_std"] = np.std(values)
        result[f"{feat}_min"] = np.min(values)
        result[f"{feat}_max"] = np.max(values)

        # Slope (linear regression coefficient)
        if len(values) > 1 and (time[-1] - time[0]) > 0:
            slope = (values[-1] - values[0]) / (time[-1] - time[0])
        else:
            slope = 0.0
        result[f"{feat}_slope"] = slope

    return result


def prepare_baseline_data(
    batches: dict[int, pd.DataFrame],
    batch_ids: list[int],
    features: list[str] = INPUT_FEATURES_EXPANDED,
    window_fraction: float = 0.25,
    normalize_y: bool = False,
    y_min: float = TARGET_MIN,
    y_max: float = TARGET_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare feature matrix and targets for baseline model.

    Args:
        batches: Dict of batch DataFrames.
        batch_ids: List of batch IDs to include.
        features: Input feature columns.
        window_fraction: Fraction of batch for early window.
        normalize_y: If True, normalize targets to [0, 1] range.
        y_min: Min value for target normalization.
        y_max: Max value for target normalization.

    Returns:
        Tuple of (X, y) where X is (n_samples, n_features) and y is (n_samples,).
    """
    X_rows = []
    y_values = []

    for batch_id in batch_ids:
        df = batches[batch_id]
        feats = extract_handcrafted_features(df, features, window_fraction)
        X_rows.append(feats)
        y_raw = get_final_target(df)
        if normalize_y:
            y_values.append(normalize_target(y_raw, y_min, y_max))
        else:
            y_values.append(y_raw)

    X_df = pd.DataFrame(X_rows)
    return X_df.values, np.array(y_values)


class RidgeBaseline:
    """Ridge regression baseline with standardization."""

    def __init__(
        self,
        alpha: float = 1.0,
        normalize_y: bool = True,
        y_min: float = TARGET_MIN,
        y_max: float = TARGET_MAX,
    ):
        """Initialize baseline model.

        Args:
            alpha: Ridge regularization strength.
            normalize_y: If True, normalize targets during training.
            y_min: Min value for target normalization.
            y_max: Max value for target normalization.
        """
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.y_min = y_min
        self.y_max = y_max
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)
        self.feature_names = None

    def fit(
        self,
        batches: dict[int, pd.DataFrame],
        train_ids: list[int],
        features: list[str] = INPUT_FEATURES_EXPANDED,
        window_fraction: float = 0.25,
    ) -> "RidgeBaseline":
        """Fit baseline model on training data.

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
        # Get normalized data for prediction
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
        alphas: list[float] | None = None,
        random_state: int = 42,
    ) -> dict:
        """Fit model with cross-validation and optional alpha tuning.

        Uses K-Fold CV to get out-of-fold predictions and optionally tunes
        the regularization strength (alpha) using RidgeCV.

        Args:
            batches: Dict of batch DataFrames.
            train_ids: Training batch IDs.
            features: Input feature columns.
            window_fraction: Fraction of batch for early window.
            n_folds: Number of cross-validation folds.
            alphas: List of alpha values to try. If None, uses fixed alpha.
            random_state: Random seed for fold splitting.

        Returns:
            Dict with CV results including per-fold metrics and OOF predictions.
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

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        fold_metrics = []
        oof_predictions = np.zeros(len(y))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit scaler on this fold's training data
            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train)
            X_val_scaled = fold_scaler.transform(X_val)

            # Use RidgeCV for alpha tuning if alphas provided
            if alphas is not None:
                fold_model = RidgeCV(alphas=alphas)
            else:
                fold_model = Ridge(alpha=self.alpha)

            fold_model.fit(X_train_scaled, y_train)
            y_pred = fold_model.predict(X_val_scaled)

            oof_predictions[val_idx] = y_pred

            # Denormalize for interpretable metrics
            if self.normalize_y:
                y_val_orig = denormalize_target(y_val, self.y_min, self.y_max)
                y_pred_orig = denormalize_target(y_pred, self.y_min, self.y_max)
            else:
                y_val_orig = y_val
                y_pred_orig = y_pred

            fold_mae = mean_absolute_error(y_val_orig, y_pred_orig)
            fold_rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))

            fold_result = {
                "fold": fold_idx + 1,
                "mae": fold_mae,
                "rmse": fold_rmse,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
            }
            if alphas is not None:
                fold_result["best_alpha"] = fold_model.alpha_

            fold_metrics.append(fold_result)

        # Compute overall CV metrics (denormalized)
        if self.normalize_y:
            y_orig = denormalize_target(y, self.y_min, self.y_max)
            oof_orig = denormalize_target(oof_predictions, self.y_min, self.y_max)
        else:
            y_orig = y
            oof_orig = oof_predictions

        cv_mae = mean_absolute_error(y_orig, oof_orig)
        cv_rmse = np.sqrt(mean_squared_error(y_orig, oof_orig))

        # Fit final model on all data
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        if alphas is not None:
            # Use RidgeCV to find best alpha on full data
            final_model = RidgeCV(alphas=alphas)
            final_model.fit(X_scaled, y)
            self.alpha = final_model.alpha_
            self.model = Ridge(alpha=self.alpha)
            self.model.fit(X_scaled, y)
        else:
            self.model.fit(X_scaled, y)

        return {
            "cv_mae": cv_mae,
            "cv_rmse": cv_rmse,
            "fold_metrics": fold_metrics,
            "oof_predictions": oof_orig,
            "y_true": y_orig,
            "batch_ids": train_ids,
            "n_folds": n_folds,
            "best_alpha": self.alpha if alphas is not None else None,
        }


def train_and_evaluate_baseline(
    batches: dict[int, pd.DataFrame],
    train_ids: list[int],
    val_ids: list[int],
    target_ids: list[int],
    alpha: float = 1.0,
    window_fraction: float = 0.25,
    features: list[str] = INPUT_FEATURES_EXPANDED,
) -> dict:
    """Train baseline and evaluate on all splits.

    Args:
        batches: Dict of batch DataFrames.
        train_ids: Training batch IDs (source).
        val_ids: Validation batch IDs (source).
        target_ids: Target domain batch IDs.
        alpha: Ridge regularization strength.
        window_fraction: Fraction of batch for early window.
        features: Input feature columns.

    Returns:
        Dict with model and metrics for each split.
    """
    model = RidgeBaseline(alpha=alpha)
    model.fit(batches, train_ids, features=features, window_fraction=window_fraction)

    results = {
        "model": model,
        "train": model.evaluate(batches, train_ids, features=features, window_fraction=window_fraction),
        "val": model.evaluate(batches, val_ids, features=features, window_fraction=window_fraction),
        "target": model.evaluate(batches, target_ids, features=features, window_fraction=window_fraction),
    }

    return results
