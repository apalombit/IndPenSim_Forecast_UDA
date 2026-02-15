"""Preprocessing pipeline for IndPenSim time-series data."""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .feature_config import (
    INPUT_FEATURES_EXPANDED,
    SIGNAL_SPARSE_OFFLINE,
    SIGNAL_SPIKY_SPARSE,
    SPARSE_OFFLINE_FEATURES,
    get_signal_type,
    get_smoothing_params,
    should_smooth,
)

# Input features for modeling (exclude target-leaking variables)
INPUT_FEATURES = [
    "DO2",       # Dissolved oxygen
    "Fs",        # Sugar feed rate
    "Fa",        # Acid flow rate
    "Fb",        # Base flow rate
    "Fg",        # Aeration rate
    "T",         # Temperature
    "pH",        # pH
    "RPM",       # Agitator RPM
    "CO2outgas", # CO2 in off-gas
    "OUR",       # Oxygen uptake rate
    "Fpaa",      # PAA flow
]

# Target variable
TARGET_COLUMN = "P"

# Target normalization range (fixed for stability across experiments)
TARGET_MIN = 0.0
TARGET_MAX = 50.0


def extract_early_window(df: pd.DataFrame, fraction: float = 0.25) -> pd.DataFrame:
    """Extract first fraction of batch by time.

    Args:
        df: Batch DataFrame with 'time' column.
        fraction: Fraction of batch duration to extract.

    Returns:
        DataFrame with early window only.
    """
    max_time = df["time"].max()
    cutoff = max_time * fraction
    return df[df["time"] <= cutoff].copy()


def get_final_target(df: pd.DataFrame, column: str = TARGET_COLUMN) -> float:
    """Get final value of target variable from batch.

    Args:
        df: Full batch DataFrame.
        column: Target column name.

    Returns:
        Final target value.
    """
    return df[column].iloc[-1]


def select_features(df: pd.DataFrame, features: list[str] = INPUT_FEATURES_EXPANDED) -> pd.DataFrame:
    """Select input features from DataFrame.

    Args:
        df: DataFrame with all columns.
        features: List of feature column names.

    Returns:
        DataFrame with only selected features.
    """
    available = [f for f in features if f in df.columns]
    return df[available].copy()


def compute_normalization_stats(
    batches: dict[int, pd.DataFrame],
    source_ids: list[int],
    features: list[str] = INPUT_FEATURES_EXPANDED,
    window_fraction: float = 0.25,
) -> dict:
    """Compute normalization statistics from source domain only.

    Args:
        batches: Dict of batch DataFrames.
        source_ids: List of source domain batch IDs.
        features: Feature columns to normalize.
        window_fraction: Fraction of batch to use for early window.

    Returns:
        Dict with 'mean' and 'std' arrays for each feature.
    """
    # Collect all source early windows
    source_data = []
    for batch_id in source_ids:
        df = batches[batch_id]
        early = extract_early_window(df, window_fraction)
        selected = select_features(early, features)
        source_data.append(selected)

    combined = pd.concat(source_data, ignore_index=True)

    return {
        "mean": combined.mean().values,
        "std": combined.std().values,
        "features": list(combined.columns),
        "y_min": TARGET_MIN,
        "y_max": TARGET_MAX,
    }


def normalize_features(
    df: pd.DataFrame,
    stats: dict,
) -> np.ndarray:
    """Normalize features using precomputed statistics.

    Args:
        df: DataFrame with features.
        stats: Dict with 'mean', 'std', and 'features' keys.

    Returns:
        Normalized numpy array.
    """
    features = stats["features"]
    X = df[features].values
    mean = stats["mean"]
    std = stats["std"]

    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)

    return (X - mean) / std


def pad_or_truncate(arr: np.ndarray, target_len: int, pad_value: float = 0.0) -> np.ndarray:
    """Pad or truncate array to target length.

    Args:
        arr: Input array of shape (T, C).
        target_len: Target sequence length.
        pad_value: Value to use for padding.

    Returns:
        Array of shape (target_len, C).
    """
    current_len = arr.shape[0]

    if current_len == target_len:
        return arr
    elif current_len > target_len:
        # Truncate
        return arr[:target_len]
    else:
        # Pad
        pad_shape = (target_len - current_len, arr.shape[1])
        padding = np.full(pad_shape, pad_value)
        return np.vstack([arr, padding])


def prepare_batch(
    df: pd.DataFrame,
    stats: dict,
    target_len: int,
    window_fraction: float = 0.25,
) -> tuple[np.ndarray, float]:
    """Prepare a single batch for modeling.

    Args:
        df: Full batch DataFrame.
        stats: Normalization statistics.
        target_len: Target sequence length after padding/truncation.
        window_fraction: Fraction of batch for early window.

    Returns:
        Tuple of (X, y) where X is (target_len, n_features) and y is scalar.
    """
    # Extract early window
    early = extract_early_window(df, window_fraction)

    # Select and normalize features
    selected = select_features(early, stats["features"])
    X = normalize_features(selected, stats)

    # Pad or truncate
    X = pad_or_truncate(X, target_len)

    # Get target and normalize
    y_raw = get_final_target(df)
    y = normalize_target(y_raw, stats.get("y_min", TARGET_MIN), stats.get("y_max", TARGET_MAX))

    return X, y


def normalize_target(y: float, y_min: float = TARGET_MIN, y_max: float = TARGET_MAX) -> float:
    """Normalize target to [0, 1] range.

    Args:
        y: Raw target value.
        y_min: Minimum of normalization range.
        y_max: Maximum of normalization range.

    Returns:
        Normalized target in [0, 1].
    """
    return (y - y_min) / (y_max - y_min)


def denormalize_target(y_norm: float, y_min: float = TARGET_MIN, y_max: float = TARGET_MAX) -> float:
    """Denormalize target from [0, 1] to original range.

    Args:
        y_norm: Normalized target value.
        y_min: Minimum of normalization range.
        y_max: Maximum of normalization range.

    Returns:
        Target in original scale.
    """
    return y_norm * (y_max - y_min) + y_min


def compute_target_length(
    batches: dict[int, pd.DataFrame],
    batch_ids: list[int],
    window_fraction: float = 0.25,
    percentile: float = 75,
) -> int:
    """Compute target sequence length based on batch statistics.

    Args:
        batches: Dict of batch DataFrames.
        batch_ids: Batch IDs to consider.
        window_fraction: Fraction of batch for early window.
        percentile: Percentile of lengths to use as target.

    Returns:
        Target sequence length.
    """
    lengths = []
    for batch_id in batch_ids:
        df = batches[batch_id]
        early = extract_early_window(df, window_fraction)
        lengths.append(len(early))

    return int(np.percentile(lengths, percentile))


# =============================================================================
# Expanded Feature Preprocessing Functions
# =============================================================================


def interpolate_sparse_feature(series: pd.Series) -> pd.Series:
    """Robust interpolation for offline measurements.

    Strategy:
    1. Linear interpolation between known values
    2. Backward-fill leading NaNs (before first measurement)
    3. Forward-fill trailing NaNs (after last measurement)

    This ensures no NaNs remain regardless of measurement pattern.

    Args:
        series: Pandas Series with sparse measurements (may contain NaNs).

    Returns:
        Series with all NaNs filled via interpolation.
    """
    result = series.copy()
    # Step 1: Linear interpolation for interior NaNs
    result = result.interpolate(method="linear")
    # Step 2: Handle edges - bfill then ffill
    result = result.bfill().ffill()
    return result


def apply_savgol_smoothing(
    values: np.ndarray, window: int, order: int
) -> np.ndarray:
    """Apply Savitzky-Golay filter with edge handling.

    Automatically adjusts window size if the input is too short.

    Args:
        values: 1D array of values to smooth.
        window: Window size for Savitzky-Golay filter (must be odd).
        order: Polynomial order for filter.

    Returns:
        Smoothed array of same shape as input.
    """
    if len(values) < window:
        # Adjust window to fit available data (must be odd)
        window = len(values) if len(values) % 2 == 1 else len(values) - 1
    if window < order + 2:
        return values  # Too short to smooth
    return savgol_filter(values, window, order)


def robust_scale_stats(values: np.ndarray) -> tuple[float, float]:
    """Compute median and IQR for robust scaling.

    Useful for zero-inflated or sparse signals where standard
    z-score scaling would be biased by many zeros.

    Args:
        values: 1D array of values.

    Returns:
        Tuple of (median, iqr). IQR defaults to 1.0 if zero.
    """
    median = float(np.median(values))
    q75, q25 = np.percentile(values, [75, 25])
    iqr = q75 - q25
    return median, iqr if iqr > 0 else 1.0


def preprocess_expanded_features(
    df: pd.DataFrame,
    features: list[str] | None = None,
    apply_smoothing: bool = False,
) -> pd.DataFrame:
    """Preprocess features with signal-type-aware transformations.

    Applies appropriate preprocessing based on signal type:
    - Sparse offline features: interpolated to fill NaNs
    - Noisy trend features: optionally smoothed with Savitzky-Golay filter

    Args:
        df: Batch DataFrame.
        features: List of feature columns. Defaults to INPUT_FEATURES_EXPANDED.
        apply_smoothing: If True, apply smoothing to noisy signals.

    Returns:
        DataFrame with preprocessed features (includes 'time' column if present).
    """
    if features is None:
        features = INPUT_FEATURES_EXPANDED

    # Start with time column if present
    if "time" in df.columns:
        result = df[["time"]].copy()
    else:
        result = pd.DataFrame(index=df.index)

    for feat in features:
        if feat not in df.columns:
            continue

        values = df[feat].copy()
        signal_type = get_signal_type(feat)

        # Handle sparse offline features
        if signal_type == SIGNAL_SPARSE_OFFLINE:
            values = interpolate_sparse_feature(values)

        # Apply smoothing if enabled and applicable
        if apply_smoothing and should_smooth(feat):
            params = get_smoothing_params(feat)
            if params is not None:
                values = pd.Series(
                    apply_savgol_smoothing(
                        values.values, params["window"], params["order"]
                    ),
                    index=values.index,
                )

        result[feat] = values

    return result


def compute_normalization_stats_expanded(
    batches: dict[int, pd.DataFrame],
    source_ids: list[int],
    features: list[str] | None = None,
    window_fraction: float = 0.25,
    apply_smoothing: bool = False,
) -> dict:
    """Compute normalization stats with signal-type awareness.

    Uses robust scaling (median/IQR) for spiky sparse features,
    standard z-score scaling for all other features.

    Args:
        batches: Dict of batch DataFrames.
        source_ids: List of source domain batch IDs.
        features: Feature columns to normalize. Defaults to INPUT_FEATURES_EXPANDED.
        window_fraction: Fraction of batch for early window.
        apply_smoothing: If True, apply smoothing before computing stats.

    Returns:
        Dict with 'features', 'signal_types', 'scaling' info, and target bounds.
    """
    if features is None:
        features = INPUT_FEATURES_EXPANDED

    source_data = []

    for batch_id in source_ids:
        df = batches[batch_id]
        early = extract_early_window(df, window_fraction)
        processed = preprocess_expanded_features(early, features, apply_smoothing)
        # Only include columns that exist
        available_features = [f for f in features if f in processed.columns]
        source_data.append(processed[available_features])

    combined = pd.concat(source_data, ignore_index=True)

    # Get actual available features
    available_features = [f for f in features if f in combined.columns]

    stats = {
        "features": available_features,
        "signal_types": {},
        "scaling": {},
        "y_min": TARGET_MIN,
        "y_max": TARGET_MAX,
    }

    for feat in available_features:
        signal_type = get_signal_type(feat)
        stats["signal_types"][feat] = signal_type

        values = combined[feat].dropna().values

        if signal_type == SIGNAL_SPIKY_SPARSE:
            # Robust scaling for sparse features
            median, iqr = robust_scale_stats(values)
            stats["scaling"][feat] = {"method": "robust", "median": median, "iqr": iqr}
        else:
            # Standard z-score scaling
            std = float(values.std())
            stats["scaling"][feat] = {
                "method": "zscore",
                "mean": float(values.mean()),
                "std": std if std > 0 else 1.0,
            }

    return stats


def normalize_features_expanded(
    df: pd.DataFrame,
    stats: dict,
) -> np.ndarray:
    """Normalize features using precomputed signal-type-aware statistics.

    Args:
        df: DataFrame with features.
        stats: Dict from compute_normalization_stats_expanded.

    Returns:
        Normalized numpy array.
    """
    features = stats["features"]
    X = df[features].values.copy().astype(float)

    for i, feat in enumerate(features):
        scaling = stats["scaling"][feat]
        if scaling["method"] == "robust":
            X[:, i] = (X[:, i] - scaling["median"]) / scaling["iqr"]
        else:  # zscore
            X[:, i] = (X[:, i] - scaling["mean"]) / scaling["std"]

    return X


def prepare_batch_expanded(
    df: pd.DataFrame,
    stats: dict,
    target_len: int,
    window_fraction: float = 0.25,
    apply_smoothing: bool = False,
) -> tuple[np.ndarray, float]:
    """Prepare a single batch for modeling with expanded features.

    Args:
        df: Full batch DataFrame.
        stats: Normalization statistics from compute_normalization_stats_expanded.
        target_len: Target sequence length after padding/truncation.
        window_fraction: Fraction of batch for early window.
        apply_smoothing: If True, apply smoothing to noisy signals.

    Returns:
        Tuple of (X, y) where X is (target_len, n_features) and y is scalar.
    """
    # Extract early window
    early = extract_early_window(df, window_fraction)

    # Preprocess features
    processed = preprocess_expanded_features(
        early, stats["features"], apply_smoothing
    )

    # Normalize
    X = normalize_features_expanded(processed, stats)

    # Pad or truncate
    X = pad_or_truncate(X, target_len)

    # Get target and normalize
    y_raw = get_final_target(df)
    y = normalize_target(y_raw, stats.get("y_min", TARGET_MIN), stats.get("y_max", TARGET_MAX))

    return X, y
