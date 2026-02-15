"""Feature configuration and signal type metadata for IndPenSim preprocessing."""

# Signal type constants
SIGNAL_STEP_WISE = "step_wise"
SIGNAL_SPIKY_SPARSE = "spiky_sparse"
SIGNAL_NOISY_TREND = "noisy_trend"
SIGNAL_NOISY_SHORT_TREND = "noisy_short_trend"
SIGNAL_TREND_JUMP = "trend_jump"
SIGNAL_SPARSE_OFFLINE = "sparse_offline"

# Feature groups by signal type
STEP_WISE_FEATURES = ["Fg", "Fs", "Fw", "pressure", "Fpaa", "Foil"]
SPIKY_SPARSE_FEATURES = ["Fa", "Fh", "Fremoved"]
NOISY_TREND_FEATURES = ["Fb", "Fc", "S", "OUR", "O2"]
NOISY_SHORT_TREND_FEATURES = ["pH", "T"]
TREND_JUMP_FEATURES = ["DO2", "V", "Wt", "CO2outgas", "CER"]
SPARSE_OFFLINE_FEATURES = ["PAA_offline", "NH3_offline", "X_offline", "Viscosity_offline"]

# Feature to signal type mapping
FEATURE_SIGNAL_TYPE = {
    **{f: SIGNAL_STEP_WISE for f in STEP_WISE_FEATURES},
    **{f: SIGNAL_SPIKY_SPARSE for f in SPIKY_SPARSE_FEATURES},
    **{f: SIGNAL_NOISY_TREND for f in NOISY_TREND_FEATURES},
    **{f: SIGNAL_NOISY_SHORT_TREND for f in NOISY_SHORT_TREND_FEATURES},
    **{f: SIGNAL_TREND_JUMP for f in TREND_JUMP_FEATURES},
    **{f: SIGNAL_SPARSE_OFFLINE for f in SPARSE_OFFLINE_FEATURES},
}

# Expanded feature set (26 features)
INPUT_FEATURES_EXPANDED = (
    STEP_WISE_FEATURES
    + SPIKY_SPARSE_FEATURES
    + NOISY_TREND_FEATURES
    + NOISY_SHORT_TREND_FEATURES
    + TREND_JUMP_FEATURES
    + SPARSE_OFFLINE_FEATURES
)

# Smoothing parameters by signal type
SMOOTHING_PARAMS = {
    SIGNAL_NOISY_TREND: {"window": 15, "order": 2},
    SIGNAL_NOISY_SHORT_TREND: {"window": 7, "order": 2},
}


def get_signal_type(feature: str) -> str:
    """Get signal type for a feature.

    Args:
        feature: Feature column name.

    Returns:
        Signal type string, or "unknown" if not found.
    """
    return FEATURE_SIGNAL_TYPE.get(feature, "unknown")


def should_smooth(feature: str) -> bool:
    """Check if feature should be smoothed (when smoothing enabled).

    Args:
        feature: Feature column name.

    Returns:
        True if feature should be smoothed.
    """
    return get_signal_type(feature) in SMOOTHING_PARAMS


def get_smoothing_params(feature: str) -> dict | None:
    """Get smoothing parameters for a feature.

    Args:
        feature: Feature column name.

    Returns:
        Dict with 'window' and 'order' keys, or None if not applicable.
    """
    signal_type = get_signal_type(feature)
    return SMOOTHING_PARAMS.get(signal_type)
