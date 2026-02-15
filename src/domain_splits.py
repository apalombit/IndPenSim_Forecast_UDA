"""Domain split assignment for source/target separation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .data_loader import PROJECT_ROOT, load_batches, get_batch_info
from .preprocessing import extract_early_window


def get_control_mode_split(variant: str = "1b") -> dict:
    """Get domain assignments based on control mode.

    Args:
        variant: Split variant:
            - "1a": Recipe (1-30) vs Operator (31-60)
            - "1b": Recipe (1-30) vs APC (61-90) [default]
            - "1c": Recipe+Operator (1-60) vs APC (61-90)

    Returns:
        Dict with 'source' and 'target' batch ID lists.
    """
    if variant == "1a":
        source = list(range(1, 31))
        target = list(range(31, 61))
    elif variant == "1b":
        source = list(range(1, 31))
        target = list(range(61, 91))
    elif variant == "1c":
        source = list(range(1, 61))
        target = list(range(61, 91))
    else:
        raise ValueError(f"Unknown variant: {variant}. Use '1a', '1b', or '1c'.")

    return {"source": source, "target": target, "variant": variant}


def compute_early_features(df: pd.DataFrame) -> dict:
    """Compute summary features from early window for clustering.

    Args:
        df: Early window DataFrame.

    Returns:
        Dict with feature values.
    """
    features = {}

    # DO2 features
    if "DO2" in df.columns:
        features["DO2_mean"] = df["DO2"].mean()
        features["DO2_std"] = df["DO2"].std()
        features["DO2_min"] = df["DO2"].min()

    # Fs features
    if "Fs" in df.columns:
        features["Fs_mean"] = df["Fs"].mean()
        features["Fs_std"] = df["Fs"].std()
        # Compute slope via linear regression
        if len(df) > 1:
            x = df["time"].values
            y = df["Fs"].values
            # Simple slope: (y[-1] - y[0]) / (x[-1] - x[0])
            features["Fs_slope"] = (y[-1] - y[0]) / (x[-1] - x[0] + 1e-8)
        else:
            features["Fs_slope"] = 0.0

    return features


def get_clustering_split(
    batches: dict[int, pd.DataFrame] | None = None,
    exclude_faults: bool = True,
    random_state: int = 42,
) -> dict:
    """Get domain assignments based on k-means clustering.

    Uses early-window (25%) statistics on DO2 and Fs to cluster batches.
    Larger cluster becomes source, smaller becomes target.

    Args:
        batches: Dict of batch DataFrames. If None, loads from default path.
        exclude_faults: If True, exclude fault batches (91-100).
        random_state: Random seed for k-means.

    Returns:
        Dict with 'source', 'target' batch ID lists and 'features' DataFrame.
    """
    if batches is None:
        batches = load_batches()

    # Get batch IDs to process
    batch_ids = sorted(batches.keys())
    if exclude_faults:
        batch_ids = [b for b in batch_ids if b <= 90]

    # Extract features for each batch
    feature_rows = []
    for batch_id in batch_ids:
        df = batches[batch_id]
        early = extract_early_window(df, fraction=0.25)
        feats = compute_early_features(early)
        feats["batch_id"] = batch_id
        feature_rows.append(feats)

    features_df = pd.DataFrame(feature_rows)

    # Standardize features for clustering
    feature_cols = ["DO2_mean", "DO2_std", "DO2_min", "Fs_mean", "Fs_std", "Fs_slope"]
    X = features_df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    features_df["cluster"] = labels

    # Assign larger cluster as source
    cluster_counts = pd.Series(labels).value_counts()
    source_cluster = cluster_counts.idxmax()
    target_cluster = 1 - source_cluster

    source_ids = features_df[features_df["cluster"] == source_cluster]["batch_id"].tolist()
    target_ids = features_df[features_df["cluster"] == target_cluster]["batch_id"].tolist()

    return {
        "source": sorted(source_ids),
        "target": sorted(target_ids),
        "features": features_df,
        "scaler": scaler,
        "kmeans": kmeans,
    }


def compute_split_overlap(split1: dict, split2: dict) -> dict:
    """Compute overlap between two split assignments.

    Args:
        split1: First split dict with 'source' and 'target' keys.
        split2: Second split dict with 'source' and 'target' keys.

    Returns:
        Dict with overlap statistics.
    """
    s1_source = set(split1["source"])
    s1_target = set(split1["target"])
    s2_source = set(split2["source"])
    s2_target = set(split2["target"])

    # Compute overlaps
    source_overlap = len(s1_source & s2_source)
    target_overlap = len(s1_target & s2_target)

    # Agreement: batches assigned same way in both splits
    agreement = source_overlap + target_overlap
    total = len((s1_source | s1_target) & (s2_source | s2_target))

    return {
        "source_source_overlap": source_overlap,
        "target_target_overlap": target_overlap,
        "source_in_split1": len(s1_source),
        "target_in_split1": len(s1_target),
        "source_in_split2": len(s2_source),
        "target_in_split2": len(s2_target),
        "agreement_count": agreement,
    }


def save_split_assignments(
    output_path: Path | str | None = None,
    control_variant: str = "1b",
) -> dict:
    """Generate and save all split assignments to JSON.

    Args:
        output_path: Path to save JSON. If None, uses default outputs path.
        control_variant: Control mode split variant to use.

    Returns:
        Dict with all split assignments.
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "outputs" / "split_assignments.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get control mode split
    control_split = get_control_mode_split(control_variant)

    # Get clustering split
    cluster_result = get_clustering_split()
    cluster_split = {
        "source": cluster_result["source"],
        "target": cluster_result["target"],
    }

    # Compute overlap
    overlap = compute_split_overlap(control_split, cluster_split)

    # Prepare output
    assignments = {
        "control_mode": {
            "variant": control_split["variant"],
            "source": control_split["source"],
            "target": control_split["target"],
        },
        "clustering": {
            "source": cluster_split["source"],
            "target": cluster_split["target"],
        },
        "overlap": overlap,
    }

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(assignments, f, indent=2)

    return assignments


def load_split_assignments(path: Path | str | None = None) -> dict:
    """Load split assignments from JSON.

    Args:
        path: Path to JSON file. If None, uses default path.

    Returns:
        Dict with split assignments.
    """
    if path is None:
        path = PROJECT_ROOT / "outputs" / "split_assignments.json"
    with open(path) as f:
        return json.load(f)
