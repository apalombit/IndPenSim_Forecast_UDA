"""Unit tests for domain_splits module."""

import pandas as pd
import pytest

from src.domain_splits import (
    compute_early_features,
    compute_split_overlap,
    extract_early_window,
    get_clustering_split,
    get_control_mode_split,
)
from src.data_loader import load_batches


class TestGetControlModeSplit:
    def test_variant_1a(self):
        split = get_control_mode_split("1a")
        assert split["source"] == list(range(1, 31))
        assert split["target"] == list(range(31, 61))
        assert split["variant"] == "1a"

    def test_variant_1b(self):
        split = get_control_mode_split("1b")
        assert split["source"] == list(range(1, 31))
        assert split["target"] == list(range(61, 91))
        assert split["variant"] == "1b"

    def test_variant_1c(self):
        split = get_control_mode_split("1c")
        assert split["source"] == list(range(1, 61))
        assert split["target"] == list(range(61, 91))
        assert split["variant"] == "1c"

    def test_invalid_variant(self):
        with pytest.raises(ValueError):
            get_control_mode_split("invalid")


class TestExtractEarlyWindow:
    def test_extracts_25_percent(self):
        df = pd.DataFrame({"time": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
        early = extract_early_window(df, fraction=0.25)
        assert early["time"].max() <= 25

    def test_extracts_50_percent(self):
        df = pd.DataFrame({"time": [0, 25, 50, 75, 100]})
        early = extract_early_window(df, fraction=0.5)
        assert early["time"].max() <= 50


class TestComputeEarlyFeatures:
    def test_returns_expected_keys(self):
        df = pd.DataFrame({
            "time": [0, 1, 2, 3, 4],
            "DO2": [10, 12, 11, 13, 14],
            "Fs": [5, 6, 7, 8, 9],
        })
        feats = compute_early_features(df)
        expected_keys = ["DO2_mean", "DO2_std", "DO2_min", "Fs_mean", "Fs_std", "Fs_slope"]
        for key in expected_keys:
            assert key in feats

    def test_feature_values(self):
        df = pd.DataFrame({
            "time": [0, 1, 2, 3, 4],
            "DO2": [10, 10, 10, 10, 10],
            "Fs": [0, 1, 2, 3, 4],
        })
        feats = compute_early_features(df)
        assert feats["DO2_mean"] == 10.0
        assert feats["DO2_std"] == 0.0
        assert feats["DO2_min"] == 10.0
        assert feats["Fs_slope"] == pytest.approx(1.0, rel=0.01)


class TestComputeSplitOverlap:
    def test_full_overlap(self):
        split1 = {"source": [1, 2, 3], "target": [4, 5, 6]}
        split2 = {"source": [1, 2, 3], "target": [4, 5, 6]}
        overlap = compute_split_overlap(split1, split2)
        assert overlap["source_source_overlap"] == 3
        assert overlap["target_target_overlap"] == 3

    def test_no_overlap(self):
        split1 = {"source": [1, 2, 3], "target": [4, 5, 6]}
        split2 = {"source": [4, 5, 6], "target": [1, 2, 3]}
        overlap = compute_split_overlap(split1, split2)
        assert overlap["source_source_overlap"] == 0
        assert overlap["target_target_overlap"] == 0


@pytest.fixture(scope="module")
def batches():
    """Load batches once for clustering tests."""
    try:
        return load_batches()
    except FileNotFoundError:
        pytest.skip("Data file not found")


class TestGetClusteringSplit:
    def test_returns_source_and_target(self, batches):
        result = get_clustering_split(batches)
        assert "source" in result
        assert "target" in result
        assert len(result["source"]) > 0
        assert len(result["target"]) > 0

    def test_no_overlap_between_source_target(self, batches):
        result = get_clustering_split(batches)
        source_set = set(result["source"])
        target_set = set(result["target"])
        assert len(source_set & target_set) == 0

    def test_excludes_fault_batches(self, batches):
        result = get_clustering_split(batches, exclude_faults=True)
        all_ids = result["source"] + result["target"]
        assert all(b <= 90 for b in all_ids)

    def test_returns_features_dataframe(self, batches):
        result = get_clustering_split(batches)
        assert "features" in result
        assert isinstance(result["features"], pd.DataFrame)
        assert "cluster" in result["features"].columns
