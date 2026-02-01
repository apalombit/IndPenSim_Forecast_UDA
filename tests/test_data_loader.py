"""Unit tests for data_loader module."""

import pandas as pd
import pytest

from src.data_loader import (
    COLUMN_MAP,
    get_batch_info,
    get_final_penicillin,
    load_batches,
    load_process_data,
    load_statistics,
)

# Skip tests if data not available
DATA_PATH = "data/Mendeley_data/100_Batches_IndPenSim_V3.csv"


@pytest.fixture(scope="module")
def process_data():
    """Load process data once for all tests."""
    try:
        return load_process_data(DATA_PATH)
    except FileNotFoundError:
        pytest.skip("Data file not found")


@pytest.fixture(scope="module")
def batches(process_data):
    """Load batches dictionary."""
    return load_batches(DATA_PATH)


class TestLoadProcessData:
    def test_loads_dataframe(self, process_data):
        assert isinstance(process_data, pd.DataFrame)
        assert len(process_data) > 0

    def test_excludes_raman_spectra(self, process_data):
        # Should not have numeric column names (Raman wavelengths)
        numeric_cols = [c for c in process_data.columns if c.isdigit()]
        assert len(numeric_cols) == 0

    def test_has_renamed_columns(self, process_data):
        expected = ["time", "batch_id", "P", "DO2", "Fs", "T", "pH"]
        for col in expected:
            assert col in process_data.columns, f"Missing column: {col}"

    def test_batch_id_range(self, process_data):
        batch_ids = process_data["batch_id"].unique()
        assert batch_ids.min() >= 1
        assert batch_ids.max() <= 100


class TestLoadBatches:
    def test_returns_dict(self, batches):
        assert isinstance(batches, dict)

    def test_has_100_batches(self, batches):
        assert len(batches) == 100

    def test_batch_keys_are_ints(self, batches):
        assert all(isinstance(k, int) for k in batches.keys())

    def test_each_batch_is_dataframe(self, batches):
        for batch_id, df in batches.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0


class TestGetBatchInfo:
    def test_returns_dataframe(self, batches):
        info = get_batch_info(batches)
        assert isinstance(info, pd.DataFrame)
        assert len(info) == 100

    def test_has_required_columns(self, batches):
        info = get_batch_info(batches)
        required = ["batch_id", "length", "duration_h", "control_mode", "is_fault"]
        for col in required:
            assert col in info.columns

    def test_control_mode_assignment(self, batches):
        info = get_batch_info(batches)
        recipe = info[info["batch_id"] <= 30]["control_mode"].unique()
        operator = info[(info["batch_id"] > 30) & (info["batch_id"] <= 60)]["control_mode"].unique()
        apc = info[(info["batch_id"] > 60) & (info["batch_id"] <= 90)]["control_mode"].unique()
        fault = info[info["batch_id"] > 90]["control_mode"].unique()

        assert list(recipe) == ["recipe"]
        assert list(operator) == ["operator"]
        assert list(apc) == ["apc"]
        assert list(fault) == ["fault"]


class TestGetFinalPenicillin:
    def test_returns_dataframe(self, batches):
        result = get_final_penicillin(batches)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100

    def test_has_required_columns(self, batches):
        result = get_final_penicillin(batches)
        assert "batch_id" in result.columns
        assert "final_P" in result.columns

    def test_values_are_positive(self, batches):
        result = get_final_penicillin(batches)
        assert (result["final_P"] > 0).all()
