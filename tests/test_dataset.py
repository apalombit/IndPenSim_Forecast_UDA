"""Unit tests for dataset module."""

import torch
import pytest

from src.dataset import IndPenSimDataset, create_dataloaders
from src.data_loader import load_batches


@pytest.fixture(scope="module")
def batches():
    try:
        return load_batches()
    except FileNotFoundError:
        pytest.skip("Data file not found")


class TestIndPenSimDataset:
    def test_creates_dataset(self, batches):
        dataset = IndPenSimDataset([1, 2, 3], batches)
        assert len(dataset) == 3

    def test_returns_tensors(self, batches):
        dataset = IndPenSimDataset([1], batches)
        X, y, domain = dataset[0]
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert isinstance(domain, torch.Tensor)

    def test_correct_shapes(self, batches):
        dataset = IndPenSimDataset([1, 2], batches, target_len=100)
        X, y, domain = dataset[0]
        assert X.shape[0] == 100
        assert X.shape[1] == dataset.n_features
        assert y.shape == ()
        assert domain.shape == ()

    def test_domain_label(self, batches):
        source_ds = IndPenSimDataset([1], batches, domain_label=0)
        target_ds = IndPenSimDataset([61], batches, domain_label=1)
        _, _, d0 = source_ds[0]
        _, _, d1 = target_ds[0]
        assert d0.item() == 0
        assert d1.item() == 1


class TestCreateDataloaders:
    def test_returns_expected_keys(self, batches):
        result = create_dataloaders(
            source_ids=list(range(1, 11)),
            target_ids=list(range(61, 71)),
            batches=batches,
            batch_size=4,
        )
        assert "train" in result
        assert "val" in result
        assert "target" in result
        assert "stats" in result

    def test_dataloaders_iterate(self, batches):
        result = create_dataloaders(
            source_ids=list(range(1, 11)),
            target_ids=list(range(61, 66)),
            batches=batches,
            batch_size=2,
        )
        # Check we can iterate
        for X, y, d in result["train"]:
            assert X.ndim == 3  # (batch, seq_len, features)
            assert y.ndim == 1  # (batch,)
            assert d.ndim == 1  # (batch,)
            break

    def test_val_split(self, batches):
        source_ids = list(range(1, 21))  # 20 batches
        result = create_dataloaders(
            source_ids=source_ids,
            target_ids=list(range(61, 71)),
            batches=batches,
            val_ratio=0.2,
            batch_size=4,
        )
        # With 20 source and 0.2 val_ratio, expect 4 val and 16 train
        assert len(result["val"].dataset) == 4
        assert len(result["train"].dataset) == 16
