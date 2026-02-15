"""PyTorch Dataset classes for IndPenSim data."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .data_loader import load_batches
from .feature_config import INPUT_FEATURES_EXPANDED
from .preprocessing import (
    compute_normalization_stats_expanded,
    compute_target_length,
    prepare_batch_expanded,
)


class IndPenSimDataset(Dataset):
    """PyTorch Dataset for IndPenSim fermentation batches.

    Each sample is (X, y, domain_label) where:
        - X: (seq_len, n_features) normalized time-series
        - y: scalar final penicillin concentration
        - domain_label: 0 for source, 1 for target
    """

    def __init__(
        self,
        batch_ids: list[int],
        batches: dict | None = None,
        stats: dict | None = None,
        target_len: int | None = None,
        window_fraction: float = 0.25,
        domain_label: int = 0,
        source_ids: list[int] | None = None,
        features: list[str] | None = None,
    ):
        """Initialize dataset.

        Args:
            batch_ids: List of batch IDs to include.
            batches: Dict of batch DataFrames. Loaded if None.
            stats: Normalization statistics. Computed from source_ids if None.
            target_len: Sequence length after padding. Computed if None.
            window_fraction: Fraction of batch for early window.
            domain_label: Domain label (0=source, 1=target).
            source_ids: Source batch IDs for computing stats (if stats is None).
            features: Input feature columns. Defaults to INPUT_FEATURES_EXPANDED.
        """
        self.batch_ids = batch_ids
        self.window_fraction = window_fraction
        self.domain_label = domain_label

        if features is None:
            features = INPUT_FEATURES_EXPANDED
        self.features = features

        # Load batches if not provided
        if batches is None:
            batches = load_batches()
        self.batches = batches

        # Compute normalization stats if not provided
        if stats is None:
            if source_ids is None:
                source_ids = batch_ids
            stats = compute_normalization_stats_expanded(
                batches, source_ids, features, window_fraction
            )
        self.stats = stats

        # Compute target length if not provided
        if target_len is None:
            target_len = compute_target_length(
                batches, batch_ids, window_fraction
            )
        self.target_len = target_len

        # Precompute all samples
        self._prepare_samples()

    def _prepare_samples(self):
        """Precompute all samples for faster access."""
        self.samples = []
        for batch_id in self.batch_ids:
            df = self.batches[batch_id]
            X, y = prepare_batch_expanded(df, self.stats, self.target_len, self.window_fraction)
            self.samples.append((X, y, batch_id))

    def __len__(self) -> int:
        return len(self.batch_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X, y, batch_id = self.samples[idx]
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(self.domain_label, dtype=torch.long),
        )

    @property
    def n_features(self) -> int:
        """Number of input features."""
        return len(self.stats["features"])

    @property
    def seq_len(self) -> int:
        """Sequence length after padding."""
        return self.target_len


def create_dataloaders(
    source_ids: list[int],
    target_ids: list[int],
    batches: dict | None = None,
    val_ratio: float = 0.2,
    batch_size: int = 16,
    window_fraction: float = 0.25,
    seed: int = 42,
    features: list[str] | None = None,
) -> dict:
    """Create train/val/test DataLoaders for domain adaptation.

    Args:
        source_ids: Source domain batch IDs.
        target_ids: Target domain batch IDs.
        batches: Dict of batch DataFrames. Loaded if None.
        val_ratio: Fraction of source for validation.
        batch_size: Batch size for DataLoaders.
        window_fraction: Fraction of batch for early window.
        seed: Random seed for splitting.
        features: Input feature columns. Defaults to INPUT_FEATURES_EXPANDED.

    Returns:
        Dict with 'train', 'val', 'target' DataLoaders and 'stats'.
    """
    if features is None:
        features = INPUT_FEATURES_EXPANDED

    if batches is None:
        batches = load_batches()

    # Split source into train/val
    rng = np.random.default_rng(seed)
    source_ids = list(source_ids)
    rng.shuffle(source_ids)

    n_val = int(len(source_ids) * val_ratio)
    val_ids = source_ids[:n_val]
    train_ids = source_ids[n_val:]

    # Compute normalization stats from training set only
    stats = compute_normalization_stats_expanded(
        batches, train_ids, features, window_fraction
    )

    # Compute target sequence length from training set
    target_len = compute_target_length(batches, train_ids, window_fraction)

    # Create datasets
    train_dataset = IndPenSimDataset(
        train_ids, batches, stats, target_len, window_fraction,
        domain_label=0, source_ids=train_ids, features=features
    )
    val_dataset = IndPenSimDataset(
        val_ids, batches, stats, target_len, window_fraction,
        domain_label=0, source_ids=train_ids, features=features
    )
    target_dataset = IndPenSimDataset(
        target_ids, batches, stats, target_len, window_fraction,
        domain_label=1, source_ids=train_ids, features=features
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)

    return {
        "train": train_loader,
        "val": val_loader,
        "target": target_loader,
        "stats": stats,
        "target_len": target_len,
        "n_features": train_dataset.n_features,
    }
