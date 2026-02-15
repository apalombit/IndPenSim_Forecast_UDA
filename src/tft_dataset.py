"""PyTorch Dataset for TFT concentration prediction with variable (T, D) sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset, DataLoader, Sampler

from .data_loader import load_batches
from .feature_config import INPUT_FEATURES_EXPANDED
from .preprocessing import (
    compute_normalization_stats_expanded,
    preprocess_expanded_features,
    normalize_features_expanded,
    TARGET_MIN,
    TARGET_MAX,
)


def interpolate_concentration(
    times: np.ndarray,
    conc: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Interpolate sparse concentration measurements using cubic spline.

    Args:
        times: Time points of observations.
        conc: Concentration values at observation times.
        target_times: Times at which to interpolate.

    Returns:
        Interpolated concentration values.
    """
    # Remove NaN values
    valid_mask = ~np.isnan(conc)
    if valid_mask.sum() < 2:
        # Not enough points for spline, use linear fill
        return np.full_like(target_times, np.nanmean(conc))

    valid_times = times[valid_mask]
    valid_conc = conc[valid_mask]

    # Create cubic spline
    cs = CubicSpline(valid_times, valid_conc, extrapolate=True)

    return cs(target_times)


def build_concentration_channels(
    df: pd.DataFrame,
    conc_column: str = "P",
    time_column: str = "time",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build concentration-related channels for TFT input.

    Creates three channels:
        1. conc_ffill: Forward-filled concentration
        2. conc_mask: Binary mask (1 where observation exists)
        3. conc_time_since_obs: Hours since last observation

    Args:
        df: Batch DataFrame with concentration and time columns.
        conc_column: Column name for concentration.
        time_column: Column name for time.

    Returns:
        Tuple of (conc_ffill, conc_mask, conc_time_since_obs) arrays.
    """
    time = df[time_column].values
    conc = df[conc_column].values.copy()

    n_steps = len(time)

    # Initialize outputs
    conc_ffill = np.zeros(n_steps)
    conc_mask = np.zeros(n_steps)
    conc_time_since = np.zeros(n_steps)

    # Track last observation
    last_conc = 0.0
    last_obs_time = time[0]  # Start from beginning

    for i in range(n_steps):
        # Check if this is a new observation (non-zero concentration change)
        if i == 0 or (not np.isnan(conc[i]) and conc[i] != conc[i-1]):
            conc_mask[i] = 1.0
            last_conc = conc[i]
            last_obs_time = time[i]

        conc_ffill[i] = last_conc
        conc_time_since[i] = time[i] - last_obs_time

    return conc_ffill, conc_mask, conc_time_since


def get_target_concentration_at_horizon(
    df: pd.DataFrame,
    T_idx: int,
    D_steps: int,
    conc_column: str = "P",
    time_column: str = "time",
) -> float:
    """Get interpolated concentration at T + D.

    Args:
        df: Batch DataFrame.
        T_idx: Index of current time T (end of input window).
        D_steps: Prediction horizon in time steps.
        conc_column: Concentration column name.
        time_column: Time column name.

    Returns:
        Concentration value at T + D (interpolated).
    """
    times = df[time_column].values
    conc = df[conc_column].values

    # Target time
    target_time = times[T_idx] + (D_steps * (times[1] - times[0]) if len(times) > 1 else D_steps)

    # If target is within batch, interpolate
    if target_time <= times[-1]:
        return float(interpolate_concentration(times, conc, np.array([target_time]))[0])
    else:
        # Extrapolate to end of batch
        return float(interpolate_concentration(times, conc, np.array([target_time]))[0])


class TFTConcentrationDataset(Dataset):
    """Dataset with variable (T, D) sampling for TFT training.

    Each sample consists of:
        - x: Input features (T_steps, n_features + 3 concentration channels)
        - y: Target concentration at T + D
        - T_steps: Length of input sequence
        - D_steps: Prediction horizon
        - mask: Padding mask
        - batch_id: Source batch identifier
    """

    def __init__(
        self,
        batch_ids: list[int],
        batches: dict | None = None,
        stats: dict | None = None,
        max_seq_len: int = 500,
        min_T_fraction: float = 0.1,
        max_T_fraction: float = 0.9,
        min_D_hours: float = 10.0,
        max_D_hours: float = 100.0,
        samples_per_batch: int = 10,
        seed: int = 42,
        features: list[str] | None = None,
        source_ids: list[int] | None = None,
        conc_column: str = "P",
        n_T_values: int | None = None,
    ):
        """Initialize TFT dataset.

        Args:
            batch_ids: List of batch IDs to include.
            batches: Dict of batch DataFrames. Loaded if None.
            stats: Normalization statistics. Computed from source_ids if None.
            max_seq_len: Maximum sequence length (for padding).
            min_T_fraction: Minimum fraction of batch to use as input (e.g., 0.1 = 10%).
            max_T_fraction: Maximum fraction of batch to use as input.
            min_D_hours: Minimum prediction horizon in hours.
            max_D_hours: Maximum prediction horizon in hours.
            samples_per_batch: Number of (T, D) samples to generate per batch.
            seed: Random seed for sampling.
            features: Input feature columns. Defaults to INPUT_FEATURES_EXPANDED.
            source_ids: Source batch IDs for computing stats.
            conc_column: Column name for concentration measurements.
            n_T_values: Number of distinct T_idx values for uniform-length batching.
                When set, samples are grouped by T_idx so each mini-batch has
                uniform sequence length (no padding). When None, old random
                T behavior is used.
        """
        self.batch_ids = batch_ids
        self.max_seq_len = max_seq_len
        self.min_T_fraction = min_T_fraction
        self.max_T_fraction = max_T_fraction
        self.min_D_hours = min_D_hours
        self.max_D_hours = max_D_hours
        self.samples_per_batch = samples_per_batch
        self.conc_column = conc_column
        self.n_T_values = n_T_values

        if features is None:
            features = list(INPUT_FEATURES_EXPANDED)
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
                batches, source_ids, features, window_fraction=1.0
            )
        self.stats = stats

        # Add concentration channel stats (for normalization)
        self._compute_conc_stats(source_ids or batch_ids)

        # Generate samples
        self.rng = np.random.default_rng(seed)
        if n_T_values is not None:
            self._generate_grouped_samples()
        else:
            self.t_groups = None
            self._generate_samples()

    def _compute_conc_stats(self, batch_ids: list[int]):
        """Compute concentration channel statistics for normalization."""
        all_conc = []
        all_time_since = []

        for batch_id in batch_ids:
            df = self.batches[batch_id]
            conc_ffill, _, conc_time_since = build_concentration_channels(
                df, self.conc_column
            )
            all_conc.extend(conc_ffill)
            all_time_since.extend(conc_time_since)

        self.conc_mean = np.mean(all_conc)
        self.conc_std = np.std(all_conc) if np.std(all_conc) > 0 else 1.0
        self.time_since_mean = np.mean(all_time_since)
        self.time_since_std = np.std(all_time_since) if np.std(all_time_since) > 0 else 1.0

    def _generate_samples(self):
        """Generate (T, D) samples for all batches."""
        self.samples = []

        for batch_id in self.batch_ids:
            df = self.batches[batch_id]
            n_steps = len(df)
            times = df["time"].values

            # Time step size (assuming uniform)
            dt = (times[1] - times[0]) if len(times) > 1 else 1.0

            for _ in range(self.samples_per_batch):
                # Sample T (input window end) as fraction of batch
                T_frac = self.rng.uniform(self.min_T_fraction, self.max_T_fraction)
                T_idx = int(T_frac * n_steps)
                T_idx = max(10, min(T_idx, n_steps - 10))  # Ensure valid range

                # Sample D (prediction horizon) in hours
                D_hours = self.rng.uniform(self.min_D_hours, self.max_D_hours)
                D_steps = int(D_hours / dt)

                # Ensure T + D doesn't exceed batch (with some tolerance)
                max_D_steps = n_steps - T_idx - 1
                if D_steps > max_D_steps:
                    D_steps = max(1, max_D_steps)

                self.samples.append({
                    "batch_id": batch_id,
                    "T_idx": T_idx,
                    "D_steps": D_steps,
                })

    def _generate_grouped_samples(self):
        """Generate samples grouped by shared T_idx for uniform-length batching.

        Samples n_T_values distinct absolute T_idx values, then for each
        (T_idx, valid_batch) pair generates random D values. Stores
        self.t_groups mapping T_idx → list of sample indices.
        """
        # Compute valid T_idx range for each batch
        batch_ranges = {}
        for batch_id in self.batch_ids:
            n_steps = len(self.batches[batch_id])
            low = max(10, int(self.min_T_fraction * n_steps))
            high = min(n_steps - 10, int(self.max_T_fraction * n_steps))
            if low <= high:
                batch_ranges[batch_id] = (low, high)

        # Find global T_idx range (intersection of all possible values)
        global_low = min(r[0] for r in batch_ranges.values())
        global_high = max(r[1] for r in batch_ranges.values())

        # Sample n_T_values distinct T_idx values from the global range
        all_possible = np.arange(global_low, global_high + 1)
        n_t = min(self.n_T_values, len(all_possible))
        chosen_T_indices = self.rng.choice(all_possible, size=n_t, replace=False)
        chosen_T_indices.sort()

        # For each T_idx, find valid batches and generate D samples
        self.samples = []
        self.t_groups: dict[int, list[int]] = {}

        # How many D samples per (T_idx, batch) pair
        d_samples_per_pair = max(1, self.samples_per_batch // n_t)

        for T_idx in chosen_T_indices:
            T_idx = int(T_idx)
            group_indices = []

            for batch_id, (low, high) in batch_ranges.items():
                if low <= T_idx <= high:
                    df = self.batches[batch_id]
                    n_steps = len(df)
                    times = df["time"].values
                    dt = (times[1] - times[0]) if len(times) > 1 else 1.0

                    for _ in range(d_samples_per_pair):
                        D_hours = self.rng.uniform(self.min_D_hours, self.max_D_hours)
                        D_steps = int(D_hours / dt)
                        max_D_steps = n_steps - T_idx - 1
                        if D_steps > max_D_steps:
                            D_steps = max(1, max_D_steps)

                        sample_idx = len(self.samples)
                        self.samples.append({
                            "batch_id": batch_id,
                            "T_idx": T_idx,
                            "D_steps": D_steps,
                        })
                        group_indices.append(sample_idx)

            if group_indices:
                self.t_groups[T_idx] = group_indices

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.

        Returns:
            Dict with keys: x, y, T_steps, D_steps, batch_id
        """
        sample = self.samples[idx]
        batch_id = sample["batch_id"]
        T_idx = sample["T_idx"]
        D_steps = sample["D_steps"]

        df = self.batches[batch_id]

        # Extract input window [0:T_idx]
        input_df = df.iloc[:T_idx].copy()

        # Preprocess and normalize features
        processed = preprocess_expanded_features(input_df, self.features)
        X_features = normalize_features_expanded(processed, self.stats)

        # Build concentration channels
        conc_ffill, conc_mask, conc_time_since = build_concentration_channels(
            input_df, self.conc_column
        )

        # Normalize concentration channels
        conc_ffill_norm = (conc_ffill - self.conc_mean) / self.conc_std
        conc_time_norm = (conc_time_since - self.time_since_mean) / self.time_since_std

        # Stack all features: (T, n_features + 3)
        X = np.concatenate([
            X_features,
            conc_ffill_norm[:, None],
            conc_mask[:, None],
            conc_time_norm[:, None],
        ], axis=1)

        # Get target concentration at T + D
        y = get_target_concentration_at_horizon(
            df, T_idx, D_steps, self.conc_column
        )

        # Normalize target
        y_norm = (y - TARGET_MIN) / (TARGET_MAX - TARGET_MIN)

        return {
            "x": torch.tensor(X, dtype=torch.float32),
            "y": torch.tensor(y_norm, dtype=torch.float32),
            "T_steps": T_idx,
            "D_steps": D_steps,
            "batch_id": batch_id,
        }

    @property
    def n_features(self) -> int:
        """Number of input features (including concentration channels)."""
        return len(self.stats["features"]) + 3  # +3 for conc channels


def tft_collate_fn(batch: list[dict]) -> dict:
    """Collate function for variable-length TFT sequences.

    Pads sequences to the maximum length in the batch and creates attention masks.

    Args:
        batch: List of sample dicts from TFTConcentrationDataset.

    Returns:
        Dict with padded tensors and masks.
    """
    # Find max sequence length in this batch
    max_len = max(sample["T_steps"] for sample in batch)
    n_features = batch[0]["x"].shape[1]
    batch_size = len(batch)

    # Initialize padded tensors
    x_padded = torch.zeros(batch_size, max_len, n_features)
    y = torch.zeros(batch_size)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)  # True = padded
    T_steps = torch.zeros(batch_size, dtype=torch.long)
    D_steps = torch.zeros(batch_size, dtype=torch.long)
    batch_ids = torch.zeros(batch_size, dtype=torch.long)

    for i, sample in enumerate(batch):
        seq_len = sample["T_steps"]
        x_padded[i, :seq_len, :] = sample["x"]
        y[i] = sample["y"]
        mask[i, :seq_len] = False  # Not padded
        T_steps[i] = seq_len
        D_steps[i] = sample["D_steps"]
        batch_ids[i] = sample["batch_id"]

    return {
        "x": x_padded,
        "y": y,
        "mask": mask,
        "T_steps": T_steps,
        "D_steps": D_steps,
        "batch_ids": batch_ids,
    }


class GroupedBatchSampler(Sampler):
    """Batch sampler that yields mini-batches from within T_idx groups.

    All samples in a mini-batch share the same T_idx, ensuring uniform
    sequence length without padding.
    """

    def __init__(
        self,
        t_groups: dict[int, list[int]],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.t_groups = t_groups
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling."""
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.rng.integers(0, 2**31) + self._epoch)

        # Collect all mini-batches across all groups
        all_batches = []
        group_keys = list(self.t_groups.keys())
        if self.shuffle:
            rng.shuffle(group_keys)

        for t_idx in group_keys:
            indices = list(self.t_groups[t_idx])
            if self.shuffle:
                rng.shuffle(indices)

            # Chunk into mini-batches
            for i in range(0, len(indices), self.batch_size):
                all_batches.append(indices[i : i + self.batch_size])

        # Shuffle the order of mini-batches across groups
        if self.shuffle:
            rng.shuffle(all_batches)

        yield from all_batches

    def __len__(self) -> int:
        total = 0
        for indices in self.t_groups.values():
            total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


def tft_collate_uniform(batch: list[dict]) -> dict:
    """Collate function for uniform-length TFT sequences (no padding needed).

    All samples in the batch have the same T_idx, so sequences can be
    stacked directly. Returns an all-False mask for API compatibility.

    Args:
        batch: List of sample dicts from TFTConcentrationDataset.

    Returns:
        Dict with stacked tensors and all-False mask.
    """
    batch_size = len(batch)

    x = torch.stack([s["x"] for s in batch])  # (B, T, C)
    y = torch.stack([s["y"] for s in batch])   # (B,)
    T_steps = torch.tensor([s["T_steps"] for s in batch], dtype=torch.long)
    D_steps = torch.tensor([s["D_steps"] for s in batch], dtype=torch.long)
    batch_ids = torch.tensor([s["batch_id"] for s in batch], dtype=torch.long)

    # All-False mask (no padding)
    mask = torch.zeros(batch_size, x.shape[1], dtype=torch.bool)

    return {
        "x": x,
        "y": y,
        "mask": mask,
        "T_steps": T_steps,
        "D_steps": D_steps,
        "batch_ids": batch_ids,
    }


def create_tft_dataloaders(
    source_ids: list[int],
    target_ids: list[int],
    batches: dict | None = None,
    val_ratio: float = 0.2,
    batch_size: int = 16,
    max_seq_len: int = 500,
    min_T_fraction: float = 0.1,
    max_T_fraction: float = 0.9,
    min_D_hours: float = 10.0,
    max_D_hours: float = 100.0,
    samples_per_batch: int = 10,
    seed: int = 42,
    features: list[str] | None = None,
    conc_column: str = "P",
    num_workers: int = 0,
    n_T_values: int | None = None,
) -> dict:
    """Create train/val/target DataLoaders for TFT training.

    Args:
        source_ids: Source domain batch IDs.
        target_ids: Target domain batch IDs.
        batches: Dict of batch DataFrames. Loaded if None.
        val_ratio: Fraction of source for validation.
        batch_size: Batch size for DataLoaders.
        max_seq_len: Maximum sequence length.
        min_T_fraction: Minimum input fraction.
        max_T_fraction: Maximum input fraction.
        min_D_hours: Minimum prediction horizon (hours).
        max_D_hours: Maximum prediction horizon (hours).
        samples_per_batch: Number of (T, D) samples per batch.
        seed: Random seed.
        features: Input feature columns.
        conc_column: Concentration column name.
        num_workers: Number of DataLoader workers.
        n_T_values: Number of distinct T_idx values for uniform-length batching.
            When set, uses GroupedBatchSampler for padding-free mini-batches.

    Returns:
        Dict with 'train', 'val', 'target' DataLoaders, 'stats', and 'collate_fn'.
    """
    if features is None:
        features = list(INPUT_FEATURES_EXPANDED)

    if batches is None:
        batches = load_batches()

    # Split source into train/val
    rng = np.random.default_rng(seed)
    source_ids = list(source_ids)
    rng.shuffle(source_ids)

    n_val = int(len(source_ids) * val_ratio)
    val_ids = source_ids[:n_val]
    train_ids = source_ids[n_val:]

    # Compute normalization stats from training set
    stats = compute_normalization_stats_expanded(
        batches, train_ids, features, window_fraction=1.0
    )

    # Create datasets
    dataset_kwargs = dict(
        batches=batches,
        stats=stats,
        max_seq_len=max_seq_len,
        min_T_fraction=min_T_fraction,
        max_T_fraction=max_T_fraction,
        min_D_hours=min_D_hours,
        max_D_hours=max_D_hours,
        samples_per_batch=samples_per_batch,
        features=features,
        source_ids=train_ids,
        conc_column=conc_column,
        n_T_values=n_T_values,
    )

    train_dataset = TFTConcentrationDataset(
        batch_ids=train_ids, seed=seed, **dataset_kwargs,
    )
    val_dataset = TFTConcentrationDataset(
        batch_ids=val_ids, seed=seed + 1, **dataset_kwargs,
    )
    target_dataset = TFTConcentrationDataset(
        batch_ids=target_ids, seed=seed + 2, **dataset_kwargs,
    )

    # Create DataLoaders
    if n_T_values is not None:
        collate = tft_collate_uniform
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=GroupedBatchSampler(
                train_dataset.t_groups, batch_size, shuffle=True, seed=seed,
            ),
            collate_fn=collate,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=GroupedBatchSampler(
                val_dataset.t_groups, batch_size, shuffle=False, seed=seed,
            ),
            collate_fn=collate,
            num_workers=num_workers,
        )
        target_loader = DataLoader(
            target_dataset,
            batch_sampler=GroupedBatchSampler(
                target_dataset.t_groups, batch_size, shuffle=False, seed=seed,
            ),
            collate_fn=collate,
            num_workers=num_workers,
        )
    else:
        collate = tft_collate_fn
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=num_workers,
        )
        target_loader = DataLoader(
            target_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=num_workers,
        )

    return {
        "train": train_loader,
        "val": val_loader,
        "target": target_loader,
        "stats": stats,
        "collate_fn": collate,
        "n_features": train_dataset.n_features,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "target_dataset": target_dataset,
    }
