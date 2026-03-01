"""Dataset for DeclineForecastModel with (T) sampling and decline targets.

Each sample observes process signals up to time T (40–90% of batch) and
provides targets for decline classification, timing, and slope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .data_loader import load_batches
from .feature_config import INPUT_FEATURES_EXPANDED
from .preprocessing import (
    compute_normalization_stats_expanded,
    preprocess_expanded_features,
    normalize_features_expanded,
)
from .tft_dataset import build_concentration_channels

# Fixed normalization constant for delta_t_break
T_MAX_NORM = 300.0


class DeclineForecastDataset(Dataset):
    """Dataset for decline forecasting with T sampling.

    Each sample:
        x:              (max_seq_len, n_ch) — process signals + conc channels
        T_frac:         scalar, T_idx / n_steps
        decline_target: 0.0 or 1.0 (binary)
        delta_target:   (t_break - t_cutoff) / T_MAX_NORM (normalized)
        slope_target:   raw slope in g/L/h
        domain_label:   0 (source) / 1 (target)
    """

    def __init__(
        self,
        batch_ids: list[int],
        fitted_params_df: pd.DataFrame,
        domain_label: int = 0,
        batches: dict | None = None,
        stats: dict | None = None,
        max_seq_len: int = 500,
        min_T_fraction: float = 0.4,
        max_T_fraction: float = 0.9,
        samples_per_batch: int = 10,
        seed: int = 42,
        features: list[str] | None = None,
        source_ids: list[int] | None = None,
        conc_column: str = "P",
        gate_threshold: float = 0.01,
    ):
        self.batch_ids = batch_ids
        self.domain_label = domain_label
        self.max_seq_len = max_seq_len
        self.min_T_fraction = min_T_fraction
        self.max_T_fraction = max_T_fraction
        self.samples_per_batch = samples_per_batch
        self.conc_column = conc_column
        self.gate_threshold = gate_threshold

        if features is None:
            features = list(INPUT_FEATURES_EXPANDED)
        self.features = features

        if batches is None:
            batches = load_batches()
        self.batches = batches

        if stats is None:
            if source_ids is None:
                source_ids = batch_ids
            stats = compute_normalization_stats_expanded(
                batches, source_ids, features, window_fraction=1.0
            )
        self.stats = stats

        # Build fitted params lookup
        self.fitted_params = {}
        for _, row in fitted_params_df.iterrows():
            bid = int(row["batch_id"])
            self.fitted_params[bid] = {
                "t_break": float(row["t_break"]),
                "slope": float(row["slope"]),
            }

        # Concentration channel stats from source batches
        self._compute_conc_stats(source_ids or batch_ids)

        # Generate samples
        self.rng = np.random.default_rng(seed)
        self._generate_samples()

    def _compute_conc_stats(self, batch_ids: list[int]):
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
        self.samples = []
        for batch_id in self.batch_ids:
            if batch_id not in self.fitted_params:
                continue
            df = self.batches[batch_id]
            n_steps = len(df)
            times = df["time"].values

            params = self.fitted_params[batch_id]
            t_break = params["t_break"]
            slope = params["slope"]
            has_decline = slope > self.gate_threshold

            # End time of the batch
            t_end = float(times[-1])

            for _ in range(self.samples_per_batch):
                T_frac = self.rng.uniform(self.min_T_fraction, self.max_T_fraction)
                T_idx = int(T_frac * n_steps)
                T_idx = max(10, min(T_idx, n_steps - 1))

                t_cutoff = float(times[T_idx - 1])

                if has_decline:
                    decline_target = 1.0
                    delta_target = (t_break - t_cutoff) / T_MAX_NORM
                    slope_target = slope
                else:
                    decline_target = 0.0
                    delta_target = (t_end - t_cutoff) / T_MAX_NORM
                    slope_target = 0.0

                self.samples.append({
                    "batch_id": batch_id,
                    "T_idx": T_idx,
                    "T_frac": T_idx / n_steps,
                    "t_cutoff": t_cutoff,
                    "decline_target": decline_target,
                    "delta_target": delta_target,
                    "slope_target": slope_target,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        batch_id = sample["batch_id"]
        T_idx = sample["T_idx"]

        df = self.batches[batch_id]

        # Input window [0:T_idx]
        input_df = df.iloc[:T_idx].copy()

        # Preprocess and normalize features
        processed = preprocess_expanded_features(input_df, self.features)
        X_features = normalize_features_expanded(processed, self.stats)
        X_features = np.nan_to_num(X_features, nan=0.0)

        # Build concentration channels
        conc_ffill, conc_mask, conc_time_since = build_concentration_channels(
            input_df, self.conc_column
        )
        conc_ffill_norm = (conc_ffill - self.conc_mean) / self.conc_std
        conc_time_norm = (conc_time_since - self.time_since_mean) / self.time_since_std

        # Stack: (T, n_features + 3)
        X = np.concatenate([
            X_features,
            conc_ffill_norm[:, None],
            conc_mask[:, None],
            conc_time_norm[:, None],
        ], axis=1)

        # Truncate or pad to fixed max_seq_len
        T_actual = X.shape[0]
        n_ch = X.shape[1]
        if T_actual > self.max_seq_len:
            X = X[T_actual - self.max_seq_len:]
        elif T_actual < self.max_seq_len:
            pad = np.zeros((self.max_seq_len - T_actual, n_ch))
            X = np.concatenate([pad, X], axis=0)

        return {
            "x": torch.tensor(X, dtype=torch.float32),
            "T_frac": torch.tensor(sample["T_frac"], dtype=torch.float32),
            "decline_target": torch.tensor(sample["decline_target"], dtype=torch.float32),
            "delta_target": torch.tensor(sample["delta_target"], dtype=torch.float32),
            "slope_target": torch.tensor(sample["slope_target"], dtype=torch.float32),
            "domain_label": torch.tensor(self.domain_label, dtype=torch.long),
            "batch_id": batch_id,
        }

    @property
    def n_features(self) -> int:
        return len(self.stats["features"]) + 3


def decline_collate_fn(batch: list[dict]) -> dict:
    return {
        "x": torch.stack([s["x"] for s in batch]),
        "T_frac": torch.stack([s["T_frac"] for s in batch]),
        "decline_target": torch.stack([s["decline_target"] for s in batch]),
        "delta_target": torch.stack([s["delta_target"] for s in batch]),
        "slope_target": torch.stack([s["slope_target"] for s in batch]),
        "domain_label": torch.stack([s["domain_label"] for s in batch]),
    }


def create_decline_dataloaders(
    source_ids: list[int],
    target_ids: list[int],
    fitted_params_df: pd.DataFrame,
    batches: dict | None = None,
    val_ratio: float = 0.2,
    batch_size: int = 32,
    max_seq_len: int = 500,
    min_T_fraction: float = 0.4,
    max_T_fraction: float = 0.9,
    samples_per_batch: int = 10,
    seed: int = 42,
    features: list[str] | None = None,
    conc_column: str = "P",
    num_workers: int = 0,
    decline_oversample: float = 2.0,
    gate_threshold: float = 0.01,
) -> dict:
    """Create train/val/target DataLoaders for decline forecasting.

    Args:
        source_ids: Source domain batch IDs.
        target_ids: Target domain batch IDs.
        fitted_params_df: Pre-fitted piece-log parameters.
        batches: Dict of batch DataFrames.
        val_ratio: Fraction of source for validation.
        batch_size: Batch size.
        max_seq_len: Fixed sequence length.
        min_T_fraction: Minimum input fraction.
        max_T_fraction: Maximum input fraction.
        samples_per_batch: Samples per batch.
        seed: Random seed.
        features: Input feature columns.
        conc_column: Concentration column name.
        num_workers: DataLoader workers.
        decline_oversample: Weight multiplier for declining batches.
        gate_threshold: Slope threshold for decline classification.

    Returns:
        Dict with 'train', 'val', 'target' DataLoaders and metadata.
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

    dataset_kwargs = dict(
        batches=batches,
        stats=stats,
        max_seq_len=max_seq_len,
        min_T_fraction=min_T_fraction,
        max_T_fraction=max_T_fraction,
        samples_per_batch=samples_per_batch,
        features=features,
        source_ids=train_ids,
        conc_column=conc_column,
        gate_threshold=gate_threshold,
    )

    train_dataset = DeclineForecastDataset(
        batch_ids=train_ids,
        fitted_params_df=fitted_params_df,
        domain_label=0,
        seed=seed,
        **dataset_kwargs,
    )
    val_dataset = DeclineForecastDataset(
        batch_ids=val_ids,
        fitted_params_df=fitted_params_df,
        domain_label=0,
        seed=seed + 1,
        **dataset_kwargs,
    )
    target_dataset = DeclineForecastDataset(
        batch_ids=target_ids,
        fitted_params_df=fitted_params_df,
        domain_label=1,
        seed=seed + 2,
        **dataset_kwargs,
    )

    collate = decline_collate_fn

    # Oversample declining batches via WeightedRandomSampler
    if decline_oversample > 1.0:
        slope_lookup = fitted_params_df.set_index("batch_id")["slope"]
        weights = []
        for sample in train_dataset.samples:
            bid = sample["batch_id"]
            slope = slope_lookup.get(bid, 0.0)
            weights.append(decline_oversample if slope > gate_threshold else 1.0)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            collate_fn=collate, num_workers=num_workers,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate, num_workers=num_workers,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=num_workers,
    )
    target_loader = DataLoader(
        target_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=num_workers,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "target": target_loader,
        "stats": stats,
        "n_features": train_dataset.n_features,
        "max_seq_len": max_seq_len,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "target_dataset": target_dataset,
    }
