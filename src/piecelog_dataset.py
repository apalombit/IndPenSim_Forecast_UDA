"""PyTorch Dataset for PieceLog-PatchTST with (T, D) sampling and fitted parameters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .data_loader import load_batches
from .feature_config import INPUT_FEATURES_EXPANDED
from .piecelog_model import PARAM_NAMES, piecelog_numpy
from .preprocessing import (
    compute_normalization_stats_expanded,
    preprocess_expanded_features,
    normalize_features_expanded,
)
from .tft_dataset import (
    build_concentration_channels,
    interpolate_concentration,
)


def compute_param_stats(
    fitted_params_df: pd.DataFrame,
    batch_ids: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute mean/std per piece-log parameter for z-score normalization.

    Stats are computed from the specified batches (typically source/train only).

    Args:
        fitted_params_df: DataFrame with columns batch_id, K, r, t0, etc.
        batch_ids: Batches to compute stats from. If None, uses all rows.

    Returns:
        Dict like {"K": {"mean": 37.06, "std": 9.02}, ...}.
    """
    df = fitted_params_df
    if batch_ids is not None:
        df = df[df["batch_id"].isin(batch_ids)]
    stats = {}
    for name in PARAM_NAMES:
        mean = float(df[name].mean())
        std = float(df[name].std())
        std = max(std, 1e-6)
        stats[name] = {"mean": mean, "std": std}
    return stats


class PieceLogDataset(Dataset):
    """Dataset with (T, D) sampling, concentration channels, and fitted piece-log params.

    Each sample:
        x:            (seq_len, 28) — 26 process signals + 3 concentration channels
        t_predict:    scalar, absolute time T+D (hours)
        y_conc:       scalar, true P(T+D)
        params_fitted: (7,) pre-fitted piece-log parameters
        domain_label: 0 (source) / 1 (target)
    """

    def __init__(
        self,
        batch_ids: list[int],
        fitted_params_df: pd.DataFrame,
        domain_label: int = 0,
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
        augment: bool = False,
        signal_noise_std: float = 0.05,
        param_noise_std: float = 0.05,
    ):
        """Initialize PieceLog dataset.

        Args:
            batch_ids: Batch IDs to include.
            fitted_params_df: DataFrame with columns batch_id, K, r, t0, lam,
                t_lag, t_break, slope from pre-fitting.
            domain_label: 0 for source, 1 for target.
            batches: Dict of batch DataFrames. Loaded if None.
            stats: Normalization statistics. Computed from source_ids if None.
            max_seq_len: Fixed sequence length. Samples are truncated (keeping
                the last max_seq_len steps) or zero-padded to this length.
                The model's seq_len should match this value.
            min_T_fraction: Minimum input window fraction.
            max_T_fraction: Maximum input window fraction.
            min_D_hours: Minimum prediction horizon (hours).
            max_D_hours: Maximum prediction horizon (hours).
            samples_per_batch: Number of (T, D) samples per batch.
            seed: Random seed.
            features: Input feature columns.
            source_ids: Source batch IDs for computing stats.
            conc_column: Concentration column name.
        """
        self.batch_ids = batch_ids
        self.domain_label = domain_label
        self.max_seq_len = max_seq_len
        self.min_T_fraction = min_T_fraction
        self.max_T_fraction = max_T_fraction
        self.min_D_hours = min_D_hours
        self.max_D_hours = max_D_hours
        self.samples_per_batch = samples_per_batch
        self.conc_column = conc_column
        self.augment = augment
        self.signal_noise_std = signal_noise_std
        self.param_noise_std = param_noise_std
        self.aug_rng = np.random.default_rng() if augment else None

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
            self.fitted_params[bid] = np.array(
                [row[p] for p in PARAM_NAMES], dtype=np.float32
            )

        # Compute concentration channel stats
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

            dt = (times[1] - times[0]) if len(times) > 1 else 1.0

            for _ in range(self.samples_per_batch):
                T_frac = self.rng.uniform(self.min_T_fraction, self.max_T_fraction)
                T_idx = int(T_frac * n_steps)
                T_idx = max(10, min(T_idx, n_steps - 10))

                D_hours = self.rng.uniform(self.min_D_hours, self.max_D_hours)
                D_steps = int(D_hours / dt)

                max_D_steps = n_steps - T_idx - 1
                if D_steps > max_D_steps:
                    D_steps = max(1, max_D_steps)

                self.samples.append({
                    "batch_id": batch_id,
                    "T_idx": T_idx,
                    "D_steps": D_steps,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.

        Returns:
            Dict with keys: x, t_predict, y_conc, params_fitted, domain_label.
        """
        sample = self.samples[idx]
        batch_id = sample["batch_id"]
        T_idx = sample["T_idx"]
        D_steps = sample["D_steps"]

        df = self.batches[batch_id]
        times = df["time"].values

        # Input window [0:T_idx]
        input_df = df.iloc[:T_idx].copy()

        # Preprocess and normalize features
        processed = preprocess_expanded_features(input_df, self.features)
        X_features = normalize_features_expanded(processed, self.stats)

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
            # Keep the last max_seq_len steps (most recent signals)
            X = X[T_actual - self.max_seq_len:]
        elif T_actual < self.max_seq_len:
            # Zero-pad at the start
            pad = np.zeros((self.max_seq_len - T_actual, n_ch))
            X = np.concatenate([pad, X], axis=0)

        # Add signal noise to process features only (not concentration channels)
        if self.augment:
            n_process = len(self.stats["features"])
            noise = self.aug_rng.normal(
                0, self.signal_noise_std, size=(X.shape[0], n_process)
            ).astype(np.float32)
            X[:, :n_process] += noise

        # Absolute prediction time
        dt = (times[1] - times[0]) if len(times) > 1 else 1.0
        t_predict = times[T_idx - 1] + D_steps * dt

        # Fitted parameters for this batch
        params_fitted = self.fitted_params[batch_id].copy()

        if self.augment:
            # Perturb params multiplicatively
            perturbation = self.aug_rng.normal(
                0, self.param_noise_std, size=7
            ).astype(np.float32)
            params_fitted *= 1.0 + perturbation

            # Clamp to physical validity
            params_fitted[0] = max(params_fitted[0], 1e-3)      # K > 0
            params_fitted[1] = max(params_fitted[1], 1e-4)      # r > 0
            params_fitted[2] = max(params_fitted[2], 0.0)       # t0 >= 0
            params_fitted[3] = max(params_fitted[3], 1e-4)      # lam > 0
            params_fitted[4] = max(params_fitted[4], 0.0)       # t_lag >= 0
            params_fitted[5] = max(params_fitted[5], params_fitted[4] + 1.0)  # t_break > t_lag
            params_fitted[6] = max(params_fitted[6], 0.0)       # slope >= 0

            # Recompute y_conc from perturbed params (self-consistent target)
            y_conc = max(
                float(piecelog_numpy(np.array([t_predict]), *params_fitted)[0]),
                0.0,
            )
        else:
            # Target concentration at t_predict via interpolation
            conc = df[self.conc_column].values
            y_conc = float(interpolate_concentration(
                times, conc, np.array([t_predict])
            )[0])

        return {
            "x": torch.tensor(X, dtype=torch.float32),
            "t_predict": torch.tensor(t_predict, dtype=torch.float32),
            "y_conc": torch.tensor(y_conc, dtype=torch.float32),
            "params_fitted": torch.tensor(params_fitted, dtype=torch.float32),
            "domain_label": torch.tensor(self.domain_label, dtype=torch.long),
            "batch_id": batch_id,
        }

    @property
    def n_features(self) -> int:
        return len(self.stats["features"]) + 3


def piecelog_collate_fn(batch: list[dict]) -> dict:
    """Collate fixed-length sequences (all already truncated/padded to max_seq_len).

    Args:
        batch: List of sample dicts from PieceLogDataset.

    Returns:
        Dict with stacked tensors.
    """
    return {
        "x": torch.stack([s["x"] for s in batch]),
        "t_predict": torch.stack([s["t_predict"] for s in batch]),
        "y_conc": torch.stack([s["y_conc"] for s in batch]),
        "params_fitted": torch.stack([s["params_fitted"] for s in batch]),
        "domain_label": torch.stack([s["domain_label"] for s in batch]),
    }


def create_piecelog_dataloaders(
    source_ids: list[int],
    target_ids: list[int],
    fitted_params_df: pd.DataFrame,
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
    augment: bool = False,
    signal_noise_std: float = 0.05,
    param_noise_std: float = 0.05,
) -> dict:
    """Create train/val/target DataLoaders for PieceLog-PatchTST.

    Args:
        source_ids: Source domain batch IDs.
        target_ids: Target domain batch IDs.
        fitted_params_df: Pre-fitted piece-log parameters.
        batches: Dict of batch DataFrames.
        val_ratio: Fraction of source for validation.
        batch_size: Batch size.
        max_seq_len: Fixed sequence length for all samples. Must match
            the model's seq_len. Samples longer than this are truncated
            (keeping last max_seq_len steps); shorter ones are zero-padded.
        min_T_fraction: Minimum input fraction.
        max_T_fraction: Maximum input fraction.
        min_D_hours: Minimum horizon (hours).
        max_D_hours: Maximum horizon (hours).
        samples_per_batch: Samples per batch.
        seed: Random seed.
        features: Input feature columns.
        conc_column: Concentration column name.
        num_workers: DataLoader workers.

    Returns:
        Dict with 'train', 'val', 'target' DataLoaders, 'stats', and 'max_seq_len'.
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
        min_D_hours=min_D_hours,
        max_D_hours=max_D_hours,
        samples_per_batch=samples_per_batch,
        features=features,
        source_ids=train_ids,
        conc_column=conc_column,
    )

    train_dataset = PieceLogDataset(
        batch_ids=train_ids,
        fitted_params_df=fitted_params_df,
        domain_label=0,
        seed=seed,
        augment=augment,
        signal_noise_std=signal_noise_std,
        param_noise_std=param_noise_std,
        **dataset_kwargs,
    )
    val_dataset = PieceLogDataset(
        batch_ids=val_ids,
        fitted_params_df=fitted_params_df,
        domain_label=0,
        seed=seed + 1,
        **dataset_kwargs,
    )
    target_dataset = PieceLogDataset(
        batch_ids=target_ids,
        fitted_params_df=fitted_params_df,
        domain_label=1,
        seed=seed + 2,
        **dataset_kwargs,
    )

    # Compute param stats from training split for z-score normalization
    param_stats = compute_param_stats(fitted_params_df, train_ids)

    collate = piecelog_collate_fn

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
        "param_stats": param_stats,
        "conc_scale": train_dataset.conc_std,
        "n_features": train_dataset.n_features,
        "max_seq_len": max_seq_len,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "target_dataset": target_dataset,
    }
