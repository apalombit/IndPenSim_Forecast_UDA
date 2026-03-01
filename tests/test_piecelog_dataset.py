"""Tests for PieceLogDataset augmentation behavior."""

import numpy as np
import pandas as pd
import pytest

from src.feature_config import INPUT_FEATURES_EXPANDED
from src.piecelog_dataset import PieceLogDataset, create_piecelog_dataloaders
from src.piecelog_model import PARAM_NAMES, piecelog_numpy
from src.preprocessing import compute_normalization_stats_expanded


# Reference piece-log parameters (realistic values)
REF_PARAMS = dict(K=40.0, r=0.02, t0=50.0, lam=0.05, t_lag=25.0, t_break=200.0, slope=0.05)


def _make_synthetic_batch(batch_id: int, n_steps: int = 200, seed: int = 0) -> pd.DataFrame:
    """Create a synthetic batch with all 26 expanded features + P + time."""
    rng = np.random.default_rng(seed + batch_id)
    time = np.linspace(0, 300, n_steps)

    # Generate concentration from piece-log model
    conc = piecelog_numpy(time, **REF_PARAMS)

    data = {"time": time, "P": conc}
    for feat in INPUT_FEATURES_EXPANDED:
        data[feat] = rng.normal(0, 1, n_steps).astype(np.float32)
    return pd.DataFrame(data)


def _make_batches_and_params(n_batches: int = 5):
    """Create synthetic batches dict and fitted_params_df."""
    batch_ids = list(range(1, n_batches + 1))
    batches = {bid: _make_synthetic_batch(bid) for bid in batch_ids}

    rows = []
    for bid in batch_ids:
        row = {"batch_id": bid, **REF_PARAMS}
        rows.append(row)
    fitted_params_df = pd.DataFrame(rows)

    features = list(INPUT_FEATURES_EXPANDED)
    stats = compute_normalization_stats_expanded(
        batches, batch_ids, features, window_fraction=1.0
    )
    return batches, fitted_params_df, stats, features, batch_ids


@pytest.fixture
def synth():
    """Shared synthetic data for tests."""
    batches, fitted_params_df, stats, features, batch_ids = _make_batches_and_params()
    return dict(
        batches=batches,
        fitted_params_df=fitted_params_df,
        stats=stats,
        features=features,
        batch_ids=batch_ids,
    )


def _make_dataset(synth, augment=False, signal_noise_std=0.05, param_noise_std=0.05):
    return PieceLogDataset(
        batch_ids=synth["batch_ids"],
        fitted_params_df=synth["fitted_params_df"],
        batches=synth["batches"],
        stats=synth["stats"],
        features=synth["features"],
        max_seq_len=100,
        samples_per_batch=3,
        seed=42,
        augment=augment,
        signal_noise_std=signal_noise_std,
        param_noise_std=param_noise_std,
    )


class TestAugmentation:
    def test_augment_false_is_deterministic(self, synth):
        ds = _make_dataset(synth, augment=False)
        s1 = ds[0]
        s2 = ds[0]
        np.testing.assert_array_equal(s1["x"].numpy(), s2["x"].numpy())
        assert s1["y_conc"].item() == s2["y_conc"].item()

    def test_augment_true_varies_across_calls(self, synth):
        ds = _make_dataset(synth, augment=True)
        s1 = ds[0]
        s2 = ds[0]
        # Signal noise makes x differ
        assert not np.array_equal(s1["x"].numpy(), s2["x"].numpy())

    def test_noise_only_on_process_features(self, synth):
        ds = _make_dataset(synth, augment=True, signal_noise_std=0.5)
        s1 = ds[0]
        s2 = ds[0]

        x1 = s1["x"].numpy()
        x2 = s2["x"].numpy()

        # conc_mask is column index 26 (0-indexed: 25 process + conc_ffill=25, mask=26)
        # It should be binary (0 or 1) — noise should NOT affect it
        n_process = len(synth["features"])  # 26
        # Concentration channels are the last 3 columns
        mask_col = n_process + 1  # conc_mask column
        mask1 = x1[:, mask_col]
        mask2 = x2[:, mask_col]
        # Mask values should be identical across calls (no noise added)
        np.testing.assert_array_equal(mask1, mask2)
        # Mask values should be 0 or 1
        assert set(np.unique(mask1)).issubset({0.0, 1.0})

    def test_params_perturbed_but_close(self, synth):
        ds_no_aug = _make_dataset(synth, augment=False)
        ds_aug = _make_dataset(synth, augment=True, param_noise_std=0.05)

        orig = ds_no_aug[0]["params_fitted"].numpy()
        perturbed = ds_aug[0]["params_fitted"].numpy()

        # Params should differ
        assert not np.array_equal(orig, perturbed)
        # But stay within ~20% (3 sigma at 5% noise)
        ratio = perturbed / np.where(orig > 1e-6, orig, 1e-6)
        assert np.all(ratio > 0.5) and np.all(ratio < 2.0)

    def test_params_physically_valid(self, synth):
        ds = _make_dataset(synth, augment=True, param_noise_std=0.1)
        for _ in range(20):
            sample = ds[0]
            p = sample["params_fitted"].numpy()
            assert p[0] >= 1e-3, f"K should be > 0, got {p[0]}"
            assert p[1] >= 1e-4, f"r should be > 0, got {p[1]}"
            assert p[2] >= 0.0, f"t0 should be >= 0, got {p[2]}"
            assert p[3] >= 1e-4, f"lam should be > 0, got {p[3]}"
            assert p[4] >= 0.0, f"t_lag should be >= 0, got {p[4]}"
            assert p[5] > p[4], f"t_break ({p[5]}) should be > t_lag ({p[4]})"
            assert p[6] >= 0.0, f"slope should be >= 0, got {p[6]}"

    def test_y_conc_non_negative(self, synth):
        ds = _make_dataset(synth, augment=True, param_noise_std=0.1)
        for _ in range(20):
            sample = ds[0]
            assert sample["y_conc"].item() >= 0.0

    def test_original_params_not_mutated(self, synth):
        ds = _make_dataset(synth, augment=True, param_noise_std=0.1)
        bid = ds.samples[0]["batch_id"]
        original = ds.fitted_params[bid].copy()

        # Access multiple times (each triggers augmentation with .copy())
        for _ in range(10):
            ds[0]

        np.testing.assert_array_equal(ds.fitted_params[bid], original)


class TestTCutoff:
    def test_t_cutoff_present_in_sample(self, synth):
        ds = _make_dataset(synth, augment=False)
        sample = ds[0]
        assert "t_cutoff" in sample

    def test_t_cutoff_less_than_t_predict(self, synth):
        ds = _make_dataset(synth, augment=False)
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["t_cutoff"].item() < sample["t_predict"].item()

    def test_t_cutoff_positive(self, synth):
        ds = _make_dataset(synth, augment=False)
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["t_cutoff"].item() > 0.0

    def test_t_cutoff_in_collated_batch(self, synth):
        from src.piecelog_dataset import piecelog_collate_fn
        ds = _make_dataset(synth, augment=False)
        batch = piecelog_collate_fn([ds[i] for i in range(min(4, len(ds)))])
        assert "t_cutoff" in batch
        assert batch["t_cutoff"].shape[0] == min(4, len(ds))


class TestFullBatch:
    def test_one_sample_per_batch(self, synth):
        ds = PieceLogDataset(
            batch_ids=synth["batch_ids"],
            fitted_params_df=synth["fitted_params_df"],
            batches=synth["batches"],
            stats=synth["stats"],
            features=synth["features"],
            max_seq_len=200,
            samples_per_batch=5,  # should be ignored
            seed=0,
            full_batch=True,
        )
        # One sample per batch that has fitted params
        n_batches_with_params = sum(
            1 for bid in synth["batch_ids"] if bid in ds.fitted_params
        )
        assert len(ds) == n_batches_with_params

    def test_T_idx_equals_n_steps(self, synth):
        ds = PieceLogDataset(
            batch_ids=synth["batch_ids"],
            fitted_params_df=synth["fitted_params_df"],
            batches=synth["batches"],
            stats=synth["stats"],
            features=synth["features"],
            max_seq_len=200,
            seed=0,
            full_batch=True,
        )
        for sample_meta in ds.samples:
            bid = sample_meta["batch_id"]
            n_steps = len(synth["batches"][bid])
            assert sample_meta["T_idx"] == n_steps
            assert sample_meta["D_steps"] == 0

    def test_t_cutoff_equals_t_predict(self, synth):
        ds = PieceLogDataset(
            batch_ids=synth["batch_ids"],
            fitted_params_df=synth["fitted_params_df"],
            batches=synth["batches"],
            stats=synth["stats"],
            features=synth["features"],
            max_seq_len=200,
            seed=0,
            full_batch=True,
        )
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["t_cutoff"].item() == pytest.approx(
                sample["t_predict"].item(), rel=1e-5
            )

    def test_t_cutoff_is_last_timestamp(self, synth):
        ds = PieceLogDataset(
            batch_ids=synth["batch_ids"],
            fitted_params_df=synth["fitted_params_df"],
            batches=synth["batches"],
            stats=synth["stats"],
            features=synth["features"],
            max_seq_len=200,
            seed=0,
            full_batch=True,
        )
        for i in range(len(ds)):
            bid = ds.samples[i]["batch_id"]
            last_t = float(synth["batches"][bid]["time"].values[-1])
            assert ds[i]["t_cutoff"].item() == pytest.approx(last_t, rel=1e-5)

    def test_t_cutoff_gte_t_break(self, synth):
        """Full-batch t_cutoff should be >= t_break for all batches."""
        ds = PieceLogDataset(
            batch_ids=synth["batch_ids"],
            fitted_params_df=synth["fitted_params_df"],
            batches=synth["batches"],
            stats=synth["stats"],
            features=synth["features"],
            max_seq_len=200,
            seed=0,
            full_batch=True,
        )
        t_break_lookup = synth["fitted_params_df"].set_index("batch_id")["t_break"]
        for i in range(len(ds)):
            bid = ds.samples[i]["batch_id"]
            t_cutoff = ds[i]["t_cutoff"].item()
            t_break = float(t_break_lookup[bid])
            assert t_cutoff >= t_break, (
                f"Batch {bid}: t_cutoff={t_cutoff:.1f} < t_break={t_break:.1f}"
            )

    def test_x_shape_is_max_seq_len(self, synth):
        ds = PieceLogDataset(
            batch_ids=synth["batch_ids"],
            fitted_params_df=synth["fitted_params_df"],
            batches=synth["batches"],
            stats=synth["stats"],
            features=synth["features"],
            max_seq_len=150,
            seed=0,
            full_batch=True,
        )
        for i in range(len(ds)):
            x = ds[i]["x"]
            assert x.shape[0] == 150

    def test_create_dataloaders_full_batch(self, synth):
        loaders = create_piecelog_dataloaders(
            source_ids=synth["batch_ids"][:3],
            target_ids=synth["batch_ids"][3:],
            fitted_params_df=synth["fitted_params_df"],
            batches=synth["batches"],
            features=synth["features"],
            max_seq_len=200,
            samples_per_batch=5,  # should be ignored
            full_batch=True,
        )
        train_ds = loaders["train_dataset"]
        val_ds = loaders["val_dataset"]
        target_ds = loaders["target_dataset"]
        # Each sample should have T_idx == n_steps
        for meta in train_ds.samples:
            n_steps = len(synth["batches"][meta["batch_id"]])
            assert meta["T_idx"] == n_steps
            assert meta["D_steps"] == 0
        for meta in val_ds.samples:
            n_steps = len(synth["batches"][meta["batch_id"]])
            assert meta["T_idx"] == n_steps
        for meta in target_ds.samples:
            n_steps = len(synth["batches"][meta["batch_id"]])
            assert meta["T_idx"] == n_steps


class TestCreateDataloadersAugment:
    def test_augment_only_on_train(self, synth):
        loaders = create_piecelog_dataloaders(
            source_ids=synth["batch_ids"][:3],
            target_ids=synth["batch_ids"][3:],
            fitted_params_df=synth["fitted_params_df"],
            batches=synth["batches"],
            features=synth["features"],
            max_seq_len=100,
            samples_per_batch=3,
            augment=True,
        )
        assert loaders["train_dataset"].augment is True
        assert loaders["val_dataset"].augment is False
        assert loaders["target_dataset"].augment is False

    def test_augment_params_forwarded(self, synth):
        loaders = create_piecelog_dataloaders(
            source_ids=synth["batch_ids"][:3],
            target_ids=synth["batch_ids"][3:],
            fitted_params_df=synth["fitted_params_df"],
            batches=synth["batches"],
            features=synth["features"],
            max_seq_len=100,
            samples_per_batch=3,
            augment=True,
            signal_noise_std=0.1,
            param_noise_std=0.2,
        )
        train_ds = loaders["train_dataset"]
        assert train_ds.signal_noise_std == 0.1
        assert train_ds.param_noise_std == 0.2
