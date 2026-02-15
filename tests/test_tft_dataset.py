"""Unit tests for TFT dataset module."""

import numpy as np
import pandas as pd
import torch
import pytest

from src.tft_dataset import (
    interpolate_concentration,
    build_concentration_channels,
    get_target_concentration_at_horizon,
    TFTConcentrationDataset,
    tft_collate_fn,
    tft_collate_uniform,
    GroupedBatchSampler,
    create_tft_dataloaders,
)
from src.data_loader import load_batches


@pytest.fixture(scope="module")
def batches():
    try:
        return load_batches()
    except FileNotFoundError:
        pytest.skip("Data file not found")


def create_dummy_batch(n_steps: int = 100) -> pd.DataFrame:
    """Create a dummy batch DataFrame for testing."""
    time = np.linspace(0, 200, n_steps)
    # Simulate concentration that increases over time
    conc = np.linspace(0, 30, n_steps) + np.random.randn(n_steps) * 0.5
    # Add some features
    return pd.DataFrame({
        "time": time,
        "P": conc,
        "DO2": np.random.randn(n_steps) * 5 + 50,
        "Fg": np.random.randn(n_steps) * 2 + 10,
        "Fs": np.random.randn(n_steps) * 1 + 5,
        "T": np.random.randn(n_steps) * 0.5 + 298,
        "pH": np.random.randn(n_steps) * 0.1 + 6.5,
    })


class TestInterpolateConcentration:
    def test_basic_interpolation(self):
        times = np.array([0, 10, 20, 30])
        conc = np.array([0, 10, 20, 30])
        target_times = np.array([5, 15, 25])
        result = interpolate_concentration(times, conc, target_times)
        # Linear data, so interpolation should give exact values
        assert result.shape == (3,)
        np.testing.assert_allclose(result, [5, 15, 25], rtol=0.1)

    def test_handles_nan_values(self):
        times = np.array([0, 10, 20, 30, 40])
        conc = np.array([0, np.nan, 20, np.nan, 40])
        target_times = np.array([10, 30])
        result = interpolate_concentration(times, conc, target_times)
        assert result.shape == (2,)
        assert np.isfinite(result).all()

    def test_extrapolation(self):
        times = np.array([0, 10, 20])
        conc = np.array([0, 10, 20])
        target_times = np.array([30])  # Beyond data range
        result = interpolate_concentration(times, conc, target_times)
        assert result.shape == (1,)
        assert np.isfinite(result).all()


class TestBuildConcentrationChannels:
    def test_output_shapes(self):
        df = create_dummy_batch(100)
        conc_ffill, conc_mask, conc_time_since = build_concentration_channels(df)
        assert conc_ffill.shape == (100,)
        assert conc_mask.shape == (100,)
        assert conc_time_since.shape == (100,)

    def test_mask_binary(self):
        df = create_dummy_batch(100)
        _, conc_mask, _ = build_concentration_channels(df)
        # Mask should be 0 or 1
        assert set(np.unique(conc_mask)).issubset({0.0, 1.0})

    def test_time_since_non_negative(self):
        df = create_dummy_batch(100)
        _, _, conc_time_since = build_concentration_channels(df)
        assert (conc_time_since >= 0).all()

    def test_ffill_no_nan(self):
        df = create_dummy_batch(100)
        conc_ffill, _, _ = build_concentration_channels(df)
        assert np.isfinite(conc_ffill).all()


class TestGetTargetConcentrationAtHorizon:
    def test_returns_float(self):
        df = create_dummy_batch(100)
        result = get_target_concentration_at_horizon(df, T_idx=50, D_steps=20)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_within_batch(self):
        df = create_dummy_batch(100)
        result = get_target_concentration_at_horizon(df, T_idx=30, D_steps=10)
        # Result should be reasonable (positive, not too large)
        assert result >= 0
        assert result < 100

    def test_extrapolation_beyond_batch(self):
        df = create_dummy_batch(100)
        # Request horizon that extends beyond batch
        result = get_target_concentration_at_horizon(df, T_idx=90, D_steps=50)
        assert np.isfinite(result)


class TestTFTConcentrationDataset:
    def test_creates_dataset(self, batches):
        dataset = TFTConcentrationDataset(
            batch_ids=[1, 2, 3],
            batches=batches,
            samples_per_batch=5,
        )
        assert len(dataset) == 15  # 3 batches * 5 samples

    def test_returns_correct_keys(self, batches):
        dataset = TFTConcentrationDataset(
            batch_ids=[1],
            batches=batches,
            samples_per_batch=1,
        )
        sample = dataset[0]
        assert "x" in sample
        assert "y" in sample
        assert "T_steps" in sample
        assert "D_steps" in sample
        assert "batch_id" in sample

    def test_x_is_tensor(self, batches):
        dataset = TFTConcentrationDataset(
            batch_ids=[1],
            batches=batches,
            samples_per_batch=1,
        )
        sample = dataset[0]
        assert isinstance(sample["x"], torch.Tensor)
        assert sample["x"].dtype == torch.float32

    def test_y_is_scalar_tensor(self, batches):
        dataset = TFTConcentrationDataset(
            batch_ids=[1],
            batches=batches,
            samples_per_batch=1,
        )
        sample = dataset[0]
        assert isinstance(sample["y"], torch.Tensor)
        assert sample["y"].shape == ()

    def test_n_features_includes_conc_channels(self, batches):
        dataset = TFTConcentrationDataset(
            batch_ids=[1],
            batches=batches,
            samples_per_batch=1,
        )
        # n_features should include +3 for concentration channels
        sample = dataset[0]
        assert sample["x"].shape[1] == dataset.n_features
        # Default features (26) + 3 concentration channels = 29
        assert dataset.n_features >= 28

    def test_x_has_no_nan(self, batches):
        dataset = TFTConcentrationDataset(
            batch_ids=[1, 2],
            batches=batches,
            samples_per_batch=5,
        )
        for i in range(len(dataset)):
            sample = dataset[i]
            assert torch.isfinite(sample["x"]).all(), f"Sample {i} has NaN/Inf"

    def test_T_steps_respects_fraction(self, batches):
        dataset = TFTConcentrationDataset(
            batch_ids=[1],
            batches=batches,
            samples_per_batch=10,
            min_T_fraction=0.2,
            max_T_fraction=0.8,
        )
        batch_len = len(batches[1])
        for i in range(len(dataset)):
            sample = dataset[i]
            T_steps = sample["T_steps"]
            assert T_steps >= 10  # Minimum enforced
            assert T_steps <= batch_len - 10  # Maximum enforced

    def test_deterministic_with_seed(self, batches):
        dataset1 = TFTConcentrationDataset(
            batch_ids=[1, 2],
            batches=batches,
            samples_per_batch=5,
            seed=42,
        )
        dataset2 = TFTConcentrationDataset(
            batch_ids=[1, 2],
            batches=batches,
            samples_per_batch=5,
            seed=42,
        )
        # Same seed should produce same samples
        for i in range(len(dataset1)):
            assert dataset1.samples[i] == dataset2.samples[i]


class TestTFTCollateFn:
    def test_pads_to_max_length(self, batches):
        dataset = TFTConcentrationDataset(
            batch_ids=[1, 2],
            batches=batches,
            samples_per_batch=5,
        )
        # Get samples with different lengths
        samples = [dataset[i] for i in range(4)]
        batch = tft_collate_fn(samples)

        max_len = max(s["T_steps"] for s in samples)
        assert batch["x"].shape[1] == max_len

    def test_output_shapes(self, batches):
        dataset = TFTConcentrationDataset(
            batch_ids=[1, 2],
            batches=batches,
            samples_per_batch=5,
        )
        samples = [dataset[i] for i in range(4)]
        batch = tft_collate_fn(samples)

        assert batch["x"].shape[0] == 4
        assert batch["y"].shape == (4,)
        assert batch["mask"].shape[0] == 4
        assert batch["T_steps"].shape == (4,)
        assert batch["D_steps"].shape == (4,)
        assert batch["batch_ids"].shape == (4,)

    def test_mask_correct(self, batches):
        dataset = TFTConcentrationDataset(
            batch_ids=[1],
            batches=batches,
            samples_per_batch=5,
        )
        samples = [dataset[i] for i in range(2)]
        batch = tft_collate_fn(samples)

        for i, sample in enumerate(samples):
            T_steps = sample["T_steps"]
            # Non-padded positions should be False (not masked)
            assert not batch["mask"][i, :T_steps].any()
            # Padded positions (if any) should be True (masked)
            if T_steps < batch["mask"].shape[1]:
                assert batch["mask"][i, T_steps:].all()

    def test_returns_tensors(self, batches):
        dataset = TFTConcentrationDataset(
            batch_ids=[1],
            batches=batches,
            samples_per_batch=2,
        )
        samples = [dataset[i] for i in range(2)]
        batch = tft_collate_fn(samples)

        assert isinstance(batch["x"], torch.Tensor)
        assert isinstance(batch["y"], torch.Tensor)
        assert isinstance(batch["mask"], torch.Tensor)
        assert batch["mask"].dtype == torch.bool


class TestCreateTFTDataloaders:
    def test_returns_expected_keys(self, batches):
        result = create_tft_dataloaders(
            source_ids=list(range(1, 11)),
            target_ids=list(range(61, 66)),
            batches=batches,
            batch_size=4,
            samples_per_batch=3,
        )
        assert "train" in result
        assert "val" in result
        assert "target" in result
        assert "stats" in result
        assert "collate_fn" in result
        assert "n_features" in result

    def test_dataloaders_iterate(self, batches):
        result = create_tft_dataloaders(
            source_ids=list(range(1, 11)),
            target_ids=list(range(61, 66)),
            batches=batches,
            batch_size=4,
            samples_per_batch=3,
        )
        # Check we can iterate
        for batch in result["train"]:
            assert batch["x"].ndim == 3  # (batch, seq_len, features)
            assert batch["y"].ndim == 1  # (batch,)
            assert batch["mask"].ndim == 2  # (batch, seq_len)
            break

    def test_val_split(self, batches):
        source_ids = list(range(1, 21))  # 20 batches
        result = create_tft_dataloaders(
            source_ids=source_ids,
            target_ids=list(range(61, 66)),
            batches=batches,
            val_ratio=0.2,
            batch_size=4,
            samples_per_batch=5,
        )
        # With 20 source and 0.2 val_ratio, expect 4 val batches, 16 train batches
        # Each batch has 5 samples
        assert len(result["val_dataset"]) == 4 * 5
        assert len(result["train_dataset"]) == 16 * 5

    def test_n_features_correct(self, batches):
        result = create_tft_dataloaders(
            source_ids=list(range(1, 6)),
            target_ids=list(range(61, 66)),
            batches=batches,
            batch_size=2,
            samples_per_batch=2,
        )
        # n_features should match what's returned by dataset
        for batch in result["train"]:
            assert batch["x"].shape[2] == result["n_features"]
            break


class TestIntegrationWithModel:
    """Test that dataset output works with TFT model."""

    def test_compatible_with_tft_model(self, batches):
        from src.tft_model import TemporalFusionTransformer

        result = create_tft_dataloaders(
            source_ids=list(range(1, 6)),
            target_ids=list(range(61, 66)),
            batches=batches,
            batch_size=4,
            samples_per_batch=3,
        )

        model = TemporalFusionTransformer(
            n_features=result["n_features"],
            d_model=32,
            d_hidden=32,
        )

        # Get a batch and pass through model
        for batch in result["train"]:
            x = batch["x"]
            mask = batch["mask"]
            D_steps = batch["D_steps"]

            # Model expects (B, T, C), horizon as (B,), and optional mask
            mu, sigma = model(x, D_steps, mask=mask)

            assert mu.shape == (x.shape[0],)
            assert sigma.shape == (x.shape[0],)
            assert (sigma > 0).all()
            break

    def test_training_step_works(self, batches):
        from src.tft_model import TemporalFusionTransformer, GaussianNLLLoss

        result = create_tft_dataloaders(
            source_ids=list(range(1, 6)),
            target_ids=list(range(61, 66)),
            batches=batches,
            batch_size=4,
            samples_per_batch=3,
        )

        model = TemporalFusionTransformer(
            n_features=result["n_features"],
            d_model=32,
            d_hidden=32,
        )
        loss_fn = GaussianNLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        for batch in result["train"]:
            optimizer.zero_grad()
            mu, sigma = model(batch["x"], batch["D_steps"], mask=batch["mask"])
            loss = loss_fn(mu, sigma, batch["y"])
            loss.backward()
            optimizer.step()

            assert torch.isfinite(loss)
            break


# --- Grouped sampling tests ---


class TestGenerateGroupedSamples:
    """Tests for _generate_grouped_samples and n_T_values parameter."""

    def test_groups_created(self, batches):
        dataset = TFTConcentrationDataset(
            batch_ids=[1, 2, 3],
            batches=batches,
            samples_per_batch=10,
            n_T_values=4,
        )
        assert dataset.t_groups is not None
        assert len(dataset.t_groups) > 0
        assert len(dataset.t_groups) <= 4

    def test_uniform_T_within_groups(self, batches):
        dataset = TFTConcentrationDataset(
            batch_ids=[1, 2, 3],
            batches=batches,
            samples_per_batch=10,
            n_T_values=4,
        )
        for t_idx, indices in dataset.t_groups.items():
            for idx in indices:
                assert dataset.samples[idx]["T_idx"] == t_idx

    def test_all_samples_in_groups(self, batches):
        dataset = TFTConcentrationDataset(
            batch_ids=[1, 2, 3],
            batches=batches,
            samples_per_batch=10,
            n_T_values=4,
        )
        all_indices = []
        for indices in dataset.t_groups.values():
            all_indices.extend(indices)
        assert sorted(all_indices) == list(range(len(dataset)))

    def test_deterministic_with_seed(self, batches):
        ds1 = TFTConcentrationDataset(
            batch_ids=[1, 2, 3],
            batches=batches,
            samples_per_batch=10,
            n_T_values=4,
            seed=42,
        )
        ds2 = TFTConcentrationDataset(
            batch_ids=[1, 2, 3],
            batches=batches,
            samples_per_batch=10,
            n_T_values=4,
            seed=42,
        )
        assert list(ds1.t_groups.keys()) == list(ds2.t_groups.keys())
        for i in range(len(ds1)):
            assert ds1.samples[i] == ds2.samples[i]

    def test_default_none_preserves_old_behavior(self, batches):
        dataset = TFTConcentrationDataset(
            batch_ids=[1, 2],
            batches=batches,
            samples_per_batch=5,
        )
        assert dataset.n_T_values is None
        assert dataset.t_groups is None
        assert len(dataset) == 10  # 2 batches * 5 samples

    def test_n_T_values_clamped_to_range(self, batches):
        """Requesting more T values than available should be clamped."""
        dataset = TFTConcentrationDataset(
            batch_ids=[1],
            batches=batches,
            samples_per_batch=10,
            n_T_values=100000,  # Way more than possible
        )
        # Should still work, just use as many as available
        assert dataset.t_groups is not None
        assert len(dataset.t_groups) > 0


class TestGroupedBatchSampler:
    """Tests for GroupedBatchSampler."""

    def test_correct_batch_count(self):
        t_groups = {100: list(range(10)), 200: list(range(10, 25))}
        sampler = GroupedBatchSampler(t_groups, batch_size=4, shuffle=False)
        batches = list(sampler)
        # Group 100: 10 items / 4 = 3 batches; Group 200: 15 items / 4 = 4 batches
        assert len(batches) == 7
        assert len(sampler) == 7

    def test_no_cross_group_mixing(self):
        t_groups = {100: list(range(10)), 200: list(range(10, 25))}
        sampler = GroupedBatchSampler(t_groups, batch_size=4, shuffle=False)
        group_100_set = set(t_groups[100])
        group_200_set = set(t_groups[200])
        for batch in sampler:
            in_100 = any(idx in group_100_set for idx in batch)
            in_200 = any(idx in group_200_set for idx in batch)
            # Each batch should be from exactly one group
            assert in_100 != in_200, "Batch mixes samples from different groups"

    def test_all_indices_yielded(self):
        t_groups = {100: list(range(10)), 200: list(range(10, 25))}
        sampler = GroupedBatchSampler(t_groups, batch_size=4, shuffle=False)
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)
        assert sorted(all_indices) == list(range(25))

    def test_shuffle_changes_order(self):
        t_groups = {100: list(range(20)), 200: list(range(20, 40))}
        s1 = GroupedBatchSampler(t_groups, batch_size=4, shuffle=True, seed=1)
        s2 = GroupedBatchSampler(t_groups, batch_size=4, shuffle=True, seed=2)
        b1 = list(s1)
        b2 = list(s2)
        # Different seeds should produce different orderings
        assert b1 != b2

    def test_set_epoch(self):
        t_groups = {100: list(range(20))}
        sampler = GroupedBatchSampler(t_groups, batch_size=4, shuffle=True, seed=42)
        sampler.set_epoch(0)
        b0 = list(sampler)
        sampler.set_epoch(1)
        b1 = list(sampler)
        assert b0 != b1


class TestTFTCollateUniform:
    """Tests for tft_collate_uniform."""

    def test_stacks_correctly(self, batches):
        dataset = TFTConcentrationDataset(
            batch_ids=[1, 2, 3],
            batches=batches,
            samples_per_batch=10,
            n_T_values=2,
        )
        # Pick samples from a single group
        t_idx = list(dataset.t_groups.keys())[0]
        indices = dataset.t_groups[t_idx][:3]
        samples = [dataset[i] for i in indices]
        result = tft_collate_uniform(samples)
        assert result["x"].shape[0] == 3
        assert result["x"].shape[1] == t_idx  # All same length

    def test_correct_keys(self, batches):
        dataset = TFTConcentrationDataset(
            batch_ids=[1],
            batches=batches,
            samples_per_batch=10,
            n_T_values=2,
        )
        t_idx = list(dataset.t_groups.keys())[0]
        indices = dataset.t_groups[t_idx][:2]
        samples = [dataset[i] for i in indices]
        result = tft_collate_uniform(samples)
        assert set(result.keys()) == {"x", "y", "mask", "T_steps", "D_steps", "batch_ids"}

    def test_mask_all_false(self, batches):
        dataset = TFTConcentrationDataset(
            batch_ids=[1, 2],
            batches=batches,
            samples_per_batch=10,
            n_T_values=2,
        )
        t_idx = list(dataset.t_groups.keys())[0]
        indices = dataset.t_groups[t_idx][:3]
        samples = [dataset[i] for i in indices]
        result = tft_collate_uniform(samples)
        assert not result["mask"].any(), "Mask should be all-False (no padding)"
        assert result["mask"].dtype == torch.bool


class TestGroupedIntegration:
    """End-to-end tests for grouped batching with DataLoader and model."""

    def test_dataloader_yields_uniform_lengths(self, batches):
        result = create_tft_dataloaders(
            source_ids=list(range(1, 11)),
            target_ids=list(range(61, 66)),
            batches=batches,
            batch_size=4,
            samples_per_batch=10,
            n_T_values=4,
        )
        for batch in result["train"]:
            # All T_steps within a batch should be the same
            assert (batch["T_steps"] == batch["T_steps"][0]).all()
            # Mask should be all-False
            assert not batch["mask"].any()
            break

    def test_model_forward_pass(self, batches):
        from src.tft_model import TemporalFusionTransformer

        result = create_tft_dataloaders(
            source_ids=list(range(1, 6)),
            target_ids=list(range(61, 66)),
            batches=batches,
            batch_size=4,
            samples_per_batch=10,
            n_T_values=4,
        )
        model = TemporalFusionTransformer(
            n_features=result["n_features"],
            d_model=32,
            d_hidden=32,
        )
        for batch in result["train"]:
            mu, sigma = model(batch["x"], batch["D_steps"], mask=batch["mask"])
            assert mu.shape == (batch["x"].shape[0],)
            assert sigma.shape == (batch["x"].shape[0],)
            assert (sigma > 0).all()
            break

    def test_training_step(self, batches):
        from src.tft_model import TemporalFusionTransformer, GaussianNLLLoss

        result = create_tft_dataloaders(
            source_ids=list(range(1, 6)),
            target_ids=list(range(61, 66)),
            batches=batches,
            batch_size=4,
            samples_per_batch=10,
            n_T_values=4,
        )
        model = TemporalFusionTransformer(
            n_features=result["n_features"],
            d_model=32,
            d_hidden=32,
        )
        loss_fn = GaussianNLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        for batch in result["train"]:
            optimizer.zero_grad()
            mu, sigma = model(batch["x"], batch["D_steps"], mask=batch["mask"])
            loss = loss_fn(mu, sigma, batch["y"])
            loss.backward()
            optimizer.step()
            assert torch.isfinite(loss)
            break

    def test_collate_fn_in_result(self, batches):
        result = create_tft_dataloaders(
            source_ids=list(range(1, 6)),
            target_ids=list(range(61, 66)),
            batches=batches,
            batch_size=4,
            samples_per_batch=10,
            n_T_values=4,
        )
        assert result["collate_fn"] is tft_collate_uniform
