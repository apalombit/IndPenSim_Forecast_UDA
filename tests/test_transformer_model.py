"""Unit tests for transformer_model module."""

import torch
import pytest

from src.transformer_model import (
    PatchEmbedding,
    PositionalEncoding,
    TransformerEncoder,
    PatchTSTRegressor,
    count_parameters,
    create_model,
)


class TestPatchEmbedding:
    def test_output_shape(self):
        embed = PatchEmbedding(n_features=11, patch_len=16, patch_stride=8, d_model=64)
        x = torch.randn(4, 100, 11)  # (B, T, C)
        out = embed(x)

        expected_n_patches = (100 - 16) // 8 + 1  # 11 patches
        assert out.shape == (4, expected_n_patches, 64)

    def test_get_num_patches(self):
        embed = PatchEmbedding(n_features=11, patch_len=16, patch_stride=8, d_model=64)
        assert embed.get_num_patches(100) == 11
        assert embed.get_num_patches(50) == 5
        assert embed.get_num_patches(16) == 1

    def test_different_patch_sizes(self):
        embed1 = PatchEmbedding(n_features=11, patch_len=32, patch_stride=16, d_model=64)
        embed2 = PatchEmbedding(n_features=11, patch_len=8, patch_stride=4, d_model=64)

        x = torch.randn(2, 100, 11)
        out1 = embed1(x)
        out2 = embed2(x)

        # Different patch sizes should produce different number of patches
        assert out1.shape[1] != out2.shape[1]


class TestPositionalEncoding:
    def test_output_shape_preserved(self):
        pos_enc = PositionalEncoding(d_model=64, max_len=100)
        x = torch.randn(4, 50, 64)
        out = pos_enc(x)
        assert out.shape == x.shape

    def test_adds_positional_info(self):
        pos_enc = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x = torch.zeros(1, 10, 64)
        out = pos_enc(x)
        # Output should not be all zeros (positional encoding added)
        assert not torch.allclose(out, x)


class TestTransformerEncoder:
    def test_output_shape_mean_pool(self):
        encoder = TransformerEncoder(
            n_features=11, seq_len=100, patch_len=16, patch_stride=8,
            d_model=64, n_heads=4, n_layers=2, pool="mean"
        )
        x = torch.randn(4, 100, 11)
        out = encoder(x)
        assert out.shape == (4, 64)

    def test_output_shape_cls_pool(self):
        encoder = TransformerEncoder(
            n_features=11, seq_len=100, patch_len=16, patch_stride=8,
            d_model=64, n_heads=4, n_layers=2, pool="cls"
        )
        x = torch.randn(4, 100, 11)
        out = encoder(x)
        assert out.shape == (4, 64)

    def test_output_shape_last_pool(self):
        encoder = TransformerEncoder(
            n_features=11, seq_len=100, patch_len=16, patch_stride=8,
            d_model=64, n_heads=4, n_layers=2, pool="last"
        )
        x = torch.randn(4, 100, 11)
        out = encoder(x)
        assert out.shape == (4, 64)

    def test_different_depths(self):
        encoder_shallow = TransformerEncoder(
            n_features=11, seq_len=100, d_model=64, n_layers=1
        )
        encoder_deep = TransformerEncoder(
            n_features=11, seq_len=100, d_model=64, n_layers=4
        )

        # Deeper model should have more parameters
        assert count_parameters(encoder_deep) > count_parameters(encoder_shallow)


class TestPatchTSTRegressor:
    def test_output_shape(self):
        model = PatchTSTRegressor(n_features=11, seq_len=100)
        x = torch.randn(4, 100, 11)
        out = model(x)
        assert out.shape == (4,)

    def test_get_features(self):
        model = PatchTSTRegressor(n_features=11, seq_len=100, d_model=64)
        x = torch.randn(4, 100, 11)
        features = model.get_features(x)
        assert features.shape == (4, 64)

    def test_forward_backward(self):
        model = PatchTSTRegressor(n_features=11, seq_len=100)
        x = torch.randn(4, 100, 11)
        y = torch.randn(4)

        # Forward
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)

        # Backward
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_batch_size_one(self):
        model = PatchTSTRegressor(n_features=11, seq_len=100)
        x = torch.randn(1, 100, 11)
        out = model(x)
        assert out.shape == (1,)


class TestCreateModel:
    def test_default_config(self):
        model = create_model(n_features=11, seq_len=100)
        assert isinstance(model, PatchTSTRegressor)

        # Test forward pass works
        x = torch.randn(2, 100, 11)
        out = model(x)
        assert out.shape == (2,)

    def test_custom_config(self):
        config = {
            "d_model": 128,
            "n_layers": 4,
            "n_heads": 8,
        }
        model = create_model(n_features=11, seq_len=100, config=config)

        # Verify config was applied
        assert model.encoder.d_model == 128

    def test_parameter_count(self):
        model = create_model(n_features=11, seq_len=100)
        n_params = count_parameters(model)
        # Should have reasonable number of parameters
        assert 10_000 < n_params < 1_000_000


class TestCountParameters:
    def test_counts_trainable_only(self):
        model = PatchTSTRegressor(n_features=11, seq_len=100)

        # Freeze some parameters
        for param in model.head.parameters():
            param.requires_grad = False

        total_before = count_parameters(
            PatchTSTRegressor(n_features=11, seq_len=100)
        )
        total_after = count_parameters(model)

        assert total_after < total_before
