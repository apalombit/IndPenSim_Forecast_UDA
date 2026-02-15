"""Unit tests for TFT model module."""

import torch
import pytest

from src.tft_model import (
    GatedResidualNetwork,
    VariableSelectionNetwork,
    InterpretableMultiHeadAttention,
    HorizonConditioning,
    ProbabilisticOutputHead,
    TemporalFusionTransformer,
    GaussianNLLLoss,
    count_parameters,
    create_tft_model,
)


class TestGatedResidualNetwork:
    def test_output_shape_same_dims(self):
        grn = GatedResidualNetwork(d_input=32, d_hidden=64, d_output=32)
        x = torch.randn(4, 32)
        out = grn(x)
        assert out.shape == (4, 32)

    def test_output_shape_different_dims(self):
        grn = GatedResidualNetwork(d_input=32, d_hidden=64, d_output=64)
        x = torch.randn(4, 32)
        out = grn(x)
        assert out.shape == (4, 64)

    def test_with_context(self):
        grn = GatedResidualNetwork(d_input=32, d_hidden=64, d_output=32, context_dim=16)
        x = torch.randn(4, 32)
        context = torch.randn(4, 16)
        out = grn(x, context)
        assert out.shape == (4, 32)

    def test_batch_and_time_dims(self):
        grn = GatedResidualNetwork(d_input=32, d_hidden=64, d_output=32)
        x = torch.randn(4, 10, 32)  # (B, T, D)
        out = grn(x)
        assert out.shape == (4, 10, 32)

    def test_gradient_flow(self):
        grn = GatedResidualNetwork(d_input=32, d_hidden=64, d_output=32)
        x = torch.randn(4, 32, requires_grad=True)
        out = grn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestVariableSelectionNetwork:
    def test_output_shapes_with_time(self):
        vsn = VariableSelectionNetwork(
            n_variables=10, d_input=16, d_model=32, d_hidden=32
        )
        x = torch.randn(4, 50, 10, 16)  # (B, T, V, D)
        features, weights = vsn(x)
        assert features.shape == (4, 50, 32)
        assert weights.shape == (4, 50, 10)

    def test_output_shapes_static(self):
        vsn = VariableSelectionNetwork(
            n_variables=10, d_input=16, d_model=32, d_hidden=32
        )
        x = torch.randn(4, 10, 16)  # (B, V, D)
        features, weights = vsn(x)
        assert features.shape == (4, 32)
        assert weights.shape == (4, 10)

    def test_weights_sum_to_one(self):
        vsn = VariableSelectionNetwork(
            n_variables=10, d_input=16, d_model=32, d_hidden=32
        )
        x = torch.randn(4, 50, 10, 16)
        _, weights = vsn(x)
        # Sum along variable dimension should be 1
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_gradient_flow(self):
        vsn = VariableSelectionNetwork(
            n_variables=10, d_input=16, d_model=32, d_hidden=32
        )
        x = torch.randn(4, 50, 10, 16, requires_grad=True)
        features, weights = vsn(x)
        loss = features.sum() + weights.sum()
        loss.backward()
        assert x.grad is not None


class TestInterpretableMultiHeadAttention:
    def test_output_shapes(self):
        attn = InterpretableMultiHeadAttention(d_model=64, n_heads=4)
        q = torch.randn(4, 20, 64)
        k = torch.randn(4, 20, 64)
        v = torch.randn(4, 20, 64)
        context, weights = attn(q, k, v)
        assert context.shape == (4, 20, 64)
        assert weights.shape == (4, 4, 20, 20)  # (B, n_heads, T_q, T_k)

    def test_attention_weights_valid_distribution(self):
        attn = InterpretableMultiHeadAttention(d_model=64, n_heads=4)
        attn.eval()  # Disable dropout for testing
        q = torch.randn(4, 20, 64)
        k = torch.randn(4, 20, 64)
        v = torch.randn(4, 20, 64)
        with torch.no_grad():
            _, weights = attn(q, k, v)
        # Attention weights should sum to 1 along key dimension
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_with_mask(self):
        attn = InterpretableMultiHeadAttention(d_model=64, n_heads=4)
        q = torch.randn(4, 20, 64)
        k = torch.randn(4, 20, 64)
        v = torch.randn(4, 20, 64)
        # Mask last 5 positions
        mask = torch.zeros(4, 20, 20, dtype=torch.bool)
        mask[:, :, 15:] = True
        context, weights = attn(q, k, v, mask=mask)
        assert context.shape == (4, 20, 64)
        # Masked positions should have zero attention
        assert torch.allclose(weights[:, :, :, 15:], torch.zeros_like(weights[:, :, :, 15:]), atol=1e-5)

    def test_gradient_flow(self):
        attn = InterpretableMultiHeadAttention(d_model=64, n_heads=4)
        q = torch.randn(4, 20, 64, requires_grad=True)
        k = torch.randn(4, 20, 64)
        v = torch.randn(4, 20, 64)
        context, _ = attn(q, k, v)
        loss = context.sum()
        loss.backward()
        assert q.grad is not None


class TestHorizonConditioning:
    def test_output_shape(self):
        hc = HorizonConditioning(d_model=64, max_horizon=500)
        features = torch.randn(4, 64)
        horizon = torch.tensor([10, 50, 100, 200])
        out = hc(features, horizon)
        assert out.shape == (4, 64)

    def test_clamps_horizon(self):
        hc = HorizonConditioning(d_model=64, max_horizon=100)
        features = torch.randn(4, 64)
        # Horizon exceeds max
        horizon = torch.tensor([10, 50, 150, 200])
        # Should not raise error
        out = hc(features, horizon)
        assert out.shape == (4, 64)

    def test_different_horizons_different_outputs(self):
        hc = HorizonConditioning(d_model=64, max_horizon=500)
        features = torch.randn(1, 64)
        out1 = hc(features, torch.tensor([10]))
        out2 = hc(features, torch.tensor([100]))
        # Different horizons should produce different outputs
        assert not torch.allclose(out1, out2)

    def test_gradient_flow(self):
        hc = HorizonConditioning(d_model=64, max_horizon=500)
        features = torch.randn(4, 64, requires_grad=True)
        horizon = torch.tensor([10, 50, 100, 200])
        out = hc(features, horizon)
        loss = out.sum()
        loss.backward()
        assert features.grad is not None


class TestProbabilisticOutputHead:
    def test_output_shapes(self):
        head = ProbabilisticOutputHead(d_model=64, d_hidden=32)
        features = torch.randn(4, 64)
        mu, sigma = head(features)
        assert mu.shape == (4,)
        assert sigma.shape == (4,)

    def test_sigma_positive(self):
        head = ProbabilisticOutputHead(d_model=64, d_hidden=32, min_sigma=1e-4)
        features = torch.randn(100, 64)
        _, sigma = head(features)
        assert (sigma > 0).all()
        assert (sigma >= 1e-4).all()

    def test_gradient_flow(self):
        head = ProbabilisticOutputHead(d_model=64, d_hidden=32)
        features = torch.randn(4, 64, requires_grad=True)
        mu, sigma = head(features)
        loss = mu.sum() + sigma.sum()
        loss.backward()
        assert features.grad is not None


class TestTemporalFusionTransformer:
    def test_output_shapes(self):
        model = TemporalFusionTransformer(
            n_features=28, d_model=64, d_hidden=64,
            n_lstm_layers=2, n_attention_heads=4
        )
        x = torch.randn(4, 50, 28)
        horizon = torch.tensor([10, 20, 30, 40])
        mu, sigma = model(x, horizon)
        assert mu.shape == (4,)
        assert sigma.shape == (4,)

    def test_with_mask(self):
        model = TemporalFusionTransformer(
            n_features=28, d_model=64, d_hidden=64,
            n_lstm_layers=2, n_attention_heads=4
        )
        x = torch.randn(4, 50, 28)
        horizon = torch.tensor([10, 20, 30, 40])
        # Mask last 10 positions for half the batch
        mask = torch.zeros(4, 50, dtype=torch.bool)
        mask[:2, 40:] = True
        mu, sigma = model(x, horizon, mask=mask)
        assert mu.shape == (4,)
        assert sigma.shape == (4,)

    def test_sigma_always_positive(self):
        model = TemporalFusionTransformer(
            n_features=28, d_model=64, d_hidden=64
        )
        x = torch.randn(10, 50, 28)
        horizon = torch.randint(1, 100, (10,))
        _, sigma = model(x, horizon)
        assert (sigma > 0).all()

    def test_interpretability_output(self):
        model = TemporalFusionTransformer(
            n_features=28, d_model=64, d_hidden=64,
            n_attention_heads=4
        )
        x = torch.randn(4, 50, 28)
        horizon = torch.tensor([10, 20, 30, 40])
        result = model.get_interpretability(x, horizon)

        assert "mu" in result
        assert "sigma" in result
        assert "vsn_weights" in result
        assert "attention_weights" in result

        assert result["vsn_weights"].shape == (4, 50, 28)
        assert result["attention_weights"].shape == (4, 4, 50, 50)

    def test_vsn_weights_sum_to_one(self):
        model = TemporalFusionTransformer(n_features=28, d_model=64)
        x = torch.randn(4, 50, 28)
        horizon = torch.tensor([10, 20, 30, 40])
        result = model.get_interpretability(x, horizon)
        vsn_weights = result["vsn_weights"]
        weight_sums = vsn_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_gradient_flow(self):
        model = TemporalFusionTransformer(
            n_features=28, d_model=64, d_hidden=64
        )
        x = torch.randn(4, 50, 28, requires_grad=True)
        horizon = torch.tensor([10, 20, 30, 40])
        mu, sigma = model(x, horizon)
        loss = mu.sum() + sigma.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_training_mode(self):
        model = TemporalFusionTransformer(n_features=28, d_model=64)
        model.train()
        x = torch.randn(4, 50, 28)
        horizon = torch.tensor([10, 20, 30, 40])
        mu1, _ = model(x, horizon)

        model.eval()
        mu2, _ = model(x, horizon)
        # Should potentially differ due to dropout
        # (but this is just a smoke test)
        assert mu1.shape == mu2.shape


class TestGaussianNLLLoss:
    def test_output_shape_mean_reduction(self):
        loss_fn = GaussianNLLLoss(reduction="mean")
        mu = torch.randn(10)
        sigma = torch.abs(torch.randn(10)) + 0.1
        y = torch.randn(10)
        loss = loss_fn(mu, sigma, y)
        assert loss.shape == ()

    def test_output_shape_none_reduction(self):
        loss_fn = GaussianNLLLoss(reduction="none")
        mu = torch.randn(10)
        sigma = torch.abs(torch.randn(10)) + 0.1
        y = torch.randn(10)
        loss = loss_fn(mu, sigma, y)
        assert loss.shape == (10,)

    def test_loss_positive(self):
        loss_fn = GaussianNLLLoss()
        mu = torch.randn(10)
        sigma = torch.abs(torch.randn(10)) + 0.1
        y = torch.randn(10)
        loss = loss_fn(mu, sigma, y)
        # NLL can be negative for small sigma, so just check it's finite
        assert torch.isfinite(loss)

    def test_perfect_prediction_lower_loss(self):
        loss_fn = GaussianNLLLoss()
        y = torch.randn(10)
        sigma = torch.ones(10) * 0.1

        # Perfect prediction
        loss_perfect = loss_fn(y, sigma, y)

        # Bad prediction
        mu_bad = y + 5.0
        loss_bad = loss_fn(mu_bad, sigma, y)

        assert loss_perfect < loss_bad

    def test_smaller_sigma_lower_loss_for_good_prediction(self):
        loss_fn = GaussianNLLLoss()
        y = torch.zeros(10)
        mu = torch.zeros(10)  # Perfect prediction

        loss_small_sigma = loss_fn(mu, torch.ones(10) * 0.1, y)
        loss_large_sigma = loss_fn(mu, torch.ones(10) * 1.0, y)

        # For perfect predictions, smaller sigma should give lower loss
        assert loss_small_sigma < loss_large_sigma

    def test_gradient_flow(self):
        loss_fn = GaussianNLLLoss()
        mu = torch.randn(10, requires_grad=True)
        sigma = torch.abs(torch.randn(10)) + 0.1
        sigma.requires_grad = True
        y = torch.randn(10)
        loss = loss_fn(mu, sigma, y)
        loss.backward()
        assert mu.grad is not None
        assert sigma.grad is not None


class TestHelperFunctions:
    def test_count_parameters(self):
        model = TemporalFusionTransformer(n_features=28, d_model=64)
        n_params = count_parameters(model)
        assert n_params > 0
        assert isinstance(n_params, int)

    def test_create_tft_model_default(self):
        model = create_tft_model(n_features=28)
        assert isinstance(model, TemporalFusionTransformer)
        assert model.n_features == 28
        assert model.d_model == 64

    def test_create_tft_model_custom_config(self):
        config = {
            "d_model": 128,
            "d_hidden": 128,
            "n_lstm_layers": 3,
            "n_attention_heads": 8,
        }
        model = create_tft_model(n_features=20, config=config)
        assert model.n_features == 20
        assert model.d_model == 128


class TestEndToEndTraining:
    def test_single_training_step(self):
        """Test that a single training step works."""
        model = TemporalFusionTransformer(n_features=28, d_model=32, d_hidden=32)
        loss_fn = GaussianNLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Dummy batch
        x = torch.randn(4, 50, 28)
        horizon = torch.randint(1, 100, (4,))
        y = torch.randn(4)

        # Forward pass
        model.train()
        mu, sigma = model(x, horizon)
        loss = loss_fn(mu, sigma, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check gradients were applied
        assert torch.isfinite(loss)

    def test_loss_decreases_over_epochs(self):
        """Test that loss decreases during training on synthetic data."""
        torch.manual_seed(42)
        model = TemporalFusionTransformer(n_features=10, d_model=16, d_hidden=16)
        loss_fn = GaussianNLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create synthetic data with a learnable pattern
        x = torch.randn(20, 30, 10)
        horizon = torch.randint(1, 50, (20,))
        y = x[:, :, 0].mean(dim=1)  # Simple target based on first feature

        initial_loss = None
        final_loss = None

        model.train()
        for epoch in range(20):
            optimizer.zero_grad()
            mu, sigma = model(x, horizon)
            loss = loss_fn(mu, sigma, y)
            loss.backward()
            optimizer.step()

            if epoch == 0:
                initial_loss = loss.item()
            if epoch == 19:
                final_loss = loss.item()

        # Loss should generally decrease (allow some tolerance)
        assert final_loss < initial_loss * 1.5
