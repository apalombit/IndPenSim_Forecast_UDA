"""Tests for CORAL loss module."""

import pytest
import torch

from src.coral_loss import coral_loss


class TestCoralLoss:

    def test_identical_distributions_zero_loss(self):
        features = torch.randn(20, 8)
        loss = coral_loss(features, features.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_distributions_positive_loss(self):
        source = torch.randn(20, 8)
        target = torch.randn(20, 8) * 3 + 2
        loss = coral_loss(source, target)
        assert loss.item() > 0

    def test_output_is_scalar(self):
        source = torch.randn(10, 4)
        target = torch.randn(10, 4)
        loss = coral_loss(source, target)
        assert loss.dim() == 0

    def test_different_sample_sizes(self):
        source = torch.randn(30, 8)
        target = torch.randn(10, 8)
        loss = coral_loss(source, target)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_gradient_flows(self):
        source = torch.randn(10, 4, requires_grad=True)
        target = torch.randn(10, 4, requires_grad=True)
        loss = coral_loss(source, target)
        loss.backward()
        assert source.grad is not None
        assert target.grad is not None

    def test_single_sample(self):
        source = torch.randn(1, 4)
        target = torch.randn(1, 4)
        loss = coral_loss(source, target)
        assert not torch.isnan(loss)

    def test_symmetric(self):
        source = torch.randn(15, 6)
        target = torch.randn(15, 6)
        loss_st = coral_loss(source, target)
        loss_ts = coral_loss(target, source)
        assert loss_st.item() == pytest.approx(loss_ts.item(), abs=1e-6)

    def test_normalization_controls_scale(self):
        """1/(4*d^2) normalization should keep loss in reasonable range."""
        torch.manual_seed(42)
        source = torch.randn(20, 64)
        target = torch.randn(20, 64) * 2
        loss = coral_loss(source, target)
        assert loss.item() < 100

    def test_zero_features(self):
        source = torch.zeros(10, 4)
        target = torch.zeros(10, 4)
        loss = coral_loss(source, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_increasing_shift_increases_loss(self):
        torch.manual_seed(0)
        source = torch.randn(50, 8)
        target_small = source + 0.1 * torch.randn_like(source)
        target_large = source * 5 + 10

        loss_small = coral_loss(source, target_small)
        loss_large = coral_loss(source, target_large)
        assert loss_large.item() > loss_small.item()
