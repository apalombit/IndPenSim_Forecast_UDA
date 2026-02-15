"""Unit tests for domain_classifier module."""

import numpy as np
import torch
import pytest

from src.domain_classifier import (
    DomainClassifier,
    extract_features,
    train_domain_classifier,
    compute_domain_statistics,
    compute_tsne_embedding,
    compute_pca_embedding,
    analyze_misclassified_batches,
)


class TestDomainClassifier:
    def test_output_shape(self):
        classifier = DomainClassifier(input_dim=64, hidden_dim=32)
        x = torch.randn(8, 64)
        out = classifier(x)
        assert out.shape == (8,)

    def test_forward_backward(self):
        classifier = DomainClassifier(input_dim=64, hidden_dim=32)
        x = torch.randn(8, 64)
        y = torch.randint(0, 2, (8,)).float()

        logits = classifier(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()

        # Check gradients exist
        for param in classifier.parameters():
            assert param.grad is not None

    def test_different_input_dims(self):
        for input_dim in [32, 64, 128]:
            classifier = DomainClassifier(input_dim=input_dim)
            x = torch.randn(4, input_dim)
            out = classifier(x)
            assert out.shape == (4,)


class TestTrainDomainClassifier:
    def test_returns_expected_keys(self):
        # Create synthetic features with clear domain separation
        np.random.seed(42)
        source_features = np.random.randn(30, 16)
        target_features = np.random.randn(30, 16) + 2  # Shifted

        features = np.vstack([source_features, target_features])
        labels = np.array([0] * 30 + [1] * 30)

        result = train_domain_classifier(
            features, labels, n_epochs=10, verbose=False
        )

        assert "classifier" in result
        assert "history" in result
        assert "accuracy" in result
        assert "auc" in result
        assert "confusion_matrix" in result
        assert "predictions" in result
        assert "probabilities" in result

    def test_accuracy_above_random(self):
        # With clear separation, accuracy should be above 50%
        np.random.seed(42)
        source_features = np.random.randn(30, 16)
        target_features = np.random.randn(30, 16) + 3  # Large shift

        features = np.vstack([source_features, target_features])
        labels = np.array([0] * 30 + [1] * 30)

        result = train_domain_classifier(
            features, labels, n_epochs=50, verbose=False
        )

        assert result["accuracy"] > 0.5

    def test_probabilities_valid_range(self):
        np.random.seed(42)
        features = np.random.randn(40, 16)
        labels = np.array([0] * 20 + [1] * 20)

        result = train_domain_classifier(
            features, labels, n_epochs=10, verbose=False
        )

        assert np.all(result["probabilities"] >= 0)
        assert np.all(result["probabilities"] <= 1)


class TestComputeDomainStatistics:
    def test_returns_expected_keys(self):
        features = np.random.randn(40, 16)
        labels = np.array([0] * 20 + [1] * 20)

        stats = compute_domain_statistics(features, labels)

        assert "source_mean" in stats
        assert "source_std" in stats
        assert "target_mean" in stats
        assert "target_std" in stats
        assert "normalized_mean_diff" in stats
        assert "mmd_approx" in stats
        assert "cov_diff_frobenius" in stats

    def test_shapes_correct(self):
        n_features = 16
        features = np.random.randn(40, n_features)
        labels = np.array([0] * 20 + [1] * 20)

        stats = compute_domain_statistics(features, labels)

        assert stats["source_mean"].shape == (n_features,)
        assert stats["target_mean"].shape == (n_features,)
        assert stats["source_cov"].shape == (n_features, n_features)

    def test_detects_shift(self):
        # Create features with clear shift
        source = np.random.randn(20, 8)
        target = np.random.randn(20, 8) + 5  # Large shift

        features = np.vstack([source, target])
        labels = np.array([0] * 20 + [1] * 20)

        stats = compute_domain_statistics(features, labels)

        # Should detect significant difference
        assert stats["mean_normalized_diff"] > 1.0


class TestComputeTsneEmbedding:
    def test_output_shape(self):
        features = np.random.randn(30, 16)
        embedding = compute_tsne_embedding(features, n_components=2)
        assert embedding.shape == (30, 2)

    def test_different_n_components(self):
        features = np.random.randn(30, 16)
        embedding = compute_tsne_embedding(features, n_components=3)
        assert embedding.shape == (30, 3)


class TestComputePcaEmbedding:
    def test_output_shape(self):
        features = np.random.randn(30, 16)
        embedding, variance = compute_pca_embedding(features, n_components=2)

        assert embedding.shape == (30, 2)
        assert len(variance) == 2

    def test_variance_sums_to_less_than_one(self):
        features = np.random.randn(30, 16)
        _, variance = compute_pca_embedding(features, n_components=2)

        # Explained variance for 2 components should be < 1
        assert variance.sum() <= 1.0
        assert np.all(variance >= 0)


class TestAnalyzeMisclassifiedBatches:
    def test_returns_expected_keys(self):
        predictions = np.array([0, 0, 1, 1, 0])
        labels = np.array([0, 1, 1, 0, 0])  # Some misclassified
        probs = np.array([0.2, 0.4, 0.8, 0.6, 0.3])

        result = analyze_misclassified_batches(predictions, labels, probs)

        assert "misclassified_indices" in result
        assert "n_misclassified" in result
        assert "uncertain_indices" in result
        assert "correct_confidence" in result
        assert "incorrect_confidence" in result

    def test_counts_misclassified(self):
        predictions = np.array([0, 0, 1, 1, 0])
        labels = np.array([0, 1, 1, 0, 0])  # 2 misclassified (idx 1, 3)
        probs = np.array([0.2, 0.4, 0.8, 0.6, 0.3])

        result = analyze_misclassified_batches(predictions, labels, probs)

        assert result["n_misclassified"] == 2
        assert 1 in result["misclassified_indices"]
        assert 3 in result["misclassified_indices"]

    def test_with_batch_ids(self):
        predictions = np.array([0, 1, 1])
        labels = np.array([0, 0, 1])  # idx 1 misclassified
        probs = np.array([0.2, 0.7, 0.8])
        batch_ids = [10, 20, 30]

        result = analyze_misclassified_batches(
            predictions, labels, probs, batch_ids
        )

        assert "misclassified_batch_ids" in result
        assert 20 in result["misclassified_batch_ids"]
