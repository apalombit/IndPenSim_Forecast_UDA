"""CORAL (CORrelation ALignment) loss for unsupervised domain adaptation.

Reference: Sun & Saenko, "Deep CORAL" (ECCV 2016 Workshops)
"""

import torch


def coral_loss(
    source_features: torch.Tensor, target_features: torch.Tensor
) -> torch.Tensor:
    """Compute CORAL loss between source and target feature distributions.

    Aligns second-order statistics (covariance matrices):
        L_coral = ||C_s - C_t||_F^2 / (4 * d^2)

    Args:
        source_features: Source domain features, shape (n_s, d).
        target_features: Target domain features, shape (n_t, d).

    Returns:
        Scalar CORAL loss tensor.
    """
    d = source_features.size(1)
    ns = source_features.size(0)
    nt = target_features.size(0)

    # Center features
    source_centered = source_features - source_features.mean(0, keepdim=True)
    target_centered = target_features - target_features.mean(0, keepdim=True)

    # Covariance matrices
    cov_source = (source_centered.T @ source_centered) / max(ns - 1, 1)
    cov_target = (target_centered.T @ target_centered) / max(nt - 1, 1)

    # Frobenius norm squared, normalized by 4*d^2
    loss = torch.sum((cov_source - cov_target) ** 2) / (4 * d * d)

    return loss
