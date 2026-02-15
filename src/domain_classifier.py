"""Domain classifier for validating domain shift in feature space."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from .transformer_model import PatchTSTRegressor
from .baseline_model import extract_handcrafted_features
from .preprocessing import INPUT_FEATURES


def extract_baseline_features(
    batches: dict[int, pd.DataFrame],
    batch_ids: list[int],
    features: list[str] = INPUT_FEATURES,
    window_fraction: float = 0.25,
) -> tuple[np.ndarray, list[str]]:
    """Extract hand-crafted features for domain classification.

    Extracts statistical features (mean, std, min, max, slope) from the early
    window of each batch, matching the baseline model's feature extraction.

    Args:
        batches: Dict of batch DataFrames keyed by batch ID.
        batch_ids: List of batch IDs to extract features from.
        features: Input feature columns to use.
        window_fraction: Fraction of batch for early window.

    Returns:
        Tuple of (feature_matrix, feature_names) where feature_matrix has shape
        (n_batches, n_features) and feature_names lists the feature names.
    """
    feature_rows = []
    feature_names = None

    for batch_id in batch_ids:
        df = batches[batch_id]
        feats = extract_handcrafted_features(df, features, window_fraction)

        if feature_names is None:
            feature_names = list(feats.keys())

        feature_rows.append(list(feats.values()))

    return np.array(feature_rows), feature_names


class DomainClassifier(nn.Module):
    """MLP classifier for domain prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1):
        """Initialize domain classifier.

        Args:
            input_dim: Input feature dimension (d_model from encoder).
            hidden_dim: Hidden layer dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape (B, input_dim).

        Returns:
            Logits of shape (B,).
        """
        return self.classifier(x).squeeze(-1)


def extract_features(
    encoder: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract encoder features from all batches.

    Args:
        encoder: Trained transformer encoder (or full model with get_features).
        data_loader: DataLoader with (X, y, domain_label) samples.
        device: Device to run on.

    Returns:
        Tuple of (features, targets, domain_labels) as numpy arrays.
    """
    encoder.eval()
    all_features = []
    all_targets = []
    all_domains = []

    with torch.no_grad():
        for X, y, domain in data_loader:
            X = X.to(device)

            # Extract features from encoder
            if hasattr(encoder, 'get_features'):
                features = encoder.get_features(X)
            elif hasattr(encoder, 'encoder'):
                features = encoder.encoder(X)
            else:
                features = encoder(X)

            all_features.append(features.cpu().numpy())
            all_targets.append(y.numpy())
            all_domains.append(domain.numpy())

    return (
        np.concatenate(all_features),
        np.concatenate(all_targets),
        np.concatenate(all_domains),
    )


def train_domain_classifier(
    features: np.ndarray,
    domain_labels: np.ndarray,
    hidden_dim: int = 32,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 16,
    val_ratio: float = 0.2,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Train domain classifier on extracted features.

    Args:
        features: Feature array of shape (n_samples, feature_dim).
        domain_labels: Domain labels (0=source, 1=target).
        hidden_dim: Hidden dimension for classifier.
        n_epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Batch size.
        val_ratio: Validation split ratio.
        device: Device to train on.
        verbose: Print progress.

    Returns:
        Dict with classifier, metrics, and predictions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split into train/val
    n_samples = len(features)
    indices = np.random.permutation(n_samples)
    n_val = int(n_samples * val_ratio)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train = torch.tensor(features[train_idx], dtype=torch.float32)
    y_train = torch.tensor(domain_labels[train_idx], dtype=torch.float32)
    X_val = torch.tensor(features[val_idx], dtype=torch.float32)
    y_val = torch.tensor(domain_labels[val_idx], dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create classifier
    input_dim = features.shape[1]
    classifier = DomainClassifier(input_dim, hidden_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(n_epochs):
        # Train
        classifier.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = classifier(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        classifier.eval()
        with torch.no_grad():
            X_val_dev = X_val.to(device)
            y_val_dev = y_val.to(device)
            val_logits = classifier(X_val_dev)
            val_loss = criterion(val_logits, y_val_dev).item()
            val_preds = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_preds == y_val_dev).float().mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.3f}")

    # Final evaluation on all data
    classifier.eval()
    with torch.no_grad():
        X_all = torch.tensor(features, dtype=torch.float32).to(device)
        all_logits = classifier(X_all).cpu().numpy()
        all_probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
        all_preds = (all_probs > 0.5).astype(int)

    accuracy = accuracy_score(domain_labels, all_preds)

    # Handle case where only one class present
    try:
        auc = roc_auc_score(domain_labels, all_probs)
    except ValueError:
        auc = 0.5

    conf_matrix = confusion_matrix(domain_labels, all_preds)

    if verbose:
        print(f"\n=== Domain Classification Results ===")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"AUC: {auc:.3f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

    return {
        "classifier": classifier,
        "history": history,
        "accuracy": accuracy,
        "auc": auc,
        "confusion_matrix": conf_matrix,
        "predictions": all_preds,
        "probabilities": all_probs,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


def compute_domain_statistics(
    features: np.ndarray,
    domain_labels: np.ndarray,
) -> dict:
    """Compute statistical measures of domain shift.

    Args:
        features: Feature array of shape (n_samples, feature_dim).
        domain_labels: Domain labels (0=source, 1=target).

    Returns:
        Dict with statistical measures.
    """
    source_mask = domain_labels == 0
    target_mask = domain_labels == 1

    source_features = features[source_mask]
    target_features = features[target_mask]

    # Per-feature statistics
    source_mean = source_features.mean(axis=0)
    source_std = source_features.std(axis=0)
    target_mean = target_features.mean(axis=0)
    target_std = target_features.std(axis=0)

    # Mean difference (normalized by source std)
    std_safe = np.where(source_std == 0, 1.0, source_std)
    normalized_diff = np.abs(target_mean - source_mean) / std_safe

    # Maximum Mean Discrepancy (simplified - using mean of squared differences)
    mmd_approx = np.mean((source_mean - target_mean) ** 2)

    # Covariance matrices
    source_cov = np.cov(source_features.T)
    target_cov = np.cov(target_features.T)

    # Frobenius norm of covariance difference (CORAL-related)
    cov_diff_norm = np.linalg.norm(source_cov - target_cov, 'fro')

    return {
        "source_mean": source_mean,
        "source_std": source_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "normalized_mean_diff": normalized_diff,
        "max_normalized_diff": normalized_diff.max(),
        "mean_normalized_diff": normalized_diff.mean(),
        "mmd_approx": mmd_approx,
        "cov_diff_frobenius": cov_diff_norm,
        "source_cov": source_cov,
        "target_cov": target_cov,
    }


def compute_tsne_embedding(
    features: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """Compute t-SNE embedding of features.

    Args:
        features: Feature array of shape (n_samples, feature_dim).
        n_components: Number of t-SNE components.
        perplexity: t-SNE perplexity parameter.
        random_state: Random seed.

    Returns:
        Embedding array of shape (n_samples, n_components).
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, len(features) - 1),
        random_state=random_state,
        max_iter=1000,
    )
    return tsne.fit_transform(features)


def compute_pca_embedding(
    features: np.ndarray,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PCA embedding of features.

    Args:
        features: Feature array of shape (n_samples, feature_dim).
        n_components: Number of PCA components.

    Returns:
        Tuple of (embedding, explained_variance_ratio).
    """
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(features)
    return embedding, pca.explained_variance_ratio_


def analyze_misclassified_batches(
    predictions: np.ndarray,
    domain_labels: np.ndarray,
    probabilities: np.ndarray,
    batch_ids: list[int] | None = None,
) -> dict:
    """Analyze which batches are hard to classify by domain.

    Args:
        predictions: Predicted domain labels.
        domain_labels: True domain labels.
        probabilities: Predicted probabilities.
        batch_ids: Optional list of batch IDs.

    Returns:
        Dict with analysis results.
    """
    correct = predictions == domain_labels

    # Find misclassified samples
    misclassified_idx = np.where(~correct)[0]

    # Find uncertain predictions (close to 0.5)
    uncertainty = np.abs(probabilities - 0.5)
    uncertain_idx = np.argsort(uncertainty)[:10]  # Top 10 most uncertain

    # Confidence for correct vs incorrect
    correct_confidence = np.abs(probabilities[correct] - 0.5).mean() if correct.sum() > 0 else 0
    incorrect_confidence = np.abs(probabilities[~correct] - 0.5).mean() if (~correct).sum() > 0 else 0

    result = {
        "misclassified_indices": misclassified_idx,
        "n_misclassified": len(misclassified_idx),
        "uncertain_indices": uncertain_idx,
        "correct_confidence": correct_confidence,
        "incorrect_confidence": incorrect_confidence,
    }

    if batch_ids is not None:
        result["misclassified_batch_ids"] = [batch_ids[i] for i in misclassified_idx]
        result["uncertain_batch_ids"] = [batch_ids[i] for i in uncertain_idx]

    return result
