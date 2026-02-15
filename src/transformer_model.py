"""Patch-based Transformer encoder for time-series regression."""

import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Convert time-series into patches and embed them."""

    def __init__(
        self,
        n_features: int,
        patch_len: int = 16,
        patch_stride: int = 8,
        d_model: int = 64,
    ):
        """Initialize patch embedding.

        Args:
            n_features: Number of input features (channels).
            patch_len: Length of each patch.
            patch_stride: Stride between patches.
            d_model: Embedding dimension.
        """
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.d_model = d_model

        # Linear projection from patch to embedding
        self.projection = nn.Linear(patch_len * n_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create patch embeddings.

        Args:
            x: Input tensor of shape (B, T, C) where
               B=batch size, T=sequence length, C=n_features.

        Returns:
            Patch embeddings of shape (B, n_patches, d_model).
        """
        B, T, C = x.shape

        # Unfold to create patches: (B, T, C) -> (B, n_patches, patch_len, C)
        # Then reshape to (B, n_patches, patch_len * C)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        # patches shape: (B, n_patches, C, patch_len)
        patches = patches.permute(0, 1, 3, 2)  # (B, n_patches, patch_len, C)
        patches = patches.reshape(B, -1, self.patch_len * C)  # (B, n_patches, patch_len * C)

        # Project to d_model
        embeddings = self.projection(patches)  # (B, n_patches, d_model)

        return embeddings

    def get_num_patches(self, seq_len: int) -> int:
        """Calculate number of patches for a given sequence length."""
        return (seq_len - self.patch_len) // self.patch_stride + 1


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: Input tensor of shape (B, seq_len, d_model).

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder for time-series.

    Architecture:
        Input (B, T, C) -> PatchEmbed -> PosEnc -> TransformerLayers -> Pool -> MLP -> y_hat
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        patch_len: int = 16,
        patch_stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        pool: str = "mean",
    ):
        """Initialize transformer encoder.

        Args:
            n_features: Number of input features.
            seq_len: Input sequence length.
            patch_len: Length of each patch.
            patch_stride: Stride between patches.
            d_model: Model dimension.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            d_ff: Feed-forward dimension.
            dropout: Dropout probability.
            pool: Pooling method ('mean', 'cls', or 'last').
        """
        super().__init__()

        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.pool = pool

        # Patch embedding
        self.patch_embed = PatchEmbedding(n_features, patch_len, patch_stride, d_model)
        self.n_patches = self.patch_embed.get_num_patches(seq_len)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=self.n_patches + 1, dropout=dropout)

        # Optional CLS token
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Encoded features of shape (B, d_model).
        """
        B = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, d_model)

        # Add CLS token if using cls pooling
        if self.pool == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer layers
        x = self.transformer(x)

        # Layer normalization
        x = self.norm(x)

        # Pooling
        if self.pool == "mean":
            x = x.mean(dim=1)  # (B, d_model)
        elif self.pool == "cls":
            x = x[:, 0, :]  # (B, d_model)
        elif self.pool == "last":
            x = x[:, -1, :]  # (B, d_model)

        return x


class PatchTSTRegressor(nn.Module):
    """Complete PatchTST model for regression.

    Combines TransformerEncoder with regression head.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        patch_len: int = 16,
        patch_stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        pool: str = "mean",
        head_hidden: int = 32,
    ):
        """Initialize PatchTST regressor.

        Args:
            n_features: Number of input features.
            seq_len: Input sequence length.
            patch_len: Length of each patch.
            patch_stride: Stride between patches.
            d_model: Model dimension.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            d_ff: Feed-forward dimension.
            dropout: Dropout probability.
            pool: Pooling method.
            head_hidden: Hidden dimension for regression head.
        """
        super().__init__()

        self.encoder = TransformerEncoder(
            n_features=n_features,
            seq_len=seq_len,
            patch_len=patch_len,
            patch_stride=patch_stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            pool=pool,
        )

        # Regression head: MLP with one hidden layer
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Predictions of shape (B,).
        """
        features = self.encoder(x)  # (B, d_model)
        output = self.head(features)  # (B, 1)
        return output.squeeze(-1)  # (B,)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract encoder features (for CORAL loss).

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Features of shape (B, d_model).
        """
        return self.encoder(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(
    n_features: int,
    seq_len: int,
    config: dict | None = None,
) -> PatchTSTRegressor:
    """Create PatchTST model with given configuration.

    Args:
        n_features: Number of input features.
        seq_len: Input sequence length.
        config: Model configuration dict. Uses defaults if None.

    Returns:
        PatchTSTRegressor model.
    """
    default_config = {
        "patch_len": 16,
        "patch_stride": 8,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 256,
        "dropout": 0.1,
        "pool": "mean",
        "head_hidden": 32,
    }

    if config is not None:
        default_config.update(config)

    return PatchTSTRegressor(
        n_features=n_features,
        seq_len=seq_len,
        **default_config,
    )
