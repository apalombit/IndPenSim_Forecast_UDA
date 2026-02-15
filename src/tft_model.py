"""Temporal Fusion Transformer for probabilistic concentration prediction."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network with skip connection and gated linear unit.

    Architecture:
        input -> Linear -> ELU -> Linear -> Dropout -> GLU -> Add(skip) -> LayerNorm
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_output: int,
        dropout: float = 0.1,
        context_dim: int | None = None,
    ):
        """Initialize GRN.

        Args:
            d_input: Input dimension.
            d_hidden: Hidden dimension.
            d_output: Output dimension.
            dropout: Dropout probability.
            context_dim: Optional context dimension for conditional GRN.
        """
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output

        # Skip connection projection if dimensions differ
        self.skip_proj = (
            nn.Linear(d_input, d_output)
            if d_input != d_output
            else nn.Identity()
        )

        # Main pathway
        self.fc1 = nn.Linear(d_input, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_output * 2)  # *2 for GLU
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_output)

        # Optional context projection
        self.context_proj = (
            nn.Linear(context_dim, d_hidden, bias=False)
            if context_dim is not None
            else None
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., d_input).
            context: Optional context tensor of shape (..., context_dim).

        Returns:
            Output tensor of shape (..., d_output).
        """
        # Skip connection
        skip = self.skip_proj(x)

        # Main pathway
        hidden = F.elu(self.fc1(x))

        # Add context if provided
        if context is not None and self.context_proj is not None:
            hidden = hidden + self.context_proj(context)

        # GLU activation
        gate_input = self.dropout(self.fc2(hidden))
        value, gate = gate_input.chunk(2, dim=-1)
        gated = value * torch.sigmoid(gate)

        # Residual connection and layer norm
        return self.layer_norm(skip + gated)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for learning soft variable importance.

    Uses GRN + Softmax to learn importance weights for each input variable.
    """

    def __init__(
        self,
        n_variables: int,
        d_input: int,
        d_model: int,
        d_hidden: int,
        dropout: float = 0.1,
    ):
        """Initialize VSN.

        Args:
            n_variables: Number of input variables (channels).
            d_input: Dimension of each variable's embedding.
            d_model: Output embedding dimension.
            d_hidden: Hidden dimension for GRNs.
            dropout: Dropout probability.
        """
        super().__init__()
        self.n_variables = n_variables
        self.d_input = d_input
        self.d_model = d_model

        # Per-variable GRN for feature transformation
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(d_input, d_hidden, d_model, dropout)
            for _ in range(n_variables)
        ])

        # GRN for computing selection weights (flattened input)
        self.selection_grn = GatedResidualNetwork(
            d_input=n_variables * d_input,
            d_hidden=d_hidden,
            d_output=n_variables,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, n_variables, d_input) or
               (B, n_variables, d_input) for static variables.

        Returns:
            Tuple of:
                - Selected features of shape (B, T, d_model) or (B, d_model)
                - Importance weights of shape (B, T, n_variables) or (B, n_variables)
        """
        has_time = x.dim() == 4

        if has_time:
            B, T, V, D = x.shape
            # Flatten time for batch processing
            x_flat = x.reshape(B * T, V, D)
        else:
            B, V, D = x.shape
            x_flat = x

        # Transform each variable through its GRN
        transformed = []
        for i in range(self.n_variables):
            var_transformed = self.variable_grns[i](x_flat[:, i, :])  # (BT, d_model)
            transformed.append(var_transformed)
        transformed = torch.stack(transformed, dim=1)  # (BT, V, d_model)

        # Compute selection weights
        x_concat = x_flat.reshape(x_flat.size(0), -1)  # (BT, V*D)
        weights = self.selection_grn(x_concat)  # (BT, V)
        weights = F.softmax(weights, dim=-1)

        # Weighted combination
        selected = (transformed * weights.unsqueeze(-1)).sum(dim=1)  # (BT, d_model)

        if has_time:
            selected = selected.reshape(B, T, self.d_model)
            weights = weights.reshape(B, T, V)
        else:
            weights = weights.reshape(B, V)

        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention that returns attention weights for interpretability.

    Uses additive attention (Bahdanau-style) for better interpretability
    compared to scaled dot-product attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        """Initialize attention module.

        Args:
            d_model: Model dimension (must be divisible by n_heads).
            n_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            q: Query tensor of shape (B, T_q, d_model).
            k: Key tensor of shape (B, T_k, d_model).
            v: Value tensor of shape (B, T_v, d_model). T_k == T_v.
            mask: Optional attention mask of shape (B, T_q, T_k) or (B, 1, T_k).
                  True values are masked (not attended to).

        Returns:
            Tuple of:
                - Context tensor of shape (B, T_q, d_model)
                - Attention weights of shape (B, n_heads, T_q, T_k)
        """
        B, T_q, _ = q.shape
        T_k = k.size(1)

        # Linear projections and reshape for multi-head
        q = self.w_q(q).view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(k).view(B, T_k, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(v).view(B, T_k, self.n_heads, self.d_head).transpose(1, 2)
        # Shapes: (B, n_heads, T_*, d_head)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, n_heads, T_q, T_k)

        # Apply mask
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, T_q, T_k)
            scores = scores.masked_fill(mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)  # (B, n_heads, T_q, T_k)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # (B, n_heads, T_q, d_head)
        context = context.transpose(1, 2).contiguous().view(B, T_q, self.d_model)

        # Output projection
        output = self.w_out(context)

        return output, attn_weights


class HorizonConditioning(nn.Module):
    """FiLM-based conditioning on prediction horizon D.

    Uses Feature-wise Linear Modulation (FiLM) to condition features
    on the prediction horizon through learned scale and shift parameters.
    """

    def __init__(
        self,
        d_model: int,
        max_horizon: int = 500,
        d_hidden: int = 32,
    ):
        """Initialize horizon conditioning.

        Args:
            d_model: Feature dimension to condition.
            max_horizon: Maximum prediction horizon (for embedding table).
            d_hidden: Hidden dimension for horizon encoder.
        """
        super().__init__()
        self.d_model = d_model
        self.max_horizon = max_horizon

        # Horizon embedding and encoder
        self.horizon_embed = nn.Embedding(max_horizon, d_hidden)

        # FiLM generators
        self.gamma_net = nn.Sequential(
            nn.Linear(d_hidden, d_model),
            nn.Tanh(),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, features: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            features: Input features of shape (B, d_model).
            horizon: Prediction horizons of shape (B,), integer values.

        Returns:
            Conditioned features of shape (B, d_model).
        """
        # Clamp horizon to valid range
        horizon = horizon.clamp(0, self.max_horizon - 1)

        # Get horizon embedding
        h_embed = self.horizon_embed(horizon)  # (B, d_hidden)

        # Compute scale and shift
        gamma = 1.0 + self.gamma_net(h_embed)  # (B, d_model), centered at 1
        beta = self.beta_net(h_embed)  # (B, d_model)

        # Apply FiLM: y = gamma * x + beta
        return gamma * features + beta


class ProbabilisticOutputHead(nn.Module):
    """Output head for Gaussian parameters (mu, sigma).

    Outputs both mean and standard deviation for probabilistic predictions.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int = 32,
        min_sigma: float = 1e-4,
    ):
        """Initialize output head.

        Args:
            d_model: Input feature dimension.
            d_hidden: Hidden dimension.
            min_sigma: Minimum sigma to ensure numerical stability.
        """
        super().__init__()
        self.min_sigma = min_sigma

        # Shared hidden layer
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
        )

        # Separate heads for mu and sigma
        self.mu_head = nn.Linear(d_hidden, 1)
        self.sigma_head = nn.Linear(d_hidden, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            features: Input features of shape (B, d_model).

        Returns:
            Tuple of:
                - mu: Mean predictions of shape (B,)
                - sigma: Standard deviation predictions of shape (B,)
        """
        hidden = self.shared(features)

        mu = self.mu_head(hidden).squeeze(-1)
        # Use softplus for positive sigma, add minimum
        sigma = F.softplus(self.sigma_head(hidden).squeeze(-1)) + self.min_sigma

        return mu, sigma


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for probabilistic time-series prediction.

    Architecture:
        Input (B, T, C) -> Per-variable Linear -> VSN -> LSTM Encoder
            -> Temporal Self-Attention -> Mean Pool -> Horizon Conditioning (FiLM)
            -> Probabilistic Output Head -> (mu, sigma)
    """

    def __init__(
        self,
        n_features: int = 28,
        d_model: int = 64,
        d_hidden: int = 64,
        n_lstm_layers: int = 2,
        n_attention_heads: int = 4,
        dropout: float = 0.1,
        max_horizon: int = 500,
    ):
        """Initialize TFT model.

        Args:
            n_features: Number of input features (channels).
            d_model: Transformer embedding dimension.
            d_hidden: Hidden dimension for GRNs.
            n_lstm_layers: Number of bidirectional LSTM layers.
            n_attention_heads: Number of attention heads.
            dropout: Dropout probability.
            max_horizon: Maximum prediction horizon.
        """
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.d_hidden = d_hidden

        # Per-variable embedding (each feature gets its own embedding)
        self.var_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])

        # Variable Selection Network
        self.vsn = VariableSelectionNetwork(
            n_variables=n_features,
            d_input=d_model,
            d_model=d_model,
            d_hidden=d_hidden,
            dropout=dropout,
        )

        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,  # Bidirectional doubles this
            num_layers=n_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_lstm_layers > 1 else 0,
        )

        # Layer norm after LSTM
        self.lstm_norm = nn.LayerNorm(d_model)

        # Temporal self-attention
        self.temporal_attention = InterpretableMultiHeadAttention(
            d_model=d_model,
            n_heads=n_attention_heads,
            dropout=dropout,
        )

        # Post-attention GRN
        self.post_attention_grn = GatedResidualNetwork(
            d_input=d_model,
            d_hidden=d_hidden,
            d_output=d_model,
            dropout=dropout,
        )

        # Horizon conditioning
        self.horizon_conditioning = HorizonConditioning(
            d_model=d_model,
            max_horizon=max_horizon,
            d_hidden=d_hidden,
        )

        # Probabilistic output head
        self.output_head = ProbabilisticOutputHead(
            d_model=d_model,
            d_hidden=d_hidden,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        horizon: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, C) where C = n_features.
            horizon: Prediction horizon of shape (B,), integer values.
            mask: Optional padding mask of shape (B, T). True = padded/masked.

        Returns:
            Tuple of (mu, sigma):
                - mu: Mean predictions of shape (B,)
                - sigma: Standard deviation of shape (B,)
        """
        B, T, C = x.shape

        # Per-variable embedding: (B, T, C) -> (B, T, C, d_model)
        var_embeds = []
        for i in range(self.n_features):
            var_embed = self.var_embeddings[i](x[:, :, i:i+1])  # (B, T, d_model)
            var_embeds.append(var_embed)
        x_embedded = torch.stack(var_embeds, dim=2)  # (B, T, C, d_model)

        # Variable selection: (B, T, C, d_model) -> (B, T, d_model)
        x_selected, vsn_weights = self.vsn(x_embedded)

        # LSTM encoding
        if mask is not None:
            # Pack padded sequence for efficient LSTM
            lengths = (~mask).sum(dim=1).cpu()
            lengths = lengths.clamp(min=1)  # Ensure at least 1
            packed = nn.utils.rnn.pack_padded_sequence(
                x_selected, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=T
            )
        else:
            lstm_out, _ = self.lstm(x_selected)

        lstm_out = self.lstm_norm(lstm_out)  # (B, T, d_model)

        # Temporal self-attention
        # Create attention mask from padding mask
        attn_mask = None
        if mask is not None:
            # Expand mask for attention: (B, T) -> (B, T, T)
            attn_mask = mask.unsqueeze(1).expand(-1, T, -1)

        attn_out, attn_weights = self.temporal_attention(
            lstm_out, lstm_out, lstm_out, mask=attn_mask
        )

        # Residual and GRN
        attn_out = self.post_attention_grn(attn_out + lstm_out)

        # Mean pooling (excluding padded positions)
        if mask is not None:
            # Mask out padded positions before mean
            attn_out = attn_out.masked_fill(mask.unsqueeze(-1), 0.0)
            lengths = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
            pooled = attn_out.sum(dim=1) / lengths
        else:
            pooled = attn_out.mean(dim=1)  # (B, d_model)

        # Horizon conditioning
        conditioned = self.horizon_conditioning(pooled, horizon)
        conditioned = self.dropout(conditioned)

        # Output head
        mu, sigma = self.output_head(conditioned)

        return mu, sigma

    def get_interpretability(
        self,
        x: torch.Tensor,
        horizon: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict:
        """Extract interpretability information.

        Args:
            x: Input tensor of shape (B, T, C).
            horizon: Prediction horizon of shape (B,).
            mask: Optional padding mask of shape (B, T).

        Returns:
            Dict containing:
                - mu: Mean predictions (B,)
                - sigma: Standard deviations (B,)
                - vsn_weights: Variable importance weights (B, T, C)
                - attention_weights: Temporal attention weights (B, n_heads, T, T)
        """
        B, T, C = x.shape

        # Per-variable embedding
        var_embeds = []
        for i in range(self.n_features):
            var_embed = self.var_embeddings[i](x[:, :, i:i+1])
            var_embeds.append(var_embed)
        x_embedded = torch.stack(var_embeds, dim=2)

        # Variable selection
        x_selected, vsn_weights = self.vsn(x_embedded)

        # LSTM encoding
        if mask is not None:
            lengths = (~mask).sum(dim=1).cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x_selected, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=T
            )
        else:
            lstm_out, _ = self.lstm(x_selected)

        lstm_out = self.lstm_norm(lstm_out)

        # Temporal self-attention
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1).expand(-1, T, -1)

        attn_out, attn_weights = self.temporal_attention(
            lstm_out, lstm_out, lstm_out, mask=attn_mask
        )

        # Residual and GRN
        attn_out = self.post_attention_grn(attn_out + lstm_out)

        # Mean pooling
        if mask is not None:
            attn_out = attn_out.masked_fill(mask.unsqueeze(-1), 0.0)
            lengths = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
            pooled = attn_out.sum(dim=1) / lengths
        else:
            pooled = attn_out.mean(dim=1)

        # Horizon conditioning and output
        conditioned = self.horizon_conditioning(pooled, horizon)
        mu, sigma = self.output_head(conditioned)

        return {
            "mu": mu,
            "sigma": sigma,
            "vsn_weights": vsn_weights,
            "attention_weights": attn_weights,
        }


class GaussianNLLLoss(nn.Module):
    """Gaussian Negative Log-Likelihood Loss.

    NLL = 0.5 * log(2*pi) + log(sigma) + 0.5 * ((y - mu) / sigma)^2
    """

    def __init__(self, reduction: str = "mean"):
        """Initialize loss.

        Args:
            reduction: Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.reduction = reduction
        self.log_2pi = math.log(2 * math.pi)

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NLL loss.

        Args:
            mu: Predicted means of shape (B,).
            sigma: Predicted standard deviations of shape (B,).
            y: Target values of shape (B,).

        Returns:
            Loss value (scalar if reduction != 'none').
        """
        # NLL = 0.5 * log(2*pi) + log(sigma) + 0.5 * ((y - mu) / sigma)^2
        nll = 0.5 * self.log_2pi + torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        return nll


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_tft_model(
    n_features: int = 28,
    config: dict | None = None,
) -> TemporalFusionTransformer:
    """Create TFT model with given configuration.

    Args:
        n_features: Number of input features.
        config: Model configuration dict. Uses defaults if None.

    Returns:
        TemporalFusionTransformer model.
    """
    default_config = {
        "d_model": 64,
        "d_hidden": 64,
        "n_lstm_layers": 2,
        "n_attention_heads": 4,
        "dropout": 0.1,
        "max_horizon": 500,
    }

    if config is not None:
        default_config.update(config)

    return TemporalFusionTransformer(
        n_features=n_features,
        **default_config,
    )
