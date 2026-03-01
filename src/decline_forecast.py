"""Decline Forecast Model: predict whether, when, and how steep decline will be.

Observes process signals up to time T (partial batch) and forecasts:
  - decline_prob: probability that decline happens (sigmoid)
  - delta_t_break: (t_break - T) / T_max, normalized time until decline
  - slope: decline rate in g/L/h (softplus, non-negative)

Uses PatchTST encoder from transformer_model.py with a multi-head output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pandas as pd

from .piecelog_patchtst import _logit, _softplus_inv
from .transformer_model import TransformerEncoder


class DeclineForecastHead(nn.Module):
    """Multi-output head for decline forecasting.

    Takes encoder features (B, d_model) + T_frac scalar → 3 outputs:
      - decline_prob ∈ [0, 1] via sigmoid
      - delta_t_break ∈ R (unconstrained, normalized by T_max)
      - slope ≥ 0 via softplus
    """

    def __init__(self, d_model: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_model + 1, hidden_dim),  # +1 for T_frac
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decline_prob_head = nn.Linear(hidden_dim, 1)
        self.delta_head = nn.Linear(hidden_dim, 1)
        self.slope_head = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()

    def forward(
        self, features: torch.Tensor, T_frac: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            features: Encoder output, shape (B, d_model).
            T_frac: Observation fraction T/n_steps, shape (B,).

        Returns:
            Dict with decline_prob (B,), delta_t_break (B,), slope (B,).
        """
        combined = torch.cat([features, T_frac.unsqueeze(-1)], dim=-1)
        h = self.shared(combined)
        decline_prob = torch.sigmoid(self.decline_prob_head(h).squeeze(-1))
        delta_t_break = self.delta_head(h).squeeze(-1)
        slope = self.softplus(self.slope_head(h).squeeze(-1))
        return {
            "decline_prob": decline_prob,
            "delta_t_break": delta_t_break,
            "slope": slope,
        }


class DeclineForecastModel(nn.Module):
    """PatchTST encoder + DeclineForecastHead for decline forecasting.

    Architecture:
        Input (B, T, 28) → TransformerEncoder → MeanPool → (B, d_model)
        concat(features, T_frac) → DeclineForecastHead → 3 outputs
    """

    def __init__(
        self,
        n_features: int = 28,
        seq_len: int = 500,
        T_max: float = 300.0,
        patch_len: int = 16,
        patch_stride: int = 8,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        head_hidden: int = 32,
    ):
        super().__init__()
        self.T_max = T_max
        self.d_model = d_model

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
        )
        self.head = DeclineForecastHead(
            d_model=d_model,
            hidden_dim=head_hidden,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, T_frac: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input signals, shape (B, seq_len, n_features).
            T_frac: Observation fraction, shape (B,).

        Returns:
            Dict with decline_prob, delta_t_break, slope, each (B,).
        """
        features = self.encoder(x)  # (B, d_model)
        return self.head(features, T_frac)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract encoder features for CORAL loss.

        Args:
            x: Input signals, shape (B, seq_len, n_features).

        Returns:
            Features of shape (B, d_model).
        """
        return self.encoder(x)


def initialize_decline_forecast_head(
    model: DeclineForecastModel,
    fitted_params_df: pd.DataFrame,
    gate_threshold: float = 0.01,
    T_max: float | None = None,
) -> None:
    """Initialize head biases so cold-start model predicts dataset means.

    Args:
        model: DeclineForecastModel to initialize.
        fitted_params_df: Fitted parameters with t_break, slope columns.
        gate_threshold: Slope below this → no decline.
        T_max: Override for normalization constant. Uses model.T_max if None.
    """
    if T_max is None:
        T_max = model.T_max

    eps = 1e-6

    # decline_prob bias: logit of fraction with decline
    gate_frac = float((fitted_params_df["slope"] > gate_threshold).mean())
    gate_frac = max(eps, min(gate_frac, 1 - eps))
    prob_bias = _logit(torch.tensor(gate_frac))

    # delta_t_break bias: mean delta across all samples
    # At initialization, assume T_frac ~ 0.65 (midpoint of [0.4, 0.9])
    # delta = (t_break - T) / T_max; with T ~ 0.65 * T_max:
    mean_t_break = float(fitted_params_df["t_break"].mean())
    mean_T = 0.65 * T_max
    delta_bias = (mean_t_break - mean_T) / T_max

    # slope bias: softplus_inv of mean slope across declining batches
    declining = fitted_params_df[fitted_params_df["slope"] > gate_threshold]
    if len(declining) > 0:
        mean_slope = float(declining["slope"].mean())
    else:
        mean_slope = 0.01
    slope_bias = _softplus_inv(mean_slope)

    # Zero task-head weights so initial outputs equal bias (shared layer keeps Xavier init)
    head = model.head
    with torch.no_grad():
        head.decline_prob_head.weight.zero_()
        head.decline_prob_head.bias.fill_(prob_bias)

        head.delta_head.weight.zero_()
        head.delta_head.bias.fill_(delta_bias)

        head.slope_head.weight.zero_()
        head.slope_head.bias.fill_(slope_bias)
