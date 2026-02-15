"""PatchTST encoder with constrained piece-log parameter head.

Architecture:
    Input (B, T, 28) -> PatchTST Encoder -> ConstrainedParamHead -> 7 params
    -> Differentiable piece-log forward -> P_hat(t_predict)
"""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn

from .piecelog_model import PARAM_NAMES, piecelog_torch
from .transformer_model import TransformerEncoder


def _softplus_inv(y: float, min_val: float = 1e-6) -> float:
    """Inverse of softplus: log(exp(y) - 1), clamped for numerical safety."""
    y = max(y, min_val)
    # For large y, softplus_inv(y) ≈ y (since exp(y) >> 1)
    if y > 20.0:
        return y
    return float(torch.log(torch.tensor(y).exp() - 1.0))


def _logit(x: torch.Tensor) -> float:
    """Logit function: log(x / (1 - x))."""
    return float(torch.log(x / (1.0 - x)))


def initialize_param_head(
    model: "PieceLogPatchTST",
    fitted_params_df: pd.DataFrame,
) -> None:
    """Initialize param_head so output matches average fitted parameters.

    Zeroes the output layer weights and sets bias to inverted-constraint
    mean values, so the head outputs constant (mean) parameters regardless
    of encoder input.
    """
    means = {p: fitted_params_df[p].mean() for p in PARAM_NAMES}
    T_max = model.param_head.T_max
    K_scale = model.param_head.K_scale

    eps = 1e-6
    raw = torch.zeros(7)
    raw[0] = _softplus_inv(means["K"] / K_scale)
    raw[1] = _softplus_inv(means["r"])
    raw[2] = _logit(torch.clamp(torch.tensor(means["t0"] / T_max), eps, 1 - eps))
    raw[3] = _softplus_inv(means["lam"])
    raw[4] = _logit(torch.clamp(torch.tensor(means["t_lag"] / (0.3 * T_max)), eps, 1 - eps))
    raw[5] = _logit(torch.clamp(torch.tensor((means["t_break"] - 0.5 * T_max) / (0.5 * T_max)), eps, 1 - eps))
    raw[6] = _softplus_inv(means["slope"])

    output_layer = model.param_head.mlp[2]
    with torch.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.copy_(raw)


class ConstrainedParameterHead(nn.Module):
    """MLP that outputs 7 physically-constrained piece-log parameters.

    Constraints:
        K      = softplus(.) * K_scale       positive, scaled ~0-60 g/L
        r      = softplus(.)                 positive growth rate
        t0     = sigmoid(.) * T_max          inflection in [0, T_max]
        lam    = softplus(.)                 positive lag rate
        t_lag  = sigmoid(.) * 0.3 * T_max    delay in [0, 30% of batch]
        t_break= 0.5*T_max + sigmoid(.)*0.5*T_max   in [50%, 100%]
        slope  = softplus(.)                 positive decline rate
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 32,
        T_max: float = 400.0,
        K_scale: float = 60.0,
    ):
        super().__init__()
        self.T_max = T_max
        self.K_scale = K_scale

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 7),
        )
        self.softplus = nn.Softplus()

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict constrained piece-log parameters.

        Args:
            features: Encoder output, shape (B, d_model).

        Returns:
            Dict with keys K, r, t0, lam, t_lag, t_break, slope, each (B,).
        """
        raw = self.mlp(features)  # (B, 7)

        K = self.softplus(raw[:, 0]) * self.K_scale
        r = self.softplus(raw[:, 1])
        t0 = torch.sigmoid(raw[:, 2]) * self.T_max
        lam = self.softplus(raw[:, 3])
        t_lag = torch.sigmoid(raw[:, 4]) * 0.3 * self.T_max
        t_break = 0.5 * self.T_max + torch.sigmoid(raw[:, 5]) * 0.5 * self.T_max
        slope = self.softplus(raw[:, 6])

        return {
            "K": K,
            "r": r,
            "t0": t0,
            "lam": lam,
            "t_lag": t_lag,
            "t_break": t_break,
            "slope": slope,
        }


class PieceLogPatchTST(nn.Module):
    """PatchTST encoder -> constrained parameters -> differentiable piece-log.

    Predicts penicillin concentration at arbitrary future time t_predict
    by first predicting the 7 piece-log parameters from process signals,
    then evaluating the parametric model.
    """

    def __init__(
        self,
        n_features: int = 28,
        seq_len: int = 200,
        T_max: float = 400.0,
        K_scale: float = 60.0,
        patch_len: int = 16,
        patch_stride: int = 8,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        head_hidden: int = 32,
    ):
        """Initialize PieceLog-PatchTST.

        Args:
            n_features: Number of input channels (26 process + 3 conc).
            seq_len: Input sequence length.
            T_max: Maximum batch duration (hours) for parameter scaling.
            K_scale: Scale for carrying capacity K.
            patch_len: Patch length for PatchTST.
            patch_stride: Patch stride.
            d_model: Transformer embedding dimension.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            d_ff: Feed-forward dimension.
            dropout: Dropout probability.
            head_hidden: Hidden dimension for parameter head.
        """
        super().__init__()

        self.n_features = n_features
        self.seq_len = seq_len
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

        self.param_head = ConstrainedParameterHead(
            d_model=d_model,
            hidden_dim=head_hidden,
            T_max=T_max,
            K_scale=K_scale,
        )

    def forward(self, x: torch.Tensor, t_predict: torch.Tensor) -> torch.Tensor:
        """Predict concentration at t_predict.

        Args:
            x: Input time-series, shape (B, T, C).
            t_predict: Prediction times, shape (B,).

        Returns:
            Predicted concentration, shape (B,).
        """
        params = self.get_parameters(x)
        return piecelog_torch(
            t_predict,
            params["K"], params["r"], params["t0"], params["lam"],
            params["t_lag"], params["t_break"], params["slope"],
        )

    def get_parameters(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract predicted piece-log parameters.

        Args:
            x: Input time-series, shape (B, T, C).

        Returns:
            Dict with 7 parameter tensors, each (B,).
        """
        features = self.encoder(x)  # (B, d_model)
        return self.param_head(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract encoder features (for CORAL loss).

        Args:
            x: Input time-series, shape (B, T, C).

        Returns:
            Features, shape (B, d_model).
        """
        return self.encoder(x)


def create_piecelog_model(
    n_features: int = 28,
    seq_len: int = 200,
    T_max: float = 400.0,
    config: dict | None = None,
) -> PieceLogPatchTST:
    """Create PieceLog-PatchTST model with given configuration.

    Args:
        n_features: Number of input features.
        seq_len: Input sequence length.
        T_max: Maximum batch duration.
        config: Override default hyperparameters.

    Returns:
        PieceLogPatchTST model.
    """
    defaults = {
        "K_scale": 60.0,
        "patch_len": 16,
        "patch_stride": 8,
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 128,
        "dropout": 0.1,
        "head_hidden": 32,
    }
    if config is not None:
        defaults.update(config)

    return PieceLogPatchTST(
        n_features=n_features,
        seq_len=seq_len,
        T_max=T_max,
        **defaults,
    )
