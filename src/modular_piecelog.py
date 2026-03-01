"""Modular PieceLog sub-models with independent encoders.

Decomposes the piece-log prediction into 3 specialized sub-models:
  - TimingModel:  t_lag (delay→growth transition)
  - GrowthModel:  K, r, t0, lam (growth-phase logistic curve)
  - DeclineModel: t_break, slope, decline_gate (optional decline phase)

Each sub-model has its own PatchTST encoder + constrained head.
CompositeModel wraps all three for end-to-end inference.
"""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn

from .piecelog_model import PARAM_NAMES, piecelog_torch
from .piecelog_patchtst import _logit, _softplus_inv
from .transformer_model import TransformerEncoder


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class TimingHead(nn.Module):
    """MLP → 1 output: t_lag = sigmoid(raw) * 0.3 * T_max."""

    def __init__(self, d_model: int, hidden_dim: int = 32, T_max: float = 400.0):
        super().__init__()
        self.T_max = T_max
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.mlp(features).squeeze(-1)  # (B,)
        t_lag = torch.sigmoid(raw) * 0.3 * self.T_max
        return {"t_lag": t_lag}


class GrowthHead(nn.Module):
    """MLP → 4 outputs: K, r, t0, lam with physical constraints."""

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
            nn.Linear(hidden_dim, 4),
        )
        self.softplus = nn.Softplus()

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.mlp(features)  # (B, 4)
        K = self.softplus(raw[:, 0]) * self.K_scale
        r = self.softplus(raw[:, 1])
        t0 = torch.sigmoid(raw[:, 2]) * self.T_max
        lam = self.softplus(raw[:, 3])
        return {"K": K, "r": r, "t0": t0, "lam": lam}


class DeclineHead(nn.Module):
    """MLP → 3 outputs: t_break, slope, decline_gate.

    t_break is parameterized as sigmoid(raw) * T_max, covering [0, T_max].
    At init (bias = logit(mean_t_break / T_max)): t_break ≈ dataset mean.
    """

    def __init__(self, d_model: int, hidden_dim: int = 32, T_max: float = 400.0):
        super().__init__()
        self.T_max = T_max
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 1, hidden_dim),  # +1 for t_cutoff_norm
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.softplus = nn.Softplus()

    def forward(
        self, features: torch.Tensor, t_cutoff_norm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        combined = torch.cat([features, t_cutoff_norm.unsqueeze(-1)], dim=-1)
        raw = self.mlp(combined)  # (B, 3)
        t_break = torch.sigmoid(raw[:, 0]) * self.T_max
        slope = self.softplus(raw[:, 1])
        decline_gate = torch.sigmoid(raw[:, 2])
        return {"t_break": t_break, "slope": slope, "decline_gate": decline_gate}


class SplitDeclineHead(nn.Module):
    """Shared hidden → 3 independent task heads (matches step4c pattern).

    Each output has its own Linear layer, eliminating gradient interference:
      t_break = sigmoid(.) * T_max
      slope   = softplus(.)
      decline_gate = sigmoid(.)
    """

    def __init__(
        self, d_model: int, hidden_dim: int = 32, T_max: float = 400.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.T_max = T_max
        self.shared = nn.Sequential(
            nn.Linear(d_model + 1, hidden_dim),  # +1 for t_cutoff_norm
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.t_break_head = nn.Linear(hidden_dim, 1)
        self.slope_head = nn.Linear(hidden_dim, 1)
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()

    def forward(
        self, features: torch.Tensor, t_cutoff_norm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        combined = torch.cat([features, t_cutoff_norm.unsqueeze(-1)], dim=-1)
        h = self.shared(combined)
        t_break = torch.sigmoid(self.t_break_head(h).squeeze(-1)) * self.T_max
        slope = self.softplus(self.slope_head(h).squeeze(-1))
        decline_gate = torch.sigmoid(self.gate_head(h).squeeze(-1))
        return {"t_break": t_break, "slope": slope, "decline_gate": decline_gate}


class UngatedDeclineHead(nn.Module):
    """MLP → 2 outputs: t_break, slope. No gate — always 1.0.

    For no-decline batches, the model learns slope → 0 via softplus.
    This eliminates the slope*gate coupling problem entirely.

    t_break is parameterized as sigmoid(raw) * T_max, covering [0, T_max].
    At init (bias = logit(mean_t_break / T_max)): t_break ≈ dataset mean.
    """

    def __init__(self, d_model: int, hidden_dim: int = 32, T_max: float = 400.0):
        super().__init__()
        self.T_max = T_max
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 1, hidden_dim),  # +1 for t_cutoff_norm
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.softplus = nn.Softplus()

    def forward(
        self, features: torch.Tensor, t_cutoff_norm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        combined = torch.cat([features, t_cutoff_norm.unsqueeze(-1)], dim=-1)
        raw = self.mlp(combined)  # (B, 2)
        t_break = torch.sigmoid(raw[:, 0]) * self.T_max
        slope = self.softplus(raw[:, 1])
        decline_gate = torch.ones(features.shape[0], device=features.device)
        return {"t_break": t_break, "slope": slope, "decline_gate": decline_gate}


# ---------------------------------------------------------------------------
# Sub-models (encoder + head)
# ---------------------------------------------------------------------------

class TimingModel(nn.Module):
    """Independent encoder → TimingHead → t_lag."""

    def __init__(
        self,
        n_features: int = 28,
        seq_len: int = 200,
        T_max: float = 400.0,
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
        self.encoder = TransformerEncoder(
            n_features=n_features, seq_len=seq_len,
            patch_len=patch_len, patch_stride=patch_stride,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, dropout=dropout,
        )
        self.head = TimingHead(d_model, hidden_dim=head_hidden, T_max=T_max)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        return self.head(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class GrowthModel(nn.Module):
    """Independent encoder → GrowthHead → K, r, t0, lam."""

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
        super().__init__()
        self.encoder = TransformerEncoder(
            n_features=n_features, seq_len=seq_len,
            patch_len=patch_len, patch_stride=patch_stride,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, dropout=dropout,
        )
        self.head = GrowthHead(
            d_model, hidden_dim=head_hidden, T_max=T_max, K_scale=K_scale,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        return self.head(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DeclineModel(nn.Module):
    """Independent encoder → DeclineHead → t_break, slope, decline_gate."""

    def __init__(
        self,
        n_features: int = 28,
        seq_len: int = 200,
        T_max: float = 400.0,
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
        self.encoder = TransformerEncoder(
            n_features=n_features, seq_len=seq_len,
            patch_len=patch_len, patch_stride=patch_stride,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, dropout=dropout,
        )
        self.head = DeclineHead(d_model, hidden_dim=head_hidden, T_max=T_max)

    def forward(
        self, x: torch.Tensor, t_cutoff_norm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        return self.head(features, t_cutoff_norm)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class UngatedDeclineModel(nn.Module):
    """Independent encoder → UngatedDeclineHead → t_break, slope (gate=1.0).

    Drop-in replacement for DeclineModel. The constant gate=1.0 means
    CompositeModel.get_parameters() works unchanged.
    """

    def __init__(
        self,
        n_features: int = 28,
        seq_len: int = 200,
        T_max: float = 400.0,
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
        self.encoder = TransformerEncoder(
            n_features=n_features, seq_len=seq_len,
            patch_len=patch_len, patch_stride=patch_stride,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, dropout=dropout,
        )
        self.head = UngatedDeclineHead(d_model, hidden_dim=head_hidden, T_max=T_max)

    def forward(
        self, x: torch.Tensor, t_cutoff_norm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        return self.head(features, t_cutoff_norm)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SplitDeclineModel(nn.Module):
    """Independent encoder → SplitDeclineHead → t_break, slope, decline_gate.

    Drop-in replacement for DeclineModel with separate task heads
    to eliminate gradient interference between outputs.
    """

    def __init__(
        self,
        n_features: int = 28,
        seq_len: int = 200,
        T_max: float = 400.0,
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
        self.encoder = TransformerEncoder(
            n_features=n_features, seq_len=seq_len,
            patch_len=patch_len, patch_stride=patch_stride,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, dropout=dropout,
        )
        self.head = SplitDeclineHead(
            d_model, hidden_dim=head_hidden, T_max=T_max, dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, t_cutoff_norm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        return self.head(features, t_cutoff_norm)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Composite model
# ---------------------------------------------------------------------------

class CompositeModel(nn.Module):
    """Wraps all 3 sub-models for full piece-log inference.

    Assembles 7 parameters from the sub-models, applies decline gate,
    and evaluates the differentiable piece-log forward model.
    """

    def __init__(
        self,
        timing_model: TimingModel,
        growth_model: GrowthModel,
        decline_model: DeclineModel,
    ):
        super().__init__()
        self.timing_model = timing_model
        self.growth_model = growth_model
        self.decline_model = decline_model

    def get_parameters(
        self,
        x: torch.Tensor,
        t_cutoff: torch.Tensor | None = None,
        hard_gate: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Get all 7 piece-log parameters from sub-models.

        Args:
            x: Input time-series, shape (B, T, C).
            t_cutoff: Absolute time at end of input window, shape (B,).
                If None, defaults to 0.9 * T_max (typical input window end).
            hard_gate: If True, use hard gate (gate >= 0.5 → slope, else 0).
                If False, use soft gate (slope * gate) for gradient flow.

        Returns:
            Dict with keys K, r, t0, lam, t_lag, t_break, slope, each (B,).
        """
        T_max = self.decline_model.head.T_max
        if t_cutoff is None:
            t_cutoff = torch.full((x.shape[0],), 0.9 * T_max, device=x.device)
        t_cutoff_norm = t_cutoff / T_max

        timing_params = self.timing_model(x)
        growth_params = self.growth_model(x)
        decline_params = self.decline_model(x, t_cutoff_norm)

        gate = decline_params["decline_gate"]
        raw_slope = decline_params["slope"]

        if hard_gate:
            slope = torch.where(gate >= 0.5, raw_slope, torch.zeros_like(raw_slope))
        else:
            slope = raw_slope * gate

        return {
            "K": growth_params["K"],
            "r": growth_params["r"],
            "t0": growth_params["t0"],
            "lam": growth_params["lam"],
            "t_lag": timing_params["t_lag"],
            "t_break": decline_params["t_break"],
            "slope": slope,
        }

    def forward(
        self,
        x: torch.Tensor,
        t_predict: torch.Tensor,
        t_cutoff: torch.Tensor | None = None,
        hard_gate: bool = False,
    ) -> torch.Tensor:
        """Predict concentration at t_predict.

        Args:
            x: Input time-series, shape (B, T, C).
            t_predict: Prediction times, shape (B,).
            t_cutoff: Absolute time at end of input window, shape (B,).
                If None, defaults to 0.9 * T_max.
            hard_gate: Use hard gate for inference.

        Returns:
            Predicted concentration, shape (B,).
        """
        params = self.get_parameters(x, t_cutoff=t_cutoff, hard_gate=hard_gate)
        return piecelog_torch(
            t_predict,
            params["K"], params["r"], params["t0"], params["lam"],
            params["t_lag"], params["t_break"], params["slope"],
        )

    def get_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract encoder features from all sub-models (for CORAL loss).

        Returns:
            Dict with keys 'timing', 'growth', 'decline', each (B, d_model).
        """
        return {
            "timing": self.timing_model.get_features(x),
            "growth": self.growth_model.get_features(x),
            "decline": self.decline_model.get_features(x),
        }


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------

def initialize_timing_head(
    model: TimingModel,
    fitted_params_df: pd.DataFrame,
) -> None:
    """Initialize TimingHead bias so output matches mean t_lag."""
    mean_t_lag = fitted_params_df["t_lag"].mean()
    T_max = model.head.T_max
    eps = 1e-6
    raw = _logit(torch.clamp(torch.tensor(mean_t_lag / (0.3 * T_max)), eps, 1 - eps))

    output_layer = model.head.mlp[2]
    with torch.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.fill_(raw)


def initialize_growth_head(
    model: GrowthModel,
    fitted_params_df: pd.DataFrame,
) -> None:
    """Initialize GrowthHead bias so output matches mean K, r, t0, lam."""
    means = {p: fitted_params_df[p].mean() for p in ["K", "r", "t0", "lam"]}
    T_max = model.head.T_max
    K_scale = model.head.K_scale
    eps = 1e-6

    raw = torch.zeros(4)
    raw[0] = _softplus_inv(means["K"] / K_scale)
    raw[1] = _softplus_inv(means["r"])
    raw[2] = _logit(torch.clamp(torch.tensor(means["t0"] / T_max), eps, 1 - eps))
    raw[3] = _softplus_inv(means["lam"])

    output_layer = model.head.mlp[2]
    with torch.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.copy_(raw)


def initialize_decline_head(
    model: DeclineModel,
    fitted_params_df: pd.DataFrame,
    gate_threshold: float = 0.01,
) -> None:
    """Initialize DeclineHead bias so output matches mean t_break, slope, gate.

    Args:
        model: DeclineModel to initialize.
        fitted_params_df: Fitted parameters DataFrame.
        gate_threshold: Slope values below this are considered "no decline".
    """
    means = {p: fitted_params_df[p].mean() for p in ["t_break", "slope"]}
    T_max = model.head.T_max
    eps = 1e-6

    # Gate target: fraction of batches with meaningful decline
    gate_frac = (fitted_params_df["slope"] > gate_threshold).mean()
    gate_frac = max(eps, min(gate_frac, 1 - eps))

    raw = torch.zeros(3)
    raw[0] = _logit(torch.clamp(torch.tensor(means["t_break"] / T_max), eps, 1 - eps))
    raw[1] = _softplus_inv(means["slope"])
    raw[2] = _logit(torch.tensor(gate_frac))

    output_layer = model.head.mlp[2]
    with torch.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.copy_(raw)


def initialize_ungated_decline_head(
    model: UngatedDeclineModel,
    fitted_params_df: pd.DataFrame,
) -> None:
    """Initialize UngatedDeclineHead bias so output matches mean t_break, slope."""
    means = {p: fitted_params_df[p].mean() for p in ["t_break", "slope"]}
    T_max = model.head.T_max
    eps = 1e-6

    raw = torch.zeros(2)
    raw[0] = _logit(torch.clamp(torch.tensor(means["t_break"] / T_max), eps, 1 - eps))
    raw[1] = _softplus_inv(means["slope"])

    output_layer = model.head.mlp[2]
    with torch.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.copy_(raw)


def initialize_split_decline_head(
    model: SplitDeclineModel,
    fitted_params_df: pd.DataFrame,
    gate_threshold: float = 0.01,
) -> None:
    """Initialize SplitDeclineHead: zero task-head weights, set biases from data.

    Matches step4c's pattern: only the 3 task-head weights are zeroed so
    initial output equals the bias. The shared layer keeps its Xavier init.

    Args:
        model: SplitDeclineModel to initialize.
        fitted_params_df: Fitted parameters DataFrame.
        gate_threshold: Slope values below this are considered "no decline".
    """
    means = {p: fitted_params_df[p].mean() for p in ["t_break", "slope"]}
    T_max = model.head.T_max
    eps = 1e-6

    # Gate target: fraction of batches with meaningful decline
    gate_frac = (fitted_params_df["slope"] > gate_threshold).mean()
    gate_frac = max(eps, min(gate_frac, 1 - eps))

    head = model.head
    with torch.no_grad():
        # t_break head
        head.t_break_head.weight.zero_()
        head.t_break_head.bias.fill_(
            _logit(torch.clamp(torch.tensor(means["t_break"] / T_max), eps, 1 - eps))
        )
        # slope head
        head.slope_head.weight.zero_()
        head.slope_head.bias.fill_(_softplus_inv(means["slope"]))
        # gate head
        head.gate_head.weight.zero_()
        head.gate_head.bias.fill_(_logit(torch.tensor(gate_frac)))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_modular_piecelog(
    n_features: int = 28,
    seq_len: int = 200,
    T_max: float = 400.0,
    config: dict | None = None,
    timing_config: dict | None = None,
    growth_config: dict | None = None,
    decline_config: dict | None = None,
) -> CompositeModel:
    """Create CompositeModel with 3 independent sub-models.

    Args:
        n_features: Number of input features.
        seq_len: Input sequence length.
        T_max: Maximum batch duration.
        config: Override default hyperparameters (shared across all sub-models).
        timing_config: Per-model overrides for TimingModel (on top of config).
        growth_config: Per-model overrides for GrowthModel (on top of config).
        decline_config: Per-model overrides for DeclineModel (on top of config).

    Returns:
        CompositeModel wrapping TimingModel, GrowthModel, DeclineModel.
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

    def _model_kwargs(override: dict | None) -> dict:
        cfg = dict(defaults)
        if override is not None:
            cfg.update(override)
        return dict(
            n_features=n_features,
            seq_len=seq_len,
            patch_len=cfg["patch_len"],
            patch_stride=cfg["patch_stride"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
            head_hidden=cfg["head_hidden"],
        )

    # Growth needs K_scale from its merged config
    growth_cfg = dict(defaults)
    if growth_config is not None:
        growth_cfg.update(growth_config)

    timing = TimingModel(T_max=T_max, **_model_kwargs(timing_config))
    growth = GrowthModel(T_max=T_max, K_scale=growth_cfg["K_scale"], **_model_kwargs(growth_config))
    decline = DeclineModel(T_max=T_max, **_model_kwargs(decline_config))

    return CompositeModel(timing, growth, decline)
