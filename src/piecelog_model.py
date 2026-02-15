"""Piece-log parametric model for penicillin concentration.

Three-phase model: delay -> growth -> decline.
  P(t) = 0                                              for t < t_lag
  P(t) = [K / (1+e^{-r(tau-t0)})] * (1-e^{-lam*tau})   for t_lag <= t < t_break
  P(t) = P(t_break) - slope*(t - t_break)               for t >= t_break

where tau = t - t_lag. Seven parameters: K, r, t0, lam, t_lag, t_break, slope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit

PARAM_NAMES = ["K", "r", "t0", "lam", "t_lag", "t_break", "slope"]


def piecelog_numpy(
    t: np.ndarray,
    K: float,
    r: float,
    t0: float,
    lam: float,
    t_lag: float,
    t_break: float,
    slope: float,
) -> np.ndarray:
    """Evaluate piece-log model (NumPy, for scipy fitting).

    Args:
        t: Time array.
        K: Carrying capacity (max concentration).
        r: Growth rate.
        t0: Inflection point (relative to t_lag).
        lam: Lag rate.
        t_lag: Delay duration where P = 0.
        t_break: Onset of decline phase.
        slope: Rate of linear decline.

    Returns:
        Concentration array.
    """
    t = np.asarray(t, dtype=float)
    result = np.zeros_like(t)

    # Phase 2: Growth (t_lag <= t < t_break)
    growth_mask = (t >= t_lag) & (t < t_break)
    if np.any(growth_mask):
        tau = t[growth_mask] - t_lag
        logistic = K / (1 + np.exp(-r * (tau - t0)))
        lag_factor = 1 - np.exp(-lam * tau)
        result[growth_mask] = logistic * lag_factor

    # Phase 3: Decline (t >= t_break)
    decline_mask = t >= t_break
    if np.any(decline_mask):
        tau_break = t_break - t_lag
        P_at_break = K / (1 + np.exp(-r * (tau_break - t0))) * (1 - np.exp(-lam * tau_break))
        result[decline_mask] = P_at_break - slope * (t[decline_mask] - t_break)
        result[result < 0] = 0  # Floor at zero

    return result


def piecelog_torch(
    t: torch.Tensor,
    K: torch.Tensor,
    r: torch.Tensor,
    t0: torch.Tensor,
    lam: torch.Tensor,
    t_lag: torch.Tensor,
    t_break: torch.Tensor,
    slope: torch.Tensor,
) -> torch.Tensor:
    """Evaluate piece-log model (PyTorch, differentiable).

    All parameter tensors should be shape (B,) or scalar.
    t should be shape (B,).

    Args:
        t: Time points, shape (B,).
        K, r, t0, lam, t_lag, t_break, slope: Parameters, each (B,).

    Returns:
        Concentration values, shape (B,).
    """
    tau = t - t_lag
    tau_break = t_break - t_lag

    # Clamp exp arguments to prevent overflow (exp(88) ~ 1e38 ~ float32 max)
    _EXP_CLAMP = 80.0

    # Growth phase value
    logistic = K / (1 + torch.exp(torch.clamp(-r * (tau - t0), -_EXP_CLAMP, _EXP_CLAMP)))
    lag_factor = 1 - torch.exp(torch.clamp(-lam * tau, -_EXP_CLAMP, _EXP_CLAMP))
    growth_val = logistic * lag_factor

    # Value at breakpoint (for decline continuity)
    logistic_break = K / (1 + torch.exp(torch.clamp(-r * (tau_break - t0), -_EXP_CLAMP, _EXP_CLAMP)))
    lag_break = 1 - torch.exp(torch.clamp(-lam * tau_break, -_EXP_CLAMP, _EXP_CLAMP))
    P_at_break = logistic_break * lag_break

    # Decline phase value
    decline_val = P_at_break - slope * (t - t_break)

    # Select phase using torch.where (differentiable)
    result = torch.where(t < t_lag, torch.zeros_like(t), growth_val)
    result = torch.where(t >= t_break, decline_val, result)
    result = torch.clamp(result, min=0.0)

    return result


def fit_piecelog(
    t: np.ndarray,
    y: np.ndarray,
    maxfev: int = 10000,
) -> dict:
    """Fit piece-log model to time-concentration data.

    Args:
        t: Time array.
        y: Concentration array.
        maxfev: Maximum function evaluations for curve_fit.

    Returns:
        Dict with 'params' (dict of 7 values), 'r_squared', 'success'.
    """
    # Remove NaN values
    valid = ~np.isnan(y) & ~np.isnan(t)
    t = t[valid].astype(float)
    y = y[valid].astype(float)

    if len(t) < 10:
        return {"params": None, "r_squared": 0.0, "success": False}

    T_max = float(t.max())

    # Initial parameter estimates
    threshold = 0.01 * y.max()
    first_significant = np.where(y > threshold)[0]
    t_lag_init = float(t[first_significant[0]]) if len(first_significant) > 0 else 10.0
    t_lag_init = max(0.1, min(t_lag_init, T_max * 0.3))

    K0 = float(y.max()) * 1.1
    t0_init = float(t[np.argmax(np.gradient(y))]) - t_lag_init
    t0_init = max(10.0, t0_init)
    r0 = 0.1
    lam0 = 0.1

    t_break_init = float(t[np.argmax(y)])
    t_break_init = max(T_max * 0.5, min(t_break_init, T_max - 1))

    # Estimate late-stage slope
    late_idx = int(len(y) * 0.7)
    if late_idx < len(y) - 1:
        late_slope_fit = np.polyfit(t[late_idx:], y[late_idx:], 1)
        slope_init = max(0.01, -late_slope_fit[0]) if late_slope_fit[0] < 0 else 0.01
    else:
        slope_init = 0.01

    p0 = [K0, r0, t0_init, lam0, t_lag_init, t_break_init, slope_init]
    bounds = (
        [0, 0.001, 0, 0.001, 0, T_max * 0.5, 0],
        [100, 2, T_max, 2, T_max * 0.3, T_max, 2],
    )

    try:
        popt, _ = curve_fit(
            piecelog_numpy, t, y,
            p0=p0,
            bounds=bounds,
            maxfev=maxfev,
        )

        y_pred = piecelog_numpy(t, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        params = dict(zip(PARAM_NAMES, popt))
        return {"params": params, "r_squared": r_squared, "success": True}

    except (RuntimeError, ValueError):
        return {"params": None, "r_squared": 0.0, "success": False}


def fit_all_batches(
    batches: dict[int, pd.DataFrame],
    exclude_faults: bool = True,
    target_col: str = "P",
    maxfev: int = 10000,
) -> pd.DataFrame:
    """Fit piece-log model to all batches.

    Args:
        batches: Dict of batch_id -> DataFrame.
        exclude_faults: If True, skip batches 91-100.
        target_col: Concentration column name.
        maxfev: Maximum function evaluations per fit.

    Returns:
        DataFrame with columns: batch_id, K, r, t0, lam, t_lag, t_break, slope, r_squared.
    """
    rows = []
    for batch_id, df in sorted(batches.items()):
        if exclude_faults and batch_id > 90:
            continue

        t = df["time"].values
        y = df[target_col].values
        result = fit_piecelog(t, y, maxfev=maxfev)

        if result["success"]:
            row = {"batch_id": batch_id, **result["params"], "r_squared": result["r_squared"]}
        else:
            row = {"batch_id": batch_id, **{p: np.nan for p in PARAM_NAMES}, "r_squared": 0.0}
        rows.append(row)

    return pd.DataFrame(rows)
