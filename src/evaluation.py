"""
Utilities for time-series splitting and forecast evaluation.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from src.returns import TRADING_DAYS


def walk_forward_forecast(
    returns: pd.Series,
    model_funcs: Dict[str, Callable[[pd.Series], float]],
    start: int,
    expanding: bool = True,
    window: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate one-step-ahead variance forecasts with walk-forward splitting.

    Parameters
    ----------
    returns : pd.Series
        Return series (chronological).
    model_funcs : dict
        Mapping model name -> callable(train_returns) -> forecast variance.
    start : int
        Index to start forecasting (size of initial training set).
    expanding : bool
        If True, training set grows; otherwise use fixed rolling window.
    window : int, optional
        Rolling window length when expanding is False.
    """
    if not expanding and window is None:
        raise ValueError("Provide window when using rolling (expanding=False).")
    forecasts = {name: [] for name in model_funcs}
    index: list[pd.Timestamp] = []
    for t in range(start, len(returns)):
        train = returns.iloc[:t] if expanding else returns.iloc[t - window : t]
        for name, func in model_funcs.items():
            forecasts[name].append(func(train))
        index.append(returns.index[t])
    forecast_df = pd.DataFrame(forecasts, index=index)
    return forecast_df


def mse_variance(pred: pd.Series, realized_var: pd.Series) -> float:
    aligned_pred, aligned_real = pred.align(realized_var, join="inner")
    return float(((aligned_pred - aligned_real) ** 2).mean())


def mae_volatility(pred_vol: pd.Series, realized_var: pd.Series) -> float:
    aligned_vol, aligned_var = pred_vol.align(realized_var, join="inner")
    return float((aligned_vol - np.sqrt(aligned_var)).abs().mean())


def qlike_loss(pred_var: pd.Series, realized_var: pd.Series, eps: float = 1e-8) -> float:
    aligned_pred, aligned_real = pred_var.align(realized_var, join="inner")
    safe_pred = aligned_pred.clip(lower=eps)
    loss = np.log(safe_pred) + aligned_real / safe_pred
    return float(loss.mean())


def windowed_realized_volatility(
    returns: pd.Series,
    windows: Sequence[int],
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """
    Compute windowed realized volatility (annualized) for multiple windows.

    RV_t(w) = sqrt(periods_per_year / w * sum_{i=0}^{w-1} r_{t-i}^2)
    """
    r2 = returns**2
    data = {}
    for w in windows:
        if w <= 0:
            raise ValueError("Window must be positive.")
        rolling_sum = r2.rolling(window=w).sum()
        rv = np.sqrt((periods_per_year / w) * rolling_sum)
        data[f"rv_{w}"] = rv
    return pd.DataFrame(data, index=returns.index)


def windowed_realized_variance(
    returns: pd.Series,
    windows: Sequence[int],
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """
    Realized variance corresponding to windowed realized volatility.
    """
    rv = windowed_realized_volatility(returns, windows, periods_per_year)
    return rv**2


def evaluate_forecasts(
    forecast_var: pd.DataFrame,
    returns: pd.Series,
    realized_windows: Optional[Sequence[int]] = None,
    annualize_pred: bool = False,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """
    Evaluate variance forecasts using MSE, MAE (on volatility), and QLIKE.

    Parameters
    ----------
    forecast_var : pd.DataFrame
        Forecasted variances (daily variance unless annualized externally).
    returns : pd.Series
        Return series.
    realized_windows : sequence of int, optional
        If provided, compute realized variance using windowed RV for each window.
        When None, fall back to single-period squared returns.
    annualize_pred : bool
        If True, forecast_var will be scaled by periods_per_year before scoring
        to match annualized realized variance.
    periods_per_year : int
        Trading periods per year for scaling.
    """
    metrics = []

    if realized_windows is None:
        realized_var = (returns**2).loc[forecast_var.index]
        for col in forecast_var.columns:
            var_pred = forecast_var[col]
            vol_pred = np.sqrt(var_pred)
            metrics.append(
                {
                    "model": col,
                    "mse_var": mse_variance(var_pred, realized_var),
                    "mae_vol": mae_volatility(vol_pred, realized_var),
                    "qlike": qlike_loss(var_pred, realized_var),
                }
            )
    else:
        realized_var_df = windowed_realized_variance(
            returns, windows=realized_windows, periods_per_year=periods_per_year
        )
        scale = periods_per_year if annualize_pred else 1.0
        for col in forecast_var.columns:
            var_pred_raw = forecast_var[col]
            var_pred = var_pred_raw * scale
            for w in realized_windows:
                realized_var = realized_var_df[f"rv_{w}"].loc[var_pred.index]
                vol_pred = np.sqrt(var_pred)
                metrics.append(
                    {
                        "model": col,
                        "window": w,
                        "mse_var": mse_variance(var_pred, realized_var),
                        "mae_vol": mae_volatility(vol_pred, realized_var),
                        "qlike": qlike_loss(var_pred, realized_var),
                    }
                )

    df_metrics = pd.DataFrame(metrics)
    if "window" in df_metrics.columns:
        df_metrics.set_index(["model", "window"], inplace=True)
    else:
        df_metrics.set_index("model", inplace=True)
    return df_metrics
