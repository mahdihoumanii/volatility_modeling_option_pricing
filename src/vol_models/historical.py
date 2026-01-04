"""
Historical/rolling volatility estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.returns import TRADING_DAYS


def rolling_vol(
    returns: pd.Series,
    window: int,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """
    Rolling standard deviation of returns.

    Parameters
    ----------
    returns : pd.Series
        Return series.
    window : int
        Rolling window length in days.
    annualize : bool
        If True, multiply by sqrt(periods_per_year).
    periods_per_year : int
        Trading periods per year for annualization.

    Returns
    -------
    pd.Series
        Volatility series.
    """
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    vol.name = f"hist_vol_{window}"
    return vol


# Backward compatibility alias
rolling_volatility = rolling_vol


def forecast_next_variance(train_returns: pd.Series, window: int) -> float:
    """
    One-step-ahead variance forecast using rolling sample variance.
    """
    recent = train_returns.dropna().tail(window)
    if len(recent) < window:
        raise ValueError("Not enough observations for the specified window.")
    variance = recent.var(ddof=1)
    return float(variance)
