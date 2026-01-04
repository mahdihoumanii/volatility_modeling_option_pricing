"""
Return calculations and utility helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log returns r_t = ln(P_t / P_{t-1}).

    Parameters
    ----------
    prices : pd.Series
        Price series indexed by datetime.

    Returns
    -------
    pd.Series
        Log returns aligned with prices (first value NaN).
    """
    shifted = prices.shift(1)
    returns = np.log(prices / shifted)
    returns.name = "log_return"
    return returns


def annualize_volatility(daily_vol: pd.Series, trading_days: int = TRADING_DAYS) -> pd.Series:
    """
    Annualize daily volatility.
    """
    return daily_vol * np.sqrt(trading_days)
