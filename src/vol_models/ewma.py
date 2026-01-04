"""
Exponentially weighted moving average variance model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.returns import TRADING_DAYS


def ewma_variance(returns: pd.Series, lam: float = 0.94, initial_variance: float | None = None) -> pd.Series:
    """
    Compute EWMA variance recursively.

    Parameters
    ----------
    returns : pd.Series
        Return series.
    lam : float
        Decay parameter lambda.
    initial_variance : float, optional
        Starting variance. Defaults to sample variance if None.

    Returns
    -------
    pd.Series
        EWMA variance series.
    """
    if not 0 < lam < 1:
        raise ValueError("lambda must be in (0,1)")
    r2 = returns.dropna() ** 2
    var = np.zeros(len(r2))
    if initial_variance is None:
        initial_variance = float(r2.iloc[:10].mean()) if len(r2) >= 10 else float(r2.mean())
    var[0] = initial_variance
    for i in range(1, len(r2)):
        var[i] = lam * var[i - 1] + (1 - lam) * r2.iloc[i - 1]
    variance = pd.Series(var, index=r2.index, name="ewma_variance")
    return variance


def ewma_vol(
    returns: pd.Series,
    lam: float = 0.94,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """
    EWMA volatility series.
    """
    variance = ewma_variance(returns, lam)
    vol = np.sqrt(variance)
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    vol.name = "ewma_vol"
    return vol


# Backward compatibility alias
ewma_volatility = ewma_vol


def forecast_next_variance(train_returns: pd.Series, lam: float = 0.94, initial_variance: float | None = None) -> float:
    """
    One-step-ahead EWMA variance forecast.
    """
    variance_series = ewma_variance(train_returns, lam=lam, initial_variance=initial_variance)
    return float(variance_series.iloc[-1])
