"""
GARCH(1,1) volatility modeling using the arch package.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from src.returns import TRADING_DAYS

try:
    from arch import arch_model
except ImportError as exc:  # pragma: no cover - defensive for environments without arch
    raise ImportError("The 'arch' package is required for GARCH models.") from exc


def fit_garch(
    returns: pd.Series,
    mean: str = "Zero",
    vol: str = "GARCH",
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    options: Optional[Dict[str, Any]] = None,
    scale: float = 100.0,
):
    """
    Fit a GARCH model to returns.

    Parameters
    ----------
    returns : pd.Series
        Return series (will be scaled by `scale` before fitting).
    mean : str
        Mean model (default Zero to avoid mean leakage).
    vol : str
        Volatility process.
    p : int
        GARCH order.
    q : int
        ARCH order.
    dist : str
        Distribution.
    options : dict, optional
        Additional fit options passed to fit().
    scale : float
        Multiplier to rescale returns (e.g., 100 for percent returns) to improve optimizer stability.

    Returns
    -------
    arch.univariate.base.ARCHModelResult
        Fitted model result (with attribute `_scale` set to the scaling factor).
    """
    options = options or {"disp": "off"}
    returns_scaled = returns.dropna() * scale
    model = arch_model(returns_scaled, mean=mean, vol=vol, p=p, q=q, dist=dist, rescale=False)
    res = model.fit(**options)
    res._scale = scale  # type: ignore[attr-defined]
    return res


def forecast_next_variance(fitted_model, horizon: int = 1) -> float:
    """
    One-step-ahead variance forecast from a fitted GARCH model.
    """
    forecast = fitted_model.forecast(horizon=horizon, reindex=False)
    variance = forecast.variance.iloc[-1, horizon - 1]
    scale = getattr(fitted_model, "_scale", 1.0)
    variance = variance / (scale**2)
    return float(variance)


def fit_garch11(
    returns: pd.Series,
    dist: str = "normal",
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS,
    scale: float = 100.0,
    fit_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fit a GARCH(1,1) model and return conditional volatility (annualized by default).

    Returns a dictionary with keys:
    - result: fitted ARCHModelResult
    - cond_vol: pd.Series of conditional volatility (annualized if requested)
    """
    fit_options = fit_options or {"disp": "off"}
    fitted = fit_garch(returns, p=1, q=1, dist=dist, options=fit_options, scale=scale)
    cond_vol = pd.Series(
        fitted.conditional_volatility / scale,
        index=fitted.conditional_volatility.index,
        name="garch11_cond_vol",
    )
    if annualize:
        cond_vol = cond_vol * (periods_per_year**0.5)
    return {"result": fitted, "cond_vol": cond_vol}
