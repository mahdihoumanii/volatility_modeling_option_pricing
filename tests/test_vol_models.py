import numpy as np
import pandas as pd
import pytest

from src.returns import log_returns
from src.vol_models import ewma, garch, historical


def test_log_returns_basic():
    prices = pd.Series([100.0, 110.0, 121.0])
    r = log_returns(prices).dropna()
    expected = [np.log(1.1), np.log(1.1)]
    assert np.allclose(r.values, expected)


def test_historical_variance_forecast_matches_sample_var():
    returns = pd.Series([0.01, -0.02, 0.015, -0.005])
    forecast = historical.forecast_next_variance(returns, window=3)
    expected = returns.tail(3).var(ddof=1)
    assert np.isclose(forecast, expected)


def test_historical_rolling_vol_alias():
    returns = pd.Series([0.01, -0.02, 0.015, -0.005])
    vol_alias = historical.rolling_vol(returns, window=2, annualize=False)
    vol_original = historical.rolling_volatility(returns, window=2, annualize=False)
    assert vol_alias.equals(vol_original)


def test_ewma_variance_recursion():
    returns = pd.Series([0.01, -0.01, 0.015, -0.02])
    variance = ewma.ewma_variance(returns, lam=0.8, initial_variance=returns.var())
    assert variance.iloc[-1] > 0
    # Check recursion property for last step
    r_prev = returns.dropna().iloc[-2] ** 2
    assert np.isclose(
        variance.iloc[-1],
        0.8 * variance.iloc[-2] + 0.2 * r_prev,
    )


def test_ewma_vol_alias():
    returns = pd.Series([0.01, -0.01, 0.015, -0.02])
    vol_alias = ewma.ewma_vol(returns, lam=0.94, annualize=False)
    vol_original = ewma.ewma_volatility(returns, lam=0.94, annualize=False)
    assert vol_alias.equals(vol_original)


def test_garch_forecast_positive_variance():
    arch = pytest.importorskip("arch")
    rng = np.random.default_rng(0)
    simulated_returns = pd.Series(rng.normal(scale=0.01, size=200))
    fitted = garch.fit_garch(simulated_returns)
    forecast_var = garch.forecast_next_variance(fitted)
    assert forecast_var > 0


def test_fit_garch11_returns_cond_vol():
    pytest.importorskip("arch")
    rng = np.random.default_rng(1)
    simulated_returns = pd.Series(rng.normal(scale=0.01, size=300))
    result = garch.fit_garch11(simulated_returns, annualize=False)
    cond_vol = result["cond_vol"]
    assert isinstance(cond_vol, pd.Series)
    assert cond_vol.name == "garch11_cond_vol"
    assert cond_vol.iloc[-1] > 0
