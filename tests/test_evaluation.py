import numpy as np
import pandas as pd

from src.evaluation import evaluate_forecasts, windowed_realized_variance, windowed_realized_volatility


def test_windowed_realized_volatility_matches_manual():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01])
    rv_df = windowed_realized_volatility(returns, windows=[2], periods_per_year=252)
    tail = returns.tail(2)
    expected_var = (252 / 2) * (tail.pow(2).sum())
    expected_vol = np.sqrt(expected_var)
    assert np.isclose(rv_df["rv_2"].iloc[-1], expected_vol)


def test_evaluate_forecasts_with_windowed_realized_variance():
    returns = pd.Series([0.01, -0.02, 0.015, -0.005])
    # Simple constant variance forecast
    forecast_index = returns.index[2:]
    forecast_var = pd.DataFrame({"const": 0.0001}, index=forecast_index)
    metrics = evaluate_forecasts(
        forecast_var,
        returns,
        realized_windows=[2, 3],
        annualize_pred=True,
        periods_per_year=252,
    )
    assert ("const", 2) in metrics.index
    assert ("const", 3) in metrics.index
    # Ensure metrics are finite numbers
    assert np.isfinite(metrics.loc[("const", 2), "mse_var"])
    assert np.isfinite(metrics.loc[("const", 3), "qlike"])
