import pandas as pd

from src.data_loader import _extract_adj_close


def test_extract_adj_close_multiindex_prefers_adj_close():
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    cols = pd.MultiIndex.from_product([["Adj Close", "Close"], ["SPY"]])
    data = pd.DataFrame([[100.0, 99.5], [101.0, 100.5]], index=idx, columns=cols)
    result = _extract_adj_close(data, "SPY")
    assert list(result.columns) == ["adj_close"]
    assert result.iloc[0, 0] == 100.0


def test_extract_adj_close_single_index_fallback_to_close():
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    data = pd.DataFrame({"Close": [10.0, 11.0]}, index=idx)
    result = _extract_adj_close(data, "SPY")
    assert list(result.columns) == ["adj_close"]
    assert result.iloc[-1, 0] == 11.0
