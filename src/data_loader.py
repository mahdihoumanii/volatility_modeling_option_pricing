"""
Utility functions to download and persist market data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


def _extract_adj_close(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Extract adjusted close (or close) as a single-column DataFrame named adj_close.
    Handles both single-index and MultiIndex columns from yfinance.
    """
    if data.empty:
        raise ValueError("Downloaded data is empty; cannot extract prices.")

    def _raise_missing() -> None:
        raise KeyError(f"Adj Close/Close not found. Available columns: {list(data.columns)}")

    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        for price_col in ("Adj Close", "Close"):
            if price_col not in level0:
                continue
            price_block = data[price_col]
            if isinstance(price_block, pd.Series):
                series = price_block
            elif ticker in price_block.columns:
                series = price_block[ticker]
            else:
                first_col = price_block.columns[0]
                series = price_block[first_col]
            return series.to_frame(name="adj_close")
        _raise_missing()

    for col in ("Adj Close", "Close"):
        if col in data.columns:
            return data[[col]].rename(columns={col: "adj_close"})

    _raise_missing()


def download_price_data(
    ticker: str = "SPY",
    start: str = "2010-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download adjusted close prices using yfinance.

    Parameters
    ----------
    ticker : str
        Instrument ticker (e.g., 'SPY').
    start : str
        Start date YYYY-MM-DD.
    end : str, optional
        End date YYYY-MM-DD. Defaults to today when None.
    interval : str
        Data interval (e.g., '1d').

    Returns
    -------
    pd.DataFrame
        DataFrame with DateTime index and an 'Adj Close' column.
    """
    data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker {ticker}")
    prices = _extract_adj_close(data, ticker)
    prices.index = pd.to_datetime(prices.index)
    prices.sort_index(inplace=True)
    return prices


def save_dataframes(raw: pd.DataFrame, processed: pd.DataFrame, base_dir: Path) -> None:
    """
    Save raw and processed data to disk.

    Parameters
    ----------
    raw : pd.DataFrame
        Raw price data.
    processed : pd.DataFrame
        Processed data including returns/volatility.
    base_dir : Path
        Base project directory containing data/raw and data/processed.
    """
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_dir / "prices.csv")
    processed.to_csv(processed_dir / "returns.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download price data and compute log returns.")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project base directory containing data/ folder",
    )
    args = parser.parse_args()

    prices = download_price_data(args.ticker, args.start, args.end)
    from src.returns import log_returns  # Local import to avoid circular deps when packaged

    returns = log_returns(prices["adj_close"]).to_frame(name="log_return")
    processed = prices.join(returns)
    save_dataframes(prices, processed, args.base_dir)
    print(f"Saved raw data to {args.base_dir / 'data' / 'raw'}")
    print(f"Saved processed data to {args.base_dir / 'data' / 'processed'}")


if __name__ == "__main__":
    main()
