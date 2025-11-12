# prepare data for data analysis
"""
prepare_data.py — Minimal data prep for index-tracking BLP (Stooq source)

Requirements:
  - pandas
  - numpy
  - pandas_datareader

Examples:
  python prepare_data.py \
      --index SPY.US \
      --tickers AAPL.US MSFT.US GOOGL.US AMZN.US META.US \
      --start 2018-01-01 --end 2024-12-31 \
      --outdir data

Notes:
  * Stooq symbols often use the ".US" suffix for US listings (e.g., AAPL.US).
  * Stooq returns data in descending date order; we sort ascending.
  * We compute simple returns: r_t = P_t / P_{t-1} - 1.
  * Outputs:
      data/prices_assets.csv
      data/prices_index.csv
      data/returns_assets.csv
      data/returns_index.csv
      data/metadata.json
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr


def _fetch_stooq(symbol: str, start: str, end: str) -> pd.Series:
    """Fetch Adjusted Close for one symbol from Stooq, sorted by date asc."""
    df = pdr.DataReader(symbol, "stooq", start=start, end=end)
    if df.empty:
        raise ValueError(f"No data returned for symbol: {symbol}")
    # Stooq is descending; keep Adjusted Close if available, else Close
    col = "Close"
    for candidate in ("Adj Close", "AdjClose", "Close"):
        if candidate in df.columns:
            col = candidate
            break
    s = df[col].sort_index()  # ascending date
    s.name = symbol
    return s


def _fetch_many(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    """Fetch a panel of prices, outer-join on dates, sort ascending."""
    series = []
    for sym in symbols:
        s = _fetch_stooq(sym, start, end)
        series.append(s)
    prices = pd.concat(series, axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    return prices


def _resample_if_needed(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Optionally resample to 'W' or 'M' using last observation."""
    if freq.upper() in ("D", "DAILY"):
        return df
    rule = {"W": "W-FRI", "M": "M"}[freq.upper()]
    return df.resample(rule).last()


def _forward_fill_limited(df: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    """Forward-fill small gaps up to max_gap days to handle holidays/missing ticks."""
    return df.ffill(limit=max_gap)


def _to_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple returns; drop first row (NaNs)."""
    rets = prices.pct_change()
    return rets.dropna(how="all")


def _intersect_dates(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only common dates to ensure alignment between assets and index."""
    common = a.index.intersection(b.index)
    return a.loc[common], b.loc[common]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Stooq data for index-tracking BLP.")
    parser.add_argument("--index", required=True, help="Index/benchmark symbol (e.g., SPY.US).")
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="List of asset symbols (e.g., AAPL.US MSFT.US ...).",
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD.")
    parser.add_argument(
        "--freq",
        default="D",
        choices=["D", "W", "M"],
        help="Sampling frequency: D (daily), W (weekly), M (monthly).",
    )
    parser.add_argument("--outdir", default="data", help="Output directory.")
    parser.add_argument(
        "--min-days",
        type=int,
        default=250,
        help="Minimum number of common observations required (after alignment).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Fetch prices
    print(f"[INFO] Fetching index: {args.index}")
    idx_prices = _fetch_stooq(args.index, args.start, args.end).to_frame()

    print(f"[INFO] Fetching assets: {', '.join(args.tickers)}")
    asset_prices = _fetch_many(args.tickers, args.start, args.end)

    # Basic cleaning
    asset_prices = _forward_fill_limited(asset_prices, max_gap=3)
    idx_prices = _forward_fill_limited(idx_prices, max_gap=3)

    # Optional resampling
    if args.freq.upper() != "D":
        print(f"[INFO] Resampling to {args.freq.upper()}")
        asset_prices = _resample_if_needed(asset_prices, args.freq)
        idx_prices = _resample_if_needed(idx_prices, args.freq)

    # Align dates
    asset_prices, idx_prices = _intersect_dates(asset_prices, idx_prices)

    # Basic sanity check
    n_obs = len(asset_prices)
    if n_obs < args.min_days:
        raise ValueError(
            f"Too few common observations after alignment: {n_obs} < {args.min_days}. "
            f"Consider adjusting date range, tickers, or frequency."
        )

    # Returns
    asset_rets = _to_simple_returns(asset_prices)
    idx_rets = _to_simple_returns(idx_prices)

    # Re-align after pct_change (first row dropped)
    asset_rets, idx_rets = _intersect_dates(asset_rets, idx_rets)

    # Save
    p_prices_assets = os.path.join(args.outdir, "prices_assets.csv")
    p_prices_index = os.path.join(args.outdir, "prices_index.csv")
    p_rets_assets = os.path.join(args.outdir, "returns_assets.csv")
    p_rets_index = os.path.join(args.outdir, "returns_index.csv")
    p_meta = os.path.join(args.outdir, "metadata.json")

    asset_prices.to_csv(p_prices_assets, index_label="Date")
    idx_prices.to_csv(p_prices_index, index_label="Date")
    asset_rets.to_csv(p_rets_assets, index_label="Date")
    idx_rets.to_csv(p_rets_index, index_label="Date")

    meta = {
        "source": "stooq",
        "index_symbol": args.index,
        "asset_symbols": args.tickers,
        "start": args.start,
        "end": args.end,
        "frequency": args.freq.upper(),
        "n_observations": int(len(asset_rets)),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "notes": "Simple returns; forward-filled small gaps (limit=3); dates intersected.",
    }
    with open(p_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print("[OK] Wrote:")
    print(f"  {p_prices_assets}")
    print(f"  {p_prices_index}")
    print(f"  {p_rets_assets}")
    print(f"  {p_rets_index}")
    print(f"  {p_meta}")


if __name__ == "__main__":
    main()
