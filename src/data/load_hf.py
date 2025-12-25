from __future__ import annotations

from datasets import load_dataset
import pandas as pd


def load_daily_ohlcv(dataset_id: str, split: str = "train") -> pd.DataFrame:
    """
    Load daily OHLCV data from Hugging Face and return a clean, sorted DataFrame.

    Expected schema for paperswithbacktest/Stocks-Daily-Price:
      symbol, date, open, high, low, close, volume, adj_close
    """
    ds = load_dataset(dataset_id, split=split)
    df = ds.to_pandas()

    required = {"symbol", "date", "open", "high", "low", "close", "volume", "adj_close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "symbol"]).copy()

    # Ensure deterministic ordering for time-series ops
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df
