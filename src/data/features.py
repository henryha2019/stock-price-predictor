from __future__ import annotations

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame, price_col: str = "close", group_col: str = "symbol") -> pd.DataFrame:
    """
    Adds log_return_1 = log(price).diff() per symbol.
    """
    df = df.sort_values([group_col, "date"]).copy()
    # transform keeps index aligned (safer than groupby().apply())
    df["log_return_1"] = df.groupby(group_col)[price_col].transform(lambda s: np.log(s).diff())
    return df


def make_tabular_features(
    df: pd.DataFrame,
    group_col: str = "symbol",
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Creates tabular features from returns + OHLCV.

    Output includes:
      - rolling stats on log_return_1
      - lagged returns
      - rolling stats on volume
      - simple candle features (range, body)
      - rolling stats on price
    """
    df = df.sort_values([group_col, "date"]).copy()

    # ---- lagged returns ----
    df["r_lag_1"] = df.groupby(group_col)["log_return_1"].shift(1)
    df["r_lag_2"] = df.groupby(group_col)["log_return_1"].shift(2)
    df["r_lag_5"] = df.groupby(group_col)["log_return_1"].shift(5)

    # ---- rolling return stats ----
    for w in (5, 10, 20, 60):
        df[f"r_mean_{w}"] = df.groupby(group_col)["log_return_1"].transform(lambda s: s.rolling(w).mean())
        df[f"r_std_{w}"] = df.groupby(group_col)["log_return_1"].transform(lambda s: s.rolling(w).std())
        df[f"r_min_{w}"] = df.groupby(group_col)["log_return_1"].transform(lambda s: s.rolling(w).min())
        df[f"r_max_{w}"] = df.groupby(group_col)["log_return_1"].transform(lambda s: s.rolling(w).max())

    # ---- volume features ----
    if "volume" in df.columns:
        df["vol_lag_1"] = df.groupby(group_col)["volume"].shift(1)
        for w in (5, 20):
            df[f"vol_mean_{w}"] = df.groupby(group_col)["volume"].transform(lambda s: s.rolling(w).mean())
            df[f"vol_std_{w}"] = df.groupby(group_col)["volume"].transform(lambda s: s.rolling(w).std())

    # ---- candle/range features (requires OHLC) ----
    if {"open", "high", "low", "close"}.issubset(df.columns):
        df["hl_range"] = (df["high"] - df["low"]).astype(float)
        df["oc_body"] = (df["close"] - df["open"]).astype(float)
        # Avoid division by zero
        df["hl_range_pct"] = df["hl_range"] / df["open"].replace(0, np.nan)
        df["oc_body_pct"] = df["oc_body"] / df["open"].replace(0, np.nan)

    # ---- price rolling stats ----
    if price_col in df.columns:
        df["price_lag_1"] = df.groupby(group_col)[price_col].shift(1)
        for w in (5, 20):
            df[f"price_mean_{w}"] = df.groupby(group_col)[price_col].transform(lambda s: s.rolling(w).mean())
            df[f"price_std_{w}"] = df.groupby(group_col)[price_col].transform(lambda s: s.rolling(w).std())

    return df
