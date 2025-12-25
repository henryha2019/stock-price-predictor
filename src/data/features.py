from __future__ import annotations

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame, price_col: str = "adj_close", group_col: str = "symbol") -> pd.DataFrame:
    """
    Adds log_return_1 = log(price).diff() per symbol.
    """
    df = df.sort_values([group_col, "date"]).copy()
    df["log_return_1"] = df.groupby(group_col)[price_col].transform(lambda s: np.log(s).diff())
    return df


def make_tabular_features(
    df: pd.DataFrame,
    group_col: str = "symbol",
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """
    Tabular features from returns + OHLCV.
    Rolling windows are capped at 10 to reduce minimum context requirements.
    """
    df = df.sort_values([group_col, "date"]).copy()

    # ---- lagged returns ----
    df["r_lag_1"] = df.groupby(group_col)["log_return_1"].shift(1)
    df["r_lag_2"] = df.groupby(group_col)["log_return_1"].shift(2)
    df["r_lag_5"] = df.groupby(group_col)["log_return_1"].shift(5)

    # ---- rolling return stats (max window = 10) ----
    for w in (3, 5, 10):
        df[f"r_mean_{w}"] = df.groupby(group_col)["log_return_1"].transform(lambda s: s.rolling(w).mean())
        df[f"r_std_{w}"] = df.groupby(group_col)["log_return_1"].transform(lambda s: s.rolling(w).std())
        df[f"r_min_{w}"] = df.groupby(group_col)["log_return_1"].transform(lambda s: s.rolling(w).min())
        df[f"r_max_{w}"] = df.groupby(group_col)["log_return_1"].transform(lambda s: s.rolling(w).max())

    # ---- volume features (max window = 10) ----
    if "volume" in df.columns:
        df["vol_lag_1"] = df.groupby(group_col)["volume"].shift(1)
        for w in (5, 10):
            df[f"vol_mean_{w}"] = df.groupby(group_col)["volume"].transform(lambda s: s.rolling(w).mean())
            df[f"vol_std_{w}"] = df.groupby(group_col)["volume"].transform(lambda s: s.rolling(w).std())

    # ---- candle/range features ----
    if {"open", "high", "low", "close"}.issubset(df.columns):
        df["hl_range"] = (df["high"] - df["low"]).astype(float)
        df["oc_body"] = (df["close"] - df["open"]).astype(float)
        df["hl_range_pct"] = df["hl_range"] / df["open"].replace(0, np.nan)
        df["oc_body_pct"] = df["oc_body"] / df["open"].replace(0, np.nan)

    # ---- price rolling stats (max window = 10) ----
    if price_col in df.columns:
        df["price_lag_1"] = df.groupby(group_col)[price_col].shift(1)
        for w in (5, 10):
            df[f"price_mean_{w}"] = df.groupby(group_col)[price_col].transform(lambda s: s.rolling(w).mean())
            df[f"price_std_{w}"] = df.groupby(group_col)[price_col].transform(lambda s: s.rolling(w).std())

    return df
