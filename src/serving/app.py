# src/serving/app.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Any, Dict, List

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.features import add_returns, make_tabular_features


@dataclass
class ModelState:
    loaded: bool
    model_uri: Optional[str] = None
    loaded_at_unix: Optional[float] = None
    last_error: Optional[str] = None


MODEL_STATE = ModelState(loaded=False)

app = FastAPI(title="Stock Price Predictor", version="0.1.0")


class OHLCVBar(BaseModel):
    date: Optional[str] = Field(None, description="YYYY-MM-DD (optional)")
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictRequest(BaseModel):
    symbol: str = Field(..., examples=["AAPL"])
    horizon: int = Field(1, ge=1, le=30)
    as_of: Optional[str] = Field(None, description="YYYY-MM-DD (optional)")

    # Minimal payload (works for basic tabular models if features are return-based)
    last_close: Optional[float] = Field(None, gt=0)
    last_returns: Optional[list[float]] = Field(None, min_length=5, max_length=240)

    # Preferred payload (works for "use all features" training):
    # Provide a trailing window of OHLCV so serving can run the same feature pipeline.
    ohlcv_window: Optional[list[OHLCVBar]] = Field(
        None,
        description="Trailing OHLCV window (recommended). Provide >= 30 rows for rolling features.",
    )


class PredictResponse(BaseModel):
    pred_return: float
    pred_price: float
    model_uri: str
    loaded_at_unix: float
    used_mode: str


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_production_model() -> None:
    """
    Loads an MLflow model from Model Registry.

    Required env vars:
      - MLFLOW_TRACKING_URI (e.g., http://<mlflow-host>:5001)
      - MODEL_NAME (e.g., stock-price-predictor)

    Optional:
      - MODEL_ALIAS (default: "prod")          # preferred
      - MODEL_STAGE (fallback: "Production")  # if stages are used instead of aliases
      - SKIP_MODEL_LOAD=1                     # for CI/tests
    """
    if _bool_env("SKIP_MODEL_LOAD", default=False):
        MODEL_STATE.loaded = False
        MODEL_STATE.last_error = "SKIP_MODEL_LOAD=1 (model load skipped)"
        return

    try:
        tracking_uri = _env("MLFLOW_TRACKING_URI")
        model_name = _env("MODEL_NAME")
        model_alias = os.getenv("MODEL_ALIAS", "prod")
        model_stage = os.getenv("MODEL_STAGE", "Production")

        mlflow.set_tracking_uri(tracking_uri)

        # Prefer alias-based deployment. Fallback to stage-based.
        model_uri = f"models:/{model_name}@{model_alias}"
        try:
            model = mlflow.pyfunc.load_model(model_uri)
        except Exception:
            model_uri = f"models:/{model_name}/{model_stage}"
            model = mlflow.pyfunc.load_model(model_uri)

        app.state.model = model
        MODEL_STATE.loaded = True
        MODEL_STATE.model_uri = model_uri
        MODEL_STATE.loaded_at_unix = time.time()
        MODEL_STATE.last_error = None

    except Exception as e:
        MODEL_STATE.loaded = False
        MODEL_STATE.last_error = str(e)
        raise


@app.on_event("startup")
def on_startup() -> None:
    # Fail fast in prod. For CI/tests, set SKIP_MODEL_LOAD=1
    load_production_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if not MODEL_STATE.loaded:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "error": MODEL_STATE.last_error})
    return {
        "status": "ready",
        "model_uri": MODEL_STATE.model_uri,
        "loaded_at_unix": MODEL_STATE.loaded_at_unix,
    }


def _coerce_input_schema_columns(model: Any) -> Optional[List[str]]:
    """
    Best-effort: read MLflow input schema column names if present.
    If not available, return None.
    """
    try:
        schema = model.metadata.get_input_schema()
        if schema is None:
            return None
        cols = []
        for c in schema.inputs():
            cols.append(c.name)
        return cols or None
    except Exception:
        return None


def _build_df_from_ohlcv_window(symbol: str, window: list[OHLCVBar]) -> pd.DataFrame:
    df = pd.DataFrame([b.model_dump() for b in window])
    df["symbol"] = symbol
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    # Ensure required columns exist
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' in ohlcv_window.")
    df = df.dropna(subset=["close"]).copy()
    return df


def build_features_from_ohlcv_window(symbol: str, window: list[OHLCVBar], horizon: int) -> pd.DataFrame:
    """
    Runs the SAME feature pipeline used in training (add_returns + make_tabular_features)
    on the provided OHLCV window, then returns:
      - a multi-row DF (sequence-friendly)
    """
    df = _build_df_from_ohlcv_window(symbol, window)

    df = add_returns(df, price_col="close", group_col="symbol")
    df = make_tabular_features(df, group_col="symbol")

    # Training sets target as shift(-horizon) of log_return_1, but for inference we only need features.
    df["horizon"] = int(horizon)

    # Drop rows that have NaNs from rolling features
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("After feature engineering, no usable rows remain. Provide a longer ohlcv_window.")
    return df


def build_minimal_features_from_returns(last_close: float, last_returns: list[float], horizon: int) -> pd.DataFrame:
    """
    Minimal fall-back features if you don't provide OHLCV.
    Works only for models trained on return-only columns.
    """
    r = np.array(last_returns, dtype=float)

    feats = {
        "r_mean_5": float(np.mean(r[-5:])),
        "r_std_5": float(np.std(r[-5:], ddof=1)) if len(r[-5:]) > 1 else 0.0,
        "r_mean_20": float(np.mean(r[-20:])),
        "r_std_20": float(np.std(r[-20:], ddof=1)) if len(r[-20:]) > 1 else 0.0,
        "r_last": float(r[-1]),
        "last_close": float(last_close),
        "horizon": int(horizon),
    }
    return pd.DataFrame([feats])


def _align_to_expected_columns(X: pd.DataFrame, expected_cols: Optional[List[str]]) -> pd.DataFrame:
    """
    If the model has an input schema, enforce those columns:
      - add missing columns as 0.0
      - drop extra columns
      - reorder
    If no schema is available, return X unchanged.
    """
    if not expected_cols:
        return X
    out = X.copy()
    for c in expected_cols:
        if c not in out.columns:
            out[c] = 0.0
    out = out[expected_cols]
    return out


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not MODEL_STATE.loaded:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {MODEL_STATE.last_error}")

    model = app.state.model
    expected_cols = _coerce_input_schema_columns(model)

    # Strategy:
    # 1) If OHLCV window provided, build full feature DF and try predict on a multi-row window (sequence mode)
    # 2) If that fails, try last row only (tabular mode)
    # 3) If no OHLCV window, use minimal return features (tabular mode)
    try:
        if req.ohlcv_window and len(req.ohlcv_window) >= int(os.getenv("MIN_OHLCV_WINDOW", "30")):
            feats_df = build_features_from_ohlcv_window(req.symbol, req.ohlcv_window, req.horizon)

            # Sequence attempt: pass the whole window (some pyfuncs expect multi-row)
            X_seq = _align_to_expected_columns(feats_df, expected_cols)
            try:
                y_hat = model.predict(X_seq)
                used_mode = "sequence"
            except Exception:
                # Tabular fallback: last row
                X_tab = X_seq.tail(1).copy()
                y_hat = model.predict(X_tab)
                used_mode = "tabular_from_window"

            pred_return = float(np.array(y_hat).reshape(-1)[0])

            # Determine last_close for price conversion
            last_close = float(req.ohlcv_window[-1].close)
            pred_price = float(last_close * np.exp(pred_return))
        else:
            # Minimal / fallback mode
            if req.last_close is None or req.last_returns is None:
                raise HTTPException(
                    status_code=422,
                    detail="Provide either ohlcv_window (recommended) or both last_close + last_returns.",
                )

            X = build_minimal_features_from_returns(float(req.last_close), req.last_returns, req.horizon)
            X = _align_to_expected_columns(X, expected_cols)

            y_hat = model.predict(X)
            pred_return = float(np.array(y_hat).reshape(-1)[0])
            pred_price = float(float(req.last_close) * np.exp(pred_return))
            used_mode = "tabular_minimal"

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return PredictResponse(
        pred_return=pred_return,
        pred_price=pred_price,
        model_uri=str(MODEL_STATE.model_uri),
        loaded_at_unix=float(MODEL_STATE.loaded_at_unix or 0),
        used_mode=used_mode,
    )
