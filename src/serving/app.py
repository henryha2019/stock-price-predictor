# src/serving/app.py
from __future__ import annotations

import os
import time
import json
import logging
import traceback
from dataclasses import dataclass
from typing import Optional, List

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from mlflow.tracking import MlflowClient

from src.data.features import add_returns, make_tabular_features

# rolling max 10 => need at least 11 rows
MIN_FEATURE_ROWS_DEFAULT = 11


@dataclass
class ModelState:
    loaded: bool
    model_uri: Optional[str] = None
    loaded_at_unix: Optional[float] = None
    last_error: Optional[str] = None
    feature_cols: Optional[List[str]] = None
    prod_run_id: Optional[str] = None


MODEL_STATE = ModelState(loaded=False)
logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Stock Price Predictor", version="0.1.0")


class OHLCVBar(BaseModel):
    date: Optional[str] = Field(None, description="YYYY-MM-DD")
    open: float
    high: float
    low: float
    close: float
    adj_close: float = Field(..., description="Adjusted close (required)")
    volume: float


class PredictRequest(BaseModel):
    symbol: str = Field(..., examples=["AAPL"])
    horizon: int = Field(1, ge=1, le=30)
    as_of: Optional[str] = Field(None, description="YYYY-MM-DD (optional)")

    last_close: Optional[float] = Field(None, gt=0)
    last_returns: Optional[list[float]] = Field(None, min_length=5, max_length=240)

    ohlcv_window: Optional[list[OHLCVBar]] = Field(
        None,
        description="Trailing OHLCV window (recommended). Must include adj_close.",
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


def _resolve_prod_run_id(model_name: str, alias: str, stage: str) -> Optional[str]:
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
        return mv.run_id
    except Exception:
        pass

    try:
        latest = client.get_latest_versions(model_name, stages=[stage])
        if latest:
            return latest[0].run_id
    except Exception:
        pass

    return None


def _load_feature_cols_from_run(run_id: str) -> Optional[List[str]]:
    try:
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="feature_cols.json")
        with open(local_path, "r") as f:
            obj = json.load(f)
        cols = obj.get("feature_cols")
        if isinstance(cols, list) and cols:
            return cols
    except Exception:
        return None
    return None


def load_production_model() -> None:
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

        model_uri = f"models:/{model_name}@{model_alias}"
        try:
            model = mlflow.pyfunc.load_model(model_uri)
        except Exception:
            model_uri = f"models:/{model_name}/{model_stage}"
            model = mlflow.pyfunc.load_model(model_uri)

        prod_run_id = _resolve_prod_run_id(model_name, model_alias, model_stage)
        feature_cols = _load_feature_cols_from_run(prod_run_id) if prod_run_id else None

        app.state.model = model
        MODEL_STATE.loaded = True
        MODEL_STATE.model_uri = model_uri
        MODEL_STATE.loaded_at_unix = time.time()
        MODEL_STATE.last_error = None
        MODEL_STATE.prod_run_id = prod_run_id
        MODEL_STATE.feature_cols = feature_cols

        if not MODEL_STATE.feature_cols:
            raise RuntimeError(
                f"feature_cols.json not found for deployed model run_id={prod_run_id}. "
                f"Retrain and ensure feature_cols.json is logged in the SAME run that registers the model."
            )

    except Exception as e:
        MODEL_STATE.loaded = False
        MODEL_STATE.last_error = str(e)
        raise


@app.on_event("startup")
def on_startup() -> None:
    load_production_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if os.getenv("SKIP_MODEL_LOAD") == "1":
        return {
            "status": "ready",
            "mode": "ci",
            "model_uri": MODEL_STATE.model_uri,
            "loaded_at_unix": MODEL_STATE.loaded_at_unix,
            "prod_run_id": MODEL_STATE.prod_run_id,
            "has_feature_cols": bool(MODEL_STATE.feature_cols),
            "note": "SKIP_MODEL_LOAD=1, readiness does not require model load",
        }
    if not MODEL_STATE.loaded:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "error": MODEL_STATE.last_error})
    return {
        "status": "ready",
        "model_uri": MODEL_STATE.model_uri,
        "loaded_at_unix": MODEL_STATE.loaded_at_unix,
        "prod_run_id": MODEL_STATE.prod_run_id,
        "has_feature_cols": bool(MODEL_STATE.feature_cols),
    }


def _build_df_from_ohlcv_window(symbol: str, window: list[OHLCVBar]) -> pd.DataFrame:
    df = pd.DataFrame([b.model_dump() for b in window])
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    required = ["open", "high", "low", "close", "adj_close", "volume", "symbol", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in ohlcv_window: {missing}")

    df = df.dropna(subset=["adj_close", "date"]).copy()
    return df


def build_features_from_ohlcv_window(symbol: str, window: list[OHLCVBar], price_col: str = "adj_close") -> pd.DataFrame:
    df = _build_df_from_ohlcv_window(symbol, window)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    df = add_returns(df, price_col=price_col, group_col="symbol")
    df = make_tabular_features(df, group_col="symbol", price_col=price_col)

    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("After feature engineering, no usable rows remain. Provide a longer ohlcv_window.")
    return df


def _to_model_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in ["date", "symbol", "target"] if c in df.columns]
    out = df.drop(columns=drop_cols, errors="ignore")
    out = out.select_dtypes(include=["number"]).copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    if len(out) == 0:
        raise ValueError("No numeric feature rows after cleaning (NaNs/infs).")
    return out


def _align_exact_order(X: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = X.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = 0.0
    out = out[feature_cols]
    return out


def _coerce_float64(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.astype("float64")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not MODEL_STATE.loaded:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {MODEL_STATE.last_error}")

    model = app.state.model

    min_rows = int(os.getenv("MIN_OHLCV_WINDOW", str(MIN_FEATURE_ROWS_DEFAULT)))
    allow_minimal = _bool_env("ALLOW_MINIMAL_RETURNS_MODE", default=False)

    try:
        if req.ohlcv_window is not None:
            if len(req.ohlcv_window) < min_rows:
                raise HTTPException(
                    status_code=422,
                    detail=f"ohlcv_window must contain at least {min_rows} rows (needed for rolling features).",
                )

            feats_df = build_features_from_ohlcv_window(req.symbol, req.ohlcv_window, price_col="adj_close")
            X_seq = _to_model_features(feats_df)
            X_seq = _align_exact_order(X_seq, MODEL_STATE.feature_cols or [])
            X_tab = _coerce_float64(X_seq.tail(min_rows).copy())

            y_hat = model.predict(X_tab)
            pred_return = float(np.array(y_hat).reshape(-1)[0])

            last_adj_close = float(req.ohlcv_window[-1].adj_close)
            pred_price = float(last_adj_close * np.exp(pred_return))
            used_mode = "tabular_from_window_adj_close"

        else:
            if not allow_minimal:
                raise HTTPException(
                    status_code=422,
                    detail="This deployed model requires ohlcv_window with adj_close. Minimal mode is disabled.",
                )

            if req.last_close is None or req.last_returns is None:
                raise HTTPException(status_code=422, detail="Provide last_close + last_returns.")

            r = np.array(req.last_returns, dtype=float)

            feats = {
                "r_mean_5": float(np.mean(r[-5:])),
                "r_std_5": float(np.std(r[-5:], ddof=1)) if len(r[-5:]) > 1 else 0.0,
                "r_mean_10": float(np.mean(r[-10:])),
                "r_std_10": float(np.std(r[-10:], ddof=1)) if len(r[-10:]) > 1 else 0.0,
                "r_last": float(r[-1]),
                "last_close": float(req.last_close),
            }
            X = pd.DataFrame([feats])
            X = _align_exact_order(X, MODEL_STATE.feature_cols or [])
            X = _coerce_float64(X)

            y_hat = model.predict(X)
            pred_return = float(np.array(y_hat).reshape(-1)[0])
            pred_price = float(float(req.last_close) * np.exp(pred_return))
            used_mode = "tabular_minimal"

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Inference failed: %s\n%s", repr(e), traceback.format_exc())
        msg = str(e).strip()
        if not msg:
            msg = repr(e)
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed ({type(e).__name__}): {msg}",
        )

    return PredictResponse(
        pred_return=pred_return,
        pred_price=pred_price,
        model_uri=str(MODEL_STATE.model_uri),
        loaded_at_unix=float(MODEL_STATE.loaded_at_unix or 0),
        used_mode=used_mode,
    )
