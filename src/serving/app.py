# src/serving/app.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


@dataclass
class ModelState:
    loaded: bool
    model_uri: Optional[str] = None
    loaded_at_unix: Optional[float] = None
    last_error: Optional[str] = None


MODEL_STATE = ModelState(loaded=False)

app = FastAPI(title="Stock Price Predictor", version="0.1.0")


class PredictRequest(BaseModel):
    symbol: str = Field(..., examples=["AAPL"])
    horizon: int = Field(1, ge=1, le=30)
    as_of: Optional[str] = Field(None, description="YYYY-MM-DD (optional)")
    last_close: float = Field(..., gt=0)
    # returns are assumed to be log-returns (or small arithmetic returns); consistent with your training target
    last_returns: list[float] = Field(..., min_length=5, max_length=240)


class PredictResponse(BaseModel):
    pred_return: float
    pred_price: float
    model_uri: str
    loaded_at_unix: float


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def load_production_model() -> None:
    """
    Loads an MLflow model from Model Registry.

    Required env vars:
      - MLFLOW_TRACKING_URI (e.g., http://<mlflow-ec2>:5000)
      - MODEL_NAME (e.g., stock-price-predictor)
    Optional:
      - MODEL_ALIAS (default: "prod")          # preferred
      - MODEL_STAGE (fallback: "Production")  # if stages are used instead of aliases
    """
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
    # Fail fast if model cannot load; ECS/ALB healthchecks will prevent traffic.
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


def build_tabular_row(last_close: float, last_returns: list[float], horizon: int) -> pd.DataFrame:
    """
    One-row feature vector for tabular models (Ridge / AutoML).
    Must match the feature columns used in training for those models.
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


def build_window_df(last_close: float, last_returns: list[float], horizon: int) -> pd.DataFrame:
    """
    Multi-row window for sequence models (your Transformer / future HF fine-tune pyfunc).

    We compute the same engineered features per timestep so the sequence model
    can take the final seq_len rows internally.
    """
    r = pd.Series(last_returns, dtype="float64")

    df = pd.DataFrame(
        {
            "r_mean_5": r.rolling(5).mean(),
            "r_std_5": r.rolling(5).std(ddof=1).fillna(0.0),
            "r_mean_20": r.rolling(20).mean(),
            "r_std_20": r.rolling(20).std(ddof=1).fillna(0.0),
            "r_last": r,
            "last_close": float(last_close),
            "horizon": int(horizon),
        }
    )

    # Rolling means need enough history; drop early NaNs.
    df = df.dropna(subset=["r_mean_5", "r_mean_20"]).reset_index(drop=True)
    return df


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not MODEL_STATE.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Need enough returns for rolling windows (20) + a decent sequence.
    if len(req.last_returns) < 25:
        raise HTTPException(status_code=400, detail="last_returns must have at least 25 values")

    X_tabular = build_tabular_row(req.last_close, req.last_returns, req.horizon)
    X_window = build_window_df(req.last_close, req.last_returns, req.horizon)

    model = app.state.model

    pred_return: float | None = None
    last_error: str | None = None

    # Try tabular first (works for Ridge/AutoML). Fall back to window (works for seq models).
    for X in (X_tabular, X_window):
        try:
            y_hat = model.predict(X)
            pred_return = float(np.array(y_hat).reshape(-1)[0])
            last_error = None
            break
        except Exception as e:
            last_error = str(e)

    if pred_return is None:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed for both tabular and window inputs: {last_error}",
        )

    pred_price = float(req.last_close * np.exp(pred_return))  # log-return -> price

    return PredictResponse(
        pred_return=pred_return,
        pred_price=pred_price,
        model_uri=str(MODEL_STATE.model_uri),
        loaded_at_unix=float(MODEL_STATE.loaded_at_unix or 0.0),
    )
