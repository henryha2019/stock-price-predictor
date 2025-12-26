from __future__ import annotations

from typing import Any, Dict

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch

from src.models.common import (
    TrainResult,
    coerce_features_float64,
    get_feature_cols_from_cfg,
    infer_and_log_signature,
    log_feature_cols_artifact,
)
from src.training.registry import register_model

def _make_chronos_input_example(
    X: pd.DataFrame,
    feature_cols: list[str],
    price_col: str,
    min_context: int,
) -> pd.DataFrame:
    """
    Build an MLflow input_example that will pass Chronos min_context validation.
    Uses the last min_context *valid* price points, and returns the corresponding
    tabular rows (all feature_cols) aligned by index.
    """
    if price_col not in X.columns:
        return X.tail(min_context).copy()

    s = pd.to_numeric(X[price_col], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()

    if len(s) >= min_context:
        idx = s.index[-min_context:]
        ex = X.loc[idx, feature_cols].copy()
    else:
        # fallback: still return something deterministic
        ex = X.tail(min_context).copy()

    # MLflow likes plain dtypes; keep it numeric
    return ex.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

def _pick_price_col(df: pd.DataFrame) -> str:
    for c in ("adj_close", "close", "price"):
        if c in df.columns:
            return c
    raise ValueError("No price column found (expected adj_close/close/price).")


class HFFineTuneModel:
    """
    Minimal Chronos wrapper (uses chronos-forecasting).
    Assumes X_train contains at least 'adj_close' or 'close' among feature cols.
    """

    def train_and_log(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        cfg: Dict[str, Any],
    ) -> TrainResult:
        from chronos import ChronosPipeline  # local import to avoid import cost if unused

        artifact_path = "model"

        feature_cols = get_feature_cols_from_cfg(cfg)
        Xtr_df = coerce_features_float64(X_train, feature_cols)

        hcfg = cfg.get("hf", {})
        model_id = hcfg.get("model_id", "amazon/chronos-t5-small")
        device = hcfg.get("device", "cpu")
        torch_dtype = hcfg.get("torch_dtype", "float32")
        num_samples = int(hcfg.get("num_samples", 256))
        quantile = float(hcfg.get("quantile", 0.5))
        min_context = int(hcfg.get("min_context", 64))

        horizon = int(cfg.get("train", {}).get("horizon", 1))

        price_col = _pick_price_col(Xtr_df)

        with mlflow.start_run(nested=True) as run:
            mlflow.set_tag("model_family", "hf_finetune")
            mlflow.log_params(
                {
                    "model_id": model_id,
                    "device": device,
                    "torch_dtype": torch_dtype,
                    "num_samples": num_samples,
                    "quantile": quantile,
                    "min_context": min_context,
                    "horizon": horizon,
                    "price_col": price_col,
                    "num_features": len(feature_cols),
                }
            )

            # Required for serving alignment
            log_feature_cols_artifact(feature_cols)

            dtype = getattr(torch, torch_dtype)
            pipe = ChronosPipeline.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype=dtype,
            )

            class _ChronosPyfunc(mlflow.pyfunc.PythonModel):
                def __init__(self, pipeline, price_col: str, horizon: int, num_samples: int, quantile: float, min_context: int):
                    self.pipeline = pipeline
                    self.price_col = price_col
                    self.horizon = int(horizon)
                    self.num_samples = int(num_samples)
                    self.quantile = float(quantile)
                    self.min_context = int(min_context)

                def predict(self, context, model_input: pd.DataFrame):
                    if self.price_col not in model_input.columns:
                        raise ValueError(f"Missing required price column: {self.price_col}")
                    s = pd.to_numeric(model_input[self.price_col], errors="coerce")
                    s = s.replace([np.inf, -np.inf], np.nan).dropna()
                    series_np = s.to_numpy(dtype=np.float32)

                    if len(series_np) < self.min_context:
                        raise ValueError(f"Need at least min_context={self.min_context} valid rows for Chronos.")

                    # Chronos expects torch.Tensor contexts
                    series_t = torch.tensor(series_np, dtype=torch.float32)

                    forecast = self.pipeline.predict(
                        [series_t],
                        prediction_length=self.horizon,
                        num_samples=self.num_samples,
                    )[0]
                    # Use quantile point forecast
                    q = np.quantile(forecast, self.quantile, axis=0)  # predicted price path
                    pred_price = float(q[-1])

                    last_price = float(series_np[-1])
                    # log-return consistent with app.py: pred_price = last_price * exp(pred_return)
                    pred_return = float(np.log(max(pred_price, 1e-12) / max(last_price, 1e-12)))

                    return np.array([pred_return], dtype=float)

            py_model = _ChronosPyfunc(pipe, price_col, horizon, num_samples, quantile, min_context)

            # Signature should allow the same tabular DF you send in serving.
            sig_kwargs = infer_and_log_signature(Xtr_df, y_train, artifact_path=artifact_path)

            # Ensure MLflow's input example has enough valid context for Chronos validation
            input_ex = _make_chronos_input_example(
                Xtr_df,
                feature_cols=feature_cols,
                price_col=price_col,
                min_context=min_context,
            )
            sig_kwargs.pop("input_example", None)

            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=py_model,
                input_example=input_ex, 
                pip_requirements=[
                    "mlflow==2.19.0",
                    "pandas",
                    "numpy",
                    "torch",
                    "chronos-forecasting==2.2.2",
                    "transformers==4.48.0",
                    "accelerate==0.34.2",
                ],
                **sig_kwargs,
            )

            version = None
            if cfg.get("mlflow", {}).get("register", True):
                version = register_model(run.info.run_id, artifact_path, cfg["mlflow"]["model_name"])
                mlflow.set_tag("registered_model_version", version)

            return TrainResult(run_id=run.info.run_id, artifact_path=artifact_path, registered_model_version=version)
