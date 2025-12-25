from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch

from src.models.common import TrainResult
from src.training.registry import register_model


def _pick_price_col(df: pd.DataFrame) -> str:
    for c in ("adj_close", "close", "price"):
        if c in df.columns:
            return c
    raise ValueError("Input DataFrame must contain one of: adj_close, close, price")


class _ChronosPyFunc(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper around Amazon Chronos.

    Input:
      DataFrame with columns:
        - symbol (required)
        - adj_close or close or price (required)
        - optional: date (ignored)

    Output:
      1D numpy array of predicted log-returns (one per symbol) for the requested horizon.
    """

    def __init__(
        self,
        model_id: str,
        horizon: int,
        device: str = "cpu",
        torch_dtype: str = "float32",
        num_samples: int = 256,
        quantile: float = 0.5,
        min_context: int = 64,
    ) -> None:
        self.model_id = model_id
        self.horizon = int(horizon)
        self.device = device
        self.torch_dtype = torch_dtype
        self.num_samples = int(num_samples)
        self.quantile = float(quantile)
        self.min_context = int(min_context)

        self.pipeline = None  # loaded in load_context

    def load_context(self, context) -> None:
        # Import here so MLflow model can carry pip requirements cleanly
        from chronos import ChronosPipeline  # type: ignore

        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        tdtype = dtype_map.get(self.torch_dtype, torch.float32)

        # device_map="cpu" is valid; on GPU you can set "cuda"
        self.pipeline = ChronosPipeline.from_pretrained(
            self.model_id,
            device_map=self.device,
            torch_dtype=tdtype,
        )

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Chronos pipeline not loaded")

        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("model_input must be a pandas DataFrame")

        if "symbol" not in model_input.columns:
            raise ValueError("model_input must contain column: symbol")

        price_col = _pick_price_col(model_input)

        df = model_input.copy()
        # Ensure deterministic ordering if date exists; otherwise respect existing order
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values(["symbol", "date"])

        preds: List[float] = []

        for sym, g in df.groupby("symbol", sort=False):
            series = g[price_col].astype(float).dropna().to_numpy()

            if len(series) < max(8, self.min_context):
                raise ValueError(
                    f"Not enough context for symbol={sym}. "
                    f"Need >= {self.min_context} rows, got {len(series)}. "
                    f"Provide a longer ohlcv_window."
                )

            last_price = float(series[-1])
            # Chronos expects a torch tensor
            ts = torch.tensor(series, dtype=torch.float32)

            # forecast shape: [num_series=1, num_samples, prediction_length]
            forecast = self.pipeline.predict(
                ts,
                prediction_length=self.horizon,
                num_samples=self.num_samples,
            )

            # Convert samples to numpy: [num_samples, horizon]
            samples = forecast[0].detach().cpu().numpy()

            # Take chosen quantile trajectory; default median (0.5)
            q = np.quantile(samples, self.quantile, axis=0)
            pred_price_h = float(q[self.horizon - 1])

            # Convert to log-return
            pred_logret = float(np.log(pred_price_h / last_price))
            preds.append(pred_logret)

        return np.array(preds, dtype=float)


class HFFineTuneModel:
    """
    For your project: this is a HF foundation-model forecaster (Chronos),
    logged to MLflow and served via the same registry+FastAPI stack.

    Note: this is zero-shot inference (no fine-tune) to keep scope tight and production-ready.
    You can later add finetuning as a separate milestone.
    """

    def train_and_log(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        cfg: Dict[str, Any],
    ) -> TrainResult:
        model_id = cfg["hf"].get("model_id", "amazon/chronos-t5-small")
        horizon = int(cfg["train"]["horizon"])

        device = cfg["hf"].get("device", "cpu")
        torch_dtype = cfg["hf"].get("torch_dtype", "float32")
        num_samples = int(cfg["hf"].get("num_samples", 256))
        quantile = float(cfg["hf"].get("quantile", 0.5))
        min_context = int(cfg["hf"].get("min_context", 64))

        artifact_path = "model"

        with mlflow.start_run(run_name="hf_chronos") as run:
            mlflow.set_tag("model_family", "hf_finetune")
            mlflow.log_params(
                {
                    "hf_model_id": model_id,
                    "horizon": horizon,
                    "device": device,
                    "torch_dtype": torch_dtype,
                    "num_samples": num_samples,
                    "quantile": quantile,
                    "min_context": min_context,
                    "note": "zero-shot chronos (no finetune)",
                }
            )

            python_model = _ChronosPyFunc(
                model_id=model_id,
                horizon=horizon,
                device=device,
                torch_dtype=torch_dtype,
                num_samples=num_samples,
                quantile=quantile,
                min_context=min_context,
            )

            # Include chronos-forecasting so the model can load ChronosPipeline.
            # Torch is required; transformers comes in as a dependency in most envs, but keeping explicit is fine.
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=python_model,
                pip_requirements=[
                    "mlflow==2.19.0",
                    "pandas",
                    "numpy",
                    "torch",
                    "chronos-forecasting",
                ],
            )

            version = None
            if cfg["mlflow"].get("register", True):
                version = register_model(run.info.run_id, artifact_path, cfg["mlflow"]["model_name"])
                mlflow.set_tag("registered_model_version", version)

            return TrainResult(
                run_id=run.info.run_id,
                artifact_path=artifact_path,
                registered_model_version=version,
            )
