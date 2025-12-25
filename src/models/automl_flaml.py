from __future__ import annotations

from typing import Any, Dict

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from flaml import AutoML

from src.models.common import (
    TrainResult,
    coerce_features_float64,
    get_feature_cols_from_cfg,
    infer_and_log_signature,
    log_feature_cols_artifact,
)
from src.training.registry import register_model


class FLAMLAutoMLModel:
    def train_and_log(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        cfg: Dict[str, Any],
    ) -> TrainResult:
        artifact_path = "model"

        feature_cols = get_feature_cols_from_cfg(cfg)
        Xtr = coerce_features_float64(X_train, feature_cols)
        Xva = coerce_features_float64(X_val, feature_cols) if X_val is not None else None

        time_budget = int(cfg.get("automl", {}).get("time_budget_s", 120))
        task = cfg.get("automl", {}).get("task", "regression")
        metric = cfg.get("automl", {}).get("metric", "mae")

        with mlflow.start_run(nested=True) as run:
            mlflow.set_tag("model_family", "automl")
            mlflow.log_params(
                {
                    "time_budget_s": time_budget,
                    "task": task,
                    "metric": metric,
                    "num_features": len(feature_cols),
                }
            )

            # Always log feature ordering artifact in SAME run
            log_feature_cols_artifact(feature_cols)

            automl = AutoML()
            automl.fit(
                X_train=Xtr,
                y_train=y_train,
                X_val=Xva,
                y_val=y_val,
                task=task,
                metric=metric,
                time_budget=time_budget,
            )

            mlflow.log_param("best_estimator", str(getattr(automl, "best_estimator", "")))
            mlflow.log_param("best_config", str(getattr(automl, "best_config", "")))

            class _FLAMLPyfunc(mlflow.pyfunc.PythonModel):
                def __init__(self, automl_model):
                    self.automl_model = automl_model

                def predict(self, context, model_input):
                    return self.automl_model.predict(model_input)

            sig_kwargs = infer_and_log_signature(Xtr, y_train, artifact_path=artifact_path)

            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=_FLAMLPyfunc(automl),
                pip_requirements=[
                    "mlflow==2.19.0",
                    "pandas",
                    "numpy",
                    # Important: keep flaml pinned to match training
                    "flaml==2.3.0",
                ],
                **sig_kwargs,
            )

            version = None
            if cfg.get("mlflow", {}).get("register", True):
                version = register_model(run.info.run_id, artifact_path, cfg["mlflow"]["model_name"])
                mlflow.set_tag("registered_model_version", version)

            return TrainResult(run_id=run.info.run_id, artifact_path=artifact_path, registered_model_version=version)
