from __future__ import annotations
from typing import Dict, Any

import mlflow
import mlflow.pyfunc
import pandas as pd
from flaml import AutoML

from src.training.registry import register_model
from src.models.common import TrainResult


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
        time_budget = int(cfg["automl"]["time_budget_s"])
        task = cfg["automl"].get("task", "regression")
        metric = cfg["automl"].get("metric", "mae")

        with mlflow.start_run(nested=True) as run:
            mlflow.set_tag("model_family", "automl")
            mlflow.log_params({"time_budget_s": time_budget, "task": task, "metric": metric})

            automl = AutoML()
            automl.fit(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                task=task,
                metric=metric,
                time_budget=time_budget,
            )

            # FLAML exposes best config/estimator
            mlflow.log_param("best_estimator", str(automl.best_estimator))
            mlflow.log_param("best_config", str(automl.best_config))

            # Log as pyfunc
            class _FLAMLPyfunc(mlflow.pyfunc.PythonModel):
                def __init__(self, automl_model):
                    self.automl_model = automl_model

                def predict(self, context, model_input):
                    return self.automl_model.predict(model_input)

            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=_FLAMLPyfunc(automl),
                pip_requirements=["flaml==2.3.0", "pandas", "numpy", "mlflow"],
            )

            version = None
            if cfg["mlflow"].get("register", True):
                version = register_model(run.info.run_id, artifact_path, cfg["mlflow"]["model_name"])
                mlflow.set_tag("registered_model_version", version)

            return TrainResult(run_id=run.info.run_id, artifact_path=artifact_path, registered_model_version=version)
