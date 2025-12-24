from __future__ import annotations
from typing import Dict, Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import Ridge

from src.training.registry import register_model
from src.models.common import TrainResult


class RidgeModel:
    def train_and_log(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        cfg: Dict[str, Any],
    ) -> TrainResult:
        alpha = float(cfg["ridge"]["alpha"])
        seed = int(cfg["ridge"].get("seed", 42))
        artifact_path = "model"

        with mlflow.start_run(nested=True) as run:
            mlflow.set_tag("model_family", "ridge")
            mlflow.log_params({"alpha": alpha, "seed": seed})

            model = Ridge(alpha=alpha, random_state=seed)
            model.fit(X_train, y_train)

            if X_val is not None and y_val is not None:
                preds = model.predict(X_val)
                # basic metrics (keep it light)
                mae = float((abs(y_val.values - preds)).mean())
                mlflow.log_metric("val_mae", mae)

            mlflow.sklearn.log_model(model, artifact_path=artifact_path)

            version = None
            if cfg["mlflow"].get("register", True):
                version = register_model(run.info.run_id, artifact_path, cfg["mlflow"]["model_name"])
                mlflow.set_tag("registered_model_version", version)

            return TrainResult(run_id=run.info.run_id, artifact_path=artifact_path, registered_model_version=version)
