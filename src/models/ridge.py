from __future__ import annotations

from typing import Dict, Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.models.common import TrainResult
from src.training.registry import register_model


class RidgeModel:
    def train_and_log(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        cfg: Dict[str, Any],
    ) -> TrainResult:
        artifact_path = "model"
        alpha = float(cfg["ridge"].get("alpha", 1.0))
        seed = int(cfg["ridge"].get("seed", 42))

        # Feature order used at training time (passed from train.py)
        feature_cols = cfg.get("features", {}).get("feature_cols")
        if not feature_cols:
            feature_cols = list(X_train.columns)

        # Enforce training order before fit (important)
        X_train = X_train.loc[:, feature_cols].copy()
        if X_val is not None:
            X_val = X_val.loc[:, feature_cols].copy()

        with mlflow.start_run(nested=True) as run:
            mlflow.set_tag("model_family", "ridge")
            mlflow.log_params({"alpha": alpha, "seed": seed})
            mlflow.log_dict({"feature_cols": list(feature_cols)}, "feature_cols.json")

            model = Ridge(alpha=alpha, random_state=seed)
            model.fit(X_train, y_train)

            # Basic validation metrics (optional)
            if X_val is not None and y_val is not None and len(X_val) > 0:
                pred = model.predict(X_val)
                mae = float(np.mean(np.abs(np.asarray(y_val) - np.asarray(pred))))
                mlflow.log_metric("mae_val", mae)

            # Log with signature + input_example to remove warnings
            input_example = X_train.head(3)
            signature = mlflow.models.infer_signature(X_train, model.predict(X_train))

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
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
