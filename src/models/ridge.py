from __future__ import annotations

from typing import Any, Dict

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.models.common import (
    TrainResult,
    coerce_features_float64,
    get_feature_cols_from_cfg,
    infer_and_log_signature,
    log_feature_cols_artifact,
)
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

        feature_cols = get_feature_cols_from_cfg(cfg)
        Xtr = coerce_features_float64(X_train, feature_cols)
        Xva = coerce_features_float64(X_val, feature_cols) if X_val is not None else None

        alpha = float(cfg.get("ridge", {}).get("alpha", 1.0))
        seed = int(cfg.get("ridge", {}).get("seed", 42))

        with mlflow.start_run(nested=True) as run:
            mlflow.set_tag("model_family", "ridge")
            mlflow.log_params({"alpha": alpha, "seed": seed, "num_features": len(feature_cols)})

            # Always log feature ordering artifact in SAME run
            log_feature_cols_artifact(feature_cols)

            model = Ridge(alpha=alpha, random_state=seed)
            model.fit(Xtr, y_train)

            # Optional: quick val metrics
            if Xva is not None and y_val is not None and len(Xva) > 0:
                pred = model.predict(Xva)
                mae = float(np.mean(np.abs(np.asarray(y_val) - np.asarray(pred))))
                mlflow.log_metric("val_mae", mae)

            sig_kwargs = infer_and_log_signature(Xtr, y_train, artifact_path=artifact_path)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                pip_requirements=[
                    "mlflow==2.19.0",
                    "pandas",
                    "numpy",
                    "scikit-learn==1.6.1",
                ],
                **sig_kwargs,
            )

            version = None
            if cfg.get("mlflow", {}).get("register", True):
                version = register_model(run.info.run_id, artifact_path, cfg["mlflow"]["model_name"])
                mlflow.set_tag("registered_model_version", version)

            return TrainResult(
                run_id=run.info.run_id,
                artifact_path=artifact_path,
                registered_model_version=version,
            )
