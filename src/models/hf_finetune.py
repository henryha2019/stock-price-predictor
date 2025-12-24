from __future__ import annotations
from typing import Dict, Any

import mlflow
import mlflow.pyfunc
import pandas as pd

from src.training.registry import register_model
from src.models.common import TrainResult


class HFFineTuneModel:
    def train_and_log(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        cfg: Dict[str, Any],
    ) -> TrainResult:
        """
        This is intentionally a skeleton until you pick a specific HF time-series model.

        Once you choose the model:
        - build sequence dataset
        - fine-tune with transformers Trainer
        - log as mlflow.pyfunc (or log the HF model directory)
        """
        artifact_path = "model"

        with mlflow.start_run(nested=True) as run:
            mlflow.set_tag("model_family", "hf_finetune")
            mlflow.log_params({"status": "skeleton", "note": "Select TS foundation model then implement."})

            # Placeholder pyfunc that raises until implemented
            class _NotImplementedPyfunc(mlflow.pyfunc.PythonModel):
                def predict(self, context, model_input):
                    raise NotImplementedError("HF fine-tune model not implemented yet. Pick a TS model and train.")

            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=_NotImplementedPyfunc(),
                pip_requirements=["transformers==4.46.3", "pandas", "numpy", "mlflow"],
            )

            version = None
            if cfg["mlflow"].get("register", True):
                version = register_model(run.info.run_id, artifact_path, cfg["mlflow"]["model_name"])
                mlflow.set_tag("registered_model_version", version)

            return TrainResult(run_id=run.info.run_id, artifact_path=artifact_path, registered_model_version=version)
