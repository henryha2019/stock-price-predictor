from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient


def register_model(run_id: str, artifact_path: str, model_name: str) -> str:
    """
    Register a model artifact from a specific run.
    Returns the registered model version as a string.
    """
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"

    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="None",
        archive_existing_versions=False,
    )
    return str(mv.version)
