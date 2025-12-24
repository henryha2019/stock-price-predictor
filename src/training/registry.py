import mlflow


def register_model(run_id: str, artifact_path: str, model_name: str) -> str:
    """
    Register a model logged under a run_id + artifact_path into MLflow Model Registry.
    Returns the registered model version string (e.g., "12").
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    return str(mv.version)
