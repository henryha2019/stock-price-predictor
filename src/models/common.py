from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature


@dataclass
class TrainResult:
    run_id: str
    artifact_path: str
    registered_model_version: Optional[str] = None


def get_feature_cols_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    feats = cfg.get("features", {}) if isinstance(cfg, dict) else {}
    cols = feats.get("feature_cols", [])
    if not isinstance(cols, list) or not cols:
        raise ValueError(
            "cfg['features']['feature_cols'] is required and must be a non-empty list. "
            "Train pipeline must set this before calling model.train_and_log()."
        )
    return cols


def coerce_features_float64(X: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Ensure:
      - exact columns exist
      - correct order
      - all float64 (prevents MLflow schema failures, esp. volume)
    Missing columns are filled with 0.0.
    Extra columns are dropped.
    """
    out = X.copy()

    for c in feature_cols:
        if c not in out.columns:
            out[c] = 0.0

    out = out[feature_cols].copy()

    # Enforce float64 for ALL features (including volume)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(0.0)
    out = out.astype("float64")

    return out


def log_feature_cols_artifact(feature_cols: List[str]) -> None:
    """
    Logs feature column order to the current MLflow run as 'feature_cols.json'.
    This artifact is required by serving to align inference columns exactly.
    """
    payload = {"feature_cols": feature_cols}
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/feature_cols.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        mlflow.log_artifact(path)


def infer_and_log_signature(
    X_example: pd.DataFrame,
    y_example: Optional[pd.Series] = None,
    *,
    artifact_path: str = "model",
) -> Dict[str, Any]:
    """
    Returns a dict suitable to pass into mlflow.<flavor>.log_model(...).
    Ensures you always get signature + input_example (eliminates warnings and helps serving).
    """
    input_example = X_example.head(5).copy()
    signature = infer_signature(input_example, None if y_example is None else y_example.head(5))
    return {"signature": signature, "input_example": input_example}
