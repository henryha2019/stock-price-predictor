from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Dict, Any

import pandas as pd


@dataclass
class TrainResult:
    run_id: str
    artifact_path: str
    registered_model_version: str | None = None


class TrainableModel(Protocol):
    def train_and_log(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        cfg: Dict[str, Any],
    ) -> TrainResult:
        ...
