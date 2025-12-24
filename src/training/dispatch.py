from __future__ import annotations
from typing import Dict, Any

from src.models.ridge import RidgeModel
from src.models.automl_flaml import FLAMLAutoMLModel
from src.models.transformer_custom import CustomTransformerModel
from src.models.hf_finetune import HFFineTuneModel


def get_model(model_family: str):
    model_family = model_family.lower()
    if model_family == "ridge":
        return RidgeModel()
    if model_family == "automl":
        return FLAMLAutoMLModel()
    if model_family in ("custom_transformer", "transformer"):
        return CustomTransformerModel()
    if model_family in ("hf_finetune", "hf"):
        return HFFineTuneModel()
    raise ValueError(f"Unknown model_family: {model_family}")
