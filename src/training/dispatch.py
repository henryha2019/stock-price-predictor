from __future__ import annotations

from typing import Any


def get_model(model_family: str, **kwargs: Any):
    mf = model_family.lower()

    if mf == "ridge":
        from src.models.ridge import RidgeModel
        return RidgeModel(**kwargs)

    if mf in ("automl", "flaml"):
        from src.models.automl_flaml import FLAMLAutoMLModel
        return FLAMLAutoMLModel(**kwargs)

    if mf in ("custom_transformer", "transformer"):
        from src.models.transformer_custom import CustomTransformerModel
        return CustomTransformerModel(**kwargs)

    if mf in ("hf_finetune", "hf"):
        from src.models.hf_finetune import HFFineTuneModel
        return HFFineTuneModel(**kwargs)

    raise ValueError(f"Unknown model_family: {model_family}")
