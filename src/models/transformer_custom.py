from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.common import (
    TrainResult,
    coerce_features_float64,
    get_feature_cols_from_cfg,
    infer_and_log_signature,
    log_feature_cols_artifact,
)
from src.training.registry import register_model


class _TinyTransformer(nn.Module):
    def __init__(self, d_in: int, d_model: int, nhead: int, num_layers: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.proj(x)
        x = self.enc(x)
        x = x[:, -1, :]
        return self.head(x).squeeze(-1)


def _to_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i - seq_len : i])
        ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)


class CustomTransformerModel:
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
        Xtr_df = coerce_features_float64(X_train, feature_cols)
        Xva_df = coerce_features_float64(X_val, feature_cols) if X_val is not None else None

        tcfg = cfg.get("transformer", {})
        seq_len = int(tcfg.get("seq_len", 30))
        epochs = int(tcfg.get("epochs", 3))
        lr = float(tcfg.get("lr", 1e-3))
        batch_size = int(tcfg.get("batch_size", 64))
        d_model = int(tcfg.get("d_model", 32))
        nhead = int(tcfg.get("nhead", 4))
        num_layers = int(tcfg.get("num_layers", 2))

        device = "cpu"

        with mlflow.start_run(nested=True) as run:
            mlflow.set_tag("model_family", "custom_transformer")
            mlflow.log_params(
                {
                    "seq_len": seq_len,
                    "epochs": epochs,
                    "lr": lr,
                    "batch_size": batch_size,
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_layers": num_layers,
                    "num_features": len(feature_cols),
                }
            )

            # Required for serving alignment
            log_feature_cols_artifact(feature_cols)

            Xtr = Xtr_df.to_numpy(dtype=np.float32)
            ytr = np.asarray(y_train, dtype=np.float32)

            Xs, ys = _to_sequences(Xtr, ytr, seq_len=seq_len)
            if len(Xs) == 0:
                raise ValueError(f"Not enough rows for seq_len={seq_len} after preprocessing.")

            ds = TensorDataset(torch.tensor(Xs), torch.tensor(ys))
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

            model = _TinyTransformer(d_in=Xs.shape[-1], d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.L1Loss()

            model.train()
            for ep in range(epochs):
                losses = []
                for xb, yb in dl:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.item()))
                mlflow.log_metric("train_mae_epoch", float(np.mean(losses)), step=ep)

            class _TxPyfunc(mlflow.pyfunc.PythonModel):
                def __init__(self, torch_model, seq_len: int):
                    self.m = torch_model.eval()
                    self.seq_len = seq_len

                def predict(self, context, model_input: pd.DataFrame):
                    X = model_input.to_numpy(dtype=np.float32)
                    if len(X) < self.seq_len:
                        raise ValueError(f"Need at least seq_len={self.seq_len} rows.")
                    x_seq = torch.tensor(X[-self.seq_len :][None, ...], dtype=torch.float32)
                    with torch.no_grad():
                        y = self.m(x_seq).cpu().numpy().reshape(-1)
                    return y

            # Signature should match the TABULAR input you send at inference
            sig_kwargs = infer_and_log_signature(Xtr_df, y_train, artifact_path=artifact_path)

            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=_TxPyfunc(model, seq_len=seq_len),
                pip_requirements=[
                    "mlflow==2.19.0",
                    "pandas",
                    "numpy",
                    "torch",
                ],
                **sig_kwargs,
            )

            version = None
            if cfg.get("mlflow", {}).get("register", True):
                version = register_model(run.info.run_id, artifact_path, cfg["mlflow"]["model_name"])
                mlflow.set_tag("registered_model_version", version)

            return TrainResult(run_id=run.info.run_id, artifact_path=artifact_path, registered_model_version=version)
