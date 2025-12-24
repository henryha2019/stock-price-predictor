from __future__ import annotations
from typing import Dict, Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.registry import register_model
from src.models.common import TrainResult


class TinyTransformer(nn.Module):
    def __init__(self, d_in: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (B, T, F)
        x = self.in_proj(x)
        h = self.encoder(x)
        return self.out(h[:, -1, :]).squeeze(-1)  # last token


def _to_sequence(X: pd.DataFrame, seq_len: int) -> tuple[np.ndarray, list[str]]:
    # Simple: treat feature rows as time-ordered and build sliding windows.
    # For multi-symbol you should build sequences per symbol; keep MVP pooled.
    values = X.values.astype(np.float32)
    n = len(values)
    seqs = []
    for i in range(seq_len, n):
        seqs.append(values[i - seq_len:i])
    return np.stack(seqs, axis=0), list(X.columns)


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

        seq_len = int(cfg["transformer"]["seq_len"])
        epochs = int(cfg["transformer"]["epochs"])
        lr = float(cfg["transformer"]["lr"])
        batch_size = int(cfg["transformer"]["batch_size"])
        device = "cuda" if torch.cuda.is_available() else "cpu"

        X_seq, cols = _to_sequence(X_train, seq_len)
        y_seq = y_train.values.astype(np.float32)[seq_len:]

        ds = TensorDataset(torch.tensor(X_seq), torch.tensor(y_seq))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        model = TinyTransformer(d_in=X_seq.shape[-1],
                                d_model=int(cfg["transformer"]["d_model"]),
                                nhead=int(cfg["transformer"]["nhead"]),
                                num_layers=int(cfg["transformer"]["num_layers"])).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        with mlflow.start_run(nested=True) as run:
            mlflow.set_tag("model_family", "custom_transformer")
            mlflow.log_params({
                "seq_len": seq_len,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "d_model": int(cfg["transformer"]["d_model"]),
                "nhead": int(cfg["transformer"]["nhead"]),
                "num_layers": int(cfg["transformer"]["num_layers"]),
                "device": device,
            })

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
                    losses.append(loss.item())
                mlflow.log_metric("train_mse", float(np.mean(losses)), step=ep)

            # Save weights + columns + seq_len
            tmp_dir = "artifacts_transformer"
            import os, json
            os.makedirs(tmp_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{tmp_dir}/weights.pt")
            with open(f"{tmp_dir}/meta.json", "w") as f:
                json.dump({"cols": cols, "seq_len": seq_len,
                           "d_model": int(cfg["transformer"]["d_model"]),
                           "nhead": int(cfg["transformer"]["nhead"]),
                           "num_layers": int(cfg["transformer"]["num_layers"])}, f)

            class _TransformerPyfunc(mlflow.pyfunc.PythonModel):
                def load_context(self, context):
                    import json
                    meta = json.load(open(context.artifacts["meta"]))
                    self.cols = meta["cols"]
                    self.seq_len = meta["seq_len"]

                    self.model = TinyTransformer(
                        d_in=len(self.cols),
                        d_model=meta["d_model"],
                        nhead=meta["nhead"],
                        num_layers=meta["num_layers"],
                    )
                    self.model.load_state_dict(torch.load(context.artifacts["weights"], map_location="cpu"))
                    self.model.eval()

                def predict(self, context, model_input):
                    # model_input is DataFrame with same cols as training features.
                    X = model_input[self.cols].values.astype(np.float32)
                    if len(X) < self.seq_len:
                        raise ValueError("Not enough rows to form a sequence window")
                    seq = X[-self.seq_len:][None, :, :]
                    with torch.no_grad():
                        pred = self.model(torch.tensor(seq))
                    return pred.numpy().reshape(-1)

            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=_TransformerPyfunc(),
                artifacts={"weights": f"{tmp_dir}/weights.pt", "meta": f"{tmp_dir}/meta.json"},
                pip_requirements=["torch==2.5.1", "pandas", "numpy", "mlflow"],
            )

            version = None
            if cfg["mlflow"].get("register", True):
                version = register_model(run.info.run_id, artifact_path, cfg["mlflow"]["model_name"])
                mlflow.set_tag("registered_model_version", version)

            return TrainResult(run_id=run.info.run_id, artifact_path=artifact_path, registered_model_version=version)
