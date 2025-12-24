import argparse
import os
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from src.data.load_hf import load_daily_ohlcv
from src.data.features import add_returns, make_tabular_features
from src.training.evaluate import mae, directional_accuracy
from src.training.registry import register_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    df = load_daily_ohlcv(cfg["data"]["hf_dataset_id"], split=cfg["data"]["split"])

    # Expected columns in df: date, symbol, close, volume, open, high, low (dataset-dependent)
    # You may need to rename columns once you pick the dataset.
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close", "symbol", "date"])

    df = add_returns(df, price_col="close", group_col="symbol")
    df = make_tabular_features(df, group_col="symbol")
    df["target"] = df.groupby("symbol")["log_return_1"].shift(-cfg["train"]["horizon"])

    feature_cols = ["r_mean_5", "r_std_5", "r_mean_20", "r_std_20", "r_last"]
    df = df.dropna(subset=feature_cols + ["target"]).copy()

    # Train on a single ticker subset for MVP
    universe = cfg["data"]["universe"]
    if universe:
        df = df[df["symbol"].isin(universe)].copy()

    df = df.sort_values(["symbol", "date"]).copy()

    # Simple approach: pool all rows (you can move to per-symbol models later)
    X = df[feature_cols]
    y = df["target"]

    tscv = TimeSeriesSplit(n_splits=cfg["train"]["n_splits"])

    with mlflow.start_run() as run:
        mlflow.log_params({
            "horizon": cfg["train"]["horizon"],
            "model": "Ridge",
            "alpha": cfg["train"]["ridge_alpha"],
            "n_splits": cfg["train"]["n_splits"],
            "universe_size": len(set(df["symbol"])),
        })

        fold_mae = []
        fold_diracc = []

        # Train final model on full data, but compute CV metrics for gating
        for i, (tr_idx, va_idx) in enumerate(tscv.split(X)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            model = Ridge(alpha=cfg["train"]["ridge_alpha"], random_state=cfg["train"]["seed"])
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)

            m = mae(y_va, pred)
            d = directional_accuracy(y_va, pred)
            fold_mae.append(m)
            fold_diracc.append(d)

            mlflow.log_metric(f"mae_fold_{i}", m)
            mlflow.log_metric(f"diracc_fold_{i}", d)

        mlflow.log_metric("mae_cv_mean", sum(fold_mae) / len(fold_mae))
        mlflow.log_metric("diracc_cv_mean", sum(fold_diracc) / len(fold_diracc))

        # Fit final model
        final_model = Ridge(alpha=cfg["train"]["ridge_alpha"], random_state=cfg["train"]["seed"])
        final_model.fit(X, y)

        artifact_path = "model"
        mlflow.sklearn.log_model(final_model, artifact_path=artifact_path)

        # Register model
        model_name = cfg["mlflow"]["model_name"]
        version = register_model(run.info.run_id, artifact_path, model_name)
        mlflow.set_tag("registered_model_version", version)

        print(f"Registered {model_name} version={version} run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
