# src/training/train.py
from __future__ import annotations

import argparse
from typing import Dict, Any, List

import yaml
import mlflow
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.data.load_hf import load_daily_ohlcv
from src.data.features import add_returns, make_tabular_features
from src.training.evaluate import mae, directional_accuracy
from src.training.dispatch import get_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and register models for stock-price-predictor.")
    p.add_argument("--config", required=True, help="Path to YAML config file.")
    p.add_argument(
        "--model_family",
        required=True,
        choices=["ridge", "automl", "custom_transformer", "hf_finetune"],
        help="Which model family to train.",
    )
    return p.parse_args()


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}. Found columns: {list(df.columns)}")


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    # ---- MLflow setup ----
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    # ---- Load data ----
    df = load_daily_ohlcv(cfg["data"]["hf_dataset_id"], split=cfg["data"]["split"])

    # Normalize/validate expected schema
    _require_columns(df, ["date", "symbol", "close"])
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["close", "symbol", "date"]).copy()

    # ---- Feature engineering ----
    df = add_returns(df, price_col="close", group_col="symbol")
    df = make_tabular_features(df, group_col="symbol")

    horizon = int(cfg["train"]["horizon"])
    df["target"] = df.groupby("symbol")["log_return_1"].shift(-horizon)

    feature_cols = ["r_mean_5", "r_std_5", "r_mean_20", "r_std_20", "r_last"]
    df = df.dropna(subset=feature_cols + ["target"]).copy()

    # MVP ticker subset
    universe = cfg["data"].get("universe", [])
    if universe:
        df = df[df["symbol"].isin(universe)].copy()

    df = df.sort_values(["symbol", "date"]).copy()

    X = df[feature_cols].copy()
    y = df["target"].copy()

    if len(X) < 200:
        raise ValueError(f"Not enough training rows after feature/target creation: {len(X)}")

    # ---- Evaluation splits (common metrics across model families) ----
    n_splits = int(cfg["train"].get("n_splits", 5))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # ---- Train + log via dispatcher ----
    model = get_model(args.model_family)

    with mlflow.start_run(run_name=args.model_family) as parent_run:
        mlflow.set_tag("project", "stock-price-predictor")
        mlflow.set_tag("model_family", args.model_family)

        # Log high-level params once, common across all
        mlflow.log_params(
            {
                "horizon": horizon,
                "n_splits": n_splits,
                "universe_size": int(df["symbol"].nunique()),
                "rows": int(len(df)),
                "features": ",".join(feature_cols),
            }
        )

        # Walk-forward CV metrics for gating/comparison (model-agnostic)
        fold_mae: List[float] = []
        fold_diracc: List[float] = []

        for i, (tr_idx, va_idx) in enumerate(tscv.split(X)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            # Important: only train/log the model ONCE (final fit), not per fold.
            # For fold metrics, we quickly fit a fresh instance per fold using the same model_family.
            # This keeps fold metrics comparable across families (works for ridge/automl).
            #
            # For transformer/HF, this "fit per fold" can be expensive. We handle that by:
            # - computing fold metrics only when cfg["train"]["cv_for_deep_models"]=true
            # - otherwise log holdout metrics only
            is_deep = args.model_family in ("custom_transformer", "hf_finetune")
            cv_for_deep = bool(cfg["train"].get("cv_for_deep_models", False))
            if is_deep and not cv_for_deep:
                break

            fold_model = get_model(args.model_family)
            # We do not register per-fold; the model implementation should honor cfg["mlflow"]["register"]
            # but to be safe we override here.
            cfg_fold = dict(cfg)
            cfg_fold["mlflow"] = dict(cfg.get("mlflow", {}))
            cfg_fold["mlflow"]["register"] = False

            res = fold_model.train_and_log(X_tr, y_tr, X_va, y_va, cfg_fold)

            # Load the just-logged fold model for predictions (ensures consistent interface)
            # The fold model is in the child run; use MLflow pyfunc.
            model_uri = f"runs:/{res.run_id}/{res.artifact_path}"
            pyfunc = mlflow.pyfunc.load_model(model_uri)
            pred = pyfunc.predict(X_va)

            m = mae(y_va, pred)
            d = directional_accuracy(y_va, pred)
            fold_mae.append(m)
            fold_diracc.append(d)

            mlflow.log_metric(f"mae_fold_{i}", m)
            mlflow.log_metric(f"diracc_fold_{i}", d)

        if fold_mae:
            mlflow.log_metric("mae_cv_mean", sum(fold_mae) / len(fold_mae))
            mlflow.log_metric("diracc_cv_mean", sum(fold_diracc) / len(fold_diracc))

        # Final train on full data (register this one)
        # Use a simple holdout for validation logging inside the model implementation.
        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_val, y_val = X.iloc[split:], y.iloc[split:]

        # Ensure final run registers (unless config says otherwise)
        cfg_final = dict(cfg)
        cfg_final["mlflow"] = dict(cfg.get("mlflow", {}))
        cfg_final["mlflow"]["register"] = bool(cfg_final["mlflow"].get("register", True))

        result = model.train_and_log(X_train, y_train, X_val, y_val, cfg_final)

        # Link child run info to parent for easy navigation
        mlflow.log_params(
            {
                "final_child_run_id": result.run_id,
                "final_artifact_path": result.artifact_path,
                "final_registered_version": result.registered_model_version or "",
            }
        )

        print(
            f"Done: model_family={args.model_family} "
            f"run_id={result.run_id} "
            f"version={result.registered_model_version}"
        )


if __name__ == "__main__":
    main()
