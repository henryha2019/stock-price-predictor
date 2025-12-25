#!/usr/bin/env bash
set -euo pipefail

# Promote a specific model version to alias "prod"
# Usage:
#   export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
#   export MODEL_NAME=stock-price-predictor
#   export MODEL_ALIAS=prod
#   ./scripts/promote_model.sh 12

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <MODEL_VERSION>"
  exit 1
fi

MODEL_VERSION="$1"
MODEL_NAME="${MODEL_NAME:-stock-price-predictor}"
ALIAS="${MODEL_ALIAS:-prod}"

python - <<PY
import os
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
client = MlflowClient()

name = os.environ.get("MODEL_NAME", "${MODEL_NAME}")
version = "${MODEL_VERSION}"
alias = os.environ.get("MODEL_ALIAS", "${ALIAS}")

client.set_registered_model_alias(name, alias, version)
print(f"Set alias {alias} -> {name} v{version}")
PY
