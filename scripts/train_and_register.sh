#!/usr/bin/env bash
set -euo pipefail

# Example usage:
#   export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
#   make train

python -m src.training.train --config configs/train.yaml
