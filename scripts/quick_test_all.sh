#!/usr/bin/env bash
set -euo pipefail

TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5001}"
MODEL_NAME="${MODEL_NAME:-stock-price-predictor}"
PORT="${PORT:-8000}"
PAYLOAD="${PAYLOAD:-payloads/ohlcv_11.json}"

export MLFLOW_TRACKING_URI="$TRACKING_URI"

run_one () {
  local family="$1"
  local cfg="$2"
  local alias="$3"

  echo "TRAIN: $family"
  out="$(python -m src.training.train --config "$cfg" --model_family "$family")"
  echo "$out"

  # Robust parsing: grab the last "Done:" line, then extract run_id + version via awk
  done_line="$(echo "$out" | grep -E "^Done: model_family=" | tail -n 1 || true)"
  if [ -z "$done_line" ]; then
    echo "ERROR: Could not find 'Done:' line in training output for $family"
    exit 1
  fi

  run_id="$(echo "$done_line" | awk '{
      for (i=1; i<=NF; i++) if ($i ~ /^run_id=/) {sub(/^run_id=/,"",$i); print $i; exit}
  }')"
  version="$(echo "$done_line" | awk '{
      for (i=1; i<=NF; i++) if ($i ~ /^version=/) {sub(/^version=/,"",$i); print $i; exit}
  }')"

  if [ -z "$run_id" ]; then
    echo "ERROR: Could not parse run_id from: $done_line"
    exit 1
  fi

  # If version parsing fails for any reason, set_alias.py will resolve by run_id
  python scripts/set_alias.py --model-name "$MODEL_NAME" --alias "$alias" --version "${version:-}" --run-id "$run_id"

  # Start server with chosen alias
  MODEL_ALIAS="$alias" MODEL_NAME="$MODEL_NAME" MIN_OHLCV_WINDOW=11 \
  PYTHONPATH=. .venv/bin/uvicorn src.serving.app:app --host 127.0.0.1 --port "$PORT" &
  pid=$!

  # Wait a moment for startup
  sleep 5

  echo "READY:"
  curl -s "http://127.0.0.1:${PORT}/ready" | python -m json.tool

  echo "PREDICT:"
  curl -s -X POST "http://127.0.0.1:${PORT}/predict" \
    -H "Content-Type: application/json" \
    -d @"$PAYLOAD" | python -m json.tool

  kill "$pid"
}

run_one ridge configs/quick/ridge.yaml quick-ridge
run_one automl configs/quick/automl.yaml quick-automl
run_one custom_transformer configs/quick/transformer.yaml quick-transformer
run_one hf_finetune configs/quick/hf.yaml quick-hf
