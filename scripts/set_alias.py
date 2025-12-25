from __future__ import annotations

import argparse
import os
from typing import Optional

from mlflow.tracking import MlflowClient


def _resolve_version_from_run_id(client: MlflowClient, model_name: str, run_id: str) -> Optional[str]:
    # Find model versions for this model that came from the given run_id
    # Works across file store + tracking servers.
    versions = client.search_model_versions(f"name = '{model_name}'")
    matches = [mv for mv in versions if getattr(mv, "run_id", None) == run_id]
    if not matches:
        return None
    # choose highest version
    best = max(matches, key=lambda mv: int(mv.version))
    return str(best.version)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--alias", required=True)
    p.add_argument("--version", default="")
    p.add_argument("--run-id", default="")
    args = p.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("Missing env var MLFLOW_TRACKING_URI")

    client = MlflowClient(tracking_uri=tracking_uri)

    version = args.version.strip()
    run_id = args.run_id.strip()

    if not version:
        if not run_id:
            raise SystemExit("Need either --version or --run-id to resolve a model version.")
        resolved = _resolve_version_from_run_id(client, args.model_name, run_id)
        if not resolved:
            raise SystemExit(f"Could not resolve model version for model={args.model_name} run_id={run_id}")
        version = resolved

    client.set_registered_model_alias(args.model_name, args.alias, version)
    print(f"Set alias '{args.alias}' -> {args.model_name} v{version}")


if __name__ == "__main__":
    main()