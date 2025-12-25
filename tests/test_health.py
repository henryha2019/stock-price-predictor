import os
os.environ["SKIP_MODEL_LOAD"] = "1"

from fastapi.testclient import TestClient
from src.serving.app import app

def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
