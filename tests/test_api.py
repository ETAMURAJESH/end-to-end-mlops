"""
tests/test_api.py — FastAPI endpoint tests.
Run with: pytest tests/ -v  (from project root)
"""

import os
import sys
from unittest.mock import patch

# Add project root to path before any src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient  # noqa: E402

from src.app import app  # noqa: E402

client = TestClient(app)

# ── Sample data ───────────────────────────────────────────────────────────────
VALID_PASSENGER = {
    "data": [
        {
            "Pclass": 3,
            "Age": 22.0,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 7.25,
            "Sex": "male",
            "Embarked": "S",
        }
    ]
}

VALID_BATCH = {
    "data": [
        {
            "Pclass": 3,
            "Age": 22.0,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 7.25,
            "Sex": "male",
            "Embarked": "S",
        },
        {
            "Pclass": 1,
            "Age": 38.0,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 71.28,
            "Sex": "female",
            "Embarked": "C",
        },
        {
            "Pclass": 2,
            "Age": 26.0,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 13.00,
            "Sex": "female",
            "Embarked": "S",
        },
    ]
}


# ── Health check ──────────────────────────────────────────────────────────────


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_schema():
    response = client.get("/health")
    body = response.json()
    assert "status" in body
    assert "model_ready" in body
    assert body["status"] == "ok"


def test_health_model_ready_is_bool():
    response = client.get("/health")
    assert isinstance(response.json()["model_ready"], bool)


# ── Model info ────────────────────────────────────────────────────────────────


def test_model_info_returns_200():
    response = client.get("/model/info")
    assert response.status_code == 200


def test_model_info_schema():
    response = client.get("/model/info")
    body = response.json()
    for key in ["model_name", "model_path", "is_retraining"]:
        assert key in body, f"Missing key: {key}"


def test_model_info_not_retraining_at_startup():
    response = client.get("/model/info")
    assert response.json()["is_retraining"] is False


# ── Predict ───────────────────────────────────────────────────────────────────


def test_predict_single_passenger():
    response = client.post("/predict", json=VALID_PASSENGER)
    assert response.status_code in (200, 503)


def test_predict_returns_correct_schema_when_model_loaded():
    response = client.post("/predict", json=VALID_PASSENGER)
    if response.status_code == 200:
        body = response.json()
        assert "predictions" in body
        assert "model_name" in body
        assert "record_count" in body
        assert body["record_count"] == 1
        assert isinstance(body["predictions"], list)


def test_predict_batch_record_count():
    response = client.post("/predict", json=VALID_BATCH)
    if response.status_code == 200:
        assert response.json()["record_count"] == 3


def test_predict_empty_data_returns_422():
    response = client.post("/predict", json={"data": []})
    assert response.status_code == 422


def test_predict_missing_data_key_returns_422():
    response = client.post("/predict", json={})
    assert response.status_code == 422


# ── Retrain ───────────────────────────────────────────────────────────────────


def test_retrain_returns_202():
    with patch("src.app._retrain_job"):
        response = client.post("/retrain")
    assert response.status_code == 202


def test_retrain_response_schema():
    with patch("src.app._retrain_job"):
        response = client.post("/retrain")
    body = response.json()
    assert "status" in body
    assert "message" in body
    assert body["status"] == "accepted"


def test_retrain_blocks_duplicate_while_running():
    from src.app import store

    store.is_retraining = True
    response = client.post("/retrain")
    store.is_retraining = False
    assert response.status_code == 409
