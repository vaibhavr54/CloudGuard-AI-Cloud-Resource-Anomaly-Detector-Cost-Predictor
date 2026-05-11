"""
tests/test_api.py
Tests for FastAPI endpoints — runs without trained models using mocks.
"""
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Mock the entire predict module BEFORE any imports ────────
# This prevents src/predict.py from trying to load .pkl files
mock_predict_module = MagicMock()

def mock_predict_single(raw_row: dict) -> dict:
    return {
        "timestamp"          : "2026-05-11T10:00:00",
        "resource_id"        : raw_row.get("resource_id", "ec2-web-01"),
        "resource_type"      : raw_row.get("resource_type", "EC2"),
        "is_anomaly"         : 0,
        "anomaly_prob"       : 0.03,
        "severity"           : "normal",
        "predicted_cost"     : 2.45,
        "clf_reasons"        : [
            {"feature": "cost_per_hour_zscore", "value": 0.1,
             "shap": -2.1, "impact": "down_anomaly"}
        ],
        "reg_reasons"        : [
            {"feature": "cpu_utilization", "value": 45.0,
             "shap": 0.6, "impact": "up_cost"}
        ],
        "cpu_utilization"    : 45.0,
        "memory_utilization" : 60.0,
        "network_in_mbps"    : 50.0,
        "cost_per_hour_actual": 2.48,
    }

def mock_engineer_single_row(raw: dict) -> dict:
    return {**raw, "cpu_mem_ratio": 0.75, "cost_per_cpu": 0.05,
            "is_business_hours": 1, "is_night": 0}

mock_predict_module.predict_single = mock_predict_single


# Patch sys.modules BEFORE importing anything from src or api
sys.modules["src.predict"] = mock_predict_module


@pytest.fixture
def client():
    """Create test client with mocked ML pipeline."""
    with patch("api.routes.predict_single", mock_predict_single), \
         patch("api.routes.engineer_single_row", mock_engineer_single_row):
        from fastapi.testclient import TestClient
        from api.main import app
        with TestClient(app) as c:
            yield c


# ── Health check ──────────────────────────────────────────────
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


# ── Stream endpoint ───────────────────────────────────────────
def test_stream_returns_prediction(client):
    with patch("api.routes.generate_realtime_row") as mock_gen:
        mock_gen.return_value = {
            "resource_id"  : "ec2-web-01",
            "resource_type": "EC2",
            "timestamp"    : "2026-05-11T10:00:00",
        }
        response = client.get("/stream")
        assert response.status_code == 200
        data = response.json()
        assert "is_anomaly"     in data
        assert "anomaly_prob"   in data
        assert "predicted_cost" in data
        assert "severity"       in data
        assert "resource_id"    in data
        assert "clf_reasons"    in data
        assert "reg_reasons"    in data


# ── Stats endpoint ────────────────────────────────────────────
def test_stats_empty(client):
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "total_anomalies"   in data
    assert "anomaly_rate"      in data
    assert "avg_cost"          in data


# ── Resources endpoint ────────────────────────────────────────
def test_resources_returns_list(client):
    response = client.get("/resources")
    assert response.status_code == 200
    data = response.json()
    assert "resources" in data
    assert len(data["resources"]) == 10


# ── History endpoint ──────────────────────────────────────────
def test_history_empty(client):
    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()
    assert "count"   in data
    assert "records" in data
    assert isinstance(data["records"], list)


def test_history_with_resource_filter(client):
    response = client.get("/history?resource_id=ec2-web-01")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data


# ── Severity values ───────────────────────────────────────────
def test_severity_values_valid(client):
    with patch("api.routes.generate_realtime_row") as mock_gen:
        mock_gen.return_value = {
            "resource_id"  : "ec2-web-01",
            "resource_type": "EC2",
            "timestamp"    : "2026-05-11T10:00:00",
        }
        response = client.get("/stream")
        data = response.json()
        assert data["severity"] in ["normal", "medium", "high", "critical"]


# ── Anomaly probability range ─────────────────────────────────
def test_anomaly_prob_in_range(client):
    with patch("api.routes.generate_realtime_row") as mock_gen:
        mock_gen.return_value = {
            "resource_id"  : "ec2-web-01",
            "resource_type": "EC2",
            "timestamp"    : "2026-05-11T10:00:00",
        }
        response = client.get("/stream")
        data = response.json()
        assert 0.0 <= data["anomaly_prob"] <= 1.0