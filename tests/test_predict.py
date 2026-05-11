"""
tests/test_predict.py
Unit tests for prediction pipeline logic — no ML models needed.
"""
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Feature engineer tests ────────────────────────────────────
def test_engineer_single_row_output_keys():
    """engineer_single_row should return all required feature keys."""
    from src.feature_engineer import engineer_single_row, reset_history

    reset_history()
    raw = {
        "resource_id"       : "ec2-web-01",
        "resource_type"     : "EC2",
        "timestamp"         : "2026-05-11T10:00:00",
        "cpu_utilization"   : 45.0,
        "memory_utilization": 60.0,
        "network_in_mbps"   : 50.0,
        "network_out_mbps"  : 30.0,
        "disk_io_mbps"      : 20.0,
        "request_count"     : 500,
        "error_rate_pct"    : 0.5,
        "cost_per_hour"     : 2.5,
    }

    result = engineer_single_row(raw)

    # Check derived features exist
    assert "cpu_mem_ratio"      in result
    assert "cost_per_cpu"       in result
    assert "total_network_mbps" in result
    assert "resource_pressure"  in result
    assert "hour_sin"           in result
    assert "hour_cos"           in result
    assert "is_business_hours"  in result
    assert "resource_id_enc"    in result


def test_engineer_single_row_ratios():
    """Derived ratios should be mathematically correct."""
    from src.feature_engineer import engineer_single_row, reset_history

    reset_history()
    raw = {
        "resource_id"       : "ec2-web-01",
        "resource_type"     : "EC2",
        "timestamp"         : "2026-05-11T14:00:00",
        "cpu_utilization"   : 80.0,
        "memory_utilization": 40.0,
        "network_in_mbps"   : 60.0,
        "network_out_mbps"  : 40.0,
        "disk_io_mbps"      : 20.0,
        "request_count"     : 500,
        "error_rate_pct"    : 1.0,
        "cost_per_hour"     : 4.0,
    }

    result = engineer_single_row(raw)

    assert result["total_network_mbps"] == pytest.approx(100.0, abs=0.1)
    assert result["resource_pressure"]  == pytest.approx(60.0,  abs=0.1)
    assert result["cpu_mem_ratio"]      == pytest.approx(2.0,   abs=0.1)


def test_engineer_rolling_history_builds():
    """Rolling stats should update as more rows are processed."""
    from src.feature_engineer import engineer_single_row, reset_history

    reset_history("ec2-web-01")

    base_row = {
        "resource_id"       : "ec2-web-01",
        "resource_type"     : "EC2",
        "timestamp"         : "2026-05-11T10:00:00",
        "cpu_utilization"   : 50.0,
        "memory_utilization": 60.0,
        "network_in_mbps"   : 50.0,
        "network_out_mbps"  : 30.0,
        "disk_io_mbps"      : 20.0,
        "request_count"     : 500,
        "error_rate_pct"    : 0.5,
        "cost_per_hour"     : 2.5,
    }

    # First row — no history yet
    r1 = engineer_single_row({**base_row})
    lag_1_before = r1["cpu_utilization_lag_1h"]

    # Second row — history now has 1 entry
    r2 = engineer_single_row({**base_row, "cpu_utilization": 70.0})
    lag_1_after = r2["cpu_utilization_lag_1h"]

    # Lag should now reflect the first row's value
    assert lag_1_after == pytest.approx(50.0, abs=1.0)


def test_resource_id_encoding():
    """Known resource IDs should encode to integers."""
    from src.feature_engineer import engineer_single_row, reset_history, RESOURCE_ID_MAP

    reset_history()
    for rid, expected_enc in RESOURCE_ID_MAP.items():
        raw = {
            "resource_id"       : rid,
            "resource_type"     : "EC2",
            "timestamp"         : "2026-05-11T10:00:00",
            "cpu_utilization"   : 45.0,
            "memory_utilization": 60.0,
            "network_in_mbps"   : 50.0,
            "network_out_mbps"  : 30.0,
            "disk_io_mbps"      : 20.0,
            "request_count"     : 500,
            "error_rate_pct"    : 0.5,
            "cost_per_hour"     : 2.5,
        }
        result = engineer_single_row(raw)
        assert result["resource_id_enc"] == expected_enc


# ── Data generator tests ──────────────────────────────────────
def test_generate_historical_data_shape():
    """Historical data should have correct dimensions."""
    from src.data_generator import generate_historical_data

    df = generate_historical_data(days=3, anomaly_rate=0.07)
    assert len(df) == 3 * 24 * 10   # 3 days × 24 hours × 10 resources
    assert "is_anomaly"    in df.columns
    assert "anomaly_type"  in df.columns
    assert "cpu_utilization" in df.columns


def test_generate_historical_anomaly_rate():
    """Anomaly rate should be approximately as specified."""
    from src.data_generator import generate_historical_data

    df = generate_historical_data(days=30, anomaly_rate=0.10)
    actual_rate = df["is_anomaly"].mean()
    # Allow ±5% tolerance
    assert 0.05 <= actual_rate <= 0.15


def test_generate_realtime_row_keys():
    """Realtime row should have all required metric keys."""
    from src.data_generator import generate_realtime_row, RESOURCES
    import random

    resource = random.choice(RESOURCES)
    row = generate_realtime_row(resource)

    required_keys = [
        "resource_id", "resource_type", "cpu_utilization",
        "memory_utilization", "network_in_mbps", "network_out_mbps",
        "disk_io_mbps", "request_count", "error_rate_pct", "cost_per_hour"
    ]
    for key in required_keys:
        assert key in row, f"Missing key: {key}"


def test_generate_realtime_row_value_ranges():
    """Generated metric values should be within realistic ranges."""
    from src.data_generator import generate_realtime_row, RESOURCES
    import random

    for _ in range(20):
        resource = random.choice(RESOURCES)
        row = generate_realtime_row(resource)

        assert 0 <= row["cpu_utilization"]    <= 100
        assert 0 <= row["memory_utilization"] <= 100
        assert row["network_in_mbps"]         >= 0
        assert row["cost_per_hour"]           >= 0
        assert row["request_count"]           >= 0


# ── Severity logic test ───────────────────────────────────────
def test_severity_thresholds():
    """Severity labels should map correctly to probability ranges."""
    def get_severity(prob, threshold=0.55):
        is_anomaly = prob >= threshold
        if not is_anomaly:
            return "normal"
        elif prob >= 0.90:
            return "critical"
        elif prob >= 0.75:
            return "high"
        else:
            return "medium"

    assert get_severity(0.10) == "normal"
    assert get_severity(0.50) == "normal"
    assert get_severity(0.60) == "medium"
    assert get_severity(0.80) == "high"
    assert get_severity(0.95) == "critical"