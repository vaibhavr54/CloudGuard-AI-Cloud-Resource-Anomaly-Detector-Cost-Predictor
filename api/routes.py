import sys
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
from fastapi import APIRouter, HTTPException

# Ensure project root is in path before any src imports
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.schemas import MetricInput, PredictionResponse
from src.predict import predict_single
from src.data_generator import generate_realtime_row, RESOURCES
from src.feature_engineer import engineer_single_row

router = APIRouter()

# In-memory history store — last 200 predictions per resource
history_store = {}
MAX_HISTORY   = 200

def _store(result: dict):
    rid = result["resource_id"]
    if rid not in history_store:
        history_store[rid] = deque(maxlen=MAX_HISTORY)
    history_store[rid].appendleft(result)


@router.post("/predict", response_model=PredictionResponse)
async def predict(payload: MetricInput):
    """
    Predict anomaly + cost for a single incoming metric row.
    Accepts raw metrics, runs feature engineering, returns prediction.
    """
    try:
        raw = payload.dict()
        raw["timestamp"] = datetime.now().isoformat()

        # Feature engineering on the raw row
        engineered = engineer_single_row(raw)

        # Run stacked prediction
        result = predict_single(engineered)
        _store(result)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream")
async def stream():
    """
    Generate one simulated real-time metric row and predict it.
    Called by frontend every 5 seconds.
    """
    try:
        import random
        resource    = random.choice(RESOURCES)
        raw_row     = generate_realtime_row(resource)
        engineered  = engineer_single_row(raw_row)
        result      = predict_single(engineered)
        _store(result)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def history(resource_id: str = None, limit: int = 50):
    """
    Return recent prediction history.
    Optional filter by resource_id.
    """
    if resource_id:
        data = list(history_store.get(resource_id, []))[:limit]
    else:
        # Merge all resources, sort by timestamp
        all_data = []
        for records in history_store.values():
            all_data.extend(list(records))
        all_data.sort(key=lambda x: x["timestamp"], reverse=True)
        data = all_data[:limit]

    return {"count": len(data), "records": data}


@router.get("/resources")
async def get_resources():
    """Return list of all known resource IDs and types."""
    from src.data_generator import RESOURCES
    return {"resources": RESOURCES}


@router.get("/stats")
async def get_stats():
    """Aggregate stats across all resources for dashboard summary cards."""
    all_records = []
    for records in history_store.values():
        all_records.extend(list(records))

    if not all_records:
        return {
            "total_predictions" : 0,
            "total_anomalies"   : 0,
            "anomaly_rate"      : 0.0,
            "avg_cost"          : 0.0,
            "critical_count"    : 0,
            "high_count"        : 0,
        }

    total      = len(all_records)
    anomalies  = sum(1 for r in all_records if r["is_anomaly"])
    costs      = [r["predicted_cost"] for r in all_records]
    critical   = sum(1 for r in all_records if r["severity"] == "critical")
    high       = sum(1 for r in all_records if r["severity"] == "high")

    return {
        "total_predictions" : total,
        "total_anomalies"   : anomalies,
        "anomaly_rate"      : round(anomalies / total * 100, 2),
        "avg_cost"          : round(sum(costs) / len(costs), 4),
        "critical_count"    : critical,
        "high_count"        : high,
    }