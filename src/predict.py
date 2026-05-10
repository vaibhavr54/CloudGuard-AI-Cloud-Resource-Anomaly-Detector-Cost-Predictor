import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
import json
import shap
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

# ── Load artifacts once at module import ──────────────────────────────
MODELS_DIR = ROOT / "models"

classifier = joblib.load(MODELS_DIR / "classifier_best.pkl")
regressor  = joblib.load(MODELS_DIR / "regressor_best.pkl")
scaler     = joblib.load(MODELS_DIR / "scaler.pkl")

with open(MODELS_DIR / "feature_columns.json") as f:
    feature_columns = json.load(f)

with open(MODELS_DIR / "classifier_threshold.json") as f:
    thresh_data = json.load(f)

with open(MODELS_DIR / "regressor_meta.json") as f:
    reg_meta = json.load(f)

clf_threshold       = thresh_data["threshold"]
reg_feature_columns = [c for c in feature_columns if c != "cost_per_hour"]
cost_idx            = feature_columns.index("cost_per_hour")

clf_explainer = shap.TreeExplainer(classifier)
reg_explainer = shap.TreeExplainer(regressor)

print(f"[predict.py] Loaded: {thresh_data['model']} + {reg_meta['model']}")


def predict_single(raw_row: dict) -> dict:
    """
    Full stacked pipeline: feature row → anomaly + cost + SHAP reasons.
    Called by FastAPI /predict and /stream endpoints.
    """

    # ── Step 1: Build feature dataframe ──────────────────────────────
    row_df = pd.DataFrame([raw_row])
    for col in feature_columns:
        if col not in row_df.columns:
            row_df[col] = 0.0
    row_df = row_df[feature_columns]

    # ── Step 2: Scale ─────────────────────────────────────────────────
    row_scaled_full = scaler.transform(row_df)
    row_scaled_reg  = np.delete(row_scaled_full, cost_idx, axis=1)

    # ── Step 3: Stage 1 — classify ────────────────────────────────────
    anomaly_proba = float(classifier.predict_proba(row_scaled_full)[0, 1])
    is_anomaly    = int(anomaly_proba >= clf_threshold)

    # ── Step 4: Stage 2 — regress (stacked) ───────────────────────────
    row_stacked    = np.column_stack([row_scaled_reg, [[anomaly_proba]]])
    predicted_cost = float(regressor.predict(row_stacked)[0])

    # ── Step 5: SHAP explanation ──────────────────────────────────────
    clf_shap_vals = clf_explainer.shap_values(row_scaled_full)
    if isinstance(clf_shap_vals, list):
        clf_sv = clf_shap_vals[1][0]
    elif len(np.array(clf_shap_vals).shape) == 3:
        clf_sv = clf_shap_vals[0, :, 1]
    else:
        clf_sv = clf_shap_vals[0]

    reg_shap_vals = reg_explainer.shap_values(row_stacked)
    if isinstance(reg_shap_vals, list):
        reg_sv = reg_shap_vals[0][0]
    elif len(np.array(reg_shap_vals).shape) == 3:
        reg_sv = reg_shap_vals[0, :, 0]
    else:
        reg_sv = reg_shap_vals[0]

    # ── Step 6: Top-5 SHAP reasons ────────────────────────────────────
    clf_top5_idx   = np.argsort(np.abs(clf_sv))[-5:][::-1]
    reg_feat_names = reg_feature_columns + ["anomaly_probability"]
    reg_top5_idx   = np.argsort(np.abs(reg_sv))[-5:][::-1]

    clf_reasons = [
        {
            "feature": feature_columns[i],
            "value"  : float(row_df.iloc[0][feature_columns[i]]),
            "shap"   : round(float(clf_sv[i]), 4),
            "impact" : "up_anomaly" if clf_sv[i] > 0 else "down_anomaly"
        }
        for i in clf_top5_idx
    ]

    reg_reasons = [
        {
            "feature": reg_feat_names[i] if i < len(reg_feat_names) else f"feat_{i}",
            "value"  : float(row_stacked[0][i]),
            "shap"   : round(float(reg_sv[i]), 4),
            "impact" : "up_cost" if reg_sv[i] > 0 else "down_cost"
        }
        for i in reg_top5_idx
    ]

    # ── Step 7: Severity ──────────────────────────────────────────────
    if not is_anomaly:
        severity = "normal"
    elif anomaly_proba >= 0.90:
        severity = "critical"
    elif anomaly_proba >= 0.75:
        severity = "high"
    else:
        severity = "medium"

    # ── Step 8: Return ────────────────────────────────────────────────
    return {
        "timestamp"     : datetime.now().isoformat(),
        "resource_id"   : raw_row.get("resource_id", "unknown"),
        "resource_type" : raw_row.get("resource_type", "unknown"),
        "is_anomaly"    : is_anomaly,
        "anomaly_prob"  : round(anomaly_proba, 4),
        "severity"      : severity,
        "predicted_cost": round(predicted_cost, 4),
        "clf_reasons"   : clf_reasons,
        "reg_reasons"   : reg_reasons,
        "cpu_utilization"    : float(row_df.iloc[0].get('cpu_utilization', 0)),
        "memory_utilization" : float(row_df.iloc[0].get('memory_utilization', 0)),
        "network_in_mbps"    : float(row_df.iloc[0].get('network_in_mbps', 0)),
        "cost_per_hour_actual": float(row_df.iloc[0].get('cost_per_hour', 0)),
    }