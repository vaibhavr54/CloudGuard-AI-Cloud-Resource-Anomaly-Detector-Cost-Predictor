import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_RAW        = BASE_DIR / "data" / "raw"
DATA_PROCESSED  = BASE_DIR / "data" / "processed"
DATA_SIMULATED  = BASE_DIR / "data" / "simulated"
MODELS_DIR      = BASE_DIR / "models"
NOTEBOOKS_DIR   = BASE_DIR / "notebooks"

# ── Model artifacts ────────────────────────────────────
CLASSIFIER_PATH     = MODELS_DIR / "classifier_best.pkl"
REGRESSOR_PATH      = MODELS_DIR / "regressor_best.pkl"
SCALER_PATH         = MODELS_DIR / "scaler.pkl"
FEATURE_COLS_PATH   = MODELS_DIR / "feature_columns.json"

# ── Feature engineering ────────────────────────────────
ROLLING_WINDOWS     = [6, 12, 24]       # in hours (assuming 1 row = 1 hour)
LAG_WINDOWS         = [1, 6, 24]        # lag steps
ZSCORE_THRESHOLD    = 3.0               # std devs to flag anomaly

# ── Real-time simulation ───────────────────────────────
STREAM_INTERVAL_SEC = 5                 # how often frontend polls API
STREAM_WINDOW_SIZE  = 50               # how many points shown on live chart

# ── Training ───────────────────────────────────────────
TEST_SIZE           = 0.15
VAL_SIZE            = 0.15
RANDOM_STATE        = 42

# ── API ────────────────────────────────────────────────
API_HOST            = "0.0.0.0"
API_PORT            = 8000
FRONTEND_ORIGIN     = "http://localhost:5500"  # VS Code Live Server port