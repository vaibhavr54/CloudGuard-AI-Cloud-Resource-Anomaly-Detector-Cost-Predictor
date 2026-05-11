# CloudGuard AI — Cloud Resource Anomaly Detector & Cost Predictor

[![CI — Test Suite](https://github.com/vaibhavr54/CloudGuard-AI-Cloud-Resource-Anomaly-Detector-Cost-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/vaibhavr54/CloudGuard-AI-Cloud-Resource-Anomaly-Detector-Cost-Predictor/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-FFD21E?logo=huggingface)](https://huggingface.co/spaces/vaibhavrakshe161/cloudguard-ai)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://github.com/vaibhavr54/CloudGuard-AI-Cloud-Resource-Anomaly-Detector-Cost-Predictor)
[![Release](https://img.shields.io/badge/Release-v1.0.0-blue)](https://github.com/vaibhavr54/CloudGuard-AI-Cloud-Resource-Anomaly-Detector-Cost-Predictor/releases/tag/v1.0.0)

> **Real-time ML system that detects cloud infrastructure anomalies and forecasts cost impact — with SHAP explainability on every prediction.**

---
<img width="1870" height="2974" alt="image" src="https://github.com/user-attachments/assets/d9225085-49f7-4647-aa43-1327c3744516" />

## What It Does

Every 5 seconds, the system:

1. **Ingests** a cloud resource metric reading (CPU, memory, network, disk, cost)
2. **Stage 1 — Anomaly Detection:** Classifies whether the reading is anomalous using XGBoost (F1 = 0.93)
3. **Stage 2 — Cost Forecasting:** Predicts the hourly cost impact using a stacked XGBoost regressor (R² = 0.99)
4. **Explains** every prediction with SHAP feature attribution — engineers see *why* an alert fired
5. **Streams** results to a live HTML dashboard with real-time metrics, cost forecasts, and anomaly feed

---

## Architecture
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/f6f1a2e3-f689-4b92-bbe6-d58c3068daae" />



---

## Results

| Stage | Model | Metric | Score |
|-------|-------|--------|-------|
| Classification | XGBoost (threshold-tuned) | **F1 Score** | **0.9348** |
| Classification | XGBoost | ROC-AUC | 0.9936 |
| Classification | XGBoost | Precision | 0.96 |
| Classification | XGBoost | Recall | 0.91 |
| Regression | XGBoost | **R²** | **0.9923** |
| Regression | XGBoost | MAE | $0.11/hr |
| Regression | XGBoost | MAPE | 6.98% |

*10 classifiers and 12 regressors benchmarked. All experiments tracked in MLflow.*

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data** | Python · NumPy | Synthetic cloud metric simulation |
| **ML Pipeline** | scikit-learn · XGBoost · imbalanced-learn | Feature engineering, model training, SMOTE |
| **Explainability** | SHAP | Feature attribution on every prediction |
| **Tracking** | MLflow | Experiment logging, model registry |
| **API** | FastAPI · Uvicorn | REST endpoints, async serving |
| **Frontend** | HTML · CSS · Vanilla JS · Chart.js | Real-time dashboard |
| **DevOps** | Docker · docker-compose · GitHub Actions | Containerization, CI/CD |
| **Testing** | pytest · pytest-asyncio · httpx · Locust | Unit tests, integration tests, load testing |

---

## Project Structure

```
CloudGuard-AI-Cloud-Resource-Anomaly-Detector-Cost-Predictor/
├── .github/workflows/       # CI/CD pipeline
│   └── ci.yml
├── api/                     # FastAPI backend
│   ├── main.py              # App factory, lifespan events, health checks
│   ├── routes.py            # /predict, /stream, /stats, /history, /resources
│   └── schemas.py           # Pydantic request/response models
├── benchmarks/              # Load testing
│   └── locustfile.py        # 100-user simulation, ~60 RPS sustained
├── data/                    # Raw, processed, simulated datasets
├── frontend/                # Live HTML dashboard
│   ├── index.html
│   ├── css/                 # animations.css, style.css
│   └── js/                  # api.js, charts.js, dashboard.js, utils.js
├── models/                  # Saved artifacts (generated at build time)
│   ├── classifier_best.pkl
│   ├── regressor_best.pkl
│   ├── scaler.pkl
│   └── feature_columns.json
├── notebooks/               # 7 Jupyter notebooks (EDA → deployment)
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_anomaly_labelling.ipynb
│   ├── 04_model_classification.ipynb
│   ├── 05_model_regression.ipynb
│   ├── 06_explainability.ipynb
│   └── 07_pipeline_integration.ipynb
├── src/                     # Core ML modules
│   ├── data_generator.py    # Synthetic AWS-like metric generation
│   ├── feature_engineer.py  # 96-feature real-time engineering
│   ├── predict.py           # Stacked inference pipeline
│   └── anomaly_labeller.py  # Anomaly detection logic
├── tests/                   # Test suite (17 tests)
│   ├── test_api.py          # API endpoint tests with mocked ML
│   └── test_predict.py      # Feature engineering & data generation tests
├── config.py                # Centralized configuration
├── docker-compose.yml       # Container orchestration
├── docker-entrypoint.sh     # Auto-training on first run
├── Dockerfile               # Production container
├── pytest.ini               # Test configuration
├── render.yaml              # Render cloud deployment config
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── runtime.txt              # Python 3.11.9
├── setup.py                 # Legacy notebook-based setup
└── train.py                 # Production training pipeline (no Jupyter)
```

---

## Quick Start

### Local Development

```bash
# 1. Clone
https://github.com/vaibhavr54/CloudGuard-AI-Cloud-Resource-Anomaly-Detector-Cost-Predictor.git
cd CloudGuard-AI-Cloud-Resource-Anomaly-Detector-Cost-Predictor

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train models (~5-8 minutes)
python train.py

# 5. Start API
python -m uvicorn api.main:app --reload --port 8000

# 6. Open dashboard
# Visit http://127.0.0.1:8000
```

### Docker (One Command)

```bash
docker-compose up --build
# Dashboard at http://localhost:8000
```

### Load Testing

```bash
pip install locust
locust -f benchmarks/locustfile.py --host http://localhost:8000
# Open http://localhost:8089, set 100 users, 20/s ramp
```

---

## Testing & Quality

| Suite | Tests | Status |
|-------|-------|--------|
| API Endpoints | 8 | ✅ Passing |
| Prediction Logic | 9 | ✅ Passing |
| **Total** | **17** | ✅ **All Green** |

CI runs on every push via GitHub Actions. See badge at top of README.

---

## Performance

Load tested with Locust (100 concurrent users, 4-6s polling interval):

| Metric | Value |
|--------|-------|
| Throughput | ~60 RPS sustained |
| Median Latency | ~650ms |
| p95 Latency | ~1,100ms |
| Failure Rate | <3% (mostly warm-up 503s) |

*Note: Latency includes synthetic data generation + feature engineering + XGBoost inference + SHAP explainability. In production with pre-computed features via streaming pipeline, expected p95 <200ms.*

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Synthetic Data** | Real CloudWatch data is proprietary. Simulation models real AWS patterns — business hour load curves, weekend reduction, 5 anomaly types at 7% base rate. |
| **Time-Based Split** | Rolling means and lag features create temporal structure. Random splitting would leak future information into training. |
| **F1 Over Accuracy** | Anomalies are 7% of data. A naive "always normal" classifier achieves 93% accuracy but zero utility. F1 balances precision and recall for imbalanced classification. |
| **Stacked Architecture** | Stage 1's anomaly probability fed as feature to Stage 2. Improved regressor R² by 0.001 and RMSE by $0.012/hr over independent models. |
| **SHAP Explainability** | Every alert includes top 5 features driving the prediction. Engineers act on specific signals ("cost Z-score spiked 4σ above 24h mean") rather than binary flags. |
| **Health Checks (200/503)** | Container orchestration depends on `/health`. Returns 503 while models train, 200 when ready — prevents traffic to incomplete deployments. |

---

## Production Hook-In

In a real AWS deployment, this pipeline would:
- **Ingest** metrics via CloudWatch `GetMetricData` API (1-minute granularity)
- **Compute** real-time cost estimates using AWS published pricing rates
- **Trigger** predictions through Lambda or ECS Fargate tasks
- **Store** per-resource rolling history in Redis for sub-100ms inference
- **Alert** via SNS/PagerDuty when anomaly probability exceeds threshold

The `feature_engineer.py` module is already designed for single-row real-time inference.

---

## Live Deployment

| Platform | Status | URL |
|----------|--------|-----|
| **Hugging Face Spaces** | 🟢 Running | [Live Demo](https://huggingface.co/spaces/vaibhavrakshe161/cloudguard-ai) |
| **Render** | ⚠️ Configured (512MB insufficient) | See `render.yaml` |
| **Local Docker** | 🟢 Working | `docker-compose up --build` |

---

## Release

- **v1.0.0** — Production-ready release with Docker, CI/CD, test suite, and Hugging Face deployment.
- See [Releases](https://github.com/vaibhavr54/CloudGuard-AI-Cloud-Resource-Anomaly-Detector-Cost-Predictor/releases) for changelog.

---

## License

MIT License — feel free to use, modify, and deploy.

---

## Author

**Vaibhav Rakshe** — [github.com/vaibhavr54](https://github.com/vaibhavr54)

Built as a B.Tech final-year project to demonstrate end-to-end ML engineering, DevOps, and production deployment skills.
