# CloudGuard AI — Cloud Resource Anomaly Detector & Cost Predictor

A real-time ML system that detects cloud infrastructure anomalies and forecasts cost impact — with SHAP explainability on every prediction.

<img width="1870" height="2974" alt="image" src="https://github.com/user-attachments/assets/d9225085-49f7-4647-aa43-1327c3744516" />

---

## What it does

Every 5 seconds, the system:
1. Ingests a cloud resource metric reading (CPU, memory, network, disk, cost)
2. **Stage 1** — classifies whether the reading is anomalous (XGBoost, F1=0.93)
3. **Stage 2** — forecasts the hourly cost impact (XGBoost, R²=0.99)
4. Explains every prediction with SHAP feature attribution
5. Streams results to a live HTML dashboard

---

## Architecture
    Simulated CloudWatch Stream (every 5s)
    ↓
    Feature Engineering (95 features)
    Rolling stats · Lag features · Z-scores · Temporal · Ratios
    ↓
    Stage 1: XGBoost Classifier → anomaly probability
    ↓ (stacked — prob fed as feature)
    Stage 2: XGBoost Regressor  → predicted cost $/hr
    ↓
    SHAP TreeExplainer → top 5 reasons per prediction
    ↓
    FastAPI backend → HTML/CSS/JS live dashboard

---

## Results

| Stage | Model | Metric | Score |
|---|---|---|---|
| Classification | XGBoost | F1 Score | 0.9348 |
| Classification | XGBoost | ROC-AUC | 0.9936 |
| Classification | XGBoost | Precision | 0.96 |
| Classification | XGBoost | Recall | 0.91 |
| Regression | XGBoost | R² | 0.9923 |
| Regression | XGBoost | MAE | $0.11/hr |
| Regression | XGBoost | MAPE | 6.98% |

10 classifiers and 12 regressors benchmarked. All experiments tracked in MLflow.

---

## Models compared

**Classifiers:** 
. Logistic Regression 
· Decision Tree 
· Random Forest 
· Extra Trees 
· Gradient Boosting 
· AdaBoost 
· XGBoost 
· KNN 
· SVC 
· Gaussian NB

**Regressors:** 
. Ridge 
· Lasso 
· ElasticNet 
· Decision Tree 
· Random Forest 
· Extra Trees 
· Gradient Boosting 
· XGBoost 
· AdaBoost 
· SVR 
· KNN 
· MLP

---

## Tech stack

| Layer | Technology |
|---|---|
| Data simulation | Python · NumPy |
| ML pipeline | scikit-learn · XGBoost · imbalanced-learn |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| Backend API | FastAPI · Uvicorn |
| Frontend | HTML · CSS · Vanilla JS · Chart.js |

---

## Project structure

## Project structure

```
CloudGuard AI — Cloud Resource Anomaly Detector & Cost Predictor/
├── data/                    # raw, processed, simulated data
├── notebooks/               # 7 Jupyter notebooks (EDA → deployment)
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_anomaly_labelling.ipynb
│   ├── 04_model_classification.ipynb
│   ├── 05_model_regression.ipynb
│   ├── 06_explainability.ipynb
│   └── 07_pipeline_integration.ipynb
├── src/                     # core ML modules
│   ├── data_generator.py
│   ├── feature_engineer.py
│   ├── predict.py
│   └── anomaly_labeller.py
├── api/                     # FastAPI backend
│   ├── main.py
│   ├── routes.py
│   └── schemas.py
├── frontend/                # live HTML dashboard
│   ├── index.html
│   ├── css/
│   └── js/
├── models/                  # saved model artifacts (generated)
├── config.py
├── requirements.txt
└── setup.py
```

## Quick start

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

# 4. Generate data and train models (~10 minutes)
python setup.py

# 5. Start API
python -m uvicorn api.main:app --reload --port 8000

# 6. Open dashboard
# Visit http://127.0.0.1:8000
```

---

## Key technical decisions

- **Why synthetic data?** Real CloudWatch data is proprietary. The simulation models real AWS metric patterns — business hour load curves, weekend reduction, 5 anomaly types at 7% base rate — giving statistically realistic data for model development.

- **Why stacked architecture?** Stage 1's anomaly probability is fed as an additional feature to Stage 2. This improved regressor R² by 0.001 and RMSE by $0.012/hr over independent models.

- **Why time-based train/test split?** The data has temporal structure (rolling means, lags). Random splitting would leak future information into training. Time-based splitting ensures the model is evaluated on genuinely unseen future data.

- **Why F1 over accuracy?** Anomalies are 7% of data. A naive "always normal" classifier achieves 93% accuracy but zero utility. F1 balances precision and recall for imbalanced classification.

- **Why SHAP?** Every alert includes the top 5 features driving the prediction. Engineers can act on specific signals ("cost Z-score spiked 4σ above 24h mean") rather than a binary flag.

---

## Live demo

Deployed on Render: 

---

## Author
GitHub: github.com/vaibhavr54
