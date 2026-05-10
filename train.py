"""
train.py — Runs the complete ML pipeline without Jupyter.
Called by Render during build: python train.py
"""
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (DATA_RAW, DATA_PROCESSED, MODELS_DIR,
                    FEATURE_COLS_PATH, SCALER_PATH, RANDOM_STATE,
                    ZSCORE_THRESHOLD, ROLLING_WINDOWS, LAG_WINDOWS)

# ── Imports ───────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier,
                               RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              roc_auc_score, mean_squared_error,
                              mean_absolute_error, r2_score)
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import shap


def step1_generate_data():
    print("\n[1/5] Generating synthetic dataset...")
    from src.data_generator import generate_historical_data
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df = generate_historical_data(days=90, anomaly_rate=0.07)
    out = DATA_RAW / "cloud_metrics_historical.csv"
    df.to_csv(out, index=False)
    print(f"     Saved {len(df):,} rows → {out}")
    return df


def step2_feature_engineering(df):
    print("\n[2/5] Feature engineering...")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = df.sort_values(["timestamp", "resource_id"]).reset_index(drop=True)

    metric_cols = ['cpu_utilization', 'memory_utilization',
                   'network_in_mbps', 'network_out_mbps',
                   'disk_io_mbps', 'cost_per_hour']

    # Rolling stats
    for window in ROLLING_WINDOWS:
        for col in metric_cols:
            grp = df.groupby('resource_id')[col]
            df[f'{col}_roll_mean_{window}h'] = grp.transform(
                lambda x: x.rolling(window, min_periods=1).mean())
            df[f'{col}_roll_std_{window}h'] = grp.transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        df[f'cpu_roll_max_{window}h'] = df.groupby('resource_id')['cpu_utilization'].transform(
            lambda x: x.rolling(window, min_periods=1).max())
        df[f'net_in_roll_max_{window}h'] = df.groupby('resource_id')['network_in_mbps'].transform(
            lambda x: x.rolling(window, min_periods=1).max())

    # Lag features
    lag_cols = ['cpu_utilization', 'memory_utilization',
                'network_in_mbps', 'cost_per_hour', 'error_rate_pct']
    for lag in LAG_WINDOWS:
        for col in lag_cols:
            df[f'{col}_lag_{lag}h'] = df.groupby('resource_id')[col].transform(
                lambda x: x.shift(lag))
    lag_feature_cols = [c for c in df.columns if '_lag_' in c]
    df[lag_feature_cols] = df[lag_feature_cols].fillna(df[lag_feature_cols].mean())

    # Derived ratios
    df['cpu_mem_ratio']      = df['cpu_utilization'] / (df['memory_utilization'] + 1e-6)
    df['net_in_out_ratio']   = df['network_in_mbps'] / (df['network_out_mbps'] + 1e-6)
    df['cost_per_cpu']       = df['cost_per_hour'] / (df['cpu_utilization'] + 1e-6)
    df['cost_per_request']   = df['cost_per_hour'] / (df['request_count'] + 1)
    df['total_network_mbps'] = df['network_in_mbps'] + df['network_out_mbps']
    df['total_errors']       = (df['error_rate_pct'] / 100) * df['request_count']
    df['resource_pressure']  = (df['cpu_utilization'] + df['memory_utilization']) / 2

    # Z-scores
    for col in ['cpu_utilization', 'memory_utilization',
                'network_in_mbps', 'cost_per_hour']:
        df[f'{col}_zscore'] = ((df[col] - df[f'{col}_roll_mean_24h']) /
                               (df[f'{col}_roll_std_24h'] + 1e-6))

    # Temporal
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 18) &
                                (df['is_weekend'] == 0)).astype(int)
    df['is_night']      = ((df['hour'] >= 23) | (df['hour'] <= 6)).astype(int)
    df['is_peak_hours'] = ((df['hour'] >= 11) & (df['hour'] <= 14) &
                            (df['is_weekend'] == 0)).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Encoding
    df = pd.get_dummies(df, columns=['resource_type'], prefix='rtype')
    resource_ids = sorted(df['resource_id'].unique())
    rid_map = {r: i for i, r in enumerate(resource_ids)}
    df['resource_id_enc'] = df['resource_id'].map(rid_map)

    # Save
    drop_cols = ['anomaly_type', 'timestamp', 'resource_id']
    df_model = df.drop(columns=drop_cols)
    df_model = df_model.fillna(df_model.mean(numeric_only=True))

    out = DATA_PROCESSED / "cloud_metrics_featured.csv"
    df_model.to_csv(out, index=False)
    print(f"     Features: {df_model.shape[1]} columns, {len(df_model):,} rows")
    return df_model


def step3_train_classifier(df):
    print("\n[3/5] Training classifiers...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment("cloud_anomaly_classification")

    TARGET = 'is_anomaly'
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    feature_columns = list(X.columns)
    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train = X.iloc[:train_end]
    X_val   = X.iloc[train_end:val_end]
    X_test  = X.iloc[val_end:]
    y_train = y.iloc[:train_end]
    y_val   = y.iloc[train_end:val_end]
    y_test  = y.iloc[val_end:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_COLS_PATH, 'w') as f:
        json.dump(feature_columns, f)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    classifiers = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=5, use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
            random_state=RANDOM_STATE, verbosity=0),
    }

    results = {}
    for name, clf in classifiers.items():
        with mlflow.start_run(run_name=name):
            clf.fit(X_train_smote, y_train_smote)
            y_pred  = clf.predict(X_val_scaled)
            y_proba = clf.predict_proba(X_val_scaled)[:, 1]
            f1      = f1_score(y_val, y_pred, zero_division=0)
            roc     = roc_auc_score(y_val, y_proba)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc)
            results[name] = {"model": clf, "f1": f1, "roc": roc, "proba": y_proba}
            print(f"     {name:<25} F1={f1:.3f}  ROC={roc:.3f}")

    # Threshold tuning on XGBoost
    xgb_res = results["XGBoost"]
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y_val, (xgb_res['proba'] >= t).astype(int), zero_division=0)
           for t in thresholds]
    best_thresh = float(thresholds[np.argmax(f1s)])

    # Evaluate on test
    xgb_clf       = xgb_res['model']
    y_test_pred   = (xgb_clf.predict_proba(X_test_scaled)[:, 1] >= best_thresh).astype(int)
    test_f1       = f1_score(y_test, y_test_pred)

    print(f"\n     Best classifier: XGBoost")
    print(f"     Threshold: {best_thresh:.2f}")
    print(f"     Test F1  : {test_f1:.4f}")

    joblib.dump(xgb_clf, MODELS_DIR / "classifier_best.pkl")
    with open(MODELS_DIR / "classifier_threshold.json", "w") as f:
        json.dump({"model": "XGBoost (threshold tuned)",
                   "threshold": best_thresh}, f)

    return xgb_clf, scaler, feature_columns, best_thresh, X_test_scaled, y_test


def step4_train_regressor(df, classifier, scaler, feature_columns,
                           clf_threshold, X_test_scaled_clf, y_clf_test):
    print("\n[4/5] Training regressors...")
    mlflow.set_experiment("cloud_anomaly_regression")

    TARGET_CLF = 'is_anomaly'
    TARGET_REG = 'cost_per_hour'

    X = df.drop(columns=[TARGET_CLF, TARGET_REG])
    y_reg = df[TARGET_REG]

    cost_idx            = feature_columns.index(TARGET_REG)
    reg_feature_columns = [c for c in feature_columns if c != TARGET_REG]
    X = X[reg_feature_columns]

    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train = X.iloc[:train_end]
    X_val   = X.iloc[train_end:val_end]
    X_test  = X.iloc[val_end:]
    y_train = y_reg.iloc[:train_end]
    y_val   = y_reg.iloc[train_end:val_end]
    y_test  = y_reg.iloc[val_end:]

    # Scale
    def scale_reg(split):
        full = split.reindex(columns=feature_columns, fill_value=0)
        scaled_full = scaler.transform(full)
        return np.delete(scaled_full, cost_idx, axis=1), scaled_full

    X_train_reg, X_train_full = scale_reg(X_train.reindex(
        columns=reg_feature_columns))
    X_val_reg,   X_val_full   = scale_reg(X_val.reindex(
        columns=reg_feature_columns))
    X_test_reg,  X_test_full  = scale_reg(X_test.reindex(
        columns=reg_feature_columns))

    # Fix scale_reg — simpler approach
    def get_scaled(split_df):
        full_df = pd.DataFrame(index=split_df.index)
        for col in feature_columns:
            if col in split_df.columns:
                full_df[col] = split_df[col]
            else:
                full_df[col] = 0.0
        scaled_full = scaler.transform(full_df[feature_columns])
        scaled_reg  = np.delete(scaled_full, cost_idx, axis=1)
        return scaled_reg, scaled_full

    X_train_reg, X_train_full = get_scaled(X_train.assign(cost_per_hour=0))
    X_val_reg,   X_val_full   = get_scaled(X_val.assign(cost_per_hour=0))
    X_test_reg,  X_test_full  = get_scaled(X_test.assign(cost_per_hour=0))

    train_proba = classifier.predict_proba(
        scaler.transform(df[feature_columns].iloc[:train_end]))[:, 1]
    val_proba   = classifier.predict_proba(
        scaler.transform(df[feature_columns].iloc[train_end:val_end]))[:, 1]
    test_proba  = classifier.predict_proba(
        scaler.transform(df[feature_columns].iloc[val_end:]))[:, 1]

    X_train_stacked = np.column_stack([X_train_reg, train_proba])
    X_val_stacked   = np.column_stack([X_val_reg,   val_proba])
    X_test_stacked  = np.column_stack([X_test_reg,  test_proba])

    regressors = {
        "Ridge"            : Ridge(alpha=1.0),
        "Random Forest"    : RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost"          : XGBRegressor(
            n_estimators=100, max_depth=5,
            random_state=RANDOM_STATE, verbosity=0),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=RANDOM_STATE),
    }

    results = {}
    for name, reg in regressors.items():
        with mlflow.start_run(run_name=name):
            reg.fit(X_train_stacked, y_train)
            y_pred = reg.predict(X_val_stacked)
            r2     = r2_score(y_val, y_pred)
            rmse   = np.sqrt(mean_squared_error(y_val, y_pred))
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("rmse", rmse)
            results[name] = {"model": reg, "r2": r2, "rmse": rmse}
            print(f"     {name:<25} R²={r2:.4f}  RMSE={rmse:.4f}")

    best_name = max(results, key=lambda n: results[n]['r2'])
    best_reg  = results[best_name]['model']

    y_test_pred = best_reg.predict(X_test_stacked)
    test_r2     = r2_score(y_test, y_test_pred)
    test_mae    = mean_absolute_error(y_test, y_test_pred)
    test_rmse   = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"\n     Best regressor: {best_name}")
    print(f"     Test R²  : {test_r2:.4f}")
    print(f"     Test MAE : ${test_mae:.4f}/hr")

    joblib.dump(best_reg, MODELS_DIR / "regressor_best.pkl")
    with open(MODELS_DIR / "regressor_meta.json", "w") as f:
        json.dump({"model": best_name, "test_r2": round(test_r2, 4),
                   "test_rmse": round(test_rmse, 4),
                   "test_mae": round(test_mae, 4),
                   "stacked": True,
                   "extra_feature": "anomaly_probability"}, f)

    return best_reg


def step5_verify():
    print("\n[5/5] Verifying pipeline...")
    from src.predict import predict_single
    from src.feature_engineer import engineer_single_row
    from src.data_generator import generate_realtime_row, RESOURCES
    import random

    resource   = random.choice(RESOURCES)
    raw_row    = generate_realtime_row(resource)
    engineered = engineer_single_row(raw_row)
    result     = predict_single(engineered)

    print(f"     Resource      : {result['resource_id']}")
    print(f"     Is anomaly    : {result['is_anomaly']}")
    print(f"     Anomaly prob  : {result['anomaly_prob']}")
    print(f"     Predicted cost: ${result['predicted_cost']}/hr")
    print(f"     Severity      : {result['severity']}")
    print("     Pipeline verified ✓")


if __name__ == "__main__":
    print("=" * 55)
    print("CloudGuard AI — Training Pipeline")
    print("=" * 55)

    df              = step1_generate_data()
    df_model        = step2_feature_engineering(df)
    clf, scaler, feature_columns, threshold, \
    X_test_clf, y_test_clf = step3_train_classifier(df_model)
    step4_train_regressor(df_model, clf, scaler,
                          feature_columns, threshold,
                          X_test_clf, y_test_clf)
    step5_verify()

    print("\n" + "=" * 55)
    print("Training complete! All models saved to models/")
    print("=" * 55)
