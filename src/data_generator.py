import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_RAW, RANDOM_STATE

np.random.seed(RANDOM_STATE)

# ── Resource definitions ───────────────────────────────
RESOURCES = [
    {"id": "ec2-web-01",    "type": "EC2",        "base_cpu": 45,  "base_mem": 60,  "base_cost": 2.50},
    {"id": "ec2-web-02",    "type": "EC2",        "base_cpu": 38,  "base_mem": 55,  "base_cost": 2.50},
    {"id": "ec2-ml-01",     "type": "EC2-GPU",    "base_cpu": 70,  "base_mem": 80,  "base_cost": 8.00},
    {"id": "rds-primary",   "type": "RDS",        "base_cpu": 30,  "base_mem": 70,  "base_cost": 3.20},
    {"id": "rds-replica",   "type": "RDS",        "base_cpu": 20,  "base_mem": 65,  "base_cost": 2.80},
    {"id": "lambda-auth",   "type": "Lambda",     "base_cpu": 15,  "base_mem": 25,  "base_cost": 0.40},
    {"id": "lambda-notify", "type": "Lambda",     "base_cpu": 10,  "base_mem": 20,  "base_cost": 0.30},
    {"id": "s3-data-lake",  "type": "S3",         "base_cpu": 5,   "base_mem": 10,  "base_cost": 1.10},
    {"id": "elasticache-01","type": "ElastiCache", "base_cpu": 25, "base_mem": 85,  "base_cost": 1.80},
    {"id": "eks-cluster-01","type": "EKS",        "base_cpu": 55,  "base_mem": 72,  "base_cost": 5.50},
]

ANOMALY_TYPES = [
    "cpu_spike",
    "memory_leak",
    "network_burst",
    "cost_explosion",
    "idle_waste",
]


def _time_of_day_factor(hour: int) -> float:
    """Business hours = higher load, nights = lower."""
    if 9 <= hour <= 18:
        return 1.0 + 0.3 * np.sin(np.pi * (hour - 9) / 9)
    elif 19 <= hour <= 23:
        return 0.75
    else:
        return 0.45


def _inject_anomaly(row: dict, anomaly_type: str) -> dict:
    """Inject a specific anomaly pattern into a row."""
    if anomaly_type == "cpu_spike":
        row["cpu_utilization"]    = min(98, row["cpu_utilization"] * np.random.uniform(2.0, 3.5))
        row["cost_per_hour"]      = row["cost_per_hour"] * np.random.uniform(1.8, 2.5)

    elif anomaly_type == "memory_leak":
        row["memory_utilization"] = min(99, row["memory_utilization"] * np.random.uniform(1.6, 2.2))
        row["cost_per_hour"]      = row["cost_per_hour"] * np.random.uniform(1.3, 1.8)

    elif anomaly_type == "network_burst":
        row["network_in_mbps"]    = row["network_in_mbps"]  * np.random.uniform(4.0, 8.0)
        row["network_out_mbps"]   = row["network_out_mbps"] * np.random.uniform(4.0, 8.0)
        row["cost_per_hour"]      = row["cost_per_hour"] * np.random.uniform(2.0, 3.0)

    elif anomaly_type == "cost_explosion":
        row["cost_per_hour"]      = row["cost_per_hour"] * np.random.uniform(4.0, 7.0)
        row["cpu_utilization"]    = min(95, row["cpu_utilization"] * np.random.uniform(1.5, 2.0))

    elif anomaly_type == "idle_waste":
        row["cpu_utilization"]    = np.random.uniform(0.5, 3.0)
        row["memory_utilization"] = np.random.uniform(5.0, 12.0)
        row["cost_per_hour"]      = row["cost_per_hour"] * np.random.uniform(0.9, 1.1)

    row["anomaly_type"] = anomaly_type
    return row


def generate_historical_data(
    days: int = 90,
    anomaly_rate: float = 0.07
) -> pd.DataFrame:
    """
    Generate 90 days of hourly cloud resource metrics.
    Each row = one resource reading for one hour.
    """
    records = []
    start_dt = datetime.now() - timedelta(days=days)

    for day_offset in range(days):
        for hour in range(24):
            ts = start_dt + timedelta(days=day_offset, hours=hour)
            tod_factor = _time_of_day_factor(hour)
            is_weekend = ts.weekday() >= 5

            for res in RESOURCES:
                noise = lambda scale: np.random.normal(0, scale)

                cpu  = np.clip(res["base_cpu"]  * tod_factor + noise(6), 1, 100)
                mem  = np.clip(res["base_mem"]  * tod_factor + noise(5), 1, 100)
                net_in  = np.clip(50 * tod_factor + noise(15), 0.5, 500)
                net_out = np.clip(30 * tod_factor + noise(10), 0.5, 300)
                disk_io = np.clip(40 * tod_factor + noise(12), 0.5, 200)
                req_count = int(np.clip(500 * tod_factor + noise(100), 0, 5000))
                err_rate  = np.clip(np.random.exponential(0.5), 0, 15)
                cost = res["base_cost"] * tod_factor * (1 + noise(0.05))

                # Weekend discount
                if is_weekend:
                    cpu  *= 0.65
                    cost *= 0.70

                row = {
                    "timestamp":           ts,
                    "resource_id":         res["id"],
                    "resource_type":       res["type"],
                    "hour":                hour,
                    "day_of_week":         ts.weekday(),
                    "is_weekend":          int(is_weekend),
                    "is_month_end":        int(ts.day >= 28),
                    "cpu_utilization":     round(cpu, 2),
                    "memory_utilization":  round(mem, 2),
                    "network_in_mbps":     round(net_in, 2),
                    "network_out_mbps":    round(net_out, 2),
                    "disk_io_mbps":        round(disk_io, 2),
                    "request_count":       req_count,
                    "error_rate_pct":      round(err_rate, 3),
                    "cost_per_hour":       round(cost, 4),
                    "is_anomaly":          0,
                    "anomaly_type":        "none",
                }

                # Inject anomaly
                if np.random.random() < anomaly_rate:
                    atype = np.random.choice(ANOMALY_TYPES)
                    row = _inject_anomaly(row, atype)
                    row["is_anomaly"] = 1

                records.append(row)

    df = pd.DataFrame(records)
    df = df.sort_values(["resource_id", "timestamp"]).reset_index(drop=True)
    return df


def generate_realtime_row(resource=None) -> dict:
    """
    Generate a single real-time metric row for API streaming.
    Called every 5 seconds by the FastAPI /stream endpoint.
    """
    if resource is None:
        resource = np.random.choice(RESOURCES)

    ts  = datetime.now()
    hour = ts.hour
    tod_factor = _time_of_day_factor(hour)
    is_weekend = ts.weekday() >= 5

    noise = lambda scale: np.random.normal(0, scale)

    cpu  = np.clip(resource["base_cpu"]  * tod_factor + noise(6), 1, 100)
    mem  = np.clip(resource["base_mem"]  * tod_factor + noise(5), 1, 100)
    net_in  = np.clip(50 * tod_factor + noise(15), 0.5, 500)
    net_out = np.clip(30 * tod_factor + noise(10), 0.5, 300)
    disk_io = np.clip(40 * tod_factor + noise(12), 0.5, 200)
    req_count = int(np.clip(500 * tod_factor + noise(100), 0, 5000))
    err_rate  = np.clip(np.random.exponential(0.5), 0, 15)
    cost = resource["base_cost"] * tod_factor * (1 + noise(0.05))

    if is_weekend:
        cpu  *= 0.65
        cost *= 0.70

    # ~8% chance of anomaly in live stream
    is_anomaly = 0
    anomaly_type = "none"
    row = {
        "timestamp":          ts.isoformat(),
        "resource_id":        resource["id"],
        "resource_type":      resource["type"],
        "hour":               hour,
        "day_of_week":        ts.weekday(),
        "is_weekend":         int(is_weekend),
        "is_month_end":       int(ts.day >= 28),
        "cpu_utilization":    round(cpu, 2),
        "memory_utilization": round(mem, 2),
        "network_in_mbps":    round(net_in, 2),
        "network_out_mbps":   round(net_out, 2),
        "disk_io_mbps":       round(disk_io, 2),
        "request_count":      req_count,
        "error_rate_pct":     round(err_rate, 3),
        "cost_per_hour":      round(cost, 4),
        "is_anomaly":         0,
        "anomaly_type":       "none",
    }

    if np.random.random() < 0.08:
        atype = np.random.choice(ANOMALY_TYPES)
        row = _inject_anomaly(row, atype)
        row["is_anomaly"] = 1

    return row


if __name__ == "__main__":
    print("Generating 90 days of historical data...")
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    df = generate_historical_data(days=90, anomaly_rate=0.07)

    out_path = DATA_RAW / "cloud_metrics_historical.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df):,} rows to {out_path}")
    print(f"\nShape: {df.shape}")
    print(f"Anomaly rate: {df['is_anomaly'].mean():.2%}")
    print(f"Resources: {df['resource_id'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"\nSample:\n{df.head(3).to_string()}")