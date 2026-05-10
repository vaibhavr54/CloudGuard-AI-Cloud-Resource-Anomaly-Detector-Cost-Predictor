import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict, deque

# In-memory rolling history per resource
# Stores last 24 readings per resource for rolling/lag computation
_resource_history = defaultdict(lambda: deque(maxlen=24))

METRIC_COLS = [
    'cpu_utilization', 'memory_utilization',
    'network_in_mbps', 'network_out_mbps',
    'disk_io_mbps', 'cost_per_hour'
]

ROLLING_WINDOWS = [6, 12, 24]
LAG_WINDOWS     = [1, 6, 24]

RESOURCE_TYPE_COLS = [
    'rtype_EC2', 'rtype_EC2-GPU', 'rtype_EKS',
    'rtype_ElastiCache', 'rtype_Lambda',
    'rtype_RDS', 'rtype_S3'
]

RESOURCE_ID_MAP = {
    'ec2-ml-01'     : 0,
    'ec2-web-01'    : 1,
    'ec2-web-02'    : 2,
    'eks-cluster-01': 3,
    'elasticache-01': 4,
    'lambda-auth'   : 5,
    'lambda-notify' : 6,
    'rds-primary'   : 7,
    'rds-replica'   : 8,
    's3-data-lake'  : 9,
}


def engineer_single_row(raw: dict) -> dict:
    """
    Apply full feature engineering to one incoming metric row.
    Uses per-resource rolling history stored in memory.
    Mirrors notebook 02 logic exactly.
    """
    rid      = raw.get("resource_id", "unknown")
    history  = _resource_history[rid]

    # ── Temporal features ─────────────────────────────────────────────
    ts    = datetime.fromisoformat(str(raw.get("timestamp",
                                               datetime.now().isoformat())))
    hour  = ts.hour
    dow   = ts.weekday()

    feat = {
        "hour"             : hour,
        "day_of_week"      : dow,
        "is_weekend"       : int(dow >= 5),
        "is_month_end"     : int(ts.day >= 28),
        "is_business_hours": int(9 <= hour <= 18 and dow < 5),
        "is_night"         : int(hour >= 23 or hour <= 6),
        "is_peak_hours"    : int(11 <= hour <= 14 and dow < 5),
        "hour_sin"         : np.sin(2 * np.pi * hour / 24),
        "hour_cos"         : np.cos(2 * np.pi * hour / 24),
        "dow_sin"          : np.sin(2 * np.pi * dow / 7),
        "dow_cos"          : np.cos(2 * np.pi * dow / 7),
    }

    # ── Raw metrics ───────────────────────────────────────────────────
    for col in METRIC_COLS:
        feat[col] = float(raw.get(col, 0.0))

    feat["request_count"]  = int(raw.get("request_count", 0))
    feat["error_rate_pct"] = float(raw.get("error_rate_pct", 0.0))

    # ── Rolling stats from history ────────────────────────────────────
    hist_df = pd.DataFrame(list(history)) if history else pd.DataFrame()

    for col in METRIC_COLS:
        for w in ROLLING_WINDOWS:
            if len(hist_df) >= 1:
                vals = hist_df[col].iloc[-w:] if col in hist_df else []
                roll_mean = float(np.mean(vals)) if len(vals) > 0 \
                            else feat[col]
                roll_std  = float(np.std(vals))  if len(vals) > 1 \
                            else 0.0
            else:
                roll_mean = feat[col]
                roll_std  = 0.0

            feat[f'{col}_roll_mean_{w}h'] = roll_mean
            feat[f'{col}_roll_std_{w}h']  = roll_std

        # Rolling max for cpu and network
    for w in ROLLING_WINDOWS:
        if len(hist_df) >= 1 and 'cpu_utilization' in hist_df:
            vals = hist_df['cpu_utilization'].iloc[-w:]
            feat[f'cpu_roll_max_{w}h'] = float(vals.max()) \
                                          if len(vals) > 0 \
                                          else feat['cpu_utilization']
            vals_net = hist_df['network_in_mbps'].iloc[-w:] \
                       if 'network_in_mbps' in hist_df else []
            feat[f'net_in_roll_max_{w}h'] = float(vals_net.max()) \
                                             if len(vals_net) > 0 \
                                             else feat['network_in_mbps']
        else:
            feat[f'cpu_roll_max_{w}h']    = feat['cpu_utilization']
            feat[f'net_in_roll_max_{w}h'] = feat['network_in_mbps']

    # ── Lag features ──────────────────────────────────────────────────
    lag_cols = ['cpu_utilization', 'memory_utilization',
                'network_in_mbps', 'cost_per_hour', 'error_rate_pct']

    for col in lag_cols:
        for lag in LAG_WINDOWS:
            if len(hist_df) >= lag and col in hist_df:
                feat[f'{col}_lag_{lag}h'] = float(
                    hist_df[col].iloc[-lag])
            else:
                feat[f'{col}_lag_{lag}h'] = feat[col]

    # ── Derived ratios ────────────────────────────────────────────────
    feat['cpu_mem_ratio']     = feat['cpu_utilization'] / \
                                (feat['memory_utilization'] + 1e-6)
    feat['net_in_out_ratio']  = feat['network_in_mbps'] / \
                                (feat['network_out_mbps'] + 1e-6)
    feat['cost_per_cpu']      = feat['cost_per_hour'] / \
                                (feat['cpu_utilization'] + 1e-6)
    feat['cost_per_request']  = feat['cost_per_hour'] / \
                                (feat['request_count'] + 1)
    feat['total_network_mbps']= feat['network_in_mbps'] + \
                                feat['network_out_mbps']
    feat['total_errors']      = (feat['error_rate_pct'] / 100) * \
                                feat['request_count']
    feat['resource_pressure'] = (feat['cpu_utilization'] + \
                                 feat['memory_utilization']) / 2

    # ── Z-score deviations ────────────────────────────────────────────
    for col in ['cpu_utilization', 'memory_utilization',
                'network_in_mbps', 'cost_per_hour']:
        mean_val = feat[f'{col}_roll_mean_24h']
        std_val  = feat[f'{col}_roll_std_24h']
        feat[f'{col}_zscore'] = (feat[col] - mean_val) / \
                                (std_val + 1e-6)

    # ── Categorical encoding ──────────────────────────────────────────
    rtype = raw.get("resource_type", "EC2")
    for col in RESOURCE_TYPE_COLS:
        feat[col] = 1 if col == f'rtype_{rtype}' else 0

    feat['resource_id_enc'] = RESOURCE_ID_MAP.get(rid, -1)

    # ── Update history ────────────────────────────────────────────────
    history.append({col: feat[col] for col in METRIC_COLS})

    # ── Add metadata passthrough ──────────────────────────────────────
    feat['resource_id']   = rid
    feat['resource_type'] = rtype

    return feat


def reset_history(resource_id: str = None):
    """Clear rolling history. Pass resource_id to clear one, None for all."""
    global _resource_history
    if resource_id:
        _resource_history[resource_id].clear()
    else:
        _resource_history.clear()