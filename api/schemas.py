from pydantic import BaseModel
from typing import List, Optional

class SHAPReason(BaseModel):
    feature : str
    value   : float
    shap    : float
    impact  : str

class PredictionResponse(BaseModel):
    timestamp      : str
    resource_id    : str
    resource_type  : str
    is_anomaly     : int
    anomaly_prob   : float
    severity       : str
    predicted_cost : float
    clf_reasons    : List[SHAPReason]
    reg_reasons    : List[SHAPReason]

class MetricInput(BaseModel):
    resource_id              : str
    resource_type            : str
    hour                     : int
    day_of_week              : int
    is_weekend               : int
    is_month_end             : int
    cpu_utilization          : float
    memory_utilization       : float
    network_in_mbps          : float
    network_out_mbps         : float
    disk_io_mbps             : float
    request_count            : int
    error_rate_pct           : float
    cost_per_hour            : float