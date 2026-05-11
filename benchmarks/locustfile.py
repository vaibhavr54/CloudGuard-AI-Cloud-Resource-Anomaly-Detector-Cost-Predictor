"""
benchmarks/locustfile.py
Load test for CloudGuard AI API endpoints.
Run: locust -f benchmarks/locustfile.py --host http://localhost:8000
"""
from locust import HttpUser, task, between
import random


class CloudGuardUser(HttpUser):
    """Simulates a dashboard user polling the API every 5 seconds."""
    wait_time = between(4, 6)  # Realistic 4-6s polling interval

    def on_start(self):
        """Fetch resources list once at session start."""
        self.client.get("/resources")

    @task(10)
    def stream_prediction(self):
        """Poll /stream — main dashboard endpoint."""
        with self.client.get("/stream", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                # Validate response structure
                if not all(k in data for k in ["is_anomaly", "anomaly_prob", "predicted_cost"]):
                    response.failure("Invalid response structure")
                # Validate probability range
                if not (0.0 <= data.get("anomaly_prob", -1) <= 1.0):
                    response.failure("anomaly_prob out of range")

    @task(3)
    def get_stats(self):
        """Poll /stats — dashboard summary cards."""
        self.client.get("/stats")

    @task(2)
    def get_history(self):
        """Poll /history — recent predictions table."""
        self.client.get("/history?limit=20")

    @task(1)
    def health_check(self):
        """Poll /health — monitoring endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code not in [200, 503]:
                response.failure(f"Unexpected status: {response.status_code}")


class BurstUser(HttpUser):
    """Simulates traffic spike — many rapid requests."""
    wait_time = between(0.1, 0.5)  # 100-500ms between requests

    @task
    def burst_stream(self):
        """High-frequency /stream calls."""
        self.client.get("/stream")