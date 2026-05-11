# ── Base image ─────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────
WORKDIR /app

# ── Copy requirements first (layer caching) ───────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────────────────────
COPY . .

# ── Create necessary directories ──────────────────────────────
RUN mkdir -p data/raw data/processed data/simulated models mlruns

# ── Make entrypoint executable ─────────────────────────────────
RUN chmod +x docker-entrypoint.sh

# ── Expose port ────────────────────────────────────────────────
EXPOSE 8000

# ── Entrypoint ─────────────────────────────────────────────────
ENTRYPOINT ["./docker-entrypoint.sh"]