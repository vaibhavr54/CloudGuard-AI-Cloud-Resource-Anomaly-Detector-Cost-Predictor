#!/bin/bash
set -e

echo "=========================================="
echo "CloudGuard AI — Starting up"
echo "=========================================="

# Check if models exist — if not, train first
if [ ! -f "models/classifier_best.pkl" ]; then
    echo ""
    echo "No trained models found. Running training pipeline..."
    echo "This takes approximately 5-8 minutes on first run."
    echo ""
    python train.py
    echo ""
    echo "Training complete. Starting API server..."
else
    echo ""
    echo "Trained models found. Skipping training."
    echo "Starting API server..."
fi

echo ""
echo "Dashboard will be available at http://localhost:8000"
echo "=========================================="

# Start FastAPI
exec python -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port ${PORT:-7860} \
    --workers 1