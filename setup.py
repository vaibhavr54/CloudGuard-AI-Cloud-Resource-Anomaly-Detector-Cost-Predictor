"""
setup.py — Run this after cloning to regenerate data and train models.
Usage: python setup.py
"""
import subprocess
import sys
from pathlib import Path

def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: command failed — {cmd}")
        sys.exit(1)

if __name__ == "__main__":
    print("=" * 55)
    print("CloudGuard AI — Setup Script")
    print("=" * 55)

    # Step 1 — Generate historical data
    print("\n[1/3] Generating synthetic dataset...")
    run("python src/data_generator.py")

    # Step 2 — Run feature engineering notebook headlessly
    print("\n[2/3] Running feature engineering + labelling...")
    run("jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb --output notebooks/02_feature_engineering.ipynb")
    run("jupyter nbconvert --to notebook --execute notebooks/03_anomaly_labelling.ipynb --output notebooks/03_anomaly_labelling.ipynb")

    # Step 3 — Train models
    print("\n[3/3] Training models...")
    run("jupyter nbconvert --to notebook --execute notebooks/04_model_classification.ipynb --output notebooks/04_model_classification.ipynb")
    run("jupyter nbconvert --to notebook --execute notebooks/05_model_regression.ipynb --output notebooks/05_model_regression.ipynb")
    run("jupyter nbconvert --to notebook --execute notebooks/06_explainability.ipynb --output notebooks/06_explainability.ipynb")
    run("jupyter nbconvert --to notebook --execute notebooks/07_pipeline_integration.ipynb --output notebooks/07_pipeline_integration.ipynb")

    print("\n" + "=" * 55)
    print("Setup complete! Run the API with:")
    print("  python -m uvicorn api.main:app --reload --port 8000")
    print("=" * 55)