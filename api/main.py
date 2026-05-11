import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from api.routes import router
from config import API_HOST, API_PORT, CLASSIFIER_PATH, REGRESSOR_PATH

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    models = {
        "classifier": CLASSIFIER_PATH,
        "regressor": REGRESSOR_PATH,
    }
    loaded = {name: path.exists() for name, path in models.items()}
    status = "ready" if all(loaded.values()) else "training_required"
    
    print("=" * 50)
    print("CloudGuard AI — Startup")
    print(f"Status: {status}")
    for name, exists in loaded.items():
        print(f"  {name}: {'loaded' if exists else 'missing'}")
    print("=" * 50)
    
    yield
    
    # Shutdown (if needed)
    print("CloudGuard AI — Shutting down")

app = FastAPI(
    title="Cloud Anomaly Detector",
    description="Real-time cloud resource anomaly detection & cost prediction",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes FIRST before static mount
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Log model load status on startup."""
    models = {
        "classifier": CLASSIFIER_PATH,
        "regressor": REGRESSOR_PATH,
    }
    loaded = {name: path.exists() for name, path in models.items()}
    status = "ready" if all(loaded.values()) else "training_required"
    
    print("=" * 50)
    print("CloudGuard AI — Startup")
    print(f"Status: {status}")
    for name, exists in loaded.items():
        print(f"  {name}: {'loaded' if exists else 'missing'}")
    print("=" * 50)


@app.get("/health")
async def health():
    """
    Health check endpoint.
    Returns 200 when models are loaded and ready for inference.
    Returns 503 when models are missing (training in progress).
    """
    classifier_ok = CLASSIFIER_PATH.exists()
    regressor_ok = REGRESSOR_PATH.exists()
    models_ready = classifier_ok and regressor_ok
    
    status_code = 200 if models_ready else 503
    status = "healthy" if models_ready else "loading"
    
    return JSONResponse(
        content={
            "status": status,
            "version": "1.0.0",
            "models_loaded": models_ready,
            "classifier": {
                "path": str(CLASSIFIER_PATH),
                "loaded": classifier_ok
            },
            "regressor": {
                "path": str(REGRESSOR_PATH),
                "loaded": regressor_ok
            }
        },
        status_code=status_code
    )


@app.get("/")
async def serve_dashboard():
    frontend_path = Path(__file__).parent.parent / "frontend"
    return FileResponse(str(frontend_path / "index.html"))


# Mount static files LAST — catches everything under /static/
frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")