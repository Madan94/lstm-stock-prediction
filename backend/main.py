"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial Forecasting API",
    description="API for directional financial forecasting with attention-based LSTM",
    version="1.0.0"
)

# CORS middleware for frontend
import os

# Get allowed origins from environment or use defaults
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:3001"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api", tags=["api"])


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Financial Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "indices": "/api/indices",
            "metrics": "/api/metrics/{index}",
            "predictions": "/api/predictions/{index}",
            "equity_curve": "/api/equity-curve/{index}",
            "attention": "/api/attention/{index}",
            "baseline_comparison": "/api/baseline-comparison/{index}"
        }
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}





