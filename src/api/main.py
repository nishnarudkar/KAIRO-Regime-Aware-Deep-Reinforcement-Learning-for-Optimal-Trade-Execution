"""
src/api/main.py — KAIRO Adaptive Execution Intelligence FastAPI Server

Main entry point for the REST API service wrapper around the research engine.

Usage:
  uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

Endpoints exposed:
  POST /api/execution/simulate
  POST /api/execution/backtest
  GET  /api/execution/{id}
  GET  /api/execution/{id}/metrics
  GET  /api/execution/{id}/trajectory
  GET  /api/regime/current
  GET  /api/baselines
  GET  /api/models
  GET  /api/experiments
  GET  /health
"""

from __future__ import annotations

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routers import execution, regime, metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KAIRO — Adaptive Execution Intelligence API",
    version="1.0.0",
    description=(
        "Production-grade RESTful API service wrapper around the KAIRO "
        "Regime-Aware Deep Reinforcement Learning Trade Execution Engine."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────────────────
app.include_router(execution.router)
app.include_router(regime.router)
app.include_router(metadata.router)


# ── Health check ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health_check():
    """Service health check endpoint."""
    return {
        "status": "ok",
        "service": "KAIRO Adaptive Execution Intelligence API",
        "version": "1.0.0",
    }


# ── Global Exception Handler ────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred while processing the request.",
            "error": str(exc),
        },
    )
