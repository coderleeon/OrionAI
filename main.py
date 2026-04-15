"""
OrionAI — Application Entry Point
====================================
Creates and configures the FastAPI application instance.

Features
--------
* CORS middleware (configurable)
* Global exception handler for unhandled errors
* Startup / shutdown lifecycle events (logging, memory init)
* Mounts the inference router under the root path
* Serves OpenAPI docs at /docs and /redoc
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from api.metrics_routes import router as metrics_router
from config.settings import settings
from utils.logger import StructuredLogger, log

_APP_LOG = StructuredLogger(agent_name="App")


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context for startup and shutdown events."""
    # ── Startup ─────────────────────────
    _APP_LOG.info(
        "OrionAI starting up",
        version=settings.api_version,
        model=settings.model_name,
        max_retries=settings.max_retries,
        cache_enabled=settings.cache_enabled,
    )

    # Pre-warm the memory store (validates Redis connection or falls back)
    from memory.store import memory  # noqa: PLC0415
    _APP_LOG.info("Memory backend initialised", backend=memory.backend_type())

    yield  # Application runs

    # ── Shutdown ─────────────────────────
    _APP_LOG.info("OrionAI shutting down gracefully")


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=(
            "OrionAI is a production-grade multi-agent inference engine. "
            "Queries are routed through a four-stage pipeline: "
            "Planner → Retriever → Executor → Critic, "
            "with automatic retry logic and structured JSON responses."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):  # type: ignore[return]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}s"
        return response

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        _APP_LOG.error(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An unexpected server error occurred.",
                "error": str(exc),
                "path": str(request.url.path),
            },
        )

    # ── Mount routers ──────────────────────────────────────────
    app.include_router(router)
    app.include_router(metrics_router)

    return app


# ── Module-level app instance (used by uvicorn) ───────────────────────────────
app = create_app()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
