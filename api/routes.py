"""
OrionAI — FastAPI Routes
==========================
All HTTP endpoint definitions for the OrionAI inference API.

Endpoints
---------
POST  /run      — Primary inference endpoint
GET   /health   — Liveness + readiness probe
GET   /session/{session_id} — Retrieve session memory
DELETE /session/{session_id} — Clear a session
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.models import HealthResponse, InferenceRequest, InferenceResponse
from config.settings import settings
from memory.store import memory
from orchestrator.engine import OrchestratorEngine
from utils.cache import cache
from utils.logger import StructuredLogger

router = APIRouter()
log = StructuredLogger(agent_name="API")

# Module-level engine instance (reused across requests — thread/async safe)
_engine = OrchestratorEngine()


# ── POST /run ─────────────────────────────────────────────────────────────────

@router.post(
    "/run",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Run the agent inference pipeline",
    description=(
        "Accepts a user query, routes it through the full 4-agent pipeline "
        "(Planner → Retriever → Executor → Critic), and returns a structured response."
    ),
    tags=["Inference"],
)
async def run_inference(
    request_body: InferenceRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> InferenceResponse:
    """
    Primary inference endpoint.

    - Validates the incoming request via Pydantic.
    - Delegates to the OrchestratorEngine.
    - Returns a structured JSON response.
    """
    client_ip = _get_client_ip(http_request)
    log.info(
        "Inference request received",
        session_id=request_body.session_id,
        query=request_body.query[:80],
        client_ip=client_ip,
    )

    try:
        pipeline_result = await _engine.run_pipeline(
            query=request_body.query,
            session_id=request_body.session_id,
        )
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Unhandled pipeline error",
            error=str(exc),
            session_id=request_body.session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        ) from exc

    response_dict = pipeline_result.to_dict()

    if pipeline_result.error:
        log.warning(
            "Pipeline returned with error",
            error=pipeline_result.error,
            session_id=request_body.session_id,
        )

    log.info(
        "Inference response dispatched",
        session_id=request_body.session_id,
        latency=pipeline_result.latency,
        retries=pipeline_result.retries,
        from_cache=pipeline_result.from_cache,
    )

    return InferenceResponse(**response_dict)


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Returns system health, version, and backend info."""
    return HealthResponse(
        status="ok",
        version=settings.api_version,
        memory_backend=memory.backend_type(),
        cache_enabled=settings.cache_enabled,
    )


# ── GET /session/{session_id} ─────────────────────────────────────────────────

@router.get(
    "/session/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Retrieve session memory",
    tags=["Memory"],
)
async def get_session(session_id: str) -> dict[str, Any]:
    """Return all stored memory for a session."""
    if not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id must not be empty.",
        )

    session_data = memory.get_session(session_id)
    if session_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found or expired.",
        )

    return {"session_id": session_id, "data": session_data}


# ── DELETE /session/{session_id} ──────────────────────────────────────────────

@router.delete(
    "/session/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete session memory",
    tags=["Memory"],
)
async def delete_session(session_id: str) -> dict[str, str]:
    """Clear a session from memory."""
    memory.delete_session(session_id)
    log.info("Session deleted via API", session_id=session_id)
    return {"status": "deleted", "session_id": session_id}


# ── GET /cache/stats ──────────────────────────────────────────────────────────

@router.get(
    "/cache/stats",
    status_code=status.HTTP_200_OK,
    summary="Query cache statistics",
    tags=["System"],
)
async def cache_stats() -> dict[str, Any]:
    """Return basic cache hit/miss statistics."""
    return cache.stats()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_client_ip(request: Request) -> str:
    """Extract client IP, respecting common proxy headers."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"
