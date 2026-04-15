"""
OrionAI — Metrics & Dashboard Routes
======================================
Exposes the metrics API consumed by the live dashboard,
and serves the dashboard HTML itself.

Endpoints
---------
GET /metrics/summary          — Aggregate stats (KPIs)
GET /metrics/history          — Rolling request history
GET /dashboard                — Serves the live HTML dashboard
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from utils.metrics import metrics

router = APIRouter(tags=["Metrics"])

_DASHBOARD_PATH = Path(__file__).parent.parent / "dashboard" / "index.html"


# ── GET /metrics/summary ──────────────────────────────────────────────────────

@router.get("/metrics/summary", response_model=None)
async def metrics_summary() -> dict[str, Any]:
    """
    Return aggregate statistics across all tracked requests.
    Consumed by the dashboard KPI cards and agent latency bars.
    """
    return metrics.get_summary()


# ── GET /metrics/history ──────────────────────────────────────────────────────

@router.get("/metrics/history", response_model=None)
async def metrics_history(
    limit: int = Query(default=50, ge=1, le=200, description="Max records to return"),
) -> list[dict[str, Any]]:
    """
    Return the most recent N request records.
    Consumed by the dashboard charts and history table.
    """
    return metrics.get_history(limit=limit)


# ── GET /dashboard ────────────────────────────────────────────────────────────

@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard() -> HTMLResponse:
    """Serve the self-contained dashboard HTML page."""
    if not _DASHBOARD_PATH.exists():
        return HTMLResponse(
            content="<h1>Dashboard not found</h1><p>dashboard/index.html is missing.</p>",
            status_code=404,
        )
    html = _DASHBOARD_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html)
