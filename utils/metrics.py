"""
OrionAI — Metrics Store
========================
Lightweight in-process metrics collector for the dashboard.

Tracks per-request:
  - total latency
  - per-agent latency breakdown
  - retry count
  - critic confidence
  - cache hits/misses
  - success/failure

Exposes aggregated stats via get_metrics() for the dashboard endpoint.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# Keep last N requests in memory for the dashboard
_HISTORY_SIZE = 200


@dataclass
class RequestRecord:
    request_id: str
    session_id: str
    query: str
    timestamp: float          # Unix epoch
    latency: float            # seconds
    retries: int
    critic_confidence: float
    from_cache: bool
    success: bool
    agent_latencies_ms: dict[str, float] = field(default_factory=dict)


class MetricsStore:
    """
    Thread-safe rolling metrics collector.
    Uses a deque so memory is bounded regardless of traffic volume.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: deque[RequestRecord] = deque(maxlen=_HISTORY_SIZE)
        self._total_requests = 0
        self._total_errors = 0
        self._total_cache_hits = 0
        self._started_at = time.time()

    def record(self, record: RequestRecord) -> None:
        """Add a completed request record to the store."""
        with self._lock:
            self._history.append(record)
            self._total_requests += 1
            if not record.success:
                self._total_errors += 1
            if record.from_cache:
                self._total_cache_hits += 1

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent `limit` request records as dicts."""
        with self._lock:
            records = list(self._history)[-limit:]
        return [_record_to_dict(r) for r in reversed(records)]

    def get_summary(self) -> dict[str, Any]:
        """Return aggregate statistics for the dashboard."""
        with self._lock:
            records = list(self._history)
            total = self._total_requests
            errors = self._total_errors
            cache_hits = self._total_cache_hits

        if not records:
            return _empty_summary(total, errors, cache_hits, self._started_at)

        latencies = [r.latency for r in records]
        retries = [r.retries for r in records]
        confidences = [r.critic_confidence for r in records if r.critic_confidence > 0]

        # Per-agent average latencies across all recent requests
        agent_totals: dict[str, list[float]] = {}
        for rec in records:
            for agent, ms in rec.agent_latencies_ms.items():
                agent_totals.setdefault(agent, []).append(ms)

        agent_avg = {
            agent: round(sum(vals) / len(vals), 2)
            for agent, vals in agent_totals.items()
            if vals
        }

        return {
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "total_requests": total,
            "total_errors": errors,
            "total_cache_hits": cache_hits,
            "error_rate": round(errors / total, 4) if total else 0.0,
            "cache_hit_rate": round(cache_hits / total, 4) if total else 0.0,
            "latency": {
                "avg_s": round(sum(latencies) / len(latencies), 4),
                "min_s": round(min(latencies), 4),
                "max_s": round(max(latencies), 4),
                "p95_s": round(_percentile(latencies, 95), 4),
            },
            "retries": {
                "avg": round(sum(retries) / len(retries), 3),
                "max": max(retries),
                "total": sum(retries),
                "retry_rate": round(sum(1 for r in retries if r > 0) / len(retries), 4),
            },
            "critic_confidence": {
                "avg": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
                "min": round(min(confidences), 4) if confidences else 0.0,
            },
            "agent_avg_latency_ms": agent_avg,
            "recent_window_size": len(records),
        }


# ── Module-level singleton ────────────────────────────────────────────────────
metrics = MetricsStore()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _record_to_dict(r: RequestRecord) -> dict[str, Any]:
    return {
        "request_id": r.request_id,
        "session_id": r.session_id,
        "query": r.query[:80] + ("…" if len(r.query) > 80 else ""),
        "timestamp": r.timestamp,
        "latency_s": round(r.latency, 4),
        "retries": r.retries,
        "critic_confidence": round(r.critic_confidence, 4),
        "from_cache": r.from_cache,
        "success": r.success,
        "agent_latencies_ms": {k: round(v, 2) for k, v in r.agent_latencies_ms.items()},
    }


def _percentile(data: list[float], pct: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * pct / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def _empty_summary(
    total: int, errors: int, cache_hits: int, started_at: float
) -> dict[str, Any]:
    return {
        "uptime_seconds": round(time.time() - started_at, 1),
        "total_requests": total,
        "total_errors": errors,
        "total_cache_hits": cache_hits,
        "error_rate": 0.0,
        "cache_hit_rate": 0.0,
        "latency": {"avg_s": 0.0, "min_s": 0.0, "max_s": 0.0, "p95_s": 0.0},
        "retries": {"avg": 0.0, "max": 0, "total": 0, "retry_rate": 0.0},
        "critic_confidence": {"avg": 0.0, "min": 0.0},
        "agent_avg_latency_ms": {},
        "recent_window_size": 0,
    }
