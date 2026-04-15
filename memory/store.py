"""
OrionAI — Memory Store
========================
Provides session-scoped persistent memory for the agent pipeline.

Strategy
--------
1. Try to connect to Redis (configured via REDIS_URL).
2. If Redis is unavailable (connection error, missing package), fall back
   transparently to an in-process dict-based store.

Stored per session
------------------
* Original query
* Planner steps
* Retriever context
* Executor results
* Critic feedback
* Retry count
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from config.settings import settings
from utils.logger import StructuredLogger

log = StructuredLogger(agent_name="MemoryStore")


# ── In-memory fallback ──────────────────────────────────────────────────────

class _InMemoryBackend:
    """Simple dict-based store with TTL simulation (best-effort)."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[str, float]] = {}  # key → (json_value, expires_at)

    def get(self, key: str) -> Optional[str]:
        entry = self._data.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del self._data[key]
            return None
        return value

    def setex(self, key: str, seconds: int, value: str) -> None:
        self._data[key] = (value, time.time() + seconds)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def ping(self) -> bool:
        return True


# ── Redis backend (optional) ────────────────────────────────────────────────

def _try_redis() -> Optional[Any]:
    """Attempt to create a Redis client; return None on failure."""
    try:
        import redis  # type: ignore[import-untyped]
        client = redis.from_url(settings.redis_url, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        log.info("Redis connection established", url=settings.redis_url)
        return client
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Redis unavailable — using in-memory fallback",
            error=str(exc),
            redis_url=settings.redis_url,
        )
        return None


# ── Public MemoryStore ───────────────────────────────────────────────────────

class MemoryStore:
    """
    Abstracted key-value memory store with Redis / in-memory fallback.

    All data is JSON-serialised before storage so complex objects survive
    the Redis serialisation boundary.
    """

    _SESSION_PREFIX = "orion:session:"

    def __init__(self) -> None:
        redis_client = _try_redis()
        self._backend = redis_client if redis_client is not None else _InMemoryBackend()
        self._using_redis = redis_client is not None

    # ── Core operations ─────────────────────────────────────────────────────

    def _key(self, session_id: str) -> str:
        return f"{self._SESSION_PREFIX}{session_id}"

    def save_session(self, session_id: str, data: dict[str, Any]) -> None:
        """Persist/overwrite the entire session payload."""
        key = self._key(session_id)
        serialised = json.dumps(data, default=str)
        self._backend.setex(key, settings.session_ttl_seconds, serialised)
        log.debug("Session saved", session_id=session_id, keys=list(data.keys()))

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Retrieve the full session payload, or None if it doesn't exist."""
        key = self._key(session_id)
        raw = self._backend.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def append_step(self, session_id: str, step: dict[str, Any]) -> None:
        """
        Append a pipeline step record to the session's step list.
        Creates the session if it doesn't exist yet.
        """
        session = self.get_session(session_id) or {"steps": [], "created_at": time.time()}
        session.setdefault("steps", []).append({**step, "timestamp": time.time()})
        self.save_session(session_id, session)

    def update_field(self, session_id: str, field: str, value: Any) -> None:
        """Update a single top-level field in the session."""
        session = self.get_session(session_id) or {}
        session[field] = value
        self.save_session(session_id, session)

    def delete_session(self, session_id: str) -> None:
        """Remove a session entirely."""
        self._backend.delete(self._key(session_id))
        log.info("Session deleted", session_id=session_id)

    def backend_type(self) -> str:
        return "redis" if self._using_redis else "in-memory"


# Module-level singleton
memory = MemoryStore()
