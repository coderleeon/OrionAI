"""
OrionAI — In-Memory TTL Cache
===============================
A simple, thread-safe in-process cache with per-entry TTL expiry.
Used to short-circuit duplicate queries before they reach the agent pipeline.

Design notes
------------
* Keyed by (query_hash, session_id) so different sessions don't share results.
* Expired entries are lazily evicted on reads.
* No external dependencies — pure stdlib.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from config.settings import settings
from utils.logger import StructuredLogger

log = StructuredLogger(agent_name="Cache")


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float  # Unix timestamp


class SimpleCache:
    """
    Thread-safe in-memory TTL cache.

    Usage
    -----
    cache = SimpleCache()

    # Store
    cache.set("my-key", result, ttl=300)

    # Retrieve (returns None if missing or expired)
    cached = cache.get("my-key")
    """

    def __init__(self, default_ttl: int | None = None) -> None:
        self._store: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl or settings.cache_ttl_seconds

    # ── Public interface ────────────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        """Return cached value or ``None`` if absent / expired."""
        if not settings.cache_enabled:
            return None

        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None

            if time.monotonic() > entry.expires_at:
                del self._store[key]
                log.debug("Cache MISS (expired)", key=key[:40])
                return None

            log.info("Cache HIT", key=key[:40])
            return entry.value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value with optional TTL override (seconds)."""
        if not settings.cache_enabled:
            return

        effective_ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            self._store[key] = _CacheEntry(
                value=value,
                expires_at=time.monotonic() + effective_ttl,
            )
        log.debug("Cache SET", key=key[:40], ttl=effective_ttl)

    def delete(self, key: str) -> None:
        """Explicitly remove a key."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Flush all entries."""
        with self._lock:
            self._store.clear()
        log.info("Cache cleared")

    def stats(self) -> dict[str, int]:
        """Return basic cache statistics."""
        now = time.monotonic()
        with self._lock:
            total = len(self._store)
            active = sum(1 for e in self._store.values() if e.expires_at > now)
        return {"total_entries": total, "active_entries": active, "expired_entries": total - active}

    # ── Static helpers ──────────────────────────────────────────────────────

    @staticmethod
    def make_key(query: str, session_id: str = "") -> str:
        """Produce a deterministic cache key from query + session."""
        raw = f"{session_id}::{query.strip().lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()


# Module-level singleton
cache = SimpleCache()
