"""
OrionAI — Structured Logger
============================
JSON-formatted logging to both stdout and a rotating file.
Every log record carries a consistent set of metadata fields
so logs can be ingested by any log aggregation system (ELK, Loki, etc.).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any

from config.settings import settings


class _JSONFormatter(logging.Formatter):
    """Formats each log record as a single-line JSON object."""

    RESERVED = {"message", "asctime", "exc_info", "exc_text", "stack_info"}

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        # Build the base payload
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }

        # Merge any extra fields passed via `extra=` kwarg
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and key not in self.RESERVED:
                payload[key] = value

        # Attach exception info if present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def _build_logger(name: str = "orion") -> logging.Logger:
    """Construct and configure the application logger (idempotent)."""
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured — return the existing logger
        return logger

    logger.setLevel(getattr(logging, settings.log_level))

    formatter = _JSONFormatter()

    # ── stdout handler ──────────────────────────────────────────────────────
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # ── rotating file handler ───────────────────────────────────────────────
    log_dir = os.path.dirname(settings.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=settings.log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent log propagation to the root logger (avoids duplicate output)
    logger.propagate = False

    return logger


class StructuredLogger:
    """
    Thin wrapper around the standard logger that automatically injects
    context fields (session_id, agent_name, request_id, etc.) into every
    log record.

    Usage
    -----
    log = StructuredLogger(agent_name="PlannerAgent", session_id="abc-123")
    log.info("Starting plan", query="explain photosynthesis")
    log.error("LLM call failed", error=str(e))
    """

    def __init__(self, *, agent_name: str = "orchestrator", session_id: str = "") -> None:
        self._logger = _build_logger()
        self._context: dict[str, Any] = {
            "agent": agent_name,
            "session_id": session_id,
        }

    def _log(self, level: int, msg: str, **extra: Any) -> None:
        merged = {**self._context, **extra}
        self._logger.log(level, msg, extra=merged)

    def debug(self, msg: str, **extra: Any) -> None:
        self._log(logging.DEBUG, msg, **extra)

    def info(self, msg: str, **extra: Any) -> None:
        self._log(logging.INFO, msg, **extra)

    def warning(self, msg: str, **extra: Any) -> None:
        self._log(logging.WARNING, msg, **extra)

    def error(self, msg: str, **extra: Any) -> None:
        self._log(logging.ERROR, msg, **extra)

    def critical(self, msg: str, **extra: Any) -> None:
        self._log(logging.CRITICAL, msg, **extra)

    def bind(self, **context: Any) -> "StructuredLogger":
        """Return a new logger with additional bound context fields."""
        new = StructuredLogger.__new__(StructuredLogger)
        new._logger = self._logger
        new._context = {**self._context, **context}
        return new


# Module-level default logger (import and use directly if no context needed)
log = StructuredLogger()
