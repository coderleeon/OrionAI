"""
OrionAI — Orchestrator Engine
================================
The central controller that drives the full agent pipeline.

Pipeline
--------
  Planner → Retriever → [Executor → Critic → (retry?)] → Response

Key responsibilities
--------------------
* Instantiate and coordinate all four agents.
* Manage the Executor–Critic retry loop (up to MAX_RETRIES).
* Track per-step and total wall-clock latency.
* Write structured step records to the memory store.
* Emit detailed structured logs at every stage.
* Return a fully populated :class:`PipelineResult`.

Retry logic
-----------
The Critic scores the Executor's answer.  If ``approved=False`` AND the retry
budget remains, the Executor is called again with:
  - the same planner steps / retriever context (no re-planning)
  - the Critic's ``feedback`` injected into its synthesis prompt
  - an incremented ``retry_number`` counter passed to both agents

This continues until:
  a) Critic approves,  OR
  b) MAX_RETRIES is exhausted (last answer is used regardless).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from config.settings import settings
from memory.store import memory
from utils.cache import cache
from utils.logger import StructuredLogger
from utils.metrics import RequestRecord, metrics

from agents.critic import CriticAgent
from agents.executor import ExecutorAgent
from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent

log = StructuredLogger(agent_name="Orchestrator")


@dataclass
class PipelineResult:
    """Structured output returned to the API layer."""

    query: str
    session_id: str
    request_id: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    retries: int = 0
    latency: float = 0.0                  # total wall-clock seconds
    critic_confidence: float = 0.0
    critic_feedback: str = ""
    agent_latencies: dict[str, float] = field(default_factory=dict)  # ms per agent
    from_cache: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "query": self.query,
            "steps": self.steps,
            "final_answer": self.final_answer,
            "retries": self.retries,
            "latency": round(self.latency, 4),
            "critic_confidence": round(self.critic_confidence, 4),
            "critic_feedback": self.critic_feedback,
            "agent_latencies_ms": {k: round(v, 2) for k, v in self.agent_latencies.items()},
            "from_cache": self.from_cache,
            "error": self.error,
        }


class OrchestratorEngine:
    """
    Drives the full OrionAI agent pipeline for a single query.

    Usage
    -----
    engine = OrchestratorEngine()
    result = await engine.run_pipeline(query="...", session_id="user-123")
    """

    def __init__(self) -> None:
        self._max_retries = settings.max_retries

    async def run_pipeline(
        self,
        *,
        query: str,
        session_id: str,
    ) -> PipelineResult:
        """
        Execute the full agent pipeline and return a :class:`PipelineResult`.

        Parameters
        ----------
        query : str
            The user's raw input question / task.
        session_id : str
            Session identifier used for memory and logging.
        """
        request_id = str(uuid.uuid4())
        t_global_start = time.perf_counter()

        bound_log = log.bind(session_id=session_id, request_id=request_id)
        bound_log.info("Pipeline started", query=query[:120])

        result = PipelineResult(
            query=query,
            session_id=session_id,
            request_id=request_id,
        )

        # ── 0. Cache check ──────────────────────────────────────────────────
        cache_key = cache.make_key(query, session_id)
        cached = cache.get(cache_key)
        if cached is not None:
            bound_log.info("Returning cached result")
            cached_result = PipelineResult(**{**cached, "from_cache": True})
            return cached_result

        # Persist query in session memory
        memory.save_session(session_id, {
            "query": query,
            "request_id": request_id,
            "steps": [],
            "status": "running",
        })

        try:
            # ── 1. Planner ──────────────────────────────────────────────────
            bound_log.info("Stage: Planner")
            planner = PlannerAgent(session_id=session_id)
            planner_result = await planner.run(query=query)
            result.agent_latencies["planner"] = planner_result.latency_ms

            planned_steps: list[dict[str, Any]] = planner_result.payload.get("steps", [])
            result.steps = planned_steps

            memory.append_step(session_id, {
                "stage": "planner",
                "output": planner_result.to_dict(),
            })
            _log_agent_result(bound_log, planner_result)

            # ── 2. Retriever ────────────────────────────────────────────────
            bound_log.info("Stage: Retriever")
            retriever = RetrieverAgent(session_id=session_id)
            retriever_result = await retriever.run(query=query, steps=planned_steps)
            result.agent_latencies["retriever"] = retriever_result.latency_ms

            context: str = retriever_result.payload.get("context", "")
            sources: list[dict] = retriever_result.payload.get("sources", [])

            memory.append_step(session_id, {
                "stage": "retriever",
                "output": retriever_result.to_dict(),
            })
            _log_agent_result(bound_log, retriever_result)

            # ── 3. Executor → Critic retry loop ─────────────────────────────
            previous_feedback = ""
            final_answer = ""
            critic_confidence = 0.0
            critic_feedback = ""
            attempt = 0
            approved = False

            while attempt <= self._max_retries and not approved:
                bound_log.info(
                    "Stage: Executor",
                    attempt=attempt,
                    max_retries=self._max_retries,
                )
                executor = ExecutorAgent(session_id=session_id)
                executor_result = await executor.run(
                    query=query,
                    steps=planned_steps,
                    context=context,
                    retry_number=attempt,
                    previous_feedback=previous_feedback,
                )
                result.agent_latencies[f"executor_attempt_{attempt}"] = executor_result.latency_ms

                final_answer = executor_result.payload.get("combined", "")
                memory.append_step(session_id, {
                    "stage": "executor",
                    "attempt": attempt,
                    "output": executor_result.to_dict(),
                })
                _log_agent_result(bound_log, executor_result)

                # ── Critic evaluation ───────────────────────────────────────
                bound_log.info("Stage: Critic", attempt=attempt)
                critic = CriticAgent(session_id=session_id)
                critic_result = await critic.run(
                    query=query,
                    answer=final_answer,
                    retry_number=attempt,
                    previous_feedback=previous_feedback,
                )
                result.agent_latencies[f"critic_attempt_{attempt}"] = critic_result.latency_ms

                critic_confidence = critic_result.payload.get("confidence", 0.0)
                critic_feedback = critic_result.payload.get("feedback", "")
                approved = critic_result.payload.get("approved", False)
                previous_feedback = critic_feedback

                memory.append_step(session_id, {
                    "stage": "critic",
                    "attempt": attempt,
                    "output": critic_result.to_dict(),
                })
                _log_agent_result(bound_log, critic_result)

                if approved:
                    bound_log.info(
                        "Critic approved answer",
                        confidence=critic_confidence,
                        attempt=attempt,
                    )
                    break

                if attempt < self._max_retries:
                    bound_log.warning(
                        "Critic rejected — retrying",
                        confidence=critic_confidence,
                        feedback=critic_feedback[:100],
                        attempt=attempt,
                    )
                else:
                    bound_log.warning(
                        "Max retries reached — using last answer",
                        confidence=critic_confidence,
                    )

                attempt += 1

            result.retries = attempt
            result.final_answer = final_answer
            result.critic_confidence = critic_confidence
            result.critic_feedback = critic_feedback

        except Exception as exc:  # noqa: BLE001
            bound_log.error("Pipeline error", error=str(exc), exc_info=True)
            result.error = str(exc)
            result.final_answer = "An internal error occurred. Please try again."

        finally:
            result.latency = time.perf_counter() - t_global_start
            memory.update_field(session_id, "status", "complete")
            memory.update_field(session_id, "latency", result.latency)

            bound_log.info(
                "Pipeline complete",
                latency=round(result.latency, 4),
                retries=result.retries,
                approved=not bool(result.error),
                from_cache=result.from_cache,
            )

        # ── Cache successful results ─────────────────────────────────────────
        if not result.error:
            result_dict = result.to_dict()
            cache.set(cache_key, result_dict)

        # ── Record metrics for dashboard ──────────────────────────────────────
        import time as _time  # noqa: PLC0415
        metrics.record(RequestRecord(
            request_id=result.request_id,
            session_id=result.session_id,
            query=result.query,
            timestamp=_time.time(),
            latency=result.latency,
            retries=result.retries,
            critic_confidence=result.critic_confidence,
            from_cache=result.from_cache,
            success=result.error is None,
            agent_latencies_ms=result.agent_latencies,
        ))

        return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _log_agent_result(logger: StructuredLogger, agent_result: Any) -> None:
    """Emit a concise log line summarising an agent result."""
    logger.debug(
        "Agent result",
        agent=agent_result.agent_name,
        success=agent_result.success,
        latency_ms=round(agent_result.latency_ms, 2),
        error=agent_result.error,
    )
