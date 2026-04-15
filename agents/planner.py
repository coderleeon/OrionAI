"""
OrionAI — Planner Agent
=========================
Breaks an incoming user query into an ordered list of actionable steps.

Responsibilities
----------------
* Analyse the query and decompose it into logical, sequenced steps.
* Each step carries a type tag (analysis | retrieval | synthesis | validation)
  that helps downstream agents understand what kind of work is expected.
* Returns a structured plan that the Orchestrator feeds to later agents.

Output schema
-------------
{
    "steps": [
        {"id": 1, "description": "...", "type": "analysis"},
        ...
    ]
}
"""

from __future__ import annotations

import time
from typing import Any

from agents.base_agent import AgentResult, BaseAgent

_SYSTEM_PROMPT = """\
You are the Planner Agent in a multi-agent AI reasoning system called OrionAI.

Your sole task is to decompose the user's query into a concise, ordered list
of discrete reasoning steps. Each step must have:
  - id     : integer (1-based, sequential)
  - description : brief, imperative sentence describing what to do
  - type   : one of [analysis, retrieval, synthesis, validation]

Rules:
- Produce between 2 and 6 steps (no more, no fewer).
- Steps must be logically ordered so each builds on the previous.
- Do NOT include conversational filler — pure task decomposition only.
- Respond ONLY with a JSON object matching this schema:
  {"steps": [{"id": int, "description": str, "type": str}]}
"""


class PlannerAgent(BaseAgent):
    """Decomposes a user query into an ordered step plan."""

    name = "PlannerAgent"

    async def run(self, *, query: str, **_: Any) -> AgentResult:  # noqa: D102
        t0 = time.perf_counter()
        self.log.info("Planning query", query=query[:120])

        user_prompt = f"User query: {query}"

        try:
            payload = await self.llm.chat_complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                session_id=self.session_id,
            )

            # Validate and normalise steps
            steps = payload.get("steps", [])
            if not steps:
                raise ValueError("Planner returned empty steps list.")

            validated_steps = _validate_steps(steps)
            latency_ms = (time.perf_counter() - t0) * 1000

            self.log.info(
                "Plan generated",
                step_count=len(validated_steps),
                latency_ms=round(latency_ms, 2),
            )
            return AgentResult(
                agent_name=self.name,
                payload={"steps": validated_steps},
                latency_ms=latency_ms,
            )

        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - t0) * 1000
            self.log.error("Planner failed", error=str(exc))
            return AgentResult(
                agent_name=self.name,
                payload={"steps": _fallback_steps(query)},
                success=False,
                error=str(exc),
                latency_ms=latency_ms,
            )


# ── Helpers ─────────────────────────────────────────────────────────────────

_VALID_TYPES = {"analysis", "retrieval", "synthesis", "validation"}


def _validate_steps(raw: list[Any]) -> list[dict[str, Any]]:
    """Normalise and validate each step dict from the LLM."""
    validated = []
    for i, step in enumerate(raw, start=1):
        if not isinstance(step, dict):
            continue
        validated.append({
            "id": step.get("id", i),
            "description": str(step.get("description", "Unnamed step")),
            "type": step.get("type", "analysis") if step.get("type") in _VALID_TYPES else "analysis",
        })
    return validated


def _fallback_steps(query: str) -> list[dict[str, Any]]:
    """Return a minimal 3-step plan on planner failure."""
    return [
        {"id": 1, "description": f"Analyse the query: {query[:80]}", "type": "analysis"},
        {"id": 2, "description": "Retrieve relevant background context", "type": "retrieval"},
        {"id": 3, "description": "Synthesise a well-reasoned answer", "type": "synthesis"},
    ]
