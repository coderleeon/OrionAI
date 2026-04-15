"""
OrionAI — Executor Agent
==========================
Performs step-by-step reasoning over the Planner's steps,
grounded by the Retriever's context.

Responsibilities
----------------
* Iterate through each planned step and call the LLM to reason about it.
* Accumulate intermediate results.
* Produce a final combined answer that synthesises all step outputs.

Output schema
-------------
{
    "results": [
        {"step_id": int, "step_description": str, "output": str}
    ],
    "combined": "..."
}
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from agents.base_agent import AgentResult, BaseAgent

_STEP_SYSTEM_PROMPT = """\
You are the Executor Agent in a multi-agent AI reasoning system called OrionAI.

Your task is to reason through ONE specific step of a multi-step analysis plan.

You will be given:
  - The original user query
  - Relevant background context
  - The specific step to execute

Instructions:
- Focus exclusively on the given step.
- Use the context provided — do not introduce unrelated information.
- Be concise and factual (2–5 sentences).
- Respond ONLY with this JSON:
  {"step_id": int, "output": "..."}
"""

_SYNTHESIS_SYSTEM_PROMPT = """\
You are the Executor Agent (synthesis phase) in OrionAI.

You have completed individual reasoning steps. Now synthesise all step
outputs into a single, coherent, well-structured final answer to the
original query.

Instructions:
- Address the original query directly.
- Integrate insights from all steps cohesively.
- Use clear, professional language.
- Length: 3–8 sentences unless more detail is warranted.
- Respond ONLY with this JSON:
  {"combined": "..."}
"""


class ExecutorAgent(BaseAgent):
    """Executes planned steps and synthesises a final answer."""

    name = "ExecutorAgent"

    async def run(  # noqa: D102
        self,
        *,
        query: str,
        steps: list[dict[str, Any]],
        context: str = "",
        **_: Any,
    ) -> AgentResult:
        t0 = time.perf_counter()
        self.log.info("Executing steps", step_count=len(steps), query=query[:80])

        step_results: list[dict[str, Any]] = []

        # ── Phase 1: Execute each step individually ─────────────────────────
        # Run step calls concurrently (they are independent reasoning units)
        tasks = [
            self._execute_step(step, query=query, context=context)
            for step in steps
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        for step, result in zip(steps, raw_results):
            if isinstance(result, Exception):
                self.log.error(
                    "Step execution failed",
                    step_id=step.get("id"),
                    error=str(result),
                )
                step_results.append({
                    "step_id": step.get("id", 0),
                    "step_description": step.get("description", ""),
                    "output": f"[Error: {result}]",
                })
            else:
                step_results.append(result)

        # ── Phase 2: Synthesise all step outputs into a final answer ────────
        combined = await self._synthesise(query=query, step_results=step_results)

        latency_ms = (time.perf_counter() - t0) * 1000

        self.log.info(
            "Execution complete",
            steps_executed=len(step_results),
            combined_chars=len(combined),
            latency_ms=round(latency_ms, 2),
        )

        return AgentResult(
            agent_name=self.name,
            payload={
                "results": step_results,
                "combined": combined,
            },
            latency_ms=latency_ms,
        )

    # ── Internal helpers ────────────────────────────────────────────────────

    async def _execute_step(
        self,
        step: dict[str, Any],
        *,
        query: str,
        context: str,
    ) -> dict[str, Any]:
        """Reason through a single step and return a structured output."""
        step_id = step.get("id", 0)
        description = step.get("description", "")

        user_prompt = (
            f"Original query: {query}\n\n"
            f"Background context:\n{context or 'No context provided.'}\n\n"
            f"Step to execute (ID={step_id}): {description}"
        )

        raw = await self.llm.chat_complete(
            system_prompt=_STEP_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            session_id=self.session_id,
        )

        output_text = raw.get("output", raw.get("result", str(raw)))

        self.log.debug("Step complete", step_id=step_id, output_chars=len(str(output_text)))

        return {
            "step_id": step_id,
            "step_description": description,
            "output": str(output_text),
        }

    async def _synthesise(
        self,
        *,
        query: str,
        step_results: list[dict[str, Any]],
    ) -> str:
        """Produce the final synthesised answer from all step outputs."""
        steps_text = "\n".join(
            f"Step {r['step_id']} ({r.get('step_description', '')}): {r['output']}"
            for r in step_results
        )
        user_prompt = f"Original query: {query}\n\nStep reasoning outputs:\n{steps_text}"

        raw = await self.llm.chat_complete(
            system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            session_id=self.session_id,
        )
        return raw.get("combined", raw.get("raw", "No synthesis produced."))
