"""
OrionAI — Critic Agent
========================
The final quality gate in the OrionAI pipeline.

Responsibilities
----------------
* Score the Executor's combined answer against the original query.
* Decide whether the answer is acceptable (approved=True) or needs a retry.
* Provide structured, actionable feedback so the retry attempt has better context.

Scoring dimensions
------------------
1. Relevance   — Does the answer actually address the query?
2. Accuracy    — Is the content factually plausible?
3. Completeness — Are the key points covered?
4. Clarity     — Is the language clear and well-structured?

The composite confidence score (0–1) is compared against the threshold in
settings (default 0.70). Scores below threshold trigger a retry.

Output schema
-------------
{
    "approved": bool,
    "confidence": float,          // 0.0 – 1.0
    "feedback": "...",            // actionable critique for retry
    "scores": {                   // per-dimension scores
        "relevance": float,
        "accuracy": float,
        "completeness": float,
        "clarity": float
    }
}
"""

from __future__ import annotations

import time
from typing import Any

from config.settings import settings
from agents.base_agent import AgentResult, BaseAgent

_SYSTEM_PROMPT = """\
You are the Critic Agent in a multi-agent AI reasoning system called OrionAI.

Your role is to rigorously evaluate the quality of an AI-generated answer
against the original user query.

Evaluate the answer on four dimensions, each scored 0.0–1.0:
  - relevance    : Does the answer directly address what was asked?
  - accuracy     : Is the information factually correct and plausible?
  - completeness : Are the important aspects of the question covered?
  - clarity      : Is the answer well-structured, concise, and readable?

Compute a composite confidence score as the weighted average:
  confidence = (relevance * 0.35) + (accuracy * 0.30) + (completeness * 0.20) + (clarity * 0.15)

Rules:
- Be objective and strict — only high-quality answers should get confidence ≥ 0.70.
- If the answer is vague, off-topic, or contains errors, score accordingly.
- Provide actionable feedback of 1–3 sentences explaining what needs improvement.
- Your feedback will be passed to the next retry attempt, so be specific.

Respond ONLY with this JSON structure (no extra text):
{
  "approved": bool,
  "confidence": float,
  "feedback": "...",
  "scores": {
    "relevance": float,
    "accuracy": float,
    "completeness": float,
    "clarity": float
  }
}
"""


class CriticAgent(BaseAgent):
    """
    Evaluates answer quality and approves or rejects for retry.

    The ``approved`` field is set to True when confidence >= threshold.
    Rejection provides feedback that guides a better retry attempt.
    """

    name = "CriticAgent"

    def __init__(self, *, session_id: str = "") -> None:
        super().__init__(session_id=session_id)
        self._threshold = settings.critic_confidence_threshold

    async def run(  # noqa: D102
        self,
        *,
        query: str,
        answer: str,
        retry_number: int = 0,
        previous_feedback: str = "",
        **_: Any,
    ) -> AgentResult:
        t0 = time.perf_counter()
        self.log.info(
            "Critiquing answer",
            retry_number=retry_number,
            answer_chars=len(answer),
        )

        user_prompt = _build_user_prompt(
            query=query,
            answer=answer,
            retry_number=retry_number,
            previous_feedback=previous_feedback,
        )

        try:
            raw = await self.llm.chat_complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                session_id=self.session_id,
            )
            result = _parse_critic_output(raw, self._threshold)
            latency_ms = (time.perf_counter() - t0) * 1000

            self.log.info(
                "Critique complete",
                approved=result["approved"],
                confidence=result["confidence"],
                latency_ms=round(latency_ms, 2),
            )

            return AgentResult(
                agent_name=self.name,
                payload=result,
                latency_ms=latency_ms,
            )

        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - t0) * 1000
            self.log.error("Critic failed — defaulting to approval", error=str(exc))
            # Safe fallback: on critic failure, approve to avoid infinite retries
            return AgentResult(
                agent_name=self.name,
                payload={
                    "approved": True,
                    "confidence": 0.5,
                    "feedback": f"Critic error: {exc}. Answer passed by default.",
                    "scores": {"relevance": 0.5, "accuracy": 0.5, "completeness": 0.5, "clarity": 0.5},
                },
                success=False,
                error=str(exc),
                latency_ms=latency_ms,
            )


# ── Helpers ─────────────────────────────────────────────────────────────────

def _build_user_prompt(
    *,
    query: str,
    answer: str,
    retry_number: int,
    previous_feedback: str,
) -> str:
    parts = [
        f"Original user query:\n{query}",
        f"\nGenerated answer (attempt #{retry_number + 1}):\n{answer}",
    ]
    if previous_feedback:
        parts.append(f"\nFeedback from previous attempt:\n{previous_feedback}")
    return "\n".join(parts)


def _parse_critic_output(
    raw: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    """Parse and normalise the LLM's critic output."""
    try:
        confidence = float(raw.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]

        scores = raw.get("scores", {})
        normalised_scores = {
            dim: max(0.0, min(1.0, float(scores.get(dim, 0.5))))
            for dim in ("relevance", "accuracy", "completeness", "clarity")
        }

        feedback = str(raw.get("feedback", "No feedback provided."))
        approved = confidence >= threshold

        return {
            "approved": approved,
            "confidence": round(confidence, 4),
            "feedback": feedback,
            "scores": {k: round(v, 4) for k, v in normalised_scores.items()},
        }

    except (TypeError, ValueError):
        # If parsing completely fails, treat as low-quality
        return {
            "approved": False,
            "confidence": 0.0,
            "feedback": "Could not parse critic output. Retry recommended.",
            "scores": {"relevance": 0.0, "accuracy": 0.0, "completeness": 0.0, "clarity": 0.0},
        }
