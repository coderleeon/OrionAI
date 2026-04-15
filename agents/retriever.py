"""
OrionAI — Retriever Agent
===========================
Fetches relevant context for the planner's steps.

In production this would query a vector database (Pinecone, Weaviate, Chroma).
Here we implement a two-tier retrieval strategy:
  1. **Keyword store** — a small in-process corpus that handles common topics.
  2. **LLM synthesis** — when no keyword match is found, the LLM synthesises
     a plausible context summary from its own knowledge.

Both tiers return the same output schema so the Executor Agent is agnostic
to which path was taken.

Output schema
-------------
{
    "context": "...",
    "sources": [{"title": "...", "relevance": float}]
}
"""

from __future__ import annotations

import time
from typing import Any

from agents.base_agent import AgentResult, BaseAgent

# ── Static knowledge corpus (simulates a vector DB) ─────────────────────────
# Each entry: (keywords, context_text, source_title, relevance_score)
_CORPUS: list[tuple[tuple[str, ...], str, str, float]] = [
    (
        ("photosynthesis", "plant", "chlorophyll", "glucose", "oxygen", "carbon dioxide"),
        (
            "Photosynthesis is the biochemical process by which chloroplast-containing organisms "
            "convert light energy into chemical energy stored as glucose. "
            "Overall equation: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂. "
            "The process has two stages: the light-dependent reactions (thylakoids) "
            "and the Calvin cycle (stroma)."
        ),
        "Campbell Biology, 12th Ed. — Chapter 10",
        0.97,
    ),
    (
        ("machine learning", "neural network", "deep learning", "gradient", "backpropagation"),
        (
            "Machine learning is a subset of AI that enables systems to learn from data "
            "without being explicitly programmed. Deep learning uses multi-layer neural networks "
            "trained via backpropagation and gradient descent to learn hierarchical representations."
        ),
        "Goodfellow et al., Deep Learning (MIT Press)",
        0.95,
    ),
    (
        ("quantum", "qubit", "superposition", "entanglement", "quantum computing"),
        (
            "Quantum computing leverages quantum mechanical phenomena—superposition and entanglement—"
            "to perform computations. A qubit can exist in states 0, 1, or both simultaneously. "
            "Quantum algorithms like Shor's (factoring) and Grover's (search) offer exponential speedups."
        ),
        "Nielsen & Chuang, Quantum Computation and Quantum Information",
        0.93,
    ),
    (
        ("climate change", "global warming", "greenhouse", "carbon", "emissions", "temperature"),
        (
            "Climate change refers to long-term shifts in global temperatures and weather patterns. "
            "Since the industrial revolution, human activities (burning fossil fuels, deforestation) "
            "have increased CO₂ levels from ~280 ppm to over 420 ppm, causing average global "
            "temperatures to rise ~1.1 °C above pre-industrial levels."
        ),
        "IPCC Sixth Assessment Report (AR6), 2021",
        0.96,
    ),
    (
        ("transformer", "attention", "bert", "gpt", "llm", "language model", "nlp"),
        (
            "Transformer models introduced the self-attention mechanism (Vaswani et al., 2017), "
            "replacing recurrence with parallelisable attention. BERT (bidirectional) pre-trains on "
            "masked language modelling; GPT (autoregressive) pre-trains on next-token prediction. "
            "Both form the foundation of modern large language models."
        ),
        "Vaswani et al., 'Attention Is All You Need' (NeurIPS 2017)",
        0.98,
    ),
]

_SYSTEM_PROMPT = """\
You are the Retriever Agent in a multi-agent AI reasoning system called OrionAI.

Given the user's query and planned steps, synthesise a concise, factually
grounded context summary that will help the Executor Agent answer accurately.

Rules:
- Draw on your knowledge to provide genuine, accurate context.
- Keep the context between 100–300 words.
- List 1–3 credible sources (real titles/authors if possible).
- Respond ONLY with this JSON structure:
  {
    "context": "...",
    "sources": [{"title": "...", "relevance": float_0_to_1}]
  }
"""


class RetrieverAgent(BaseAgent):
    """Fetches or synthesises context relevant to the user query."""

    name = "RetrieverAgent"

    async def run(  # noqa: D102
        self,
        *,
        query: str,
        steps: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> AgentResult:
        t0 = time.perf_counter()
        self.log.info("Retrieving context", query=query[:120])

        # ── Tier 1: keyword match in corpus ────────────────────────────────
        matched = _keyword_search(query)
        if matched:
            context_text, sources = matched
            latency_ms = (time.perf_counter() - t0) * 1000
            self.log.info(
                "Context retrieved from corpus",
                sources=[s["title"] for s in sources],
                latency_ms=round(latency_ms, 2),
            )
            return AgentResult(
                agent_name=self.name,
                payload={"context": context_text, "sources": sources},
                latency_ms=latency_ms,
            )

        # ── Tier 2: LLM synthesis ───────────────────────────────────────────
        steps_text = ""
        if steps:
            steps_text = "\n".join(f"  {s['id']}. {s['description']}" for s in steps)

        user_prompt = (
            f"Query: {query}\n"
            f"Planned steps:\n{steps_text or '(not provided)'}"
        )

        try:
            payload = await self.llm.chat_complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                session_id=self.session_id,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            self.log.info(
                "Context synthesised by LLM",
                context_chars=len(payload.get("context", "")),
                latency_ms=round(latency_ms, 2),
            )
            return AgentResult(
                agent_name=self.name,
                payload={
                    "context": payload.get("context", "No context available."),
                    "sources": payload.get("sources", []),
                },
                latency_ms=latency_ms,
            )

        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - t0) * 1000
            self.log.error("Retriever failed", error=str(exc))
            return AgentResult(
                agent_name=self.name,
                payload={"context": "Context unavailable.", "sources": []},
                success=False,
                error=str(exc),
                latency_ms=latency_ms,
            )


# ── Helpers ─────────────────────────────────────────────────────────────────

def _keyword_search(
    query: str,
) -> tuple[str, list[dict[str, Any]]] | None:
    """
    Scan the corpus for keyword overlap with the query.
    Returns (context, sources) if a sufficiently relevant entry is found.
    """
    query_lower = query.lower()
    best_score = 0
    best_entry = None

    for keywords, context, source_title, base_relevance in _CORPUS:
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > best_score:
            best_score = score
            best_entry = (context, source_title, base_relevance)

    if best_score >= 1 and best_entry:
        context, source_title, relevance = best_entry
        return context, [{"title": source_title, "relevance": relevance}]

    return None
