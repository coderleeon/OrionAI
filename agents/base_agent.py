"""
OrionAI — Base Agent
======================
Abstract base class that every agent in the pipeline must inherit from.

Responsibilities
----------------
* Enforce a consistent async ``run()`` contract.
* Provide shared LLM client instantiation.
* Bind per-agent structured logging.
* Define shared typed output envelope.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from utils.inference import LLMClient
from utils.logger import StructuredLogger


class AgentResult:
    """
    Lightweight container for an agent's output.

    Attributes
    ----------
    agent_name : str
        Which agent produced this result.
    payload : dict[str, Any]
        The agent-specific structured output.
    success : bool
        Whether the agent completed without error.
    error : str | None
        Error message if ``success`` is False.
    latency_ms : float
        Wall-clock time taken by this agent (milliseconds).
    """

    __slots__ = ("agent_name", "payload", "success", "error", "latency_ms")

    def __init__(
        self,
        *,
        agent_name: str,
        payload: dict[str, Any],
        success: bool = True,
        error: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> None:
        self.agent_name = agent_name
        self.payload = payload
        self.success = success
        self.error = error
        self.latency_ms = latency_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_name,
            "success": self.success,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            **self.payload,
        }


class BaseAgent(ABC):
    """
    Abstract base for all OrionAI agents.

    Subclasses must implement :meth:`run`.

    Parameters
    ----------
    session_id : str
        Used to scope logging / memory operations.
    """

    #: Subclasses set this to identify themselves in logs / results.
    name: str = "BaseAgent"

    def __init__(self, *, session_id: str = "") -> None:
        self.session_id = session_id
        self.llm = LLMClient(role=self.name.lower().replace("agent", "").strip())
        self.log = StructuredLogger(agent_name=self.name, session_id=session_id)

    @abstractmethod
    async def run(self, **kwargs: Any) -> AgentResult:
        """
        Execute the agent's primary task.

        All inputs are passed as keyword arguments to keep the signature
        flexible as the pipeline evolves.

        Returns
        -------
        AgentResult
            Always return an :class:`AgentResult` — never raise from here.
            Capture exceptions internally and set ``success=False``.
        """
        ...
