"""Rollout entry point: one rollout is one bounded agentic loop."""

from __future__ import annotations

from typing import Any

from .agent import Agent
from .environment import AlphaMiningEnvironment
from .types import AgentRun


def rollout(
    task: Any,
    environment: AlphaMiningEnvironment,
    agent: Agent,
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentRun:
    """Run one paper-defined agentic loop and produce at most one transition."""

    return agent.run(
        task,
        environment,
        metadata=metadata,
    )
