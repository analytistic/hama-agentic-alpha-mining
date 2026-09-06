"""Runtime records and structural protocols for one HAMA rollout."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

ToolSchema = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """One OpenAI-compatible function call emitted by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Message:
    """One canonical message in an agentic-loop transcript."""

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    output: Any = None
    error: str | None = None
    reasoning_content: str = ""
    usage: dict[str, int] | None = None
    finish_reason: str | None = None
    raw: Any = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class ToolRunResult:
    """Normalized feedback returned by a tool or market environment."""

    call_id: str
    name: str
    output: Any = None
    error: str | None = None
    terminated: bool = False
    stop_reason: str | None = None


class Tool(Protocol):
    """Structural interface for tools owned by the HAMA runtime."""

    name: str

    @property
    def schema(self) -> ToolSchema: ...

    def execute(self, call: ToolCall) -> ToolRunResult: ...


@dataclass(frozen=True)
class FunctionTool:
    """A concrete OpenAI function tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[[ToolCall], ToolRunResult]

    @property
    def schema(self) -> ToolSchema:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class CompletionModel(Protocol):
    """A model capable of one OpenAI-compatible completion turn."""

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] = (),
    ) -> Message: ...


@dataclass(frozen=True)
class Factor:
    """One executable alpha factor represented by a Qlib expression."""

    name: str
    expression: str


FactorPool = tuple[Factor, ...]


@dataclass(frozen=True)
class MarketState:
    """The state observed at the beginning of one agentic loop."""

    factor_pool: FactorPool
    market_information: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Skill:
    """One progressively disclosed skill omega_i=(d_i, sigma_i)."""

    id: str
    description: str
    strategy: str


@dataclass(frozen=True)
class MemoryEntry:
    """One memory e_j=(k_j, epsilon_j, v_j)."""

    id: str
    key: str
    factor_edit: str
    evaluation: Mapping[str, Any]
    value: str


class FactorEditOperation(StrEnum):
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"


@dataclass(frozen=True)
class FactorEdit:
    """A single atomic edit of the current factor pool."""

    operation: FactorEditOperation
    factor: Factor | None = None
    target: str | None = None


@dataclass(frozen=True)
class Transition:
    """The environment transition produced by the final factor edit."""

    state: MarketState
    action: FactorEdit
    next_state: MarketState
    reward: float
    accepted: bool
    previous_score: float
    next_score: float
    terminated: bool
    redundancy: float | None = None
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HarnessTrace:
    """One realized dependency path z_t in a HAMA trajectory."""

    index: int
    state: MarketState
    skill: Skill
    memory_query: str
    retrieved_memory: tuple[MemoryEntry, ...]
    action: FactorEdit
    next_state: MarketState
    reward: float
    transition: Transition
    interaction: tuple[Message, ...] = ()


@dataclass
class AgentRun:
    """The complete record of exactly one agentic-loop rollout."""

    messages: list[Message]
    stop_reason: str
    steps: int
    answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    loaded_skill: Skill | None = None
    memory_query: str | None = None
    retrieved_memory: tuple[MemoryEntry, ...] = ()
    transition: Transition | None = None
    traces: tuple[HarnessTrace, ...] = ()

    @property
    def usage(self) -> dict[str, int]:
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for message in self.messages:
            if message.usage is None:
                continue
            for key in totals:
                totals[key] += int(message.usage.get(key, 0))
        return totals
