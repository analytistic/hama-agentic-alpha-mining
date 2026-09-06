"""Skill and memory components of the HAMA harness."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import FunctionTool, MemoryEntry, Skill, ToolCall, ToolRunResult


@dataclass(frozen=True)
class SkillLibrary:
    """Immutable skill snapshot exposed through progressive disclosure."""

    skills: tuple[Skill, ...]

    def __post_init__(self) -> None:
        identifiers = [skill.id for skill in self.skills]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("skill ids must be unique")

    def descriptions(self) -> list[dict[str, str]]:
        return [
            {"id": skill.id, "description": skill.description} for skill in self.skills
        ]

    def load(self, skill_id: str) -> Skill:
        for skill in self.skills:
            if skill.id == skill_id:
                return skill
        raise KeyError(f"unknown skill: {skill_id}")


@dataclass(frozen=True)
class MemoryBank:
    """Immutable memory snapshot retrieved by an explicit query action."""

    entries: tuple[MemoryEntry, ...] = ()

    def __post_init__(self) -> None:
        identifiers = [entry.id for entry in self.entries]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("memory entry ids must be unique")

    def retrieve(self, query: str, *, top_k: int = 3) -> tuple[MemoryEntry, ...]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_tokens = _tokens(query)
        ranked = sorted(
            self.entries,
            key=lambda entry: (
                _overlap(query_tokens, _tokens(f"{entry.key} {entry.value}")),
                entry.id,
            ),
            reverse=True,
        )
        return tuple(ranked[:top_k])


@dataclass(frozen=True)
class Harness:
    """The rollout-frozen harness Theta=(Omega,E)."""

    skills: SkillLibrary
    memory: MemoryBank


def create_load_skill_tool(
    library: SkillLibrary,
    loaded: dict[str, Skill],
) -> FunctionTool:
    """Expose descriptions while loading exactly one selected strategy."""

    descriptions = library.descriptions()

    def execute(call: ToolCall) -> ToolRunResult:
        skill = library.load(str(call.arguments["skill_id"]))
        if loaded:
            raise RuntimeError("one skill has already been loaded at this state")
        loaded[skill.id] = skill
        return ToolRunResult(
            call_id=call.id,
            name=call.name,
            output={
                "skill_id": skill.id,
                "description": skill.description,
                "strategy": skill.strategy,
            },
        )

    return FunctionTool(
        name="load_skill",
        description=(
            "Select one skill from its exposed description and load the hidden "
            "strategy into the current rollout context."
        ),
        parameters={
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "string",
                    "enum": [item["id"] for item in descriptions],
                }
            },
            "required": ["skill_id"],
            "additionalProperties": False,
        },
        execute=execute,
    )


def create_load_memory_tool(
    memory: MemoryBank,
    loaded: dict[str, object],
) -> FunctionTool:
    """Execute one query-parameterized memory-loading action."""

    def execute(call: ToolCall) -> ToolRunResult:
        if loaded:
            raise RuntimeError("memory has already been loaded at this state")
        query = str(call.arguments["query"])
        top_k = int(call.arguments.get("top_k", 3))
        entries = memory.retrieve(query, top_k=top_k)
        loaded["query"] = query
        loaded["entries"] = entries
        return ToolRunResult(
            call_id=call.id,
            name=call.name,
            output={
                "query": query,
                "entries": [
                    {
                        "id": entry.id,
                        "key": entry.key,
                        "factor_edit": entry.factor_edit,
                        "evaluation": dict(entry.evaluation),
                        "value": entry.value,
                    }
                    for entry in entries
                ],
            },
        )

    return FunctionTool(
        name="load_memory",
        description=(
            "Generate a query and retrieve outcome-grounded factor-edit cases "
            "from the external memory bank."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        execute=execute,
    )


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
