"""Advantage-conditioned semantic gradients and atomic harness updates."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Callable

from .attribution import (
    ComponentAttribution,
    ComponentAttributor,
    InvocationKind,
    InvocationRef,
    attribution_payload,
)
from .advantage import AdvantageBatch
from .harness import Harness, MemoryBank, SkillLibrary
from .types import CompletionModel, MemoryEntry, Message, Skill, ToolSchema


class HarnessComponent(StrEnum):
    SKILL = "skill"
    MEMORY = "memory"
    SKILL_DESCRIPTION = "skill_description"
    SKILL_STRATEGY = "skill_strategy"
    MEMORY_KEY = "memory_key"
    MEMORY_VALUE = "memory_value"


@dataclass(frozen=True, order=True)
class ParameterRef:
    """One optimizable natural-language parameter in Theta=(Omega,E)."""

    component: HarnessComponent
    item_id: str


@dataclass(frozen=True)
class AggregatedSemanticGradient:
    """One cross-rollout gradient for one harness parameter."""

    parameter: ParameterRef
    feedback: str
    rationale: str
    # The number of independent rollout-level conclusions, not raw segments.
    source_count: int
    segment_count: int
    total_weight: float


@dataclass(frozen=True)
class ParameterEdit:
    """One atomic natural-language parameter update."""

    parameter: ParameterRef
    before: str
    after: str


@dataclass(frozen=True)
class AtomicGradientProposal:
    """One scored, single-parameter semantic revision direction."""

    component: InvocationRef
    parameter: ParameterRef
    rationale: str
    feedback: str
    support_ids: tuple[str, ...]
    conflict_ids: tuple[str, ...]
    score: float


@dataclass(frozen=True)
class HarnessOptimizationResult:
    """Artifacts produced by the prefill-attributed optimizer."""

    harness: Harness
    attributions: tuple[ComponentAttribution, ...]
    proposals: tuple[AtomicGradientProposal, ...]
    selected_gradients: tuple[AggregatedSemanticGradient, ...]
    edits: tuple[ParameterEdit, ...]


class ConflictAwareSemanticGradientEngine:
    """Generate and score atomic gradients from attributed shared evidence."""

    def __init__(
        self,
        model: CompletionModel,
        *,
        conflict_penalty: float = 1.0,
        max_correction_evidence: int = 6,
        max_preservation_evidence: int = 3,
        min_score: float = 0.0,
    ) -> None:
        if conflict_penalty < 0.0:
            raise ValueError("conflict_penalty must be non-negative")
        if max_correction_evidence < 1 or max_preservation_evidence < 1:
            raise ValueError("evidence limits must be positive")
        self.model = model
        self.conflict_penalty = conflict_penalty
        self.max_correction_evidence = max_correction_evidence
        self.max_preservation_evidence = max_preservation_evidence
        self.min_score = min_score

    def generate(
        self,
        harness: Harness,
        attributions: Sequence[ComponentAttribution],
    ) -> tuple[AtomicGradientProposal, ...]:
        proposals: list[AtomicGradientProposal] = []
        for component, evidence in ComponentAttributor.group(attributions).items():
            correction = sorted(
                (item for item in evidence if item.correction_weight > 0.0),
                key=lambda item: item.correction_weight,
                reverse=True,
            )[: self.max_correction_evidence]
            if not correction:
                continue
            preservation = sorted(
                (item for item in evidence if item.preservation_weight > 0.0),
                key=lambda item: item.preservation_weight,
                reverse=True,
            )[: self.max_preservation_evidence]
            allowed_parameters = _component_parameters(component)
            payload = _structured_completion(
                self.model,
                system=(
                    "You are HAMA's semantic-gradient engine. Propose a small "
                    "set of atomic revisions for one shared harness component. "
                    "Each proposal must modify exactly one allowed text field. "
                    "Use correction evidence to diagnose a recurring failure and "
                    "preservation evidence to avoid destroying successful behavior. "
                    "Return revision directions, not rewritten parameter text. "
                    "For every proposal, cite the evidence IDs it resolves and "
                    "the positive evidence IDs with which it conflicts."
                ),
                prompt=json.dumps(
                    {
                        "component": {
                            "kind": component.kind.value,
                            "item_id": component.item_id,
                        },
                        "allowed_parameters": [
                            {
                                "component": parameter.component.value,
                                "item_id": parameter.item_id,
                                "current_value": _get_parameter(harness, parameter),
                            }
                            for parameter in allowed_parameters
                        ],
                        "correction_evidence": [
                            attribution_payload(item) for item in correction
                        ],
                        "preservation_evidence": [
                            attribution_payload(item) for item in preservation
                        ],
                    },
                    ensure_ascii=False,
                    default=str,
                ),
                tool_name="submit_atomic_gradients",
                tool_description="Return atomic semantic-gradient proposals.",
                parameters={
                    "type": "object",
                    "properties": {
                        "proposals": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "component": {
                                        "type": "string",
                                        "enum": [
                                            item.value for item in HarnessComponent
                                        ],
                                    },
                                    "item_id": {"type": "string"},
                                    "rationale": {"type": "string"},
                                    "feedback": {"type": "string"},
                                    "support_ids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "conflict_ids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": [
                                    "component",
                                    "item_id",
                                    "rationale",
                                    "feedback",
                                    "support_ids",
                                    "conflict_ids",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["proposals"],
                    "additionalProperties": False,
                },
            )
            raw_proposals = payload.get("proposals")
            if not isinstance(raw_proposals, list):
                raise TypeError("semantic-gradient engine returned no proposals list")
            correction_by_id = {item.evidence_id: item for item in correction}
            preservation_by_id = {item.evidence_id: item for item in preservation}
            for raw in raw_proposals:
                if not isinstance(raw, dict):
                    raise TypeError("each semantic-gradient proposal must be an object")
                parameter = ParameterRef(
                    HarnessComponent(str(raw["component"])),
                    str(raw["item_id"]),
                )
                if parameter not in allowed_parameters:
                    continue
                support_ids = tuple(
                    item_id
                    for item_id in dict.fromkeys(map(str, raw["support_ids"]))
                    if item_id in correction_by_id
                )
                conflict_ids = tuple(
                    item_id
                    for item_id in dict.fromkeys(map(str, raw["conflict_ids"]))
                    if item_id in preservation_by_id
                )
                if not support_ids:
                    continue
                score = sum(
                    correction_by_id[item_id].correction_weight
                    for item_id in support_ids
                ) - self.conflict_penalty * sum(
                    preservation_by_id[item_id].preservation_weight
                    for item_id in conflict_ids
                )
                rationale = str(raw["rationale"]).strip()
                feedback = str(raw["feedback"]).strip()
                if not rationale or not feedback:
                    raise ValueError(
                        "proposal rationale and feedback must not be empty"
                    )
                proposals.append(
                    AtomicGradientProposal(
                        component=component,
                        parameter=parameter,
                        rationale=rationale,
                        feedback=feedback,
                        support_ids=support_ids,
                        conflict_ids=conflict_ids,
                        score=score,
                    )
                )
        return tuple(proposals)

    def select(
        self,
        proposals: Sequence[AtomicGradientProposal],
    ) -> tuple[AggregatedSemanticGradient, ...]:
        """Keep one positive, highest-scoring atomic update per component."""

        grouped: dict[InvocationRef, list[AtomicGradientProposal]] = defaultdict(list)
        for proposal in proposals:
            grouped[proposal.component].append(proposal)
        selected: list[AggregatedSemanticGradient] = []
        for component in sorted(grouped):
            best = max(grouped[component], key=lambda item: item.score)
            if best.score <= self.min_score:
                continue
            selected.append(
                AggregatedSemanticGradient(
                    parameter=best.parameter,
                    feedback=best.feedback,
                    rationale=best.rationale,
                    source_count=len(best.support_ids),
                    segment_count=len(best.support_ids) + len(best.conflict_ids),
                    total_weight=best.score,
                )
            )
        return tuple(selected)


class HarnessEvolutionOptimizer:
    """Evolve both the contents and cardinality of skill and memory libraries."""

    def __init__(
        self,
        model: CompletionModel,
        history_provider: Callable[[], str] | None = None,
        *,
        max_changes: int = 4,
    ) -> None:
        self.model = model
        self.history_provider = history_provider
        self.max_changes = max_changes

    def update(
        self,
        harness: Harness,
        gradients: Sequence[AggregatedSemanticGradient],
        *,
        batch: AdvantageBatch,
        attributions: Sequence[ComponentAttribution],
    ) -> tuple[Harness, tuple[ParameterEdit, ...]]:
        payload = _structured_completion(
            self.model,
            system=(
                "You are HAMA's library-evolution optimizer. Improve the harness "
                "from grounded rollout evidence. Update an existing text field "
                "for a local correction; create a new skill when a distinct, "
                "reusable procedure should be routed separately; create a memory "
                "for a concrete factor edit and observed evaluation; remove only "
                "a clearly harmful or duplicate item. Use concise snake_case IDs. "
                f"Return at most {self.max_changes} atomic changes."
            ),
            prompt=json.dumps(
                {
                    "current_harness": _harness_payload(harness),
                    "semantic_gradients": [
                        {
                            "component": item.parameter.component.value,
                            "item_id": item.parameter.item_id,
                            "feedback": item.feedback,
                            "rationale": item.rationale,
                            "weight": item.total_weight,
                        }
                        for item in gradients
                    ],
                    "trajectory": [
                        {
                            "step": record.step_index,
                            "skill": record.trace.skill.id,
                            "memory": [entry.id for entry in record.trace.retrieved_memory],
                            "query": record.trace.memory_query,
                            "action": str(record.trace.action),
                            "reward": record.trace.reward,
                            "advantage": record.advantage,
                        }
                        for record in batch.records
                    ],
                    "component_credit": [attribution_payload(item) for item in attributions],
                    "recent_git_history": (
                        self.history_provider() if self.history_provider else ""
                    ),
                },
                ensure_ascii=False,
                default=str,
            ),
            tool_name="submit_harness_changes",
            tool_description="Return atomic library evolution operations.",
            parameters=_evolution_schema(self.max_changes),
        )
        changes = payload.get("changes", [])
        if not isinstance(changes, list):
            raise TypeError("harness changes must be a list")
        return _apply_library_changes(harness, changes[: self.max_changes])


def optimize_harness_with_attribution(
    *,
    harness: Harness,
    batch: AdvantageBatch,
    attributor: ComponentAttributor,
    semantic_engine: ConflictAwareSemanticGradientEngine,
    edit_optimizer: HarnessEvolutionOptimizer,
) -> HarnessOptimizationResult:
    """Run prefill attribution, conflict-aware aggregation, and atomic edits."""

    attributions = attributor.attribute(batch)
    proposals = semantic_engine.generate(harness, attributions)
    selected = semantic_engine.select(proposals)
    updated, edits = edit_optimizer.update(
        harness, selected, batch=batch, attributions=attributions
    )
    return HarnessOptimizationResult(
        harness=updated,
        attributions=attributions,
        proposals=proposals,
        selected_gradients=selected,
        edits=edits,
    )


def _harness_payload(harness: Harness) -> dict[str, Any]:
    return {
        "skills": [
            {"id": item.id, "description": item.description, "strategy": item.strategy}
            for item in harness.skills.skills
        ],
        "memory": [
            {
                "id": item.id,
                "key": item.key,
                "factor_edit": item.factor_edit,
                "evaluation": dict(item.evaluation),
                "value": item.value,
            }
            for item in harness.memory.entries
        ],
    }


def _evolution_schema(max_changes: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "changes": {
                "type": "array",
                "maxItems": max_changes,
                "items": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["update_parameter", "upsert_skill", "upsert_memory", "remove_skill", "remove_memory"],
                        },
                        "component": {"type": "string"},
                        "item_id": {"type": "string"},
                        "updated_text": {"type": "string"},
                        "description": {"type": "string"},
                        "strategy": {"type": "string"},
                        "key": {"type": "string"},
                        "factor_edit": {"type": "string"},
                        "evaluation": {"type": "object"},
                        "value": {"type": "string"},
                    },
                    "required": ["operation", "item_id"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["changes"],
        "additionalProperties": False,
    }


def _apply_library_changes(
    harness: Harness, changes: Sequence[Mapping[str, Any]]
) -> tuple[Harness, tuple[ParameterEdit, ...]]:
    skills = {item.id: item for item in harness.skills.skills}
    memory = {item.id: item for item in harness.memory.entries}
    edits: list[ParameterEdit] = []
    for change in changes:
        operation, item_id = str(change["operation"]), str(change["item_id"]).strip()
        if not item_id:
            continue
        if operation == "update_parameter":
            parameter = ParameterRef(HarnessComponent(str(change["component"])), item_id)
            current = Harness(SkillLibrary(tuple(skills.values())), MemoryBank(tuple(memory.values())))
            before = _get_parameter(current, parameter)
            after = str(change.get("updated_text", "")).strip()
            current = _set_parameter(current, parameter, after)
            skills = {item.id: item for item in current.skills.skills}
            memory = {item.id: item for item in current.memory.entries}
            edits.append(ParameterEdit(parameter, before, after))
        elif operation == "upsert_skill":
            after_skill = Skill(item_id, str(change.get("description", "")).strip(), str(change.get("strategy", "")).strip())
            if not after_skill.description or not after_skill.strategy:
                continue
            before = json.dumps(skills[item_id].__dict__, ensure_ascii=False) if item_id in skills else ""
            skills[item_id] = after_skill
            edits.append(ParameterEdit(ParameterRef(HarnessComponent.SKILL, item_id), before, json.dumps(after_skill.__dict__, ensure_ascii=False)))
        elif operation == "upsert_memory":
            after_memory = MemoryEntry(item_id, str(change.get("key", "")).strip(), str(change.get("factor_edit", "")).strip(), change.get("evaluation", {}), str(change.get("value", "")).strip())
            if not after_memory.key or not after_memory.value:
                continue
            before = json.dumps(memory[item_id].__dict__, ensure_ascii=False, default=str) if item_id in memory else ""
            memory[item_id] = after_memory
            edits.append(ParameterEdit(ParameterRef(HarnessComponent.MEMORY, item_id), before, json.dumps(after_memory.__dict__, ensure_ascii=False, default=str)))
        elif operation == "remove_skill" and item_id in skills and len(skills) > 1:
            before = json.dumps(skills.pop(item_id).__dict__, ensure_ascii=False)
            edits.append(ParameterEdit(ParameterRef(HarnessComponent.SKILL, item_id), before, ""))
        elif operation == "remove_memory" and item_id in memory:
            before = json.dumps(memory.pop(item_id).__dict__, ensure_ascii=False, default=str)
            edits.append(ParameterEdit(ParameterRef(HarnessComponent.MEMORY, item_id), before, ""))
    return Harness(SkillLibrary(tuple(skills.values())), MemoryBank(tuple(memory.values()))), tuple(edits)




def _component_parameters(component: InvocationRef) -> tuple[ParameterRef, ...]:
    if component.kind == InvocationKind.SKILL:
        return (
            ParameterRef(HarnessComponent.SKILL_DESCRIPTION, component.item_id),
            ParameterRef(HarnessComponent.SKILL_STRATEGY, component.item_id),
        )
    if component.kind == InvocationKind.MEMORY:
        return (
            ParameterRef(HarnessComponent.MEMORY_KEY, component.item_id),
            ParameterRef(HarnessComponent.MEMORY_VALUE, component.item_id),
        )
    raise ValueError(f"unknown invocation kind: {component.kind}")




def _get_parameter(harness: Harness, parameter: ParameterRef) -> str:
    if parameter.component in {
        HarnessComponent.SKILL_DESCRIPTION,
        HarnessComponent.SKILL_STRATEGY,
    }:
        skill = harness.skills.load(parameter.item_id)
        return (
            skill.description
            if parameter.component == HarnessComponent.SKILL_DESCRIPTION
            else skill.strategy
        )
    entry = _memory_entry(harness, parameter.item_id)
    return (
        entry.key if parameter.component == HarnessComponent.MEMORY_KEY else entry.value
    )


def _set_parameter(
    harness: Harness,
    parameter: ParameterRef,
    value: str,
) -> Harness:
    if parameter.component in {
        HarnessComponent.SKILL_DESCRIPTION,
        HarnessComponent.SKILL_STRATEGY,
    }:
        skills: list[Skill] = []
        found = False
        for skill in harness.skills.skills:
            if skill.id != parameter.item_id:
                skills.append(skill)
                continue
            found = True
            skills.append(
                replace(
                    skill,
                    description=(
                        value
                        if parameter.component == HarnessComponent.SKILL_DESCRIPTION
                        else skill.description
                    ),
                    strategy=(
                        value
                        if parameter.component == HarnessComponent.SKILL_STRATEGY
                        else skill.strategy
                    ),
                )
            )
        if not found:
            raise KeyError(f"unknown skill: {parameter.item_id}")
        return replace(harness, skills=SkillLibrary(tuple(skills)))

    entries: list[MemoryEntry] = []
    found = False
    for entry in harness.memory.entries:
        if entry.id != parameter.item_id:
            entries.append(entry)
            continue
        found = True
        entries.append(
            replace(
                entry,
                key=(
                    value
                    if parameter.component == HarnessComponent.MEMORY_KEY
                    else entry.key
                ),
                value=(
                    value
                    if parameter.component == HarnessComponent.MEMORY_VALUE
                    else entry.value
                ),
            )
        )
    if not found:
        raise KeyError(f"unknown memory entry: {parameter.item_id}")
    return replace(harness, memory=MemoryBank(tuple(entries)))


def _memory_entry(harness: Harness, item_id: str) -> MemoryEntry:
    for entry in harness.memory.entries:
        if entry.id == item_id:
            return entry
    raise KeyError(f"unknown memory entry: {item_id}")


def _structured_completion(
    model: CompletionModel,
    *,
    system: str,
    prompt: str,
    tool_name: str,
    tool_description: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    schema: ToolSchema = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": dict(parameters),
        },
    }
    response = model.complete(
        [Message(role="system", content=system), Message(role="user", content=prompt)],
        [schema],
    )
    for call in response.tool_calls:
        if call.name == tool_name:
            return dict(call.arguments)
    if response.content.strip():
        try:
            value = json.loads(response.content)
        except json.JSONDecodeError as error:
            raise ValueError(f"model did not call {tool_name}") from error
        if isinstance(value, dict):
            return value
    raise ValueError(f"model did not call {tool_name}")
