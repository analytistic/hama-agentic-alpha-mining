"""Advantage-conditioned semantic gradients and atomic harness updates."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from statistics import fmean
from typing import Any, Callable

from .attribution import (
    ComponentAttribution,
    ComponentAttributor,
    InvocationKind,
    InvocationRef,
    attribution_payload,
)
from .advantage import AdvantageBatch, AdvantageRecord
from .harness import Harness, MemoryBank, SkillLibrary
from .types import CompletionModel, MemoryEntry, Message, Skill, ToolSchema


class HarnessComponent(StrEnum):
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
class SemanticGradient:
    """One trace-attributed natural-language gradient."""

    parameter: ParameterRef
    diagnosis: str
    feedback: str
    advantage: float
    rollout_index: int
    step_index: int

    @property
    def weight(self) -> float:
        return abs(self.advantage)


@dataclass(frozen=True)
class TrajectorySemanticGradient:
    """One parameter-level conclusion formed within a single rollout."""

    rollout_index: int
    parameter: ParameterRef
    feedback: str
    rationale: str
    step_indices: tuple[int, ...]
    source_count: int
    total_weight: float
    mean_advantage: float


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
class HarnessOptimizationResult:
    harness: Harness
    gradients: tuple[SemanticGradient, ...]
    trajectory_gradients: tuple[TrajectorySemanticGradient, ...]
    aggregated_gradients: tuple[AggregatedSemanticGradient, ...]
    edits: tuple[ParameterEdit, ...]


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
class AttributedHarnessOptimizationResult:
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


class SemanticGradientEngine:
    """Generate segment gradients, then aggregate trajectory-first."""

    def __init__(
        self,
        model: CompletionModel,
        *,
        min_abs_advantage: float = 1e-6,
    ) -> None:
        if min_abs_advantage < 0.0:
            raise ValueError("min_abs_advantage must be non-negative")
        self.model = model
        self.min_abs_advantage = min_abs_advantage

    def generate(self, record: AdvantageRecord) -> tuple[SemanticGradient, ...]:
        """Generate gradients only for parameters on the realized trace."""

        if abs(record.advantage) < self.min_abs_advantage:
            return ()
        allowed = _trace_parameters(record)
        payload = _structured_completion(
            self.model,
            system=(
                "You are the semantic-gradient engine for HAMA. Trace the "
                "realized outcome backward and return concise, actionable "
                "natural-language gradients only for responsible parameters. "
                "Attribute skill-selection errors to skill descriptions; "
                "execution or query-construction errors to skill strategies; "
                "retrieval mismatches to memory keys; and inaccurate empirical "
                "guidance to memory values. The query and retrieved result are "
                "intermediate variables, never optimization parameters. A "
                "negative advantage calls for correction; a positive advantage "
                "calls for preserving and sharpening what caused success."
            ),
            prompt=json.dumps(
                {
                    "trace": _record_payload(record),
                    "allowed_parameters": [
                        {
                            "component": parameter.component.value,
                            "item_id": parameter.item_id,
                            "current_value": _trace_parameter_value(record, parameter),
                        }
                        for parameter in allowed
                    ],
                },
                ensure_ascii=False,
                default=str,
            ),
            tool_name="submit_semantic_gradients",
            tool_description="Return trace-attributed semantic gradients.",
            parameters={
                "type": "object",
                "properties": {
                    "gradients": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "component": {
                                    "type": "string",
                                    "enum": [item.value for item in HarnessComponent],
                                },
                                "item_id": {"type": "string"},
                                "diagnosis": {"type": "string"},
                                "feedback": {"type": "string"},
                            },
                            "required": [
                                "component",
                                "item_id",
                                "diagnosis",
                                "feedback",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["gradients"],
                "additionalProperties": False,
            },
        )
        raw_gradients = payload.get("gradients")
        if not isinstance(raw_gradients, list):
            raise TypeError("semantic-gradient engine returned no gradients list")

        allowed_set = set(allowed)
        seen: set[ParameterRef] = set()
        gradients: list[SemanticGradient] = []
        for item in raw_gradients:
            if not isinstance(item, dict):
                raise TypeError("each semantic gradient must be an object")
            parameter = ParameterRef(
                HarnessComponent(str(item["component"])),
                str(item["item_id"]),
            )
            if parameter not in allowed_set:
                raise ValueError(f"gradient targets an untraced parameter: {parameter}")
            if parameter in seen:
                raise ValueError(f"duplicate gradient for parameter: {parameter}")
            diagnosis = str(item["diagnosis"]).strip()
            feedback = str(item["feedback"]).strip()
            if not diagnosis or not feedback:
                raise ValueError("semantic diagnosis and feedback must not be empty")
            seen.add(parameter)
            gradients.append(
                SemanticGradient(
                    parameter=parameter,
                    diagnosis=diagnosis,
                    feedback=feedback,
                    advantage=record.advantage,
                    rollout_index=record.rollout_index,
                    step_index=record.step_index,
                )
            )
        return tuple(gradients)

    def generate_batch(
        self,
        batch: AdvantageBatch,
    ) -> tuple[SemanticGradient, ...]:
        return tuple(
            gradient for record in batch.records for gradient in self.generate(record)
        )

    def aggregate_trajectories(
        self,
        gradients: Sequence[SemanticGradient],
    ) -> tuple[TrajectorySemanticGradient, ...]:
        """Consolidate ordered segments inside each rollout before grouping."""

        grouped: dict[int, list[SemanticGradient]] = defaultdict(list)
        for gradient in gradients:
            grouped[gradient.rollout_index].append(gradient)

        results: list[TrajectorySemanticGradient] = []
        for rollout_index in sorted(grouped):
            items = sorted(
                grouped[rollout_index],
                key=lambda item: (item.step_index, item.parameter),
            )
            if len(items) == 1:
                results.append(_trajectory_gradient(items[0].parameter, items))
                continue

            payload = _structured_completion(
                self.model,
                system=(
                    "You are HAMA's within-trajectory semantic-gradient "
                    "aggregator. Read all ordered segment gradients from exactly "
                    "one rollout as correlated evidence. Resolve earlier and "
                    "later outcomes, avoid counting repeated appearances as "
                    "independent votes, and return at most one conclusion per "
                    "affected parameter. Omit parameters with no net actionable "
                    "evidence. Do not edit parameter text."
                ),
                prompt=json.dumps(
                    {
                        "rollout_index": rollout_index,
                        "ordered_segment_gradients": [
                            {
                                "step_index": item.step_index,
                                "parameter": {
                                    "component": item.parameter.component.value,
                                    "item_id": item.parameter.item_id,
                                },
                                "diagnosis": item.diagnosis,
                                "feedback": item.feedback,
                                "advantage": item.advantage,
                                "weight": item.weight,
                            }
                            for item in items
                        ],
                    },
                    ensure_ascii=False,
                ),
                tool_name="submit_trajectory_gradients",
                tool_description=(
                    "Return rollout-level semantic gradients after consolidating "
                    "all ordered segments."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "gradients": {
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
                                },
                                "required": [
                                    "component",
                                    "item_id",
                                    "rationale",
                                    "feedback",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["gradients"],
                    "additionalProperties": False,
                },
            )
            raw_gradients = payload.get("gradients")
            if not isinstance(raw_gradients, list):
                raise TypeError("trajectory aggregator returned no gradients list")
            allowed = {item.parameter for item in items}
            seen: set[ParameterRef] = set()
            for raw in raw_gradients:
                if not isinstance(raw, dict):
                    raise TypeError("each trajectory gradient must be an object")
                parameter = ParameterRef(
                    HarnessComponent(str(raw["component"])),
                    str(raw["item_id"]),
                )
                if parameter not in allowed:
                    raise ValueError(
                        f"trajectory gradient targets an untraced parameter: {parameter}"
                    )
                if parameter in seen:
                    raise ValueError(
                        f"duplicate trajectory gradient for parameter: {parameter}"
                    )
                seen.add(parameter)
                feedback = str(raw["feedback"]).strip()
                rationale = str(raw["rationale"]).strip()
                if not feedback or not rationale:
                    raise ValueError("trajectory gradient must not be empty")
                parameter_items = [
                    item for item in items if item.parameter == parameter
                ]
                results.append(
                    _trajectory_gradient(
                        parameter,
                        parameter_items,
                        feedback=feedback,
                        rationale=rationale,
                    )
                )
        return tuple(results)

    def aggregate_rollouts(
        self,
        gradients: Sequence[TrajectorySemanticGradient],
    ) -> tuple[AggregatedSemanticGradient, ...]:
        """Aggregate one conclusion per rollout across independent rollouts."""

        grouped: dict[ParameterRef, list[TrajectorySemanticGradient]] = defaultdict(
            list
        )
        for gradient in gradients:
            grouped[gradient.parameter].append(gradient)

        results: list[AggregatedSemanticGradient] = []
        for parameter in sorted(grouped):
            items = sorted(grouped[parameter], key=lambda item: item.rollout_index)
            rollout_indices = [item.rollout_index for item in items]
            if len(rollout_indices) != len(set(rollout_indices)):
                raise ValueError(
                    f"multiple trajectory conclusions for one rollout: {parameter}"
                )
            total_weight = sum(item.total_weight for item in items)
            segment_count = sum(item.source_count for item in items)
            if len(items) == 1:
                results.append(
                    AggregatedSemanticGradient(
                        parameter=parameter,
                        feedback=items[0].feedback,
                        rationale=items[0].rationale,
                        source_count=1,
                        segment_count=segment_count,
                        total_weight=total_weight,
                    )
                )
                continue

            payload = _structured_completion(
                self.model,
                system=(
                    "You are HAMA's cross-rollout semantic-gradient aggregator. "
                    "Each input is already one consolidated conclusion from an "
                    "independent rollout. Treat each rollout as one evidence "
                    "unit, resolve agreement and conflict using its signed mean "
                    "advantage, and emit one concise parameter gradient. Do not "
                    "recount its underlying segments as independent votes and "
                    "do not edit parameter text."
                ),
                prompt=json.dumps(
                    {
                        "parameter": {
                            "component": parameter.component.value,
                            "item_id": parameter.item_id,
                        },
                        "trajectory_gradients": [
                            {
                                "rollout_index": item.rollout_index,
                                "rationale": item.rationale,
                                "feedback": item.feedback,
                                "mean_advantage": item.mean_advantage,
                                "evidence_weight": item.total_weight,
                                "step_indices": item.step_indices,
                            }
                            for item in items
                        ],
                    },
                    ensure_ascii=False,
                ),
                tool_name="submit_group_gradient",
                tool_description=(
                    "Return one gradient consolidated across independent rollouts."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "rationale": {"type": "string"},
                        "feedback": {"type": "string"},
                    },
                    "required": ["rationale", "feedback"],
                    "additionalProperties": False,
                },
            )
            feedback = str(payload["feedback"]).strip()
            rationale = str(payload["rationale"]).strip()
            if not feedback or not rationale:
                raise ValueError("group semantic gradient must not be empty")
            results.append(
                AggregatedSemanticGradient(
                    parameter=parameter,
                    feedback=feedback,
                    rationale=rationale,
                    source_count=len(items),
                    segment_count=segment_count,
                    total_weight=total_weight,
                )
            )
        return tuple(results)


class HarnessEditOptimizer:
    """Apply aggregated gradients as one atomic edit per parameter."""

    def __init__(
        self,
        model: CompletionModel,
        history_provider: Callable[[], str] | None = None,
    ) -> None:
        self.model = model
        self.history_provider = history_provider

    def update(
        self,
        harness: Harness,
        gradients: Sequence[AggregatedSemanticGradient],
    ) -> tuple[Harness, tuple[ParameterEdit, ...]]:
        updated = harness
        edits: list[ParameterEdit] = []
        seen: set[ParameterRef] = set()
        for gradient in gradients:
            if gradient.parameter in seen:
                raise ValueError(
                    f"optimizer received duplicate parameter: {gradient.parameter}"
                )
            seen.add(gradient.parameter)
            before = _get_parameter(updated, gradient.parameter)
            payload = _structured_completion(
                self.model,
                system=(
                    "You are the HAMA edit-optimizer engine. Apply the supplied "
                    "aggregated semantic gradient to exactly one current text "
                    "parameter. Preserve useful content, make the smallest "
                    "coherent update, and return only the complete replacement "
                    "text for that parameter."
                ),
                prompt=json.dumps(
                    {
                        "parameter": {
                            "component": gradient.parameter.component.value,
                            "item_id": gradient.parameter.item_id,
                        },
                        "current_value": before,
                        "aggregated_gradient": gradient.feedback,
                        "rationale": gradient.rationale,
                        "recent_edit_history": (
                            self.history_provider() if self.history_provider else ""
                        ),
                    },
                    ensure_ascii=False,
                ),
                tool_name="submit_parameter_edit",
                tool_description="Return the complete updated parameter text.",
                parameters={
                    "type": "object",
                    "properties": {"updated_text": {"type": "string"}},
                    "required": ["updated_text"],
                    "additionalProperties": False,
                },
            )
            after = str(payload["updated_text"]).strip()
            if not after:
                raise ValueError("updated harness parameter must not be empty")
            updated = _set_parameter(updated, gradient.parameter, after)
            edits.append(ParameterEdit(gradient.parameter, before, after))
        return updated, tuple(edits)


def optimize_harness(
    *,
    harness: Harness,
    batch: AdvantageBatch,
    semantic_engine: SemanticGradientEngine,
    edit_optimizer: HarnessEditOptimizer,
) -> HarnessOptimizationResult:
    gradients = semantic_engine.generate_batch(batch)
    trajectory_gradients = semantic_engine.aggregate_trajectories(gradients)
    aggregated = semantic_engine.aggregate_rollouts(trajectory_gradients)
    updated, edits = edit_optimizer.update(harness, aggregated)
    return HarnessOptimizationResult(
        updated,
        gradients,
        trajectory_gradients,
        aggregated,
        edits,
    )


def optimize_harness_with_attribution(
    *,
    harness: Harness,
    batch: AdvantageBatch,
    attributor: ComponentAttributor,
    semantic_engine: ConflictAwareSemanticGradientEngine,
    edit_optimizer: HarnessEditOptimizer,
) -> AttributedHarnessOptimizationResult:
    """Run prefill attribution, conflict-aware aggregation, and atomic edits."""

    attributions = attributor.attribute(batch)
    proposals = semantic_engine.generate(harness, attributions)
    selected = semantic_engine.select(proposals)
    updated, edits = edit_optimizer.update(harness, selected)
    return AttributedHarnessOptimizationResult(
        harness=updated,
        attributions=attributions,
        proposals=proposals,
        selected_gradients=selected,
        edits=edits,
    )


def _trajectory_gradient(
    parameter: ParameterRef,
    items: Sequence[SemanticGradient],
    *,
    feedback: str | None = None,
    rationale: str | None = None,
) -> TrajectorySemanticGradient:
    if not items:
        raise ValueError("trajectory gradient requires at least one segment")
    rollout_indices = {item.rollout_index for item in items}
    if len(rollout_indices) != 1:
        raise ValueError("trajectory gradient cannot mix rollouts")
    if any(item.parameter != parameter for item in items):
        raise ValueError("trajectory gradient cannot mix parameters")
    return TrajectorySemanticGradient(
        rollout_index=items[0].rollout_index,
        parameter=parameter,
        feedback=feedback if feedback is not None else items[0].feedback,
        rationale=rationale if rationale is not None else items[0].diagnosis,
        step_indices=tuple(sorted({item.step_index for item in items})),
        source_count=len(items),
        total_weight=sum(item.weight for item in items),
        mean_advantage=fmean(item.advantage for item in items),
    )


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


def _trace_parameters(record: AdvantageRecord) -> tuple[ParameterRef, ...]:
    parameters = [
        ParameterRef(HarnessComponent.SKILL_DESCRIPTION, record.trace.skill.id),
        ParameterRef(HarnessComponent.SKILL_STRATEGY, record.trace.skill.id),
    ]
    for entry in record.trace.retrieved_memory:
        parameters.extend(
            [
                ParameterRef(HarnessComponent.MEMORY_KEY, entry.id),
                ParameterRef(HarnessComponent.MEMORY_VALUE, entry.id),
            ]
        )
    return tuple(parameters)


def _trace_parameter_value(record: AdvantageRecord, parameter: ParameterRef) -> str:
    if parameter.component == HarnessComponent.SKILL_DESCRIPTION:
        return record.trace.skill.description
    if parameter.component == HarnessComponent.SKILL_STRATEGY:
        return record.trace.skill.strategy
    for entry in record.trace.retrieved_memory:
        if entry.id != parameter.item_id:
            continue
        if parameter.component == HarnessComponent.MEMORY_KEY:
            return entry.key
        if parameter.component == HarnessComponent.MEMORY_VALUE:
            return entry.value
    raise KeyError(parameter)


def _record_payload(record: AdvantageRecord) -> dict[str, Any]:
    trace = record.trace
    transition = trace.transition
    return {
        "rollout_index": record.rollout_index,
        "step_index": record.step_index,
        "state_factor_pool": [
            {"name": factor.name, "expression": factor.expression}
            for factor in trace.state.factor_pool
        ],
        "skill": {
            "id": trace.skill.id,
            "description": trace.skill.description,
            "strategy": trace.skill.strategy,
        },
        "memory_query": trace.memory_query,
        "interaction": [
            {
                "role": message.role,
                "content": message.content,
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments}
                    for call in message.tool_calls
                ],
                "tool_name": message.name,
                "tool_output": message.output,
                "tool_error": message.error,
            }
            for message in trace.interaction
        ],
        "retrieved_memory": [
            {
                "id": entry.id,
                "key": entry.key,
                "factor_edit": entry.factor_edit,
                "evaluation": dict(entry.evaluation),
                "value": entry.value,
            }
            for entry in trace.retrieved_memory
        ],
        "factor_action": {
            "operation": trace.action.operation.value,
            "target": trace.action.target,
            "factor": (
                {
                    "name": trace.action.factor.name,
                    "expression": trace.action.factor.expression,
                }
                if trace.action.factor is not None
                else None
            ),
        },
        "outcome": {
            "accepted": transition.accepted,
            "redundancy": transition.redundancy,
            "previous_score": transition.previous_score,
            "next_score": transition.next_score,
            "reward": transition.reward,
            "evaluation": dict(transition.info),
        },
        "return_to_go": record.return_to_go,
        "advantage": record.advantage,
    }


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
