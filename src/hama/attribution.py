"""Advantage-weighted leave-one-component-out harness attribution."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .advantage import AdvantageBatch, AdvantageRecord
from .prefill import PrefillScorer
from .types import FactorEdit, HarnessTrace, MemoryEntry, Message, Skill


class InvocationKind(StrEnum):
    SKILL = "skill"
    MEMORY = "memory"


@dataclass(frozen=True, order=True)
class InvocationRef:
    """One shared harness component invoked by a transition."""

    kind: InvocationKind
    item_id: str


@dataclass(frozen=True)
class ComponentAttribution:
    """One advantage-weighted component influence observation."""

    evidence_id: str
    component: InvocationRef
    rollout_index: int
    step_index: int
    record: AdvantageRecord
    full_logprob: float
    masked_logprob: float
    influence: float
    credit: float

    @property
    def correction_weight(self) -> float:
        return max(-self.credit, 0.0)

    @property
    def preservation_weight(self) -> float:
        return max(self.credit, 0.0)


class ComponentAttributor:
    """Attribute each transition to its invoked skill and memory entries."""

    def __init__(self, scorer: PrefillScorer) -> None:
        self.scorer = scorer

    def attribute(self, batch: AdvantageBatch) -> tuple[ComponentAttribution, ...]:
        results: list[ComponentAttribution] = []
        for record in batch.records:
            components = _invoked_components(record.trace)
            if not components:
                continue
            full_input = render_prefill(record.trace)
            masked_inputs = [
                render_prefill(record.trace, masked=component)
                for component in components
            ]
            score_many = getattr(self.scorer, "score_many", None)
            if callable(score_many):
                all_scores = score_many([full_input, *masked_inputs])
                full_logprob = float(all_scores[0])
                masked_scores = all_scores[1:]
            else:
                full_logprob = self.scorer.score(*full_input)
                masked_scores = tuple(
                    self.scorer.score(masked_prefix, masked_continuation)
                    for masked_prefix, masked_continuation in masked_inputs
                )
            for component, masked_logprob in zip(
                components, masked_scores, strict=True
            ):
                influence = full_logprob - float(masked_logprob)
                results.append(
                    ComponentAttribution(
                        evidence_id=(
                            f"r{record.rollout_index}:t{record.step_index}:"
                            f"{component.kind.value}:{component.item_id}"
                        ),
                        component=component,
                        rollout_index=record.rollout_index,
                        step_index=record.step_index,
                        record=record,
                        full_logprob=full_logprob,
                        masked_logprob=float(masked_logprob),
                        influence=influence,
                        credit=record.advantage * influence,
                    )
                )
        return tuple(results)

    @staticmethod
    def group(
        attributions: Sequence[ComponentAttribution],
    ) -> Mapping[InvocationRef, tuple[ComponentAttribution, ...]]:
        grouped: dict[InvocationRef, list[ComponentAttribution]] = defaultdict(list)
        for attribution in attributions:
            grouped[attribution.component].append(attribution)
        return {
            component: tuple(
                sorted(items, key=lambda item: (item.rollout_index, item.step_index))
            )
            for component, items in sorted(grouped.items())
        }


def render_prefill(
    trace: HarnessTrace,
    *,
    masked: InvocationRef | None = None,
) -> tuple[str, str]:
    """Render the causal context and recorded factor edit for teacher forcing."""

    skill = _skill_payload(trace.skill, masked)
    memories = [_memory_payload(entry, masked) for entry in trace.retrieved_memory]
    redactions = _redactions(trace, masked)
    context = {
        "instruction": (
            "Given the recorded state and harness context, emit the factor-pool "
            "modification that the agent chose."
        ),
        "state": {
            "factor_pool": [
                {"name": factor.name, "expression": factor.expression}
                for factor in trace.state.factor_pool
            ],
            "market_information": dict(trace.state.market_information),
        },
        "loaded_skill": skill,
        "memory_query": trace.memory_query,
        "retrieved_memory": memories,
        "interaction_before_submission": _prior_interaction(trace.interaction),
        "factor_pool_action": None,
    }
    prefix = json.dumps(context, ensure_ascii=False, default=str)
    for source, replacement in redactions:
        if source:
            prefix = prefix.replace(source, replacement)
    marker = "null}"
    if not prefix.endswith(marker):
        raise RuntimeError("unexpected serialized prefill context")
    prefix = prefix[: -len(marker)]
    continuation = (
        json.dumps(
            _action_payload(trace.action),
            ensure_ascii=False,
            sort_keys=True,
        )
        + "}"
    )
    return prefix, continuation


def attribution_payload(item: ComponentAttribution) -> dict[str, Any]:
    """Compact evidence supplied to the semantic-gradient engine."""

    trace = item.record.trace
    return {
        "evidence_id": item.evidence_id,
        "rollout_index": item.rollout_index,
        "step_index": item.step_index,
        "advantage": item.record.advantage,
        "influence": item.influence,
        "credit": item.credit,
        "correction_weight": item.correction_weight,
        "preservation_weight": item.preservation_weight,
        "factor_pool_before": [factor.name for factor in trace.state.factor_pool],
        "factor_action": _action_payload(trace.action),
        "factor_pool_after": [factor.name for factor in trace.next_state.factor_pool],
        "reward": trace.reward,
    }


def _invoked_components(trace: HarnessTrace) -> tuple[InvocationRef, ...]:
    values = [InvocationRef(InvocationKind.SKILL, trace.skill.id)]
    values.extend(
        InvocationRef(InvocationKind.MEMORY, entry.id)
        for entry in trace.retrieved_memory
    )
    return tuple(values)


def _skill_payload(skill: Skill, masked: InvocationRef | None) -> dict[str, str]:
    if masked == InvocationRef(InvocationKind.SKILL, skill.id):
        return {
            "id": skill.id,
            "description": "[MASKED_SKILL_DESCRIPTION]",
            "strategy": "[MASKED_SKILL_STRATEGY]",
        }
    return {
        "id": skill.id,
        "description": skill.description,
        "strategy": skill.strategy,
    }


def _memory_payload(
    entry: MemoryEntry,
    masked: InvocationRef | None,
) -> dict[str, Any]:
    if masked == InvocationRef(InvocationKind.MEMORY, entry.id):
        return {
            "id": entry.id,
            "key": "[MASKED_MEMORY_KEY]",
            "factor_edit": "[MASKED_MEMORY_EDIT]",
            "evaluation": "[MASKED_MEMORY_EVALUATION]",
            "value": "[MASKED_MEMORY_VALUE]",
        }
    return {
        "id": entry.id,
        "key": entry.key,
        "factor_edit": entry.factor_edit,
        "evaluation": dict(entry.evaluation),
        "value": entry.value,
    }


def _redactions(
    trace: HarnessTrace,
    masked: InvocationRef | None,
) -> tuple[tuple[str, str], ...]:
    if masked is None:
        return ()
    if masked.kind == InvocationKind.SKILL and masked.item_id == trace.skill.id:
        return (
            (trace.skill.description, "[MASKED_SKILL_DESCRIPTION]"),
            (trace.skill.strategy, "[MASKED_SKILL_STRATEGY]"),
        )
    if masked.kind == InvocationKind.MEMORY:
        for entry in trace.retrieved_memory:
            if entry.id == masked.item_id:
                return (
                    (entry.key, "[MASKED_MEMORY_KEY]"),
                    (entry.factor_edit, "[MASKED_MEMORY_EDIT]"),
                    (entry.value, "[MASKED_MEMORY_VALUE]"),
                )
    return ()


def _prior_interaction(messages: Sequence[Message]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for message in messages:
        if any(call.name == "submit_factor_edit" for call in message.tool_calls):
            break
        if message.name in {"load_skill", "load_memory"}:
            continue
        output.append(
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
        )
    return output


def _action_payload(action: FactorEdit) -> dict[str, Any]:
    return {
        "operation": action.operation.value,
        "target": action.target,
        "factor": (
            None
            if action.factor is None
            else {"name": action.factor.name, "expression": action.factor.expression}
        ),
    }
