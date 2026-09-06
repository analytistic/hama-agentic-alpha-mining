import json

from hama import (
    AdvantageBatch,
    AdvantageRecord,
    ComponentAttributor,
    ConflictAwareSemanticGradientEngine,
    Factor,
    FactorEdit,
    FactorEditOperation,
    Harness,
    HarnessEvolutionOptimizer,
    HarnessTrace,
    MarketState,
    MemoryBank,
    MemoryEntry,
    Message,
    Skill,
    SkillLibrary,
    ToolCall,
    Transition,
    optimize_harness_with_attribution,
    render_prefill,
)


def make_batch():
    skill = Skill("skill-1", "momentum selection", "use a short horizon")
    memory = MemoryEntry(
        "memory-1",
        "short momentum",
        "add short momentum",
        {"ic": -0.01},
        "short horizons are robust",
    )
    state = MarketState((Factor("close", "$close"),), {})
    action = FactorEdit(
        FactorEditOperation.ADD,
        factor=Factor("mom", "$close / Ref($close, 5) - 1"),
    )
    next_state = MarketState(state.factor_pool + (action.factor,), {})
    transition = Transition(
        state=state,
        action=action,
        next_state=next_state,
        reward=-0.02,
        accepted=True,
        previous_score=0.03,
        next_score=0.01,
        terminated=True,
    )
    trace = HarnessTrace(
        index=0,
        state=state,
        skill=skill,
        memory_query="short momentum",
        retrieved_memory=(memory,),
        action=action,
        next_state=next_state,
        reward=-0.02,
        transition=transition,
    )
    record = AdvantageRecord(0, 0, trace, -0.02, -1.0)
    return AdvantageBatch(1, 1.0, (record,)), skill, memory




class MaskAwareScorer:
    def __init__(self):
        self.calls = []

    def score(self, prefix, continuation):
        self.calls.append((prefix, continuation))
        if "MASKED_SKILL" in prefix:
            return -3.0
        if "MASKED_MEMORY" in prefix:
            return -0.5
        return -1.0

    def score_many(self, items):
        return tuple(self.score(prefix, continuation) for prefix, continuation in items)


class AtomicGradientModel:
    def complete(self, messages, tools):
        assert tools[0]["function"]["name"] == "submit_atomic_gradients"
        prompt = json.loads(messages[1].content)
        evidence_id = prompt["correction_evidence"][0]["evidence_id"]
        return Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    "atomic",
                    "submit_atomic_gradients",
                    {
                        "proposals": [
                            {
                                "component": "skill_strategy",
                                "item_id": "skill-1",
                                "rationale": "The short horizon repeatedly supports poor edits.",
                                "feedback": "Require comparison across several horizons.",
                                "support_ids": [evidence_id],
                                "conflict_ids": [],
                            }
                        ]
                    },
                )
            ],
        )


class SkillUpdateEvolutionModel:
    def complete(self, messages, tools):
        assert tools[0]["function"]["name"] == "submit_harness_changes"
        return Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    "update",
                    "submit_harness_changes",
                    {
                        "changes": [
                            {
                                "operation": "update_parameter",
                                "component": "skill_strategy",
                                "item_id": "skill-1",
                                "updated_text": "Backtest multiple horizons and submit the most stable one.",
                            }
                        ]
                    },
                )
            ],
        )


def test_prefill_attribution_and_conflict_aware_update():
    batch, skill, memory = make_batch()
    harness = Harness(SkillLibrary((skill,)), MemoryBank((memory,)))
    scorer = MaskAwareScorer()

    result = optimize_harness_with_attribution(
        harness=harness,
        batch=batch,
        attributor=ComponentAttributor(scorer),
        semantic_engine=ConflictAwareSemanticGradientEngine(AtomicGradientModel()),
        edit_optimizer=HarnessEvolutionOptimizer(SkillUpdateEvolutionModel()),
    )

    assert len(scorer.calls) == 3  # full, masked skill, masked memory
    by_kind = {item.component.kind.value: item for item in result.attributions}
    assert by_kind["skill"].influence == 2.0
    assert by_kind["skill"].credit == -2.0
    assert by_kind["memory"].influence == -0.5
    assert by_kind["memory"].credit == 0.5
    assert len(result.proposals) == 1
    assert result.proposals[0].score == 2.0
    assert result.harness.skills.load("skill-1").strategy == (
        "Backtest multiple horizons and submit the most stable one."
    )


class LibraryEvolutionModel:
    def complete(self, messages, tools):
        assert tools[0]["function"]["name"] == "submit_harness_changes"
        return Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    "evolve",
                    "submit_harness_changes",
                    {
                        "changes": [
                            {
                                "operation": "upsert_skill",
                                "item_id": "redundancy_control",
                                "description": "Reject redundant factor candidates.",
                                "strategy": "Compare the candidate with every pool member.",
                            },
                            {
                                "operation": "upsert_memory",
                                "item_id": "failed_short_momentum",
                                "key": "short momentum failure",
                                "factor_edit": "add five-day momentum",
                                "evaluation": {"reward": -0.02},
                                "value": "Short momentum degraded the pool in this split.",
                            },
                        ]
                    },
                )
            ],
        )


def test_evolution_optimizer_can_expand_skill_and_memory_libraries():
    batch, skill, memory = make_batch()
    harness = Harness(SkillLibrary((skill,)), MemoryBank((memory,)))
    updated, edits = HarnessEvolutionOptimizer(LibraryEvolutionModel()).update(
        harness, (), batch=batch, attributions=()
    )

    assert updated.skills.load("redundancy_control").strategy.startswith("Compare")
    assert {entry.id for entry in updated.memory.entries} == {
        "memory-1",
        "failed_short_momentum",
    }
    assert [edit.parameter.component.value for edit in edits] == ["skill", "memory"]


def test_render_prefill_masks_only_selected_component():
    batch, _, _ = make_batch()
    trace = batch.records[0].trace
    full_prefix, continuation = render_prefill(trace)
    assert "use a short horizon" in full_prefix
    assert "short horizons are robust" in full_prefix
    assert '"operation": "add"' in continuation
