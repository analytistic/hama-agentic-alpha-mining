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
    HarnessEditOptimizer,
    HarnessTrace,
    MarketState,
    MemoryBank,
    MemoryEntry,
    Message,
    SemanticGradientEngine,
    Skill,
    SkillLibrary,
    ToolCall,
    Transition,
    optimize_harness,
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


class GradientModel:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        assert tools[0]["function"]["name"] == "submit_semantic_gradients"
        return Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    "gradient",
                    "submit_semantic_gradients",
                    {
                        "gradients": [
                            {
                                "component": "skill_strategy",
                                "item_id": "skill-1",
                                "diagnosis": "The execution overused a noisy horizon.",
                                "feedback": "Compare several longer horizons before submission.",
                            }
                        ]
                    },
                )
            ],
        )


class EditModel:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        assert tools[0]["function"]["name"] == "submit_parameter_edit"
        return Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    "edit",
                    "submit_parameter_edit",
                    {
                        "updated_text": (
                            "Backtest multiple horizons and submit the most stable one."
                        )
                    },
                )
            ],
        )


def test_semantic_engine_and_edit_optimizer_are_separate():
    batch, skill, memory = make_batch()
    harness = Harness(SkillLibrary((skill,)), MemoryBank((memory,)))
    gradient_model = GradientModel()
    edit_model = EditModel()

    result = optimize_harness(
        harness=harness,
        batch=batch,
        semantic_engine=SemanticGradientEngine(gradient_model),
        edit_optimizer=HarnessEditOptimizer(edit_model),
    )

    assert gradient_model.calls == 1
    assert edit_model.calls == 1
    assert len(result.gradients) == 1
    assert len(result.trajectory_gradients) == 1
    assert len(result.edits) == 1
    updated_skill = result.harness.skills.load("skill-1")
    assert updated_skill.description == skill.description
    assert updated_skill.strategy == (
        "Backtest multiple horizons and submit the most stable one."
    )
    assert result.harness.memory.entries == harness.memory.entries


def make_hierarchical_batch():
    _, skill, memory = make_batch()
    records = []
    for rollout_index in range(2):
        factor_pool = (Factor("close", "$close"),)
        for step_index in range(2):
            state = MarketState(factor_pool, {})
            factor = Factor(
                f"candidate-{rollout_index}-{step_index}",
                f"expression-{rollout_index}-{step_index}",
            )
            action = FactorEdit(FactorEditOperation.ADD, factor=factor)
            factor_pool = factor_pool + (factor,)
            next_state = MarketState(factor_pool, {})
            advantage = -1.0 if rollout_index == 0 else 1.0
            transition = Transition(
                state=state,
                action=action,
                next_state=next_state,
                reward=advantage,
                accepted=True,
                previous_score=0.0,
                next_score=advantage,
                terminated=step_index == 1,
            )
            trace = HarnessTrace(
                index=step_index,
                state=state,
                skill=skill,
                memory_query="short momentum",
                retrieved_memory=(memory,),
                action=action,
                next_state=next_state,
                reward=advantage,
                transition=transition,
            )
            records.append(
                AdvantageRecord(
                    rollout_index,
                    step_index,
                    trace,
                    advantage,
                    advantage,
                )
            )
    return AdvantageBatch(2, 1.0, tuple(records)), skill, memory


class HierarchicalGradientModel:
    def __init__(self):
        self.calls = []

    def complete(self, messages, tools):
        name = tools[0]["function"]["name"]
        self.calls.append(name)
        prompt = json.loads(messages[1].content)
        if name == "submit_semantic_gradients":
            trace = prompt["trace"]
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        "segment",
                        name,
                        {
                            "gradients": [
                                {
                                    "component": "skill_strategy",
                                    "item_id": "skill-1",
                                    "diagnosis": (
                                        f"segment {trace['rollout_index']}/"
                                        f"{trace['step_index']}"
                                    ),
                                    "feedback": "calibrate the candidate",
                                }
                            ]
                        },
                    )
                ],
            )
        if name == "submit_trajectory_gradients":
            rollout_index = prompt["rollout_index"]
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        "trajectory",
                        name,
                        {
                            "gradients": [
                                {
                                    "component": "skill_strategy",
                                    "item_id": "skill-1",
                                    "rationale": f"rollout {rollout_index} summary",
                                    "feedback": f"trajectory {rollout_index} feedback",
                                }
                            ]
                        },
                    )
                ],
            )
        assert name == "submit_group_gradient"
        assert len(prompt["trajectory_gradients"]) == 2
        return Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    "group",
                    name,
                    {
                        "rationale": "two independent trajectory conclusions",
                        "feedback": "use robust calibration across candidates",
                    },
                )
            ],
        )


def test_optimization_aggregates_within_trajectory_before_across_rollouts():
    batch, skill, memory = make_hierarchical_batch()
    harness = Harness(SkillLibrary((skill,)), MemoryBank((memory,)))
    model = HierarchicalGradientModel()

    result = optimize_harness(
        harness=harness,
        batch=batch,
        semantic_engine=SemanticGradientEngine(model),
        edit_optimizer=HarnessEditOptimizer(EditModel()),
    )

    assert model.calls == [
        "submit_semantic_gradients",
        "submit_semantic_gradients",
        "submit_semantic_gradients",
        "submit_semantic_gradients",
        "submit_trajectory_gradients",
        "submit_trajectory_gradients",
        "submit_group_gradient",
    ]
    assert len(result.gradients) == 4
    assert len(result.trajectory_gradients) == 2
    assert all(item.source_count == 2 for item in result.trajectory_gradients)
    assert len(result.aggregated_gradients) == 1
    group_gradient = result.aggregated_gradients[0]
    assert group_gradient.source_count == 2
    assert group_gradient.segment_count == 4


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


def test_prefill_attribution_and_conflict_aware_update():
    batch, skill, memory = make_batch()
    harness = Harness(SkillLibrary((skill,)), MemoryBank((memory,)))
    scorer = MaskAwareScorer()

    result = optimize_harness_with_attribution(
        harness=harness,
        batch=batch,
        attributor=ComponentAttributor(scorer),
        semantic_engine=ConflictAwareSemanticGradientEngine(AtomicGradientModel()),
        edit_optimizer=HarnessEditOptimizer(EditModel()),
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


def test_render_prefill_masks_only_selected_component():
    batch, _, _ = make_batch()
    trace = batch.records[0].trace
    full_prefix, continuation = render_prefill(trace)
    assert "use a short horizon" in full_prefix
    assert "short horizons are robust" in full_prefix
    assert '"operation": "add"' in continuation
