from pathlib import Path

import pytest
from hama import (
    Agent,
    AlphaMiningEnvironment,
    Factor,
    Harness,
    MarketState,
    MemoryBank,
    MemoryEntry,
    Message,
    Skill,
    SkillLibrary,
    ToolCall,
    rollout,
)

MOMENTUM = "$close / Ref($close, 20) - 1"


class FakeEvaluator:
    def evaluate_factors(self, factors, *, start_time=None, end_time=None):
        return {
            factor.name: {"ic": 0.04 if factor.expression == MOMENTUM else 0.01}
            for factor in factors
        }

    def evaluate_pool(self, factor_pool):
        scores = {"$close": 0.01, MOMENTUM: 0.04}
        return {
            "score": sum(scores.get(factor.expression, 0.0) for factor in factor_pool)
        }

    def redundancy(self, factor, factor_pool):
        return 0.2


def make_harness() -> Harness:
    return Harness(
        skills=SkillLibrary(
            (
                Skill(
                    id="refine_momentum",
                    description="Improve a momentum factor without duplicating the pool.",
                    strategy="Backtest several horizons and retain the most stable expression.",
                ),
            )
        ),
        memory=MemoryBank(
            (
                MemoryEntry(
                    id="case-1",
                    key="momentum horizon stability",
                    factor_edit="replace 5-day momentum with 20-day momentum",
                    evaluation={"ic": 0.04},
                    value="Longer horizons were less noisy on this data split.",
                ),
            )
        ),
    )


def make_environment(tmp_path: Path) -> AlphaMiningEnvironment:
    return AlphaMiningEnvironment(
        state=MarketState(
            factor_pool=(Factor(name="close", expression="$close"),),
            market_information={"universe": "csi300", "split": "train"},
        ),
        evaluator=FakeEvaluator(),
        workspace=tmp_path,
    )


def tool_names(tools):
    return [tool["function"]["name"] for tool in tools]


class PaperLoopModel:
    def __init__(self):
        self.calls = 0
        self.exposed_tools = []

    def complete(self, messages, tools):
        self.calls += 1
        names = tool_names(tools)
        self.exposed_tools.append(names)
        if self.calls == 1:
            assert names == ["load_skill"]
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="skill-1",
                        name="load_skill",
                        arguments={"skill_id": "refine_momentum"},
                    )
                ],
            )
        if self.calls == 2:
            assert names == ["load_memory"]
            assert "Backtest several horizons" in str(messages[-1].output)
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="memory-1",
                        name="load_memory",
                        arguments={"query": "stable momentum horizon", "top_k": 1},
                    )
                ],
            )
        if self.calls == 3:
            assert "backtest_factors" in names
            assert "submit_factor_edit" in names
            assert "Longer horizons were less noisy" in str(messages[-1].output)
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="backtest-1",
                        name="backtest_factors",
                        arguments={
                            "factors": [
                                {"name": "momentum_20d", "expression": MOMENTUM}
                            ]
                        },
                    )
                ],
            )
        return Message(
            role="assistant",
            content="Submit the tested candidate.",
            tool_calls=[
                ToolCall(
                    id="submit-1",
                    name="submit_factor_edit",
                    arguments={
                        "operation": "add",
                        "target": None,
                        "factor": {
                            "name": "momentum_20d",
                            "expression": MOMENTUM,
                        },
                    },
                )
            ],
        )


def test_rollout_follows_paper_skill_memory_factor_transition(tmp_path: Path):
    model = PaperLoopModel()
    environment = make_environment(tmp_path)
    agent = Agent(
        model=model,
        system_prompt="Mine one useful alpha factor.",
        harness=make_harness(),
    )

    run = rollout({"goal": "improve the current pool"}, environment, agent)

    assert run.stop_reason == "horizon_reached"
    assert run.steps == 4
    assert run.loaded_skill is not None
    assert run.loaded_skill.id == "refine_momentum"
    assert run.memory_query == "stable momentum horizon"
    assert [entry.id for entry in run.retrieved_memory] == ["case-1"]
    assert run.transition is not None
    assert len(run.traces) == 1
    assert any(
        message.name == "backtest_factors" for message in run.traces[0].interaction
    )
    assert run.transition.accepted is True
    assert run.transition.reward == pytest.approx(0.04)
    assert [factor.name for factor in run.transition.next_state.factor_pool] == [
        "close",
        "momentum_20d",
    ]
    assert model.exposed_tools[0] == ["load_skill"]
    assert model.exposed_tools[1] == ["load_memory"]


class TwoTransitionModel:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        phase = (self.calls - 1) % 3
        if phase == 0:
            if self.calls == 4:
                assert "'reward': 0.04" in str(messages[-1].output)
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        f"skill-{self.calls}",
                        "load_skill",
                        {"skill_id": "refine_momentum"},
                    )
                ],
            )
        if phase == 1:
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        f"memory-{self.calls}",
                        "load_memory",
                        {"query": "momentum"},
                    )
                ],
            )
        if self.calls == 3:
            arguments = {
                "operation": "add",
                "target": None,
                "factor": {"name": "momentum_20d", "expression": MOMENTUM},
            }
        else:
            arguments = {"operation": "remove", "target": "close", "factor": None}
        return Message(
            role="assistant",
            tool_calls=[
                ToolCall(f"submit-{self.calls}", "submit_factor_edit", arguments)
            ],
        )


def test_one_rollout_repeats_three_actions_in_one_context(tmp_path: Path):
    environment = AlphaMiningEnvironment(
        state=MarketState(
            factor_pool=(Factor(name="close", expression="$close"),),
            market_information={"universe": "csi300"},
        ),
        evaluator=FakeEvaluator(),
        workspace=tmp_path,
        horizon=2,
    )
    model = TwoTransitionModel()
    agent = Agent(
        model=model,
        system_prompt="Mine factors.",
        harness=make_harness(),
    )

    run = rollout("improve twice", environment, agent)

    assert run.stop_reason == "horizon_reached"
    assert run.steps == 6
    assert len(run.traces) == 2
    assert [trace.index for trace in run.traces] == [0, 1]
    assert [trace.reward for trace in run.traces] == pytest.approx([0.04, -0.01])
    assert [factor.name for factor in run.transition.next_state.factor_pool] == [
        "momentum_20d"
    ]


class EndlessPaperLoopModel:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall("skill", "load_skill", {"skill_id": "refine_momentum"})
                ],
            )
        if self.calls == 2:
            return Message(
                role="assistant",
                tool_calls=[ToolCall("memory", "load_memory", {"query": "momentum"})],
            )
        return Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id=f"read-{self.calls}",
                    name="read",
                    arguments={"path": "note.txt"},
                )
            ],
        )


def test_agent_stops_after_default_fifty_steps(tmp_path: Path):
    (tmp_path / "note.txt").write_text("still working", encoding="utf-8")
    model = EndlessPaperLoopModel()
    agent = Agent(
        model=model,
        system_prompt="Keep inspecting.",
        harness=make_harness(),
    )

    run = rollout("work", make_environment(tmp_path), agent)

    assert model.calls == 50
    assert run.steps == 50
    assert run.stop_reason == "max_steps"
    assert run.transition is None


def test_agent_rejects_more_than_fifty_steps():
    with pytest.raises(ValueError, match="between 1 and 50"):
        Agent(
            model=EndlessPaperLoopModel(),
            system_prompt="test",
            harness=make_harness(),
            max_steps=51,
        )


class TextBeforeToolModel(PaperLoopModel):
    def complete(self, messages, tools):
        if not hasattr(self, "replied_with_text"):
            self.replied_with_text = True
            return Message(role="assistant", content="I should inspect the task first.")
        return super().complete(messages, tools)


def test_agent_recovers_when_model_omits_required_tool_call(tmp_path: Path):
    model = TextBeforeToolModel()
    run = rollout(
        "improve the pool",
        make_environment(tmp_path),
        Agent(model=model, system_prompt="Mine one factor.", harness=make_harness()),
    )

    assert run.stop_reason == "horizon_reached"
    assert len(run.traces) == 1
    assert any(
        message.role == "user" and "calling load_skill" in message.content
        for message in run.messages
    )
