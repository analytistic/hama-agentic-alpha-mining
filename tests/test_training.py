from pathlib import Path
from types import SimpleNamespace

import pytest
from hama import (
    AgentRun,
    Factor,
    FactorEdit,
    FactorEditOperation,
    HamaTrainer,
    Harness,
    HarnessEditOptimizer,
    HarnessTrace,
    MarketState,
    MemoryBank,
    Message,
    SemanticGradientEngine,
    Skill,
    SkillLibrary,
    ToolCall,
    TrainingConfig,
    Transition,
    evaluate_policy,
)
from hama.persistence import load_harness, save_harness


def make_harness():
    return Harness(
        SkillLibrary((Skill("skill", "description", "old strategy"),)),
        MemoryBank(()),
    )


def make_run(harness, reward):
    state = MarketState((Factor("base", "$close"),), {})
    action = FactorEdit(
        FactorEditOperation.ADD,
        factor=Factor("candidate", "candidate-expression"),
    )
    next_state = MarketState(state.factor_pool + (action.factor,), {})
    transition = Transition(
        state=state,
        action=action,
        next_state=next_state,
        reward=reward,
        accepted=True,
        previous_score=0.0,
        next_score=reward,
        terminated=True,
    )
    trace = HarnessTrace(
        index=0,
        state=state,
        skill=harness.skills.load("skill"),
        memory_query="query",
        retrieved_memory=(),
        action=action,
        next_state=next_state,
        reward=reward,
        transition=transition,
    )
    return AgentRun(
        messages=[],
        stop_reason="horizon_reached",
        steps=3,
        transition=transition,
        traces=(trace,),
    )


class FakeAgent:
    def __init__(self, run):
        self._run = run

    def run(self, task, environment, metadata=None):
        self._run.metadata.update(metadata or {})
        return self._run


class GradientAndAggregationModel:
    def complete(self, messages, tools):
        name = tools[0]["function"]["name"]
        if name == "submit_semantic_gradients":
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        "gradient",
                        name,
                        {
                            "gradients": [
                                {
                                    "component": "skill_strategy",
                                    "item_id": "skill",
                                    "diagnosis": "The strategy needs calibration.",
                                    "feedback": "Compare candidates more carefully.",
                                }
                            ]
                        },
                    )
                ],
            )
        assert name == "submit_group_gradient"
        return Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    "aggregate",
                    name,
                    {
                        "rationale": "Both signed outcomes support calibration.",
                        "feedback": "Use backtests to calibrate the candidate choice.",
                    },
                )
            ],
        )


class EditModel:
    def complete(self, messages, tools):
        return Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    "edit",
                    "submit_parameter_edit",
                    {"updated_text": "Calibrate every candidate with a backtest."},
                )
            ],
        )


def test_trainer_collects_one_rollout_then_updates_frozen_harness(tmp_path: Path):
    harness = make_harness()

    trainer = HamaTrainer(
        harness=harness,
        initial_state=MarketState((Factor("base", "$close"),), {}),
        task="mine",
        agent_factory=lambda frozen: FakeAgent(make_run(frozen, -1.0)),
        environment_factory=lambda round_index, current_state: object(),
        semantic_engine=SemanticGradientEngine(GradientAndAggregationModel()),
        edit_optimizer=HarnessEditOptimizer(EditModel()),
        config=TrainingConfig(
            rounds=1,
            checkpoint_dir=tmp_path / "checkpoints",
        ),
    )

    result = trainer.train()

    assert len(result.rounds) == 1
    assert result.rounds[0].mean_discounted_return == pytest.approx(-1.0)
    assert len(result.rounds[0].optimization.trajectory_gradients) == 1
    assert result.rounds[0].optimization.aggregated_gradients[0].source_count == 1
    assert result.harness.skills.load("skill").strategy == (
        "Calibrate every candidate with a backtest."
    )
    assert harness.skills.load("skill").strategy == "old strategy"
    checkpoint = tmp_path / "checkpoints" / "round-0000"
    assert (checkpoint / "training.json").exists()
    assert load_harness(checkpoint / "harness") == result.harness


def test_training_rollouts_inherit_the_previous_final_factor_pool():
    harness = make_harness()
    received_states = []

    def environment_factory(round_index, previous_state):
        received_states.append(previous_state)
        return SimpleNamespace(state=previous_state)

    trainer = HamaTrainer(
        harness=harness,
        initial_state=MarketState((Factor("base", "$close"),), {}),
        task="mine",
        agent_factory=lambda frozen: FakeAgent(make_run(frozen, 1.0)),
        environment_factory=environment_factory,
        semantic_engine=SemanticGradientEngine(GradientAndAggregationModel()),
        edit_optimizer=HarnessEditOptimizer(EditModel()),
        config=TrainingConfig(rounds=2),
    )

    result = trainer.train()

    assert received_states[0] == MarketState((Factor("base", "$close"),), {})
    assert [factor.name for factor in received_states[1].factor_pool] == [
        "base",
        "candidate",
    ]
    assert result.final_factor_pool == received_states[1].factor_pool


def test_evaluation_never_updates_harness():
    harness = make_harness()
    rewards = iter([1.0, 3.0])
    result = evaluate_policy(
        task="evaluate",
        harness=harness,
        agent_factory=lambda frozen: FakeAgent(make_run(frozen, next(rewards))),
        environment_factory=lambda rollout_index: object(),
        num_rollouts=2,
    )

    assert result.mean_discounted_return == pytest.approx(2.0)
    assert result.std_discounted_return == pytest.approx(1.0)
    assert result.horizon_completion_rate == pytest.approx(1.0)
    assert harness.skills.load("skill").strategy == "old strategy"


def test_harness_directory_round_trip_preserves_skill_resources(tmp_path: Path):
    directory = tmp_path / "harness"
    resource = directory / "skills" / "skill" / "scripts" / "helper.py"
    resource.parent.mkdir(parents=True)
    resource.write_text("print('preserved')\n", encoding="utf-8")

    path = save_harness(make_harness(), directory)

    assert path == directory.resolve()
    assert (directory / "skills" / "manifest.json").exists()
    assert (directory / "skills" / "skill" / "SKILL.md").read_text(
        encoding="utf-8"
    ).strip() == "old strategy"
    assert resource.read_text(encoding="utf-8") == "print('preserved')\n"
    assert load_harness(directory) == make_harness()


def test_harness_rejects_single_json_file(tmp_path: Path):
    with pytest.raises(ValueError, match="directory"):
        save_harness(make_harness(), tmp_path / "harness.json")
    with pytest.raises(NotADirectoryError):
        load_harness(tmp_path / "harness.json")
