from pathlib import Path
import pytest
from hama import (
    AgentRun,
    Factor,
    FactorEdit,
    FactorEditOperation,
    Harness,
    HarnessTrace,
    MarketState,
    MemoryBank,
    Skill,
    SkillLibrary,
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
