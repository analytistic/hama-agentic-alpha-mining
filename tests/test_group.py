import pytest
from hama import (
    AgentRun,
    EMAReturnBaseline,
    Factor,
    FactorEdit,
    FactorEditOperation,
    HarnessTrace,
    MarketState,
    Skill,
    Transition,
    estimate_ema_advantages,
)


def make_run(rewards):
    skill = Skill("skill", "description", "strategy")
    state = MarketState((Factor("base", "$close"),), {})
    traces = []
    for index, reward in enumerate(rewards):
        action = FactorEdit(
            FactorEditOperation.ADD,
            factor=Factor(f"factor-{index}", f"expression-{index}"),
        )
        next_state = MarketState(
            state.factor_pool + (action.factor,),
            state.market_information,
        )
        transition = Transition(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            accepted=True,
            previous_score=0.0,
            next_score=reward,
            terminated=index == len(rewards) - 1,
        )
        traces.append(
            HarnessTrace(
                index=index,
                state=state,
                skill=skill,
                memory_query="query",
                retrieved_memory=(),
                action=action,
                next_state=next_state,
                reward=reward,
                transition=transition,
            )
        )
        state = next_state
    return AgentRun(
        messages=[],
        stop_reason="horizon_reached",
        steps=len(rewards) * 3,
        traces=tuple(traces),
        transition=traces[-1].transition,
    )


def test_advantage_uses_only_the_historical_ema_baseline():
    baseline = EMAReturnBaseline(update_rate=0.5)
    first_batch = estimate_ema_advantages(make_run([1.0, 1.0]), baseline)
    batch = estimate_ema_advantages(make_run([3.0, 1.0]), baseline)

    assert [record.return_to_go for record in first_batch.records] == pytest.approx(
        [2.0, 1.0]
    )
    assert [record.baseline for record in first_batch.records] == pytest.approx(
        [0.0, 0.0]
    )
    assert [record.advantage for record in first_batch.records] == pytest.approx(
        [2.0, 1.0]
    )
    assert batch.horizon == 2
    assert [record.return_to_go for record in batch.records] == pytest.approx([4.0, 1.0])
    assert [record.baseline for record in batch.records] == pytest.approx([1.0, 0.5])
    assert [record.advantage for record in batch.records] == pytest.approx([3.0, 0.5])
    assert baseline.values == pytest.approx({0: 2.5, 1: 0.75})


def test_baseline_update_rate_must_be_valid():
    with pytest.raises(ValueError, match="update_rate"):
        EMAReturnBaseline(update_rate=0.0)
