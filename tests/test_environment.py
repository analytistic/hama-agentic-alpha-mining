import pytest
from hama import (
    AlphaMiningEnvironment,
    Factor,
    FactorEdit,
    FactorEditOperation,
    MarketState,
)


class RecordingEvaluator:
    def __init__(self, redundancy=0.0):
        self.redundancy_value = redundancy
        self.comparison_pools = []

    def evaluate_factors(self, factors, *, start_time=None, end_time=None):
        return {}

    def evaluate_pool(self, factor_pool):
        return {"score": float(len(factor_pool))}

    def redundancy(self, factor, factor_pool):
        self.comparison_pools.append(factor_pool)
        return self.redundancy_value


def test_redundant_addition_is_rejected_with_zero_reward(tmp_path):
    base = Factor("base", "$close")
    evaluator = RecordingEvaluator(redundancy=0.9)
    environment = AlphaMiningEnvironment(
        state=MarketState((base,), {}),
        evaluator=evaluator,
        workspace=tmp_path,
        redundancy_threshold=0.7,
        exploration_acceptance=0.0,
    )

    transition = environment.submit(
        FactorEdit(
            FactorEditOperation.ADD,
            factor=Factor("duplicate", "$close * 2"),
        )
    )

    assert transition.accepted is False
    assert transition.reward == 0.0
    assert transition.next_state.factor_pool == (base,)


def test_replace_checks_redundancy_after_excluding_target(tmp_path):
    target = Factor("target", "$close")
    retained = Factor("retained", "$volume")
    evaluator = RecordingEvaluator(redundancy=0.1)
    environment = AlphaMiningEnvironment(
        state=MarketState((target, retained), {}),
        evaluator=evaluator,
        workspace=tmp_path,
    )

    transition = environment.submit(
        FactorEdit(
            FactorEditOperation.REPLACE,
            target="target",
            factor=Factor("replacement", "$high - $low"),
        )
    )

    assert evaluator.comparison_pools == [(retained,)]
    assert transition.accepted is True
    assert [factor.name for factor in transition.next_state.factor_pool] == [
        "replacement",
        "retained",
    ]


def test_factor_action_must_be_one_well_formed_atomic_edit(tmp_path):
    environment = AlphaMiningEnvironment(
        state=MarketState((Factor("base", "$close"),), {}),
        evaluator=RecordingEvaluator(),
        workspace=tmp_path,
    )

    with pytest.raises(ValueError, match="add requires factor and forbids target"):
        environment.submit(
            FactorEdit(
                FactorEditOperation.ADD,
                factor=Factor("new", "$volume"),
                target="base",
            )
        )
