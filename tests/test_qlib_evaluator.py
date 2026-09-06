import numpy as np
import pandas as pd
import pytest
from hama import Factor, QlibEvaluatorConfig, QlibFactorEvaluator


def make_loader_factory():
    dates = pd.date_range("2024-01-02", periods=4, freq="D")
    instruments = [f"stock-{index}" for index in range(6)]
    index = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    base = np.tile(np.arange(1.0, 7.0), len(dates))
    values = {
        "positive": base,
        "scaled_positive": base * 2.0,
        "negative": -base,
        "noise": np.tile(np.array([2.0, 5.0, 1.0, 6.0, 3.0, 4.0]), len(dates)),
    }
    calls = []

    class FakeLoader:
        def __init__(self, config):
            self.config = config

        def load(self, instruments, start_time, end_time):
            calls.append((instruments, start_time, end_time))
            expressions, names = self.config["feature"]
            columns = {}
            for expression, name in zip(expressions, names, strict=True):
                columns[("feature", name)] = values[expression]
            columns[("label", "target")] = base
            return pd.DataFrame(columns, index=index)

    return FakeLoader, calls


def test_qlib_evaluator_computes_ic_pool_reward_and_redundancy():
    loader_factory, calls = make_loader_factory()
    evaluator = QlibFactorEvaluator(
        QlibEvaluatorConfig(
            provider_uri="unused-in-test",
            instruments="csi300",
            start_time="2024-01-01",
            end_time="2024-01-31",
            min_pairs=5,
            portfolio_top_k=1,
            buy_cost=0.001,
            sell_cost=0.002,
        ),
        initialize_qlib=False,
        loader_factory=loader_factory,
    )
    positive = Factor("positive", "positive")
    negative = Factor("negative", "negative")

    factor_result = evaluator.evaluate_factors((positive, negative))
    assert factor_result["factors"]["positive"]["ic"] == pytest.approx(1.0)
    assert factor_result["factors"]["negative"]["ic"] == pytest.approx(-1.0)

    positive_pool = evaluator.evaluate_pool((positive,))
    cancelled_pool = evaluator.evaluate_pool((positive, negative))
    assert positive_pool["score"] == pytest.approx(1.0)
    assert positive_pool["gross_total_return"] == pytest.approx(7.0**4 - 1.0)
    assert positive_pool["total_transaction_cost"] == pytest.approx(0.001)
    assert positive_pool["turnover"] == pytest.approx(0.0)
    assert positive_pool["annualized_excess_return"] > 0.0
    assert positive_pool["max_drawdown"] == pytest.approx(0.0)
    assert positive_pool["n_portfolio_dates"] == 4
    assert positive_pool["split"]["portfolio"]["benchmark"] == ("universe_equal_weight")
    assert cancelled_pool["score"] == pytest.approx(0.0)

    redundancy = evaluator.redundancy(
        Factor("scaled", "scaled_positive"),
        (positive,),
    )
    assert redundancy == pytest.approx(1.0)
    assert calls


def test_qlib_evaluator_caches_repeated_pool_evaluation():
    loader_factory, calls = make_loader_factory()
    evaluator = QlibFactorEvaluator(
        QlibEvaluatorConfig(
            provider_uri="unused-in-test",
            instruments="csi300",
            start_time="2024-01-01",
            end_time="2024-01-31",
        ),
        initialize_qlib=False,
        loader_factory=loader_factory,
    )
    pool = (Factor("positive", "positive"),)

    first = evaluator.evaluate_pool(pool)
    second = evaluator.evaluate_pool(pool)

    assert first == second
    assert len(calls) == 1


def test_factor_backtest_window_is_sliced_and_cannot_escape_split():
    loader_factory, _ = make_loader_factory()
    evaluator = QlibFactorEvaluator(
        QlibEvaluatorConfig(
            provider_uri="unused-in-test",
            instruments="csi300",
            start_time="2024-01-01",
            end_time="2024-01-31",
        ),
        initialize_qlib=False,
        loader_factory=loader_factory,
    )

    result = evaluator.evaluate_factors(
        (Factor("positive", "positive"),),
        start_time="2024-01-03",
        end_time="2024-01-04",
    )

    assert result["factors"]["positive"]["n_dates"] == 2
    with pytest.raises(ValueError, match="inside the evaluator split"):
        evaluator.evaluate_factors(
            (Factor("positive", "positive"),),
            start_time="2023-12-01",
        )


def test_qlib_evaluator_rejects_zero_window_max_min_clamps():
    loader_factory, _ = make_loader_factory()
    evaluator = QlibFactorEvaluator(
        QlibEvaluatorConfig(
            provider_uri="unused-in-test",
            instruments="csi300",
            start_time="2024-01-01",
            end_time="2024-01-31",
        ),
        initialize_qlib=False,
        loader_factory=loader_factory,
    )

    with pytest.raises(ValueError, match=r"Max\(x, 0\).+Greater"):
        evaluator.evaluate_factors((Factor("bad", "Max(positive, 0)"),))
    with pytest.raises(ValueError, match=r"Min\(x, 0\).+Less"):
        evaluator.evaluate_factors((Factor("bad", "Min(positive, 0)"),))
