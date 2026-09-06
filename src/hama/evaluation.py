"""Evaluation without harness updates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Any

from .agent import Agent
from .environment import AlphaMiningEnvironment, FactorEvaluator
from .harness import Harness

AgentFactory = Callable[[Harness], Agent]
EnvironmentFactory = Callable[[int], AlphaMiningEnvironment]
from .rollout import rollout
from .types import AgentRun, FactorPool


@dataclass(frozen=True)
class PolicyEvaluation:
    runs: tuple[AgentRun, ...]
    mean_discounted_return: float
    std_discounted_return: float
    horizon_completion_rate: float
    accepted_transition_rate: float


def evaluate_policy(
    *,
    task: Any,
    harness: Harness,
    agent_factory: AgentFactory,
    environment_factory: EnvironmentFactory,
    num_rollouts: int,
    gamma: float = 1.0,
) -> PolicyEvaluation:
    """Run a frozen harness on held-out environments without optimization."""

    if num_rollouts < 1:
        raise ValueError("num_rollouts must be at least 1")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    runs = tuple(
        rollout(
            task,
            environment_factory(index),
            agent_factory(harness),
            metadata={"evaluation_rollout_index": index},
        )
        for index in range(num_rollouts)
    )
    returns = [
        sum((gamma**index) * trace.reward for index, trace in enumerate(run.traces))
        for run in runs
    ]
    transitions = [trace.transition for run in runs for trace in run.traces]
    return PolicyEvaluation(
        runs=runs,
        mean_discounted_return=fmean(returns),
        std_discounted_return=pstdev(returns) if len(returns) > 1 else 0.0,
        horizon_completion_rate=(
            sum(
                bool(run.traces and run.traces[-1].transition.terminated)
                for run in runs
            )
            / len(runs)
        ),
        accepted_transition_rate=(
            sum(transition.accepted for transition in transitions) / len(transitions)
            if transitions
            else 0.0
        ),
    )


def evaluate_factor_pool_splits(
    factor_pool: FactorPool,
    evaluators: Mapping[str, FactorEvaluator],
) -> dict[str, Mapping[str, Any]]:
    """Evaluate one fixed final factor pool on train/valid/test evaluators."""

    if not evaluators:
        raise ValueError("at least one split evaluator is required")
    return {
        split: evaluator.evaluate_pool(factor_pool)
        for split, evaluator in evaluators.items()
    }
