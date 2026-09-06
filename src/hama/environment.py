"""Paper-aligned alpha-mining environment owned by HAMA."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from .tools import create_analysis_tools, create_submit_factor_edit_tool
from .types import (
    Factor,
    FactorEdit,
    FactorEditOperation,
    FactorPool,
    FunctionTool,
    MarketState,
    Transition,
)


class FactorEvaluator(Protocol):
    """Evaluate factors on the fixed data split owned by the environment."""

    def evaluate_factors(
        self,
        factors: Sequence[Factor],
        *,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> Mapping[str, Any]: ...

    def evaluate_pool(self, factor_pool: FactorPool) -> Mapping[str, Any]: ...

    def redundancy(self, factor: Factor, factor_pool: FactorPool) -> float: ...


class AlphaMiningEnvironment:
    """Stationary factor-pool MDP used by one HAMA rollout.

    Market state stays fixed while the agent reasons and invokes analysis tools.
    Each successful ``submit_factor_edit`` performs one MDP transition. A
    rollout terminates after the configured number of transitions.
    """

    def __init__(
        self,
        *,
        state: MarketState,
        evaluator: FactorEvaluator,
        workspace: str | Path,
        horizon: int = 1,
        redundancy_threshold: float = 0.7,
        exploration_acceptance: float = 0.0,
        tool_timeout: float = 60.0,
        rng: random.Random | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if not 0.0 <= redundancy_threshold <= 1.0:
            raise ValueError("redundancy_threshold must be in [0, 1]")
        if not 0.0 <= exploration_acceptance <= 1.0:
            raise ValueError("exploration_acceptance must be in [0, 1]")
        self.initial_state = state
        self.state = state
        self.evaluator = evaluator
        self.horizon = horizon
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.redundancy_threshold = redundancy_threshold
        self.exploration_acceptance = exploration_acceptance
        self.tool_timeout = tool_timeout
        self.rng = rng or random.Random()
        self.transitions: list[Transition] = []
        self._analysis_tools = create_analysis_tools(
            workspace=self.workspace,
            evaluator=self.evaluator,
            timeout=self.tool_timeout,
        )
        self._submit_tool = create_submit_factor_edit_tool(self.submit)

    @property
    def analysis_tools(self) -> list[FunctionTool]:
        return list(self._analysis_tools)

    @property
    def submit_tool(self) -> FunctionTool:
        return self._submit_tool

    @property
    def transition(self) -> Transition | None:
        """Most recent transition, retained as a convenience alias."""

        return self.transitions[-1] if self.transitions else None

    @property
    def terminated(self) -> bool:
        return len(self.transitions) >= self.horizon

    def reset(self) -> dict[str, Any]:
        self.state = self.initial_state
        self.transitions = []
        return self.observation()

    def observation(self) -> dict[str, Any]:
        return {
            "factor_pool": [asdict(factor) for factor in self.state.factor_pool],
            "market_information": dict(self.state.market_information),
            "transition_index": len(self.transitions),
            "remaining_transitions": self.horizon - len(self.transitions),
        }

    def submit(self, edit: FactorEdit) -> Transition:
        """Apply exactly one add/remove/replace edit and compute marginal reward."""

        if self.terminated:
            raise RuntimeError("the rollout horizon has already been reached")
        previous_pool = self.state.factor_pool
        candidate_pool, comparison_pool = _apply_edit(previous_pool, edit)
        redundancy: float | None = None
        accepted = True

        if edit.operation in {FactorEditOperation.ADD, FactorEditOperation.REPLACE}:
            assert edit.factor is not None
            redundancy = float(self.evaluator.redundancy(edit.factor, comparison_pool))
            if redundancy > self.redundancy_threshold:
                accepted = self.rng.random() < self.exploration_acceptance

        next_pool = candidate_pool if accepted else previous_pool
        previous_eval = self.evaluator.evaluate_pool(previous_pool)
        next_eval = self.evaluator.evaluate_pool(next_pool)
        previous_score = float(previous_eval["score"])
        next_score = float(next_eval["score"])
        reward = next_score - previous_score if accepted else 0.0
        next_state = MarketState(
            factor_pool=next_pool,
            market_information=self.state.market_information,
        )
        transition = Transition(
            state=self.state,
            action=edit,
            next_state=next_state,
            reward=reward,
            accepted=accepted,
            previous_score=previous_score,
            next_score=next_score,
            terminated=len(self.transitions) + 1 >= self.horizon,
            redundancy=redundancy,
            info={"previous": dict(previous_eval), "next": dict(next_eval)},
        )
        self.transitions.append(transition)
        self.state = next_state
        return transition


def _apply_edit(
    factor_pool: FactorPool,
    edit: FactorEdit,
) -> tuple[FactorPool, FactorPool]:
    factors = list(factor_pool)
    names = [factor.name for factor in factors]

    if edit.operation == FactorEditOperation.ADD:
        if edit.factor is None or edit.target is not None:
            raise ValueError("add requires factor and forbids target")
        if edit.factor.name in names:
            raise ValueError(f"factor already exists: {edit.factor.name}")
        factors.append(edit.factor)
        return tuple(factors), factor_pool

    if edit.operation == FactorEditOperation.REMOVE:
        if edit.factor is not None or edit.target is None:
            raise ValueError("remove requires target and forbids factor")
        if edit.target not in names:
            raise ValueError(f"unknown factor target: {edit.target}")
        index = names.index(edit.target)
        factors.pop(index)
        candidate = tuple(factors)
        return candidate, candidate

    if edit.operation == FactorEditOperation.REPLACE:
        if edit.factor is None or edit.target is None:
            raise ValueError("replace requires both target and factor")
        if edit.target not in names:
            raise ValueError(f"unknown factor target: {edit.target}")
        if edit.factor.name in names and edit.factor.name != edit.target:
            raise ValueError(f"factor already exists: {edit.factor.name}")
        index = names.index(edit.target)
        factors.pop(index)
        comparison_pool = tuple(factors)
        factors.insert(index, edit.factor)
        return tuple(factors), comparison_pool

    raise ValueError(f"unsupported factor edit: {edit.operation}")
