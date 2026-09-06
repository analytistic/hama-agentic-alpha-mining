"""End-to-end single-rollout semantic harness training."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from .attribution import ComponentAttributor
from .environment import AlphaMiningEnvironment, FactorEvaluator
from .advantage import (
    AdvantageBatch,
    EMAReturnBaseline,
    estimate_ema_advantages,
)
from .agent import Agent
from .harness import Harness
from .optimization import (
    HarnessOptimizationResult,
    ConflictAwareSemanticGradientEngine,
    HarnessEvolutionOptimizer,
    optimize_harness_with_attribution,
)
from .persistence import factor_pool_to_list, save_factor_pool, save_harness, write_json
from .types import AgentRun, FactorPool, MarketState
from .rollout import rollout
from .repository import HarnessRepository

AgentFactory = Callable[[Harness], Agent]
TrainingEnvironmentFactory = Callable[[int, MarketState], AlphaMiningEnvironment]


@dataclass(frozen=True)
class TrainingConfig:
    rounds: int
    gamma: float = 1.0
    baseline_update_rate: float = 0.1
    baseline_initial_value: float = 0.0
    baseline_initial_values: Mapping[int, float] | None = None
    max_rollout_attempts: int = 2
    checkpoint_dir: str | Path | None = None
    tensorboard_dir: str | Path | None = None

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds must be at least 1")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 < self.baseline_update_rate <= 1.0:
            raise ValueError("baseline_update_rate must be in (0, 1]")
        if self.max_rollout_attempts < 1:
            raise ValueError("max_rollout_attempts must be at least 1")


@dataclass(frozen=True)
class TrainingRound:
    """One rollout followed by prefill-attributed harness optimization."""

    index: int
    run: AgentRun
    advantages: AdvantageBatch
    optimization: HarnessOptimizationResult

    @property
    def mean_discounted_return(self) -> float:
        returns = [
            record.return_to_go
            for record in self.advantages.records
            if record.step_index == 0
        ]
        return fmean(returns) if returns else 0.0


@dataclass(frozen=True)
class TrainingResult:
    harness: Harness
    rounds: tuple[TrainingRound, ...]
    final_factor_pool: FactorPool
    best_factor_pool: FactorPool
    best_selection_score: float
    best_round: int


class HamaTrainer:
    """End-to-end trainer for advantage-weighted component attribution."""

    def __init__(
        self,
        *,
        harness: Harness,
        initial_state: MarketState,
        task: Any,
        agent_factory: AgentFactory,
        environment_factory: TrainingEnvironmentFactory,
        attributor: ComponentAttributor,
        semantic_engine: ConflictAwareSemanticGradientEngine,
        edit_optimizer: HarnessEvolutionOptimizer,
        config: TrainingConfig,
        selection_evaluator: FactorEvaluator | None = None,
        harness_repository: HarnessRepository | None = None,
    ) -> None:
        self.harness = harness
        self.initial_state = initial_state
        self.task = task
        self.agent_factory = agent_factory
        self.environment_factory = environment_factory
        self.attributor = attributor
        self.semantic_engine = semantic_engine
        self.edit_optimizer = edit_optimizer
        self.config = config
        self.selection_evaluator = selection_evaluator
        self.harness_repository = harness_repository

    def train(self) -> TrainingResult:
        rounds: list[TrainingRound] = []
        best_pool: FactorPool = ()
        best_score = float("-inf")
        best_round = -1
        writer = _summary_writer(self.config.tensorboard_dir)
        baseline = EMAReturnBaseline(
            self.config.baseline_update_rate,
            self.config.baseline_initial_value,
        )
        current_state = self.initial_state
        try:
            for round_index in range(self.config.rounds):
                frozen_harness = self.harness
                for attempt in range(self.config.max_rollout_attempts):
                    environment = self.environment_factory(round_index, current_state)
                    run = rollout(
                        self.task,
                        environment,
                        self.agent_factory(frozen_harness),
                        metadata={
                            "rollout_index": 0,
                            "training_round": round_index,
                            "attempt": attempt,
                        },
                    )
                    if run.traces:
                        break
                else:
                    raise RuntimeError(
                        f"round {round_index} produced no transition after "
                        f"{self.config.max_rollout_attempts} attempts"
                    )
                current_state = (
                    run.traces[-1].next_state if run.traces else environment.state
                )
                advantages = estimate_ema_advantages(
                    run,
                    baseline,
                    gamma=self.config.gamma,
                )
                optimization = optimize_harness_with_attribution(
                    harness=frozen_harness,
                    batch=advantages,
                    attributor=self.attributor,
                    semantic_engine=self.semantic_engine,
                    edit_optimizer=self.edit_optimizer,
                )
                record = TrainingRound(
                    round_index, run, advantages, optimization
                )
                rounds.append(record)
                self.harness = optimization.harness
                if self.harness_repository is not None:
                    self.harness_repository.commit(
                        round_index,
                        self.harness,
                        optimization.edits,
                        optimization.selected_gradients,
                    )
                selection = self._select_factor_pool(record)
                if selection is not None and selection[0] > best_score:
                    best_score, best_pool, best_metrics = selection
                    best_round = round_index
                    self._save_best_pool(
                        best_pool,
                        best_score,
                        best_round,
                        best_metrics,
                    )
                self._checkpoint(record)
                _log_attributed_round(writer, record, selection)
        finally:
            if writer is not None:
                writer.close()
        return TrainingResult(
            self.harness,
            tuple(rounds),
            current_state.factor_pool,
            best_pool,
            best_score,
            best_round,
        )

    def _select_factor_pool(
        self,
        record: TrainingRound,
    ) -> tuple[float, FactorPool, dict[str, Any]] | None:
        if not record.run.traces:
            return None
        pool = record.run.traces[-1].next_state.factor_pool
        if self.selection_evaluator is not None:
            metrics = dict(self.selection_evaluator.evaluate_pool(pool))
            score = float(metrics["score"])
        else:
            score = record.advantages.records[0].return_to_go
            metrics = {"score": score}
        return score, pool, metrics

    def _save_best_pool(
        self,
        pool: FactorPool,
        score: float,
        round_index: int,
        metrics: dict[str, Any],
    ) -> None:
        if self.config.checkpoint_dir is None:
            return
        directory = Path(self.config.checkpoint_dir).resolve().parent
        save_factor_pool(pool, directory / "best-factor-pool.json")
        write_json(
            directory / "best-factor-pool-metadata.json",
            {
                "selection_split": (
                    "validation" if self.selection_evaluator is not None else "train"
                ),
                "selection_score": score,
                "round": round_index,
                "metrics": metrics,
            },
        )

    def _checkpoint(self, record: TrainingRound) -> None:
        if self.config.checkpoint_dir is None:
            return
        directory = Path(self.config.checkpoint_dir).resolve()
        round_directory = directory / f"round-{record.index:04d}"
        save_harness(record.optimization.harness, round_directory / "harness")
        write_json(
            round_directory / "trajectory.json",
            _trajectory_payload(record.run, record.advantages),
        )
        write_json(
            round_directory / "rollout-factor-pools.json",
            {
                "initial_factors": (
                    factor_pool_to_list(record.run.traces[0].state.factor_pool)
                    if record.run.traces
                    else []
                ),
                "final_factors": (
                    factor_pool_to_list(
                        record.run.traces[-1].next_state.factor_pool
                    )
                    if record.run.traces
                    else []
                ),
            },
        )
        write_json(
            round_directory / "attribution.json",
            {
                "round": record.index,
                "mean_discounted_return": record.mean_discounted_return,
                "attributions": [
                    {
                        "evidence_id": item.evidence_id,
                        "component": item.component.kind.value,
                        "item_id": item.component.item_id,
                        "rollout_index": item.rollout_index,
                        "step_index": item.step_index,
                        "advantage": item.record.advantage,
                        "baseline": item.record.baseline,
                        "full_logprob": item.full_logprob,
                        "masked_logprob": item.masked_logprob,
                        "influence": item.influence,
                        "credit": item.credit,
                    }
                    for item in record.optimization.attributions
                ],
                "proposals": [
                    {
                        "component": item.component.kind.value,
                        "item_id": item.component.item_id,
                        "parameter": item.parameter.component.value,
                        "rationale": item.rationale,
                        "feedback": item.feedback,
                        "support_ids": list(item.support_ids),
                        "conflict_ids": list(item.conflict_ids),
                        "score": item.score,
                    }
                    for item in record.optimization.proposals
                ],
                "edits": [
                    {
                        "component": edit.parameter.component.value,
                        "item_id": edit.parameter.item_id,
                        "before": edit.before,
                        "after": edit.after,
                    }
                    for edit in record.optimization.edits
                ],
                "harness_path": "harness",
            },
        )


def _summary_writer(path: str | Path | None):
    if path is None:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise ImportError(
            "TensorBoard logging requires: uv sync --extra attribution"
        ) from error
    return SummaryWriter(log_dir=str(Path(path).resolve()))


def _trajectory_payload(run: AgentRun, advantages: AdvantageBatch) -> dict[str, Any]:
    credit_by_step = {item.step_index: item for item in advantages.records}
    return {
        "stop_reason": run.stop_reason,
        "model_steps": run.steps,
        "usage": run.usage,
        "steps": [
            {
                "step_index": trace.index,
                "factor_pool_before": factor_pool_to_list(trace.state.factor_pool),
                "skill": {
                    "id": trace.skill.id,
                    "description": trace.skill.description,
                    "strategy": trace.skill.strategy,
                },
                "memory_query": trace.memory_query,
                "memory": [
                    {
                        "id": entry.id,
                        "key": entry.key,
                        "value": entry.value,
                    }
                    for entry in trace.retrieved_memory
                ],
                "action": {
                    "operation": trace.action.operation.value,
                    "target": trace.action.target,
                    "factor": (
                        {
                            "name": trace.action.factor.name,
                            "expression": trace.action.factor.expression,
                        }
                        if trace.action.factor is not None
                        else None
                    ),
                },
                "accepted": trace.transition.accepted,
                "redundancy": trace.transition.redundancy,
                "previous_score": trace.transition.previous_score,
                "next_score": trace.transition.next_score,
                "reward": trace.reward,
                "return_to_go": credit_by_step[trace.index].return_to_go,
                "baseline": credit_by_step[trace.index].baseline,
                "advantage": credit_by_step[trace.index].advantage,
                "factor_pool_after": factor_pool_to_list(trace.next_state.factor_pool),
                "interaction": [
                    {
                        "role": message.role,
                        "content": message.content,
                        "tool_calls": [
                            {"name": call.name, "arguments": call.arguments}
                            for call in message.tool_calls
                        ],
                        "tool_name": message.name,
                        "tool_output": message.output,
                        "tool_error": message.error,
                    }
                    for message in trace.interaction
                ],
            }
            for trace in run.traces
        ],
    }


def _log_attributed_round(writer, record, selection) -> None:
    if writer is None:
        return
    step = record.index
    returns = [
        item.return_to_go for item in record.advantages.records if item.step_index == 0
    ]
    transitions = [trace.transition for trace in record.run.traces]
    usage = record.run.usage
    writer.add_scalar("train/return_mean", fmean(returns), step)
    writer.add_scalar(
        "train/return_std", pstdev(returns) if len(returns) > 1 else 0.0, step
    )
    writer.add_scalar(
        "train/reward_mean",
        fmean(item.reward for item in transitions) if transitions else 0.0,
        step,
    )
    writer.add_scalar(
        "train/baseline_mean",
        fmean(item.baseline for item in record.advantages.records),
        step,
    )
    writer.add_scalar(
        "train/advantage_mean",
        fmean(item.advantage for item in record.advantages.records),
        step,
    )
    writer.add_scalar(
        "train/accepted_transition_rate",
        fmean(float(item.accepted) for item in transitions) if transitions else 0.0,
        step,
    )
    writer.add_scalar(
        "train/final_factor_count",
        len(record.run.traces[-1].next_state.factor_pool)
        if record.run.traces
        else 0.0,
        step,
    )
    for key, value in usage.items():
        writer.add_scalar(f"tokens/{key}", value, step)
    attributions = record.optimization.attributions
    writer.add_scalar("attribution/count", len(attributions), step)
    writer.add_scalar(
        "attribution/correction_weight",
        sum(item.correction_weight for item in attributions),
        step,
    )
    writer.add_scalar(
        "attribution/preservation_weight",
        sum(item.preservation_weight for item in attributions),
        step,
    )
    writer.add_scalar("optimizer/proposals", len(record.optimization.proposals), step)
    writer.add_scalar("optimizer/edits", len(record.optimization.edits), step)
    if selection is not None:
        writer.add_scalar("selection/best_pool_score", selection[0], step)
        for name in (
            "ic",
            "icir",
            "rank_ic",
            "rank_icir",
            "coverage",
            "gross_total_return",
            "total_return",
            "benchmark_total_return",
            "gross_annualized_return",
            "annualized_return",
            "benchmark_annualized_return",
            "annualized_excess_return",
            "sharpe",
            "information_ratio",
            "max_drawdown",
            "turnover",
            "total_transaction_cost",
            "average_transaction_cost",
            "n_portfolio_dates",
        ):
            value = selection[2].get(name)
            if isinstance(value, (int, float)):
                writer.add_scalar(f"selection/{name}", value, step)
