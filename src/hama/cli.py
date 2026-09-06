"""Command-line entry points for HAMA training and evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .agent import Agent
from .attribution import ComponentAttributor
from .environment import AlphaMiningEnvironment
from .evaluation import evaluate_policy
from .model import Model
from .optimization import (
    ConflictAwareSemanticGradientEngine,
    HarnessEvolutionOptimizer,
)
from .persistence import (
    factor_pool_from_list,
    load_factor_pool,
    load_harness,
    read_json,
    save_factor_pool,
    save_harness,
    write_json,
)
from .prefill import QwenPrefillScorer
from .qlib_evaluator import QlibEvaluatorConfig, QlibFactorEvaluator
from .report import build_training_report
from .repository import HarnessRepository
from .training import HamaTrainer, TrainingConfig
from .types import MarketState


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hama")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="optimize skill and memory")
    train_parser.add_argument("config", type=Path)
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate a frozen harness on one held-out split"
    )
    evaluate_parser.add_argument("config", type=Path)
    evaluate_parser.add_argument("--harness", type=Path)
    evaluate_parser.add_argument("--factor-pool", type=Path)
    evaluate_parser.add_argument("--split", default="test")
    backtest_parser = subparsers.add_parser(
        "backtest", help="evaluate one fixed factor pool without agent rollouts"
    )
    backtest_parser.add_argument("config", type=Path)
    backtest_parser.add_argument("--factor-pool", type=Path, required=True)
    backtest_parser.add_argument("--split", default="test")
    report_parser = subparsers.add_parser(
        "report", help="build one offline HTML report from checkpoint directories"
    )
    report_parser.add_argument("output", type=Path)
    report_parser.add_argument("run_dirs", type=Path, nargs="+")
    report_parser.add_argument("--title", default="HAMA training record")

    arguments = parser.parse_args(argv)
    if arguments.command == "report":
        build_training_report(
            arguments.output,
            arguments.run_dirs,
            title=arguments.title,
        )
        return 0
    config = read_json(arguments.config)
    if arguments.command == "train":
        _train(config, arguments.config.parent)
    elif arguments.command == "backtest":
        _backtest_fixed_pool(
            config,
            arguments.config.parent,
            arguments.factor_pool,
            arguments.split,
        )
    else:
        _evaluate(
            config,
            arguments.config.parent,
            arguments.harness,
            arguments.factor_pool,
            arguments.split,
        )
    return 0


def _backtest_fixed_pool(
    config: Mapping[str, Any],
    config_dir: Path,
    factor_pool_path: Path,
    split: str,
) -> None:
    """Evaluate a persisted factor pool without invoking the agent policy."""

    output_dir = _path(config_dir, config.get("output_dir", "hama-output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_pool = load_factor_pool(factor_pool_path)
    metrics = dict(_evaluator(config, split).evaluate_pool(factor_pool))
    write_json(
        output_dir / f"backtest-{split}.json",
        {
            "split": split,
            "factor_pool": [
                {"name": factor.name, "expression": factor.expression}
                for factor in factor_pool
            ],
            "metrics": metrics,
        },
    )


def _train(config: Mapping[str, Any], config_dir: Path) -> None:
    output_dir = _path(config_dir, config.get("output_dir", "hama-output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    harness = _load_configured_harness(config, config_dir)
    model = _model(config)
    try:
        evaluator = _evaluator(config, "train")
        qlib_config = _object(config["qlib"], "qlib")
        split_config = _object(qlib_config["splits"], "qlib.splits")
        selection_evaluator = (
            _evaluator(config, "valid") if "valid" in split_config else None
        )
        state = _state(config)
        environment_config = _object(config.get("environment", {}), "environment")
        training = _object(config["training"], "training")
        agent_config = _object(config.get("agent", {}), "agent")

        prefill_scorer = _prefill_scorer(config)
        harness_repository = HarnessRepository(output_dir / "harness-repo")
        harness_repository.initialize(harness)
        trainer = HamaTrainer(
            harness=harness,
            initial_state=state,
            task=config.get("task", "Improve the current alpha-factor pool."),
            agent_factory=lambda frozen: Agent(
                model=model,
                system_prompt=str(
                    agent_config.get(
                        "system_prompt",
                        "You are an agentic alpha-mining researcher.",
                    )
                ),
                harness=frozen,
                max_steps=int(agent_config.get("max_steps", 50)),
                max_analysis_calls_per_state=int(
                    agent_config.get("max_analysis_calls_per_state", 5)
                ),
            ),
            environment_factory=lambda round_index, current_state: (
                AlphaMiningEnvironment(
                    state=current_state,
                    evaluator=evaluator,
                    workspace=(
                        output_dir
                        / "workspaces"
                        / f"round-{round_index:04d}"
                    ),
                    horizon=int(environment_config.get("horizon", 1)),
                    redundancy_threshold=float(
                        environment_config.get("redundancy_threshold", 0.7)
                    ),
                    exploration_acceptance=float(
                        environment_config.get("exploration_acceptance", 0.0)
                    ),
                    tool_timeout=float(environment_config.get("tool_timeout", 60.0)),
                )
            ),
            attributor=ComponentAttributor(prefill_scorer),
            semantic_engine=ConflictAwareSemanticGradientEngine(
                model,
                conflict_penalty=float(training.get("conflict_penalty", 1.0)),
                max_correction_evidence=int(training.get("max_correction_evidence", 6)),
                max_preservation_evidence=int(
                    training.get("max_preservation_evidence", 3)
                ),
                min_score=float(training.get("min_gradient_score", 0.0)),
            ),
            edit_optimizer=HarnessEvolutionOptimizer(
                model,
                harness_repository.history,
                max_changes=int(training.get("max_harness_changes", 4)),
            ),
            config=TrainingConfig(
                rounds=int(training["rounds"]),
                gamma=float(training.get("gamma", 1.0)),
                baseline_update_rate=float(
                    training.get("baseline_update_rate", 0.1)
                ),
                baseline_initial_value=float(
                    training.get("baseline_initial_value", 0.0)
                ),
                baseline_initial_values={
                    int(step): float(value)
                    for step, value in _object(
                        training.get("baseline_initial_values", {}),
                        "training.baseline_initial_values",
                    ).items()
                },
                max_rollout_attempts=int(training.get("max_rollout_attempts", 2)),
                checkpoint_dir=output_dir / "checkpoints",
                tensorboard_dir=(
                    output_dir / "tensorboard"
                    if bool(training.get("tensorboard", False))
                    else None
                ),
            ),
            selection_evaluator=selection_evaluator,
            harness_repository=harness_repository,
        )
        result = trainer.train()
        final_harness = save_harness(result.harness, output_dir / "harness-final")
        final_pool_path = save_factor_pool(
            result.final_factor_pool,
            output_dir / "final-factor-pool.json",
        )
        if result.best_factor_pool:
            save_factor_pool(
                result.best_factor_pool,
                output_dir / "best-factor-pool.json",
            )
        write_json(
            output_dir / "training-summary.json",
            {
                "rounds": len(result.rounds),
                "mean_discounted_return": [
                    item.mean_discounted_return for item in result.rounds
                ],
                "edits_per_round": [
                    len(item.optimization.edits) for item in result.rounds
                ],
                "harness_path": str(final_harness),
                "final_factor_pool_path": str(final_pool_path),
                "best_factor_pool_path": str(output_dir / "best-factor-pool.json"),
                "best_selection_score": result.best_selection_score,
                "best_round": result.best_round,
                "selection_split": (
                    "valid" if selection_evaluator is not None else "train"
                ),
                "tensorboard_path": str(output_dir / "tensorboard"),
            },
        )
    finally:
        model.close()


def _evaluate(
    config: Mapping[str, Any],
    config_dir: Path,
    harness_path: Path | None,
    factor_pool_path: Path | None,
    split: str,
) -> None:
    output_dir = _path(config_dir, config.get("output_dir", "hama-output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    final_harness = output_dir / "harness-final"
    harness = (
        load_harness(harness_path)
        if harness_path is not None
        else (
            load_harness(final_harness)
            if final_harness.exists()
            else _load_configured_harness(config, config_dir)
        )
    )
    model = _model(config)
    try:
        evaluator = _evaluator(config, split)
        state = _state(config)
        environment_config = _object(config.get("environment", {}), "environment")
        evaluation_config = _object(config.get("evaluation", {}), "evaluation")
        agent_config = _object(config.get("agent", {}), "agent")
        result = evaluate_policy(
            task=config.get("task", "Improve the current alpha-factor pool."),
            harness=harness,
            agent_factory=lambda frozen: Agent(
                model=model,
                system_prompt=str(
                    agent_config.get(
                        "system_prompt",
                        "You are an agentic alpha-mining researcher.",
                    )
                ),
                harness=frozen,
                max_steps=int(agent_config.get("max_steps", 50)),
                max_analysis_calls_per_state=int(
                    agent_config.get("max_analysis_calls_per_state", 5)
                ),
            ),
            environment_factory=lambda rollout_index: AlphaMiningEnvironment(
                state=state,
                evaluator=evaluator,
                workspace=(
                    output_dir
                    / "evaluation-workspaces"
                    / split
                    / f"rollout-{rollout_index:04d}"
                ),
                horizon=int(environment_config.get("horizon", 1)),
                redundancy_threshold=float(
                    environment_config.get("redundancy_threshold", 0.7)
                ),
                exploration_acceptance=0.0,
                tool_timeout=float(environment_config.get("tool_timeout", 60.0)),
            ),
            num_rollouts=int(evaluation_config.get("num_rollouts", 1)),
            gamma=float(evaluation_config.get("gamma", 1.0)),
        )
        configured_pool_path = factor_pool_path
        if configured_pool_path is None:
            candidate = output_dir / "best-factor-pool.json"
            configured_pool_path = candidate if candidate.exists() else None
        fixed_pool = (
            load_factor_pool(configured_pool_path)
            if configured_pool_path is not None
            else state.factor_pool
        )
        fixed_pool_metrics = evaluator.evaluate_pool(fixed_pool)
        write_json(
            output_dir / f"evaluation-{split}.json",
            {
                "split": split,
                "mean_discounted_return": result.mean_discounted_return,
                "std_discounted_return": result.std_discounted_return,
                "horizon_completion_rate": result.horizon_completion_rate,
                "accepted_transition_rate": result.accepted_transition_rate,
                "fixed_factor_pool": [
                    {"name": factor.name, "expression": factor.expression}
                    for factor in fixed_pool
                ],
                "fixed_factor_pool_metrics": dict(fixed_pool_metrics),
                "rollouts": [
                    {
                        "stop_reason": run.stop_reason,
                        "steps": run.steps,
                        "rewards": [trace.reward for trace in run.traces],
                        "final_factor_pool": (
                            [
                                {
                                    "name": factor.name,
                                    "expression": factor.expression,
                                }
                                for factor in run.transition.next_state.factor_pool
                            ]
                            if run.transition is not None
                            else []
                        ),
                        "usage": run.usage,
                    }
                    for run in result.runs
                ],
            },
        )
    finally:
        model.close()


def _model(config: Mapping[str, Any]) -> Model:
    value = _object(config["model"], "model")
    return Model(
        model=str(value["name"]),
        base_url=str(value["base_url"]) if value.get("base_url") else None,
        timeout=float(value.get("timeout", 180.0)),
        max_retries=int(value.get("max_retries", 3)),
        retry_backoff=float(value.get("retry_backoff", 2.0)),
        temperature=(
            float(value["temperature"])
            if value.get("temperature") is not None
            else None
        ),
        max_tokens=(
            int(value["max_tokens"]) if value.get("max_tokens") is not None else None
        ),
    )


def _prefill_scorer(config: Mapping[str, Any]) -> QwenPrefillScorer:
    value = _object(config["prefill"], "prefill")
    device_map = value.get("device_map", "auto")
    if device_map is not None and not isinstance(device_map, (str, dict)):
        raise TypeError("prefill.device_map must be a string, object, or null")
    return QwenPrefillScorer(
        model_name=str(value["model"]),
        device_map=device_map,
        torch_dtype=value.get("torch_dtype", "auto"),
        trust_remote_code=bool(value.get("trust_remote_code", False)),
        max_length=(
            int(value["max_length"]) if value.get("max_length") is not None else None
        ),
    )


def _evaluator(config: Mapping[str, Any], split: str) -> QlibFactorEvaluator:
    data = _object(config["qlib"], "qlib")
    portfolio = _object(data.get("portfolio", {}), "qlib.portfolio")
    splits = _object(data["splits"], "qlib.splits")
    if split not in splits:
        raise KeyError(f"unknown Qlib split: {split}")
    window = _object(splits[split], f"qlib.splits.{split}")
    return QlibFactorEvaluator(
        QlibEvaluatorConfig(
            provider_uri=data["provider_uri"],
            instruments=data["instruments"],
            start_time=str(window["start_time"]),
            end_time=str(window["end_time"]),
            label=str(data.get("label", "Ref($close, -1) / $close - 1")),
            region=str(data.get("region", "cn")),
            min_pairs=int(data.get("min_pairs", 5)),
            score_metric=str(data.get("score_metric", "ic")),
            portfolio_top_k=int(portfolio.get("top_k", 50)),
            annualization_factor=int(portfolio.get("annualization_factor", 252)),
            buy_cost=float(portfolio.get("buy_cost", 0.0005)),
            sell_cost=float(portfolio.get("sell_cost", 0.001)),
        )
    )


def _state(config: Mapping[str, Any]) -> MarketState:
    return MarketState(
        factor_pool=factor_pool_from_list(config.get("initial_factor_pool", [])),
        market_information=_object(
            config.get("market_information", {}), "market_information"
        ),
    )


def _load_configured_harness(config: Mapping[str, Any], config_dir: Path):
    if not config.get("harness_path"):
        raise KeyError("config must define a Harness directory in harness_path")
    return load_harness(_path(config_dir, config["harness_path"]))


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _path(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
