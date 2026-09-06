"""Concrete Qlib evaluator for HAMA factor and factor-pool rewards."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .types import Factor, FactorPool

DEFAULT_LABEL = "Ref($close, -1) / $close - 1"


class FactorEvaluationError(RuntimeError):
    """Raised when Qlib cannot evaluate an expression on the configured split."""


class FactorCombiner(Protocol):
    """Fixed downstream combination procedure c(X; F)."""

    def combine(self, features: pd.DataFrame) -> pd.Series: ...


@dataclass(frozen=True)
class EqualWeightZScoreCombiner:
    """Cross-sectionally z-score factors and average them with fixed weights."""

    clip: float | None = 3.0

    def combine(self, features: pd.DataFrame) -> pd.Series:
        if features.empty or features.shape[1] == 0:
            return pd.Series(index=features.index, dtype=float, name="pool_score")
        level = _datetime_level(features.index)
        grouped = features.groupby(level=level, sort=False)
        means = grouped.transform("mean")
        stds = grouped.transform(lambda values: values.std(ddof=0)).replace(0.0, np.nan)
        normalized = (features - means) / stds
        if self.clip is not None:
            normalized = normalized.clip(-self.clip, self.clip)
        return normalized.mean(axis=1, skipna=True).rename("pool_score")


@dataclass(frozen=True)
class QlibEvaluatorConfig:
    """One immutable Qlib data split used for reward computation."""

    provider_uri: str | Mapping[str, str]
    instruments: str | Sequence[str]
    start_time: str
    end_time: str
    label: str = DEFAULT_LABEL
    region: str = "cn"
    min_pairs: int = 5
    score_metric: str = "ic"
    portfolio_top_k: int = 50
    annualization_factor: int = 252
    buy_cost: float = 0.0005
    sell_cost: float = 0.001

    def __post_init__(self) -> None:
        if self.min_pairs < 2:
            raise ValueError("min_pairs must be at least 2")
        if self.score_metric not in {"ic", "rank_ic", "abs_ic", "abs_rank_ic"}:
            raise ValueError("score_metric must be ic, rank_ic, abs_ic, or abs_rank_ic")
        if self.portfolio_top_k < 1:
            raise ValueError("portfolio_top_k must be positive")
        if self.annualization_factor < 1:
            raise ValueError("annualization_factor must be positive")
        if self.buy_cost < 0.0 or self.sell_cost < 0.0:
            raise ValueError("transaction costs must be non-negative")


LoaderFactory = Callable[[Mapping[str, Any]], Any]


class QlibFactorEvaluator:
    """Evaluate Qlib expressions and the fixed combined factor-pool signal.

    The same immutable split is used for the old and candidate pools, so the
    environment reward is exactly their marginal IC contribution. Expression
    values are cached because the old pool is evaluated repeatedly.
    """

    def __init__(
        self,
        config: QlibEvaluatorConfig,
        *,
        combiner: FactorCombiner | None = None,
        initialize_qlib: bool = True,
        loader_factory: LoaderFactory | None = None,
    ) -> None:
        self.config = config
        self.combiner = combiner or EqualWeightZScoreCombiner()
        self._series_cache: dict[str, pd.Series] = {}
        self._label: pd.Series | None = None
        self._pool_cache: dict[tuple[str, ...], dict[str, Any]] = {}

        if loader_factory is None:
            if initialize_qlib:
                _initialize_qlib(config)
            from qlib.data.dataset.loader import QlibDataLoader

            self._loader_factory: LoaderFactory = lambda value: QlibDataLoader(
                config=value
            )
        else:
            self._loader_factory = loader_factory

    def evaluate_factors(
        self,
        factors: Sequence[Factor],
        *,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> Mapping[str, Any]:
        """Return independent IC diagnostics for every supplied factor."""

        _validate_factors(factors)
        if not factors:
            return {"factors": {}, "split": self._split_metadata()}
        features, label = self._load(factors)
        features, label = self._slice(features, label, start_time, end_time)
        results = {
            factor.name: {
                "expression": factor.expression,
                **_signal_metrics(features[factor.name], label, self.config.min_pairs),
            }
            for factor in factors
        }
        split = self._split_metadata()
        split["evaluation_start_time"] = start_time or self.config.start_time
        split["evaluation_end_time"] = end_time or self.config.end_time
        return {"factors": results, "split": split}

    def evaluate_pool(self, factor_pool: FactorPool) -> Mapping[str, Any]:
        """Evaluate the fixed combination c(X; F) and expose its reward score."""

        _validate_factors(factor_pool)
        cache_key = tuple(factor.expression for factor in factor_pool)
        cached = self._pool_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        if not factor_pool:
            result = {
                "score": 0.0,
                "ic": 0.0,
                "icir": 0.0,
                "rank_ic": 0.0,
                "rank_icir": 0.0,
                "coverage": 0.0,
                "n_dates": 0,
                "factor_count": 0,
                "score_metric": self.config.score_metric,
                **_empty_portfolio_metrics(),
                "split": self._split_metadata(),
            }
            self._pool_cache[cache_key] = result
            return dict(result)

        features, label = self._load(factor_pool)
        combined = self.combiner.combine(features)
        metrics = _signal_metrics(combined, label, self.config.min_pairs)
        portfolio_metrics = _portfolio_metrics(combined, label, self.config)
        score = _score_from_metrics(metrics, self.config.score_metric)
        result = {
            "score": score,
            **metrics,
            **portfolio_metrics,
            "factor_count": len(factor_pool),
            "score_metric": self.config.score_metric,
            "split": self._split_metadata(),
        }
        self._pool_cache[cache_key] = result
        return dict(result)

    def redundancy(self, factor: Factor, factor_pool: FactorPool) -> float:
        """Return max absolute mean daily correlation with the comparison pool."""

        if not factor_pool:
            return 0.0
        _validate_factors((factor, *factor_pool), allow_duplicate_names=True)
        all_factors = (Factor("__candidate__", factor.expression),) + tuple(
            Factor(f"__pool_{index}__", item.expression)
            for index, item in enumerate(factor_pool)
        )
        features, _ = self._load(all_factors)
        candidate = features["__candidate__"]
        correlations = [
            _mean_daily_correlation(
                candidate,
                features[f"__pool_{index}__"],
                self.config.min_pairs,
            )
            for index in range(len(factor_pool))
        ]
        finite = [abs(value) for value in correlations if math.isfinite(value)]
        return max(finite, default=0.0)

    def _load(self, factors: Sequence[Factor]) -> tuple[pd.DataFrame, pd.Series]:
        missing = [
            factor for factor in factors if factor.expression not in self._series_cache
        ]
        if missing:
            names = [f"factor_{index}" for index in range(len(missing))]
            loader_config = {
                "feature": ([factor.expression for factor in missing], names),
                "label": ([self.config.label], ["target"]),
            }
            try:
                raw = self._loader_factory(loader_config).load(
                    instruments=self.config.instruments,
                    start_time=self.config.start_time,
                    end_time=self.config.end_time,
                )
            except Exception as error:
                expressions = [factor.expression for factor in missing]
                raise FactorEvaluationError(
                    f"Qlib failed to evaluate expressions {expressions}: {error}"
                ) from error
            feature_frame, label = _split_qlib_frame(raw, names)
            for factor, name in zip(missing, names, strict=True):
                self._series_cache[factor.expression] = feature_frame[name].rename(
                    factor.expression
                )
            if self._label is None:
                self._label = label.rename("target")

        if self._label is None:
            raise FactorEvaluationError("Qlib returned no target label")
        columns = {
            factor.name: self._series_cache[factor.expression] for factor in factors
        }
        frame = pd.concat(columns, axis=1).replace([np.inf, -np.inf], np.nan)
        label = self._label.reindex(frame.index).replace([np.inf, -np.inf], np.nan)
        return frame, label

    def _split_metadata(self) -> dict[str, Any]:
        return {
            "instruments": self.config.instruments,
            "start_time": self.config.start_time,
            "end_time": self.config.end_time,
            "label": self.config.label,
            "portfolio": {
                "strategy": "top_k_equal_weight",
                "top_k": self.config.portfolio_top_k,
                "benchmark": "universe_equal_weight",
                "annualization_factor": self.config.annualization_factor,
                "buy_cost": self.config.buy_cost,
                "sell_cost": self.config.sell_cost,
            },
        }

    def _slice(
        self,
        features: pd.DataFrame,
        label: pd.Series,
        start_time: str | None,
        end_time: str | None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        start = pd.Timestamp(start_time or self.config.start_time)
        end = pd.Timestamp(end_time or self.config.end_time)
        configured_start = pd.Timestamp(self.config.start_time)
        configured_end = pd.Timestamp(self.config.end_time)
        if start < configured_start or end > configured_end:
            raise ValueError("backtest window must stay inside the evaluator split")
        if start > end:
            raise ValueError("backtest start_time must not exceed end_time")
        level = _datetime_level(features.index)
        dates = pd.to_datetime(features.index.get_level_values(level))
        mask = (dates >= start) & (dates <= end)
        sliced_features = features.loc[mask]
        if sliced_features.empty:
            raise FactorEvaluationError(
                "the requested backtest window contains no rows"
            )
        return sliced_features, label.reindex(sliced_features.index)


def _initialize_qlib(config: QlibEvaluatorConfig) -> None:
    import qlib

    provider_uri: str | dict[str, str]
    if isinstance(config.provider_uri, str):
        provider_uri = str(Path(config.provider_uri).expanduser())
    else:
        provider_uri = {
            frequency: str(Path(path).expanduser())
            for frequency, path in config.provider_uri.items()
        }
    qlib.init(provider_uri=provider_uri, region=config.region)


def _split_qlib_frame(
    raw: Any,
    feature_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series]:
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        raise FactorEvaluationError("Qlib returned an empty data frame")

    if isinstance(raw.columns, pd.MultiIndex):
        try:
            features = raw.xs("feature", axis=1, level=0, drop_level=True)
            labels = raw.xs("label", axis=1, level=0, drop_level=True)
        except KeyError as error:
            raise FactorEvaluationError(
                "Qlib output does not contain feature and label column groups"
            ) from error
    else:
        missing = [name for name in [*feature_names, "target"] if name not in raw]
        if missing:
            raise FactorEvaluationError(f"Qlib output is missing columns: {missing}")
        features = raw.loc[:, list(feature_names)]
        labels = raw.loc[:, ["target"]]

    missing_features = [name for name in feature_names if name not in features]
    if missing_features or "target" not in labels:
        raise FactorEvaluationError(
            f"Qlib output is missing factors={missing_features} or target"
        )
    features = features.loc[:, list(feature_names)].astype(float)
    label = labels["target"].astype(float)
    return features, label


def _signal_metrics(
    signal: pd.Series,
    label: pd.Series,
    min_pairs: int,
) -> dict[str, Any]:
    aligned = pd.concat(
        [signal.rename("signal"), label.rename("target")], axis=1
    ).replace([np.inf, -np.inf], np.nan)
    valid_count = int(aligned["signal"].notna().sum())
    coverage = valid_count / len(aligned) if len(aligned) else 0.0
    level = _datetime_level(aligned.index)
    daily_ic: list[float] = []
    daily_rank_ic: list[float] = []
    for _, group in aligned.groupby(level=level, sort=False):
        pair = group.dropna()
        if (
            len(pair) < min_pairs
            or pair["signal"].nunique() < 2
            or pair["target"].nunique() < 2
        ):
            continue
        ic = pair["signal"].corr(pair["target"])
        rank_ic = pair["signal"].rank().corr(pair["target"].rank())
        if math.isfinite(ic):
            daily_ic.append(float(ic))
        if math.isfinite(rank_ic):
            daily_rank_ic.append(float(rank_ic))

    ic_mean, ic_std = _mean_std(daily_ic)
    rank_mean, rank_std = _mean_std(daily_rank_ic)
    return {
        "ic": ic_mean,
        "ic_std": ic_std,
        "icir": ic_mean / ic_std if ic_std > 0.0 else 0.0,
        "rank_ic": rank_mean,
        "rank_ic_std": rank_std,
        "rank_icir": rank_mean / rank_std if rank_std > 0.0 else 0.0,
        "coverage": float(coverage),
        "n_dates": len(daily_ic),
    }


def _portfolio_metrics(
    signal: pd.Series,
    label: pd.Series,
    config: QlibEvaluatorConfig,
) -> dict[str, Any]:
    """Backtest a deterministic daily Top-K, equal-weight portfolio."""

    aligned = pd.concat(
        [signal.rename("signal"), label.rename("target")], axis=1
    ).replace([np.inf, -np.inf], np.nan)
    level = _datetime_level(aligned.index)
    previous_weights = pd.Series(dtype=float)
    gross_returns: list[float] = []
    net_returns: list[float] = []
    benchmark_returns: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []

    for _, group in aligned.groupby(level=level, sort=True):
        pair = group.dropna()
        if pair.empty:
            continue
        selected = pair.nlargest(min(config.portfolio_top_k, len(pair)), "signal")
        asset_index = _asset_index(selected.index)
        current_weights = pd.Series(1.0 / len(selected), index=asset_index, dtype=float)
        union = previous_weights.index.union(current_weights.index)
        delta = current_weights.reindex(
            union, fill_value=0.0
        ) - previous_weights.reindex(union, fill_value=0.0)
        buy_turnover = float(delta.clip(lower=0.0).sum())
        sell_turnover = float((-delta.clip(upper=0.0)).sum())
        transaction_cost = (
            config.buy_cost * buy_turnover + config.sell_cost * sell_turnover
        )
        gross_return = float(selected["target"].mean())
        benchmark_return = float(pair["target"].mean())

        gross_returns.append(gross_return)
        net_returns.append(gross_return - transaction_cost)
        benchmark_returns.append(benchmark_return)
        costs.append(transaction_cost)
        # Report conventional one-way turnover; initial portfolio construction
        # remains charged above but is not treated as recurring turnover.
        if not previous_weights.empty:
            turnovers.append(0.5 * (buy_turnover + sell_turnover))
        previous_weights = current_weights

    if not net_returns:
        return _empty_portfolio_metrics()

    gross = np.asarray(gross_returns, dtype=float)
    net = np.asarray(net_returns, dtype=float)
    benchmark = np.asarray(benchmark_returns, dtype=float)
    excess = net - benchmark
    annualization = config.annualization_factor
    return {
        "gross_total_return": _total_return(gross),
        "total_return": _total_return(net),
        "benchmark_total_return": _total_return(benchmark),
        "gross_annualized_return": _annualized_return(gross, annualization),
        "annualized_return": _annualized_return(net, annualization),
        "benchmark_annualized_return": _annualized_return(benchmark, annualization),
        "annualized_excess_return": float(np.mean(excess) * annualization),
        "sharpe": _annualized_ratio(net, annualization),
        "information_ratio": _annualized_ratio(excess, annualization),
        "max_drawdown": _max_drawdown(net),
        "turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "total_transaction_cost": float(np.sum(costs)),
        "average_transaction_cost": float(np.mean(costs)),
        "n_portfolio_dates": len(net_returns),
    }


def _empty_portfolio_metrics() -> dict[str, Any]:
    return {
        "gross_total_return": 0.0,
        "total_return": 0.0,
        "benchmark_total_return": 0.0,
        "gross_annualized_return": 0.0,
        "annualized_return": 0.0,
        "benchmark_annualized_return": 0.0,
        "annualized_excess_return": 0.0,
        "sharpe": 0.0,
        "information_ratio": 0.0,
        "max_drawdown": 0.0,
        "turnover": 0.0,
        "total_transaction_cost": 0.0,
        "average_transaction_cost": 0.0,
        "n_portfolio_dates": 0,
    }


def _total_return(returns: np.ndarray) -> float:
    return float(np.prod(1.0 + returns) - 1.0)


def _annualized_return(returns: np.ndarray, periods: int) -> float:
    growth = float(np.prod(1.0 + returns))
    if growth <= 0.0:
        return -1.0
    return float(growth ** (periods / len(returns)) - 1.0)


def _annualized_ratio(returns: np.ndarray, periods: int) -> float:
    if len(returns) < 2:
        return 0.0
    standard_deviation = float(np.std(returns, ddof=1))
    if standard_deviation == 0.0:
        return 0.0
    return float(np.sqrt(periods) * np.mean(returns) / standard_deviation)


def _max_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], wealth)))
    drawdowns = wealth / peaks[1:] - 1.0
    return float(-np.min(drawdowns)) if len(drawdowns) else 0.0


def _mean_daily_correlation(
    left: pd.Series,
    right: pd.Series,
    min_pairs: int,
) -> float:
    aligned = pd.concat([left.rename("left"), right.rename("right")], axis=1)
    level = _datetime_level(aligned.index)
    values: list[float] = []
    for _, group in aligned.groupby(level=level, sort=False):
        pair = group.replace([np.inf, -np.inf], np.nan).dropna()
        if (
            len(pair) < min_pairs
            or pair["left"].nunique() < 2
            or pair["right"].nunique() < 2
        ):
            continue
        value = pair["left"].corr(pair["right"])
        if math.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else 0.0


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def _score_from_metrics(metrics: Mapping[str, Any], metric: str) -> float:
    absolute = metric.startswith("abs_")
    key = metric[4:] if absolute else metric
    value = float(metrics[key])
    return abs(value) if absolute else value


def _datetime_level(index: pd.Index) -> str | int:
    if isinstance(index, pd.MultiIndex) and "datetime" in index.names:
        return "datetime"
    return 0


def _asset_index(index: pd.Index) -> pd.Index:
    if isinstance(index, pd.MultiIndex):
        if "instrument" in index.names:
            return index.get_level_values("instrument")
        if index.nlevels > 1:
            return index.get_level_values(1)
    return index


def _validate_factors(
    factors: Sequence[Factor],
    *,
    allow_duplicate_names: bool = False,
) -> None:
    if not all(factor.name.strip() and factor.expression.strip() for factor in factors):
        raise ValueError("factor names and expressions must not be empty")
    for factor in factors:
        _validate_qlib_expression(factor.expression)
    if not allow_duplicate_names:
        names = [factor.name for factor in factors]
        if len(names) != len(set(names)):
            raise ValueError("factor names must be unique within a pool")


def _validate_qlib_expression(expression: str) -> None:
    """Reject element-wise clamp syntax that Qlib reads as a rolling window."""

    compact = "".join(expression.split())
    for operator in ("Max", "Min"):
        start = 0
        marker = f"{operator}("
        while (index := compact.find(marker, start)) >= 0:
            depth = 1
            comma = None
            cursor = index + len(marker)
            while cursor < len(compact) and depth:
                char = compact[cursor]
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                elif char == "," and depth == 1:
                    comma = cursor
                if depth == 0 and comma is not None:
                    argument = compact[comma + 1 : cursor]
                    if argument in {"0", "0.0", "+0", "-0"}:
                        raise ValueError(
                            f"{operator}(x, 0) is a rolling-window expression in "
                            f"Qlib; use {'Greater' if operator == 'Max' else 'Less'}(x, 0) "
                            "for an element-wise clamp"
                        )
                    break
                cursor += 1
            start = index + len(marker)
