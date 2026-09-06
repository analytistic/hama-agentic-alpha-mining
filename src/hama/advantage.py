"""Single-rollout return-to-go and historical EMA advantages."""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import AgentRun, HarnessTrace


@dataclass(frozen=True)
class AdvantageRecord:
    """Return-to-go and historical-baseline advantage for one transition."""

    rollout_index: int
    step_index: int
    trace: HarnessTrace
    return_to_go: float
    advantage: float
    baseline: float = 0.0


@dataclass(frozen=True)
class AdvantageBatch:
    """Advantages for one complete trajectory."""

    horizon: int
    gamma: float
    records: tuple[AdvantageRecord, ...]

    def for_rollout(self, rollout_index: int = 0) -> tuple[AdvantageRecord, ...]:
        return tuple(
            record for record in self.records if record.rollout_index == rollout_index
        )


@dataclass
class EMAReturnBaseline:
    """Per-step running baselines computed only from preceding rollouts."""

    update_rate: float = 0.1
    initial_value: float = 0.0
    values: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.update_rate <= 1.0:
            raise ValueError("update_rate must be in (0, 1]")

    def value(self, step_index: int) -> float:
        return self.values.get(step_index, self.initial_value)

    def update(self, returns: tuple[float, ...]) -> None:
        eta = self.update_rate
        for step_index, return_to_go in enumerate(returns):
            previous = self.value(step_index)
            self.values[step_index] = (1.0 - eta) * previous + eta * return_to_go


def estimate_ema_advantages(
    run: AgentRun,
    baseline: EMAReturnBaseline,
    *,
    gamma: float = 1.0,
) -> AdvantageBatch:
    """Estimate ``A_t = G_t - b_t`` and then update ``b_t`` by EMA."""

    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if not run.traces:
        raise ValueError("the rollout must contain at least one transition")

    returns = [0.0] * len(run.traces)
    running = 0.0
    for step_index in range(len(run.traces) - 1, -1, -1):
        running = run.traces[step_index].reward + gamma * running
        returns[step_index] = running

    frozen_baselines = tuple(baseline.value(index) for index in range(len(returns)))
    records = tuple(
        AdvantageRecord(
            rollout_index=0,
            step_index=step_index,
            trace=run.traces[step_index],
            return_to_go=return_to_go,
            advantage=return_to_go - frozen_baselines[step_index],
            baseline=frozen_baselines[step_index],
        )
        for step_index, return_to_go in enumerate(returns)
    )
    baseline.update(tuple(returns))
    return AdvantageBatch(
        horizon=len(run.traces),
        gamma=gamma,
        records=records,
    )
