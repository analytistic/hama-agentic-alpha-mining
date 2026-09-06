"""Paper-aligned agentic loop for one HAMA rollout."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any

from .environment import AlphaMiningEnvironment
from .harness import Harness, create_load_memory_tool, create_load_skill_tool
from .types import (
    AgentRun,
    CompletionModel,
    HarnessTrace,
    MemoryEntry,
    Message,
    Skill,
    Tool,
    ToolCall,
    ToolRunResult,
)

MAX_ROLLOUT_STEPS = 50
DEFAULT_MAX_ANALYSIS_CALLS_PER_STATE = 5
StepHook = Callable[[Message], None]


class LoopPhase(StrEnum):
    """The three actions in the paper's per-state policy factorization."""

    SKILL = "skill_loading"
    MEMORY = "memory_loading"
    FACTOR = "factor_pool_modification"


class Agent:
    """Execute one HAMA trajectory with a frozen harness and one context.

    At every MDP state, the runtime exposes actions in the paper's order:
    skill loading, query-parameterized memory loading, and one atomic
    factor-pool modification. After the environment returns ``s_{t+1}`` and
    ``r_{t+1}``, the same message context continues at the next state. The run
    ends at the environment horizon or after at most 50 model completions.
    """

    def __init__(
        self,
        *,
        model: CompletionModel,
        system_prompt: str,
        harness: Harness,
        max_steps: int = MAX_ROLLOUT_STEPS,
        max_analysis_calls_per_state: int = DEFAULT_MAX_ANALYSIS_CALLS_PER_STATE,
        on_step: StepHook | None = None,
    ) -> None:
        if not 1 <= max_steps <= MAX_ROLLOUT_STEPS:
            raise ValueError(f"max_steps must be between 1 and {MAX_ROLLOUT_STEPS}")
        if not harness.skills.skills:
            raise ValueError("the skill library must contain at least one skill")
        if max_analysis_calls_per_state < 0:
            raise ValueError("max_analysis_calls_per_state must be non-negative")
        self.model = model
        self.system_prompt = system_prompt
        self.harness = harness
        self.max_steps = max_steps
        self.max_analysis_calls_per_state = max_analysis_calls_per_state
        self.on_step = on_step

    def run(
        self,
        task: Any,
        environment: AlphaMiningEnvironment,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRun:
        """Run one bounded trajectory ``tau=(s_0,...,s_T)``."""

        observation = environment.reset()
        current_skill: dict[str, Skill] = {}
        current_memory: dict[str, object] = {}
        traces: list[HarnessTrace] = []
        phase = LoopPhase.SKILL
        messages = [
            Message(role="system", content=self._system_message()),
            Message(
                role="user",
                content=_task_message(
                    task,
                    observation,
                    self.harness.skills.descriptions(),
                ),
            ),
        ]
        trace_message_start = 1
        analysis_calls = 0
        force_required_action = False
        last_answer = ""

        for step in range(1, self.max_steps + 1):
            tools = self._phase_tools(
                phase=phase,
                current_skill=current_skill,
                current_memory=current_memory,
                environment=environment,
                analysis_calls=analysis_calls,
                force_required_action=force_required_action,
            )
            bound_tools = _bind_tools(tools)
            assistant = self.model.complete(
                messages,
                [tool.schema for tool in bound_tools.values()],
            )
            if assistant.role != "assistant":
                raise ValueError("model.complete() must return an assistant message")
            messages.append(assistant)
            self._notify(assistant)
            last_answer = assistant.content

            if not assistant.tool_calls:
                messages.append(
                    Message(
                        role="user",
                        content=(
                            f"Complete the current {phase.value} phase now by "
                            f"calling {_required_tool_name(phase)}."
                        ),
                    )
                )
                force_required_action = True
                continue

            phase_succeeded = False
            submitted = False
            submission_result: ToolRunResult | None = None
            stage_closed = False

            for call in assistant.tool_calls:
                if stage_closed:
                    result = ToolRunResult(
                        call_id=call.id,
                        name=call.name,
                        error="not executed after the factor-pool action completed",
                    )
                else:
                    result = _execute_tool(call, bound_tools.get(call.name))

                tool_message = Message(
                    role="tool",
                    tool_call_id=result.call_id,
                    name=result.name,
                    output=result.output,
                    error=result.error,
                )
                messages.append(tool_message)
                self._notify(tool_message)

                succeeded = result.error is None and call.name == _required_tool_name(
                    phase
                )
                phase_succeeded = phase_succeeded or succeeded
                if phase == LoopPhase.FACTOR and call.name != "submit_factor_edit":
                    analysis_calls += 1
                if succeeded and phase == LoopPhase.FACTOR:
                    submitted = True
                    submission_result = result
                    stage_closed = True

            if submitted:
                trace = _make_trace(
                    index=len(traces),
                    current_skill=current_skill,
                    current_memory=current_memory,
                    environment=environment,
                    interaction=messages[trace_message_start:],
                )
                traces.append(trace)
                if trace.transition.terminated:
                    return self._result(
                        messages=messages,
                        answer=last_answer,
                        stop_reason=(
                            submission_result.stop_reason
                            if submission_result is not None
                            else None
                        )
                        or "horizon_reached",
                        steps=step,
                        metadata=metadata,
                        current_skill=current_skill,
                        current_memory=current_memory,
                        environment=environment,
                        traces=traces,
                    )

                current_skill = {}
                current_memory = {}
                phase = LoopPhase.SKILL
                analysis_calls = 0
                force_required_action = False
                trace_message_start = len(messages) - 1
                continue

            if phase_succeeded and phase == LoopPhase.SKILL:
                phase = LoopPhase.MEMORY
                force_required_action = False
            elif phase_succeeded and phase == LoopPhase.MEMORY:
                phase = LoopPhase.FACTOR
                force_required_action = False

        return self._result(
            messages=messages,
            answer=last_answer,
            stop_reason="max_steps",
            steps=self.max_steps,
            metadata=metadata,
            current_skill=current_skill,
            current_memory=current_memory,
            environment=environment,
            traces=traces,
        )

    def _system_message(self) -> str:
        return (
            f"{self.system_prompt.rstrip()}\n\n"
            "At every state, follow the HAMA action order: load exactly one "
            "skill, load memory with exactly one query, then use analysis tools "
            "as needed and submit exactly one add, remove, or replace edit. A "
            "submission changes at most one factor. After the environment "
            "returns the next state and reward, repeat this order until the "
            "environment reports that the rollout horizon is reached. Factor "
            "expressions must use Qlib syntax: raw fields start with '$' and "
            "time-series operators use Qlib names, for example "
            "'Ref($close, 5) / $close - 1'. Never use bare fields such as "
            "'close' or operators such as 'delay'."
            f" At each state, at most {self.max_analysis_calls_per_state} "
            "analysis-tool calls are available; submit the best supported edit "
            "when that budget is exhausted."
        )

    def _phase_tools(
        self,
        *,
        phase: LoopPhase,
        current_skill: dict[str, Skill],
        current_memory: dict[str, object],
        environment: AlphaMiningEnvironment,
        analysis_calls: int,
        force_required_action: bool,
    ) -> list[Tool]:
        if phase == LoopPhase.SKILL:
            return [create_load_skill_tool(self.harness.skills, current_skill)]
        if phase == LoopPhase.MEMORY:
            return [create_load_memory_tool(self.harness.memory, current_memory)]
        if force_required_action or analysis_calls >= self.max_analysis_calls_per_state:
            return [environment.submit_tool]
        return [*environment.analysis_tools, environment.submit_tool]

    @staticmethod
    def _result(
        *,
        messages: list[Message],
        answer: str,
        stop_reason: str,
        steps: int,
        metadata: dict[str, Any] | None,
        current_skill: Mapping[str, Skill],
        current_memory: Mapping[str, object],
        environment: AlphaMiningEnvironment,
        traces: Sequence[HarnessTrace],
    ) -> AgentRun:
        last_trace = traces[-1] if traces else None
        skill = next(iter(current_skill.values()), None)
        if skill is None and last_trace is not None:
            skill = last_trace.skill

        entries = _memory_entries(current_memory)
        if not entries and last_trace is not None:
            entries = last_trace.retrieved_memory
        query = current_memory.get("query")
        if query is None and last_trace is not None:
            query = last_trace.memory_query

        return AgentRun(
            messages=messages,
            answer=answer,
            stop_reason=stop_reason,
            steps=steps,
            metadata=dict(metadata or {}),
            loaded_skill=skill,
            memory_query=str(query) if query is not None else None,
            retrieved_memory=entries,
            transition=environment.transition,
            traces=tuple(traces),
        )

    def _notify(self, message: Message) -> None:
        if self.on_step is not None:
            self.on_step(message)


def _make_trace(
    *,
    index: int,
    current_skill: Mapping[str, Skill],
    current_memory: Mapping[str, object],
    environment: AlphaMiningEnvironment,
    interaction: Sequence[Message],
) -> HarnessTrace:
    skill = next(iter(current_skill.values()), None)
    if skill is None:
        raise RuntimeError("factor action completed without a loaded skill")
    query = current_memory.get("query")
    if query is None:
        raise RuntimeError("factor action completed without a memory query")
    transition = environment.transition
    if transition is None:
        raise RuntimeError("factor action completed without an environment transition")
    return HarnessTrace(
        index=index,
        state=transition.state,
        skill=skill,
        memory_query=str(query),
        retrieved_memory=_memory_entries(current_memory),
        action=transition.action,
        next_state=transition.next_state,
        reward=transition.reward,
        transition=transition,
        interaction=tuple(interaction),
    )


def _memory_entries(memory: Mapping[str, object]) -> tuple[MemoryEntry, ...]:
    entries = memory.get("entries", ())
    if isinstance(entries, tuple) and all(
        isinstance(entry, MemoryEntry) for entry in entries
    ):
        return entries
    return ()


def _required_tool_name(phase: LoopPhase) -> str:
    if phase == LoopPhase.SKILL:
        return "load_skill"
    if phase == LoopPhase.MEMORY:
        return "load_memory"
    return "submit_factor_edit"


def _bind_tools(tools: Sequence[Tool]) -> dict[str, Tool]:
    bound: dict[str, Tool] = {}
    for tool in tools:
        if tool.name in bound:
            raise ValueError(f"duplicate tool name: {tool.name}")
        bound[tool.name] = tool
    return bound


def _execute_tool(call: ToolCall, tool: Tool | None) -> ToolRunResult:
    if tool is None:
        return ToolRunResult(
            call_id=call.id,
            name=call.name,
            error=f"tool {call.name!r} is unavailable in the current loop phase",
        )
    try:
        result = tool.execute(call)
    except Exception as error:  # noqa: BLE001 - tool failures are returned to the agent
        return ToolRunResult(
            call_id=call.id,
            name=call.name,
            error=f"{type(error).__name__}: {error}",
        )
    return ToolRunResult(
        call_id=str(result.call_id),
        name=str(result.name),
        output=result.output,
        error=result.error,
        terminated=bool(result.terminated),
        stop_reason=result.stop_reason,
    )


def _task_message(
    task: Any,
    observation: Mapping[str, Any],
    skill_descriptions: Sequence[Mapping[str, str]],
) -> str:
    task_text = (
        task
        if isinstance(task, str)
        else json.dumps(
            task,
            ensure_ascii=False,
            default=str,
        )
    )
    return (
        f"Task:\n{task_text}\n\n"
        "Initial state s_0:\n"
        f"{json.dumps(dict(observation), ensure_ascii=False, default=str)}\n\n"
        "Exposed skill descriptions (strategies remain hidden until loaded):\n"
        f"{json.dumps(list(skill_descriptions), ensure_ascii=False, default=str)}"
    )
