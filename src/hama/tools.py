"""Tools owned by the paper-aligned HAMA environment."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .types import (
    Factor,
    FactorEdit,
    FactorEditOperation,
    FunctionTool,
    ToolCall,
    ToolRunResult,
    Transition,
)

if TYPE_CHECKING:
    from .environment import FactorEvaluator


def create_analysis_tools(
    *,
    workspace: Path,
    evaluator: FactorEvaluator,
    timeout: float,
) -> list[FunctionTool]:
    return [
        _read_tool(workspace),
        _write_tool(workspace),
        _edit_tool(workspace),
        _bash_tool(workspace, timeout),
        _backtest_tool(evaluator),
    ]


def create_submit_factor_edit_tool(
    submit: Callable[[FactorEdit], Transition],
) -> FunctionTool:
    def execute(call: ToolCall) -> ToolRunResult:
        edit = _parse_factor_edit(call.arguments)
        transition = submit(edit)
        return ToolRunResult(
            call_id=call.id,
            name=call.name,
            output={
                "accepted": transition.accepted,
                "reward": transition.reward,
                "previous_score": transition.previous_score,
                "next_score": transition.next_score,
                "redundancy": transition.redundancy,
                "terminated": transition.terminated,
                "next_factor_pool": [
                    asdict(factor) for factor in transition.next_state.factor_pool
                ],
            },
            terminated=transition.terminated,
            stop_reason="horizon_reached" if transition.terminated else None,
        )

    return FunctionTool(
        name="submit_factor_edit",
        description=(
            "Submit exactly one atomic factor-pool edit to complete the current "
            "MDP transition. Valid operations are add, remove, and replace."
        ),
        parameters={
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "remove", "replace"]},
                "target": {
                    "type": ["string", "null"],
                    "description": "Existing factor name for remove or replace.",
                },
                "factor": {
                    "anyOf": [_factor_schema(), {"type": "null"}],
                    "description": "New factor for add or replace.",
                },
            },
            "required": ["operation", "target", "factor"],
            "additionalProperties": False,
        },
        execute=execute,
    )


def _backtest_tool(evaluator: FactorEvaluator) -> FunctionTool:
    def execute(call: ToolCall) -> ToolRunResult:
        factors = _parse_factors(call.arguments.get("factors"))
        output = evaluator.evaluate_factors(
            factors,
            start_time=call.arguments.get("start_time"),
            end_time=call.arguments.get("end_time"),
        )
        return _success(call, dict(output))

    return FunctionTool(
        name="backtest_factors",
        description=(
            "Evaluate candidate factors independently on the environment's fixed "
            "historical training split. This does not change the factor pool. "
            "Expressions must use Qlib syntax, e.g. 'Ref($close, 5) / $close - 1'; "
            "raw fields require '$' and bare names such as 'close' are invalid."
        ),
        parameters={
            "type": "object",
            "properties": {
                "factors": {
                    "type": "array",
                    "items": _factor_schema(),
                    "minItems": 1,
                },
                "start_time": {
                    "type": ["string", "null"],
                    "description": "Optional inclusive start inside the fixed split.",
                },
                "end_time": {
                    "type": ["string", "null"],
                    "description": "Optional inclusive end inside the fixed split.",
                },
            },
            "required": ["factors"],
            "additionalProperties": False,
        },
        execute=execute,
    )


def _read_tool(workspace: Path) -> FunctionTool:
    def execute(call: ToolCall) -> ToolRunResult:
        path = _workspace_path(workspace, call.arguments["path"])
        return _success(call, {"content": path.read_text(encoding="utf-8")})

    return FunctionTool(
        name="read",
        description="Read a UTF-8 file from this rollout's isolated workspace.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        execute=execute,
    )


def _write_tool(workspace: Path) -> FunctionTool:
    def execute(call: ToolCall) -> ToolRunResult:
        path = _workspace_path(workspace, call.arguments["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        content = str(call.arguments["content"])
        path.write_text(content, encoding="utf-8")
        return _success(
            call, {"path": str(path.relative_to(workspace)), "size": len(content)}
        )

    return FunctionTool(
        name="write",
        description="Write a UTF-8 file in this rollout's isolated workspace.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        execute=execute,
    )


def _edit_tool(workspace: Path) -> FunctionTool:
    def execute(call: ToolCall) -> ToolRunResult:
        path = _workspace_path(workspace, call.arguments["path"])
        content = path.read_text(encoding="utf-8")
        old = str(call.arguments["old_text"])
        if content.count(old) != 1:
            raise ValueError("old_text must occur exactly once")
        path.write_text(
            content.replace(old, str(call.arguments["new_text"]), 1), encoding="utf-8"
        )
        return _success(call, {"path": str(path.relative_to(workspace))})

    return FunctionTool(
        name="edit",
        description="Replace one exact occurrence in a workspace file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
        execute=execute,
    )


def _bash_tool(workspace: Path, timeout: float) -> FunctionTool:
    def execute(call: ToolCall) -> ToolRunResult:
        try:
            result = subprocess.run(
                ["/bin/sh", "-lc", str(call.arguments["command"])],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return ToolRunResult(
                call_id=call.id,
                name=call.name,
                output={"stdout": error.stdout or "", "stderr": error.stderr or ""},
                error=f"process timed out after {timeout} seconds",
            )
        return _success(
            call,
            {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            },
        )

    return FunctionTool(
        name="bash",
        description="Run shell or Python analysis inside this rollout's isolated workspace.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
        execute=execute,
    )


def _parse_factor_edit(arguments: dict[str, Any]) -> FactorEdit:
    operation = FactorEditOperation(str(arguments["operation"]))
    raw_factor = arguments.get("factor")
    factor = None
    if raw_factor is not None:
        if not isinstance(raw_factor, dict):
            raise TypeError("factor must be an object or null")
        factor = Factor(
            name=str(raw_factor["name"]), expression=str(raw_factor["expression"])
        )
    target = arguments.get("target")
    return FactorEdit(
        operation=operation,
        factor=factor,
        target=str(target) if target is not None else None,
    )


def _parse_factors(value: Any) -> tuple[Factor, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("factors must be a non-empty list")
    return tuple(
        Factor(name=str(item["name"]), expression=str(item["expression"]))
        for item in value
    )


def _factor_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"name": {"type": "string"}, "expression": {"type": "string"}},
        "required": ["name", "expression"],
        "additionalProperties": False,
    }


def _workspace_path(workspace: Path, value: Any) -> Path:
    path = (workspace / str(value)).resolve()
    if path != workspace and workspace not in path.parents:
        raise ValueError("path escapes the rollout workspace")
    return path


def _success(call: ToolCall, output: Any) -> ToolRunResult:
    return ToolRunResult(call_id=call.id, name=call.name, output=output)
