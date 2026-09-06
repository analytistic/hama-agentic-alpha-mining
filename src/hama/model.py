"""OpenAI Chat Completions compatible model adapter."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any, Self

import httpx
from dotenv import find_dotenv, load_dotenv

from .types import Message, ToolCall, ToolSchema


class Model:
    """Call any endpoint implementing the OpenAI Chat Completions protocol.

    ``base_url`` should normally end in ``/v1``. Authentication is optional so
    that the same adapter works with local OpenAI-compatible inference servers.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        env_file: str | PathLike[str] | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if max_retries < 0 or retry_backoff < 0:
            raise ValueError("retry settings must be non-negative")
        _load_environment(env_file)
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        configured_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        )
        self.endpoint = _chat_completions_endpoint(configured_url)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=timeout)

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] = (),
    ) -> Message:
        """Return one assistant message, including zero or more tool calls."""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_openai_message(message) for message in messages],
            **self.extra_body,
        }
        if tools:
            payload["tools"] = list(tools)
            payload.setdefault(
                "tool_choice",
                (
                    {
                        "type": "function",
                        "function": {"name": tools[0]["function"]["name"]},
                    }
                    if len(tools) == 1
                    else "auto"
                ),
            )
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(self.endpoint, headers=headers, json=payload)
                response.raise_for_status()
                return _parse_completion(response.json())
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt >= self.max_retries:
                    raise
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                if attempt >= self.max_retries or not (
                    status in {408, 429} or status >= 500
                ):
                    raise
            time.sleep(self.retry_backoff * (2**attempt))
        raise RuntimeError("unreachable model retry state")

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _chat_completions_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _load_environment(env_file: str | PathLike[str] | None) -> None:
    """Load secrets from ``.env`` without overriding process environment."""

    if env_file is not None:
        load_dotenv(dotenv_path=env_file, override=False)
        return
    discovered = find_dotenv(usecwd=True)
    if discovered:
        load_dotenv(dotenv_path=discovered, override=False)


def _to_openai_message(message: Message) -> dict[str, Any]:
    if message.role == "assistant":
        value: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or None,
        }
        if message.tool_calls:
            value["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
        return value

    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": json.dumps(
                {"ok": message.ok, "output": message.output, "error": message.error},
                ensure_ascii=False,
                default=str,
            ),
        }

    return {"role": message.role, "content": message.content}


def _parse_completion(data: Any) -> Message:
    if not isinstance(data, dict) or not data.get("choices"):
        raise ValueError("OpenAI-compatible response contains no choices")
    choice = data["choices"][0]
    raw_message = choice.get("message") or {}
    if raw_message.get("role", "assistant") != "assistant":
        raise ValueError(
            "OpenAI-compatible response did not return an assistant message"
        )

    calls: list[ToolCall] = []
    for item in raw_message.get("tool_calls") or []:
        function = item.get("function") or {}
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON arguments for tool {function.get('name')!r}"
                ) from error
        elif isinstance(arguments, dict):
            parsed_arguments = arguments
        else:
            raise TypeError("tool arguments must be a JSON object")
        if not isinstance(parsed_arguments, dict):
            raise TypeError("tool arguments must decode to a JSON object")
        calls.append(
            ToolCall(
                id=str(item.get("id") or ""),
                name=str(function.get("name") or ""),
                arguments=parsed_arguments,
            )
        )

    usage = data.get("usage")
    normalized_usage = None
    if isinstance(usage, dict):
        normalized_usage = {
            key: int(usage.get(key, 0))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }

    return Message(
        role="assistant",
        content=raw_message.get("content") or "",
        reasoning_content=raw_message.get("reasoning_content") or "",
        tool_calls=calls,
        usage=normalized_usage,
        finish_reason=choice.get("finish_reason"),
        raw=data,
    )
