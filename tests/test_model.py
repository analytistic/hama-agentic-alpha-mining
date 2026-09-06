import httpx
from hama.model import Model
from hama.types import Message


def test_model_loads_api_key_from_dotenv_and_parses_tool_calls(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=secret-from-env\n", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["payload"] = __import__("json").loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "inspect",
                                        "arguments": '{"field":"$close"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = Model(
        model="test-model",
        base_url="https://example.test/v1",
        env_file=env_file,
        client=client,
    )
    result = model.complete(
        [Message(role="user", content="inspect")],
        [
            {
                "type": "function",
                "function": {
                    "name": "inspect",
                    "description": "Inspect data.",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["authorization"] == "Bearer secret-from-env"
    assert seen["payload"]["tool_choice"] == {
        "type": "function",
        "function": {"name": "inspect"},
    }
    assert result.tool_calls[0].name == "inspect"
    assert result.tool_calls[0].arguments == {"field": "$close"}
    assert result.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }


def test_process_environment_takes_precedence_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "process-secret")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-secret\n", encoding="utf-8")

    model = Model(
        model="test-model",
        base_url="https://example.test/v1",
        env_file=env_file,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(500))
        ),
    )

    assert model.api_key == "process-secret"
