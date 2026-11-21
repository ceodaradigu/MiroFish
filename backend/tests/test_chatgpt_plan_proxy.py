"""Contract tests for the local plan adapter. No account or API call is used."""

from __future__ import annotations

import json

import httpx
import pytest
from flask import Flask
from openai import OpenAI

from app.api.chatgpt_proxy import chatgpt_proxy_bp
from app.config import Config
from app.services.chatgpt_plan_proxy import (
    ProxyPlanUsageError,
    ProxyRequestError,
    ProxyUpstreamError,
    build_responses_request,
    completion_sse,
    create_completion,
)


def completed_event(*, output, usage=None):
    return {
        "type": "response.completed",
        "response": {
            "id": "resp_test", "status": "completed", "created_at": 123,
            "model": "gpt-test", "output": output, "usage": usage,
        },
    }


def test_text_request_uses_stateless_streamed_responses_and_json_mode():
    request, downstream_stream, omitted = build_responses_request({
        "model": "gpt-test",
        "messages": [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Return a JSON object"},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 200,
    })

    assert downstream_stream is False
    assert omitted == ["max_tokens", "temperature"]
    assert request == {
        "model": "gpt-test", "input": [{"role": "user", "content": "Return a JSON object"}],
        "instructions": "Be concise", "store": False, "stream": True,
        "text": {"format": {"type": "json_object"}},
    }


def test_function_tools_and_history_round_trip_to_responses_items():
    request, _, _ = build_responses_request({
        "model": "gpt-test",
        "messages": [
            {"role": "user", "content": "Make a post"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "create_post", "arguments": '{"text":"hello"}'},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "name": "create_post", "content": '{"post_id":1}'},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "create_post", "description": "Write a post",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
            },
        }],
        "tool_choice": "auto",
    })

    assert request["tools"] == [{
        "type": "namespace", "name": "mirofish_actions",
        "description": "Functions implemented locally by the MiroFish social simulation",
        "tools": [{
            "type": "function", "name": "create_post", "description": "Write a post",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
            "strict": False,
        }],
    }]
    assert request["input"][1:] == [
        {
            "type": "function_call", "call_id": "call_1", "name": "create_post",
            "arguments": '{"text":"hello"}', "namespace": "mirofish_actions",
        },
        {
            "type": "function_call_output", "call_id": "call_1",
            "output": '{"post_id":1}', "namespace": "mirofish_actions",
        },
    ]


@pytest.mark.parametrize("extra", [
    {"top_p": 0.5},
    {"tools": [{"type": "web_search"}]},
    {"tool_choice": {"type": "function", "function": {"name": "x"}}},
    {"n": 2},
    {"response_format": {"type": "json_schema", "json_schema": {}}},
])
def test_unsupported_fields_or_tool_shapes_fail_closed(extra):
    with pytest.raises(ProxyRequestError):
        build_responses_request({
            "model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], **extra,
        })


def test_completed_text_is_returned_in_chat_completion_shape():
    calls = []

    def stream_factory(**kwargs):
        calls.append(kwargs)
        return iter([
            {"type": "response.output_text.delta", "delta": "hello"},
            completed_event(output=[{
                "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            }], usage={"input_tokens": 4, "output_tokens": 1, "total_tokens": 5}),
        ])

    completion, stream, ignored = create_completion({
        "model": "gpt-test", "messages": [{"role": "user", "content": "hi"}],
    }, stream_factory=stream_factory)

    assert calls[0]["store"] is False and calls[0]["stream"] is True
    assert stream is False and ignored == []
    assert completion["choices"] == [{
        "index": 0, "message": {"role": "assistant", "content": "hello"},
        "finish_reason": "stop",
    }]
    assert completion["usage"] == {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}


def test_function_output_is_returned_as_chat_tool_call_and_sse():
    def stream_factory(**_):
        return iter([completed_event(output=[{
            "type": "function_call", "namespace": "mirofish_actions",
            "name": "create_post", "call_id": "call_9", "arguments": '{"text":"hello"}',
        }])])

    completion, _, _ = create_completion({
        "model": "gpt-test", "messages": [{"role": "user", "content": "post"}],
        "stream": True,
    }, stream_factory=stream_factory)
    assert completion["choices"][0]["finish_reason"] == "tool_calls"
    assert completion["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "create_post"
    chunks = list(completion_sse(completion, include_usage=True))
    assert chunks[-1] == "data: [DONE]\n\n"
    assert any('"tool_calls"' in chunk for chunk in chunks)
    assert json.loads(chunks[-3][6:])["choices"][0]["finish_reason"] == "tool_calls"


def test_openai_sdk_can_parse_adapter_tool_call_stream_without_network():
    completion = {
        "id": "resp_test", "object": "chat.completion", "created": 123,
        "model": "gpt-test", "usage": None,
        "choices": [{
            "index": 0, "finish_reason": "tool_calls",
            "message": {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "create_post", "arguments": '{"text":"hello"}'},
            }]},
        }],
    }

    def handle(_request):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            text="".join(completion_sse(completion)),
        )

    client = OpenAI(
        api_key="fake-local-token", base_url="http://127.0.0.1:5001/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handle)), max_retries=0,
    )
    chunks = list(client.chat.completions.create(
        model="gpt-test", messages=[{"role": "user", "content": "post"}], stream=True,
    ))
    tool_delta = next(chunk.choices[0].delta.tool_calls[0] for chunk in chunks if chunk.choices and chunk.choices[0].delta.tool_calls)
    assert tool_delta.function.name == "create_post"
    assert tool_delta.function.arguments == '{"text":"hello"}'
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


@pytest.mark.parametrize("events", [
    [],
    [{"type": "response.failed", "response": {"error": {"message": "secret"}}}],
    [completed_event(output=[{"type": "refusal", "text": "no"}])],
])
def test_incomplete_or_unsupported_upstream_result_never_becomes_chat_success(events):
    with pytest.raises(ProxyUpstreamError):
        create_completion({
            "model": "gpt-test", "messages": [{"role": "user", "content": "hi"}],
        }, stream_factory=lambda **_: iter(events))


@pytest.mark.parametrize(("code", "status", "has_usage_url"), [
    ("subscription_sharing_usage_limit_exceeded", 429, True),
    ("subscription_sharing_usage_unavailable", 503, False),
    ("another_upstream_failure", 502, False),
])
def test_streamed_usage_failure_has_safe_http_recovery_and_no_retry(
    monkeypatch, code, status, has_usage_url,
):
    app = Flask(__name__)
    app.register_blueprint(chatgpt_proxy_bp)
    client = app.test_client()
    monkeypatch.setenv("MIROFISH_PLAN_PROXY_TOKEN", "local-only-test-token")
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "gpt-test")
    calls = []

    def fake_upstream(**kwargs):
        calls.append(kwargs)
        return iter([
            {"type": "response.output_text.delta", "delta": "partial-private-text"},
            {"type": "response.failed", "response": {
                "error": {"code": code, "message": "private-upstream-diagnostic"},
            }},
        ])

    monkeypatch.setattr("app.services.chatgpt_plan_proxy._default_stream_factory", fake_upstream)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        headers={"Authorization": "Bearer local-only-test-token"},
    )

    assert response.status_code == status
    assert response.json["error"]["type"] == ("upstream_error" if status == 502 else "plan_usage_error")
    assert (response.json["error"].get("usage_url") == "https://chatgpt.com/settings/usage") is has_usage_url
    assert "private-upstream-diagnostic" not in response.get_data(as_text=True)
    assert "partial-private-text" not in response.get_data(as_text=True)
    assert len(calls) == 1
    assert calls[0]["store"] is False and calls[0]["stream"] is True


@pytest.mark.parametrize("code", [
    "subscription_sharing_usage_limit_exceeded",
    "subscription_sharing_usage_unavailable",
])
def test_plan_usage_error_is_distinct_from_generic_upstream_failure(code):
    event = {"type": "response.failed", "response": {"error": {"code": code}}}
    with pytest.raises(ProxyPlanUsageError) as raised:
        create_completion({
            "model": "gpt-test", "messages": [{"role": "user", "content": "hi"}],
        }, stream_factory=lambda **_: iter([event]))
    assert raised.value.code == code


def test_http_route_requires_loopback_secret_and_rejects_browser_origin(monkeypatch):
    app = Flask(__name__)
    app.register_blueprint(chatgpt_proxy_bp)
    client = app.test_client()
    body = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}
    monkeypatch.setenv("MIROFISH_PLAN_PROXY_TOKEN", "local-only-test-token")
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "gpt-test")
    monkeypatch.setattr("app.api.chatgpt_proxy.create_completion", lambda _: ({
        "id": "resp_test", "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    }, False, []))

    assert client.post("/v1/chat/completions", json=body).status_code == 403
    headers = {"Authorization": "Bearer local-only-test-token"}
    assert client.post("/v1/chat/completions", json=body, headers={**headers, "Origin": "https://evil.example"}).status_code == 403
    assert client.post("/v1/chat/completions", json=body, headers=headers, environ_overrides={"REMOTE_ADDR": "192.0.2.1"}).status_code == 403
    assert client.post("/v1/chat/completions", json={**body, "model": "other-model"}, headers=headers).status_code == 422
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "")
    assert client.post("/v1/chat/completions", json=body, headers=headers).status_code == 422
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "gpt-test")
    response = client.post("/v1/chat/completions", json=body, headers=headers)
    assert response.status_code == 200
    assert response.json["choices"][0]["message"]["content"] == "ok"
    assert response.headers["Cache-Control"] == "no-store"
