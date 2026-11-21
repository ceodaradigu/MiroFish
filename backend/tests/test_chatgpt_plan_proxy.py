"""Contract tests for the local plan adapter. No account or API call is used."""

from __future__ import annotations

import json

import httpx
import pytest
from flask import Flask
from openai import OpenAI

from app.api.chatgpt_proxy import chatgpt_proxy_bp
from app.config import Config
import app.services.chatgpt_plan_proxy as plan_proxy
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


def test_json_mode_keeps_explicit_format_instruction_in_responses_input():
    request, _, _ = build_responses_request({
        "model": "gpt-test",
        "messages": [
            {"role": "system", "content": "Return valid JSON with an ontology."},
            {"role": "user", "content": "Create an ontology for these workshops."},
        ],
        "response_format": {"type": "json_object"},
    })

    assert request["instructions"] == "Return valid JSON with an ontology."
    assert request["input"] == [{
        "role": "user",
        "content": "Create an ontology for these workshops.\n\nReturn a JSON object.",
    }]


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


def test_admission_failure_keeps_safe_diagnostics_without_logging_provider_text(caplog):
    class AdmissionError(RuntimeError):
        status_code = 400
        request_id = "req-test-123"
        body = {"error": {
            "code": "subscription_sharing_unsupported_capability",
            "param": "text.format",
            "type": "invalid_request_error",
            "message": "Invalid input contains private request and bearer text",
        }}

        def __str__(self):
            return "private request and bearer text"

    def reject(**_kwargs):
        raise AdmissionError()

    with caplog.at_level("ERROR"):
        with pytest.raises(ProxyUpstreamError) as raised:
            create_completion({
                "model": "gpt-test", "messages": [{"role": "user", "content": "private prompt"}],
            }, stream_factory=reject)

    assert raised.value.http_status == 400
    assert raised.value.code == "subscription_sharing_unsupported_capability"
    assert raised.value.param == "text.format"
    assert raised.value.request_id is None
    assert raised.value.provider_type == "invalid_request_error"
    assert raised.value.category == "invalid_input"
    assert "stage=request" in caplog.text
    assert "provider_type=invalid_request_error category=invalid_input" in caplog.text
    assert "body_shape=object[error]/error[code,message,param,type]" in caplog.text
    assert "private request" not in caplog.text
    assert "private prompt" not in caplog.text


def test_stream_error_classifies_input_failure_without_provider_prose(caplog):
    class StreamError(RuntimeError):
        body = {
            "code": None, "param": "input", "type": "invalid_request_error",
            "message": "Input should be valid; private provider prose",
        }

    def stream(**_kwargs):
        def events():
            raise StreamError("private provider prose")
            yield None  # pragma: no cover

        return events()

    with caplog.at_level("ERROR"):
        with pytest.raises(ProxyUpstreamError) as raised:
            create_completion({
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "private prompt"}],
            }, stream_factory=stream)

    assert raised.value.http_status == 502
    assert raised.value.param == "input"
    assert raised.value.provider_type == "invalid_request_error"
    assert raised.value.category == "invalid_input"
    assert "stage=stream" in caplog.text
    assert "provider_type=invalid_request_error category=invalid_input" in caplog.text
    assert "private provider prose" not in caplog.text
    assert "private prompt" not in caplog.text


def test_unknown_upstream_identifiers_cannot_leak_in_diagnostics(caplog):
    class UpstreamError(RuntimeError):
        status_code = 400
        request_id = "req_private_token_made_up"
        body = {"error": {
            "code": "sk_fake_secret_code", "param": "sk_fake_secret_param",
            "type": "invalid_request_error", "message": "private provider prose",
            "sk_fake_secret_key": "private",
        }}

    with caplog.at_level("ERROR"):
        with pytest.raises(ProxyUpstreamError) as raised:
            create_completion({
                "model": "gpt-test", "messages": [{"role": "user", "content": "hello"}],
            }, stream_factory=lambda **_: (_ for _ in ()).throw(UpstreamError()))

    assert raised.value.code is None
    assert raised.value.param is None
    assert "sk_fake_secret" not in caplog.text
    assert "private provider prose" not in caplog.text


def test_opt_in_error_skeleton_is_redacted_and_used_once(monkeypatch, caplog):
    secret_email = "user@example.com"
    secret_token = "sk-super-secret"
    fictional_text = "Fictional workshop invoice details"

    class StreamError(RuntimeError):
        body = {
            "code": None, "param": "input", "type": "invalid_request_error",
            "message": f"Input must be an array: {secret_email}; Bearer {secret_token}; {fictional_text}.",
        }

    def stream(**_kwargs):
        def events():
            raise StreamError("private upstream text")
            yield None  # pragma: no cover

        return events()

    payload = {"model": "gpt-test", "messages": [{"role": "user", "content": fictional_text}]}
    monkeypatch.setattr(plan_proxy, "_error_message_diagnostic", plan_proxy._OneShotErrorMessageDiagnostic(True))
    with caplog.at_level("ERROR"):
        for _ in range(2):
            with pytest.raises(ProxyUpstreamError):
                create_completion(payload, stream_factory=stream)

    skeletons = [record.message for record in caplog.records if "one-shot error skeleton" in record.message]
    assert len(skeletons) == 1
    assert "input must" in skeletons[0]
    assert "array" in skeletons[0]
    for secret in (secret_email, "Bearer", secret_token, fictional_text, "private upstream text"):
        assert secret not in caplog.text


def test_error_skeleton_disabled_by_default(monkeypatch, caplog):
    monkeypatch.setattr(plan_proxy, "_error_message_diagnostic", plan_proxy._OneShotErrorMessageDiagnostic(False))
    with caplog.at_level("ERROR"):
        plan_proxy._log_upstream_failure("stream", body={"message": "Input must be an array: private secret"})
    assert "one-shot error skeleton" not in caplog.text
    assert "private secret" not in caplog.text


def test_proxy_returns_safe_upstream_status_code_and_request_id(monkeypatch):
    app = Flask(__name__)
    app.register_blueprint(chatgpt_proxy_bp)
    client = app.test_client()
    monkeypatch.setenv("MIROFISH_PLAN_PROXY_TOKEN", "local-only-test-token")
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "gpt-test")
    monkeypatch.setattr(
        "app.api.chatgpt_proxy.create_completion",
        lambda _payload: (_ for _ in ()).throw(ProxyUpstreamError(
            "private provider message", http_status=403,
            code="subscription_sharing_user_not_eligible", request_id="req-test-456",
        )),
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer local-only-test-token"},
    )
    assert response.status_code == 403
    assert response.json["error"]["code"] == "subscription_sharing_user_not_eligible"
    assert response.json["error"]["request_id"] == "req-test-456"
    assert "private provider message" not in response.get_data(as_text=True)
    assert response.headers["Cache-Control"] == "no-store"


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
