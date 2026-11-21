"""One-shot plan diagnostics never expose model text or credentials."""

from __future__ import annotations

from app import create_app
from app.api import chatgpt_plan
from app.config import Config
from app.services.chatgpt_plan_auth import auth
from app.services.chatgpt_plan_proxy import ProxyUpstreamError
from app.services.plan_work_guard import plan_work_guard


LOCAL = "http://127.0.0.1:5001"


def _client(monkeypatch, *, connected=True, enabled=True):
    monkeypatch.setattr(Config, "CHATGPT_PLAN_MODE", True)
    monkeypatch.setattr(Config, "MEMORY_BACKEND", "local")
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "gpt-visible-test")
    monkeypatch.setattr(auth, "status", lambda: {"plan_usage_enabled": connected})
    monkeypatch.setattr(chatgpt_plan, "_diagnostic_once", chatgpt_plan._OneShotDiagnostic(enabled))
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_diagnostic_is_disabled_by_default_and_get_never_infers(monkeypatch):
    calls = []
    monkeypatch.setattr(chatgpt_plan, "create_completion", lambda *_args: calls.append(True))
    client = _client(monkeypatch, enabled=False)

    assert client.post("/api/chatgpt/self-test", base_url=LOCAL).status_code == 404
    assert client.get("/api/chatgpt/self-test", base_url=LOCAL).status_code == 405
    assert calls == []


def test_diagnostic_sends_one_fixed_minimal_request_and_returns_no_text(monkeypatch):
    calls = []

    def completed(payload):
        calls.append(payload)
        return ({"choices": [{"message": {"content": "private model output"}}]}, False, [])

    monkeypatch.setattr(chatgpt_plan, "create_completion", completed)
    client = _client(monkeypatch)

    first = client.post("/api/chatgpt/self-test", base_url=LOCAL)
    second = client.post("/api/chatgpt/self-test", base_url=LOCAL)

    assert first.status_code == 200
    assert first.json == {"ok": True}
    assert first.headers["Cache-Control"] == "no-store"
    assert second.status_code == 404
    assert calls == [{
        "model": "gpt-visible-test",
        "messages": [{"role": "user", "content": "Say exactly: Hello, world!"}],
        "stream": False,
    }]
    assert "private model output" not in first.get_data(as_text=True)
    assert not plan_work_guard.has_work()


def test_diagnostic_requires_connection_before_consuming_once(monkeypatch):
    calls = []
    monkeypatch.setattr(chatgpt_plan, "create_completion", lambda *_args: calls.append(True))
    client = _client(monkeypatch, connected=False)

    disconnected = client.post("/api/chatgpt/self-test", base_url=LOCAL)
    assert disconnected.status_code == 503
    assert chatgpt_plan._diagnostic_once.available()
    assert calls == []
    assert not plan_work_guard.has_work()


def test_diagnostic_failure_exposes_only_safe_machine_fields(monkeypatch):
    def rejected(_payload):
        raise ProxyUpstreamError(
            "private provider prose with a token",
            http_status=502,
            code="unsupported_input",
            param="input",
            request_id="req_example",
            provider_type="invalid_request_error",
            category="invalid_input",
        )

    monkeypatch.setattr(chatgpt_plan, "create_completion", rejected)
    client = _client(monkeypatch)

    response = client.post("/api/chatgpt/self-test", base_url=LOCAL)
    assert response.status_code == 502
    assert response.json == {"ok": False, "error": {
        "type": "upstream_error", "code": "unsupported_input",
        "param": "input", "request_id": "req_example",
        "provider_type": "invalid_request_error", "category": "invalid_input",
    }}
    assert "private provider prose" not in response.get_data(as_text=True)
    assert not chatgpt_plan._diagnostic_once.available()
    assert not plan_work_guard.has_work()


def test_diagnostic_rejects_cross_site_before_consuming_once(monkeypatch):
    monkeypatch.setattr(chatgpt_plan, "create_completion", lambda *_args: (_ for _ in ()).throw(AssertionError("called")))
    client = _client(monkeypatch)

    response = client.post(
        "/api/chatgpt/self-test", base_url=LOCAL,
        headers={"Origin": "https://outside.example"},
    )
    assert response.status_code == 403
    assert chatgpt_plan._diagnostic_once.available()
