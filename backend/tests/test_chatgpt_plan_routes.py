"""Flask route integration for optional local ChatGPT plan use.

All identity, model discovery, and inference boundaries are mocked. These
tests make no OpenAI or Zep requests and contain no usable credentials.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from app import create_app
from app.config import Config
from app.services.chatgpt_plan_auth import ChatGPTPlanAuthError, auth
from app.services.simulation_runner import SimulationRunner
from app.services.plan_work_guard import plan_work_guard, retain_request_plan_work, release_plan_work


LOCAL = "http://127.0.0.1:5001"
FRONTEND = "http://127.0.0.1:3000"


@pytest.fixture
def app_for_mode(monkeypatch):
    def build(plan_mode: bool):
        monkeypatch.setattr(Config, "CHATGPT_PLAN_MODE", plan_mode)
        monkeypatch.setattr(Config, "MEMORY_BACKEND", "zep")
        monkeypatch.setattr(Config, "LLM_MODEL_NAME", "")
        monkeypatch.setattr(Config, "ZEP_API_KEY", None)
        # Model selection writes these process variables; monkeypatch restores
        # their previous values even when the route changes them directly.
        monkeypatch.setenv("MIROFISH_PLAN_SELECTED_MODEL", "")
        monkeypatch.setenv("LLM_MODEL_NAME", "")
        app = create_app()
        app.config.update(TESTING=True)
        return app

    return build


def test_normal_mode_exposes_only_disabled_status(app_for_mode):
    client = app_for_mode(False).test_client()

    assert client.get("/api/chatgpt/status").json == {"planMode": False}
    assert client.get("/api/chatgpt/start").status_code == 404
    assert client.get("/api/chatgpt/start-new").status_code == 404
    assert client.get("/api/chatgpt/models").status_code == 404
    assert client.post("/v1/chat/completions", json={}).status_code == 404


def test_plan_status_is_safe_and_reflects_required_connections(app_for_mode, monkeypatch):
    monkeypatch.setattr(auth, "status", lambda: {
        "registered": True, "identity_verified": False, "pending_registration": True,
        "connected": False, "plan_usage_enabled": False,
    })
    client = app_for_mode(True).test_client()

    response = client.get("/api/chatgpt/status", base_url=LOCAL)
    assert response.status_code == 200
    assert response.json == {
        "planMode": True, "registered": True, "identityVerified": False,
        "pendingRegistration": True, "connected": False, "planUsageEnabled": False,
        "memoryBackend": "zep", "memoryReady": False,
        "zepConfigured": False, "selectedModel": "",
    }
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Referrer-Policy"] == "no-referrer"

    monkeypatch.setattr(auth, "status", lambda: {
        "registered": True, "identity_verified": True, "pending_registration": False,
        "connected": True, "plan_usage_enabled": True,
    })
    monkeypatch.setattr(Config, "ZEP_API_KEY", "fake-zep-test-value")
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "gpt-visible-test")
    state = client.get("/api/chatgpt/status", base_url=LOCAL).json
    assert state["connected"] and state["planUsageEnabled"] and state["memoryReady"] and state["zepConfigured"]
    assert state["registered"] and state["identityVerified"] and not state["pendingRegistration"]
    assert state["selectedModel"] == "gpt-visible-test"
    assert not any(key in state for key in ("access_token", "refresh_token", "subject", "client_id"))


def test_new_account_route_starts_separate_local_authorization(app_for_mode, monkeypatch):
    calls = []

    def begin_new(redirect_uri):
        calls.append(redirect_uri)
        return "https://auth.openai.com/mock-new-account?state=opaque-new-test-state"

    monkeypatch.setattr(auth, "begin_new_registration", begin_new)
    client = app_for_mode(True).test_client()
    response = client.get("/api/chatgpt/start-new", base_url=LOCAL)
    assert response.status_code == 302
    assert response.headers["Location"].startswith("https://auth.openai.com/mock-new-account")
    assert calls == [LOCAL + "/api/chatgpt/callback"]
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_start_and_cross_site_callback_use_exact_local_redirect(app_for_mode, monkeypatch):
    calls = []

    def begin(redirect_uri):
        calls.append(("begin", redirect_uri))
        return "https://auth.openai.com/mock-authorize?state=opaque-test-state"

    def finish(query, redirect_uri):
        calls.append(("finish", dict(query), redirect_uri))
        return {"connected": True, "plan_usage_enabled": True}

    monkeypatch.setattr(auth, "begin", begin)
    monkeypatch.setattr(auth, "finish", finish)
    app = app_for_mode(True)
    client = app.test_client()

    started = client.get("/api/chatgpt/start", base_url=LOCAL)
    assert started.status_code == 302
    assert started.headers["Location"].startswith("https://auth.openai.com/mock-authorize")
    assert calls[0] == ("begin", LOCAL + "/api/chatgpt/callback")

    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "previous-choice")
    callback = client.get(
        "/api/chatgpt/callback?state=opaque-test-state&code=one-use-test-code",
        base_url=LOCAL, headers={"Sec-Fetch-Site": "cross-site"},
    )
    assert callback.status_code == 303
    assert callback.headers["Location"] == FRONTEND + "/"
    assert calls[1] == (
        "finish", {"state": "opaque-test-state", "code": "one-use-test-code"},
        LOCAL + "/api/chatgpt/callback",
    )
    assert callback.headers["Cache-Control"] == "no-store"


def test_callback_failure_returns_safe_error_without_changing_model(app_for_mode, monkeypatch):
    def fail(_query, _redirect_uri):
        raise ChatGPTPlanAuthError("La respuesta de ChatGPT no coincide con la solicitud.")

    monkeypatch.setattr(auth, "finish", fail)
    client = app_for_mode(True).test_client()
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "prior-choice")

    response = client.get("/api/chatgpt/callback?state=wrong", base_url=LOCAL)
    assert response.status_code == 400
    assert "no coincide" in response.json["error"]
    assert Config.LLM_MODEL_NAME == "prior-choice"
    assert response.headers["Cache-Control"] == "no-store"


def test_models_and_selection_allow_only_visible_model(app_for_mode, monkeypatch):
    visible = [
        {"slug": "gpt-visible-test", "display_name": "Visible test model"},
        {"slug": "gpt-other-test", "display_name": "Other test model"},
    ]
    selections = []

    def choose(model):
        selections.append(model)
        return model in {item["slug"] for item in visible}

    monkeypatch.setattr(auth, "list_models", lambda: visible)
    monkeypatch.setattr(auth, "select_model", choose)
    client = app_for_mode(True).test_client()

    listed = client.get("/api/chatgpt/models", base_url=LOCAL)
    assert listed.status_code == 200
    assert listed.json == {"models": visible}
    assert listed.headers["Cache-Control"] == "no-store"

    unknown = client.post("/api/chatgpt/model", base_url=LOCAL, json={"model": "unlisted-test"})
    assert unknown.status_code == 400
    assert Config.LLM_MODEL_NAME == ""

    selected = client.post("/api/chatgpt/model", base_url=LOCAL, json={"model": "gpt-visible-test"})
    assert selected.status_code == 200
    assert selected.json == {"selectedModel": "gpt-visible-test"}
    assert selections == ["unlisted-test", "gpt-visible-test"]


def test_models_fail_without_account_access(app_for_mode, monkeypatch):
    def unavailable():
        raise ChatGPTPlanAuthError("Conecta y autoriza primero tu plan ChatGPT.")

    monkeypatch.setattr(auth, "list_models", unavailable)
    monkeypatch.setattr(auth, "select_model", lambda _model: unavailable())
    client = app_for_mode(True).test_client()
    assert client.get("/api/chatgpt/models", base_url=LOCAL).status_code == 503
    assert client.post("/api/chatgpt/model", base_url=LOCAL, json={"model": "gpt-test"}).status_code == 503


def test_work_gate_requires_plan_zep_and_model_without_running_handler(app_for_mode, monkeypatch):
    state = {"registered": True, "connected": False, "plan_usage_enabled": False}
    monkeypatch.setattr(auth, "status", lambda: state.copy())
    app = app_for_mode(True)
    app.add_url_rule("/api/graph/_route-gate-probe", view_func=lambda: ("", 204), methods=["POST"])
    client = app.test_client()
    work = lambda: client.post("/api/graph/_route-gate-probe", base_url=LOCAL)

    assert work().status_code == 503
    state["connected"] = state["plan_usage_enabled"] = True
    assert work().status_code == 503  # Zep and model missing.
    monkeypatch.setattr(Config, "ZEP_API_KEY", "fake-zep-test-value")
    assert work().status_code == 503  # Model missing.
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "gpt-visible-test")
    assert work().status_code == 204


def test_plan_mode_rejects_wrong_host_remote_and_cross_site_api_calls(app_for_mode, monkeypatch):
    monkeypatch.setattr(auth, "status", lambda: {
        "registered": False, "connected": False, "plan_usage_enabled": False,
    })
    client = app_for_mode(True).test_client()

    assert client.get("/api/chatgpt/status", base_url="http://localhost:5001").status_code == 403
    assert client.get(
        "/api/chatgpt/status", base_url=LOCAL,
        environ_overrides={"REMOTE_ADDR": "192.0.2.1"},
    ).status_code == 403
    assert client.get(
        "/api/chatgpt/start", base_url=LOCAL,
        headers={"Sec-Fetch-Site": "cross-site"},
    ).status_code == 403
    assert client.get(
        "/api/chatgpt/start-new", base_url=LOCAL,
        headers={"Sec-Fetch-Site": "cross-site"},
    ).status_code == 403
    assert client.get(
        "/api/chatgpt/callback?state=x", base_url=LOCAL,
        headers={"Origin": "https://outside.example", "Sec-Fetch-Site": "cross-site"},
    ).status_code == 403
    assert client.post(
        "/api/chatgpt/model", base_url=LOCAL, json={"model": "gpt-test"},
        headers={"Origin": "https://outside.example"},
    ).status_code == 403


def test_active_simulation_blocks_connection_and_model_changes(app_for_mode, monkeypatch):
    monkeypatch.setattr(SimulationRunner, "get_running_simulations", lambda: ["sim-active-test"])

    def unexpected(*_args, **_kwargs):
        raise AssertionError("A busy route must not call account methods")

    monkeypatch.setattr(auth, "begin", unexpected)
    monkeypatch.setattr(auth, "begin_new_registration", unexpected)
    monkeypatch.setattr(auth, "finish", unexpected)
    monkeypatch.setattr(auth, "list_models", unexpected)
    monkeypatch.setattr(auth, "select_model", unexpected)
    client = app_for_mode(True).test_client()
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "prior-visible-test")
    os.environ["MIROFISH_PLAN_SELECTED_MODEL"] = "prior-visible-test"
    os.environ["LLM_MODEL_NAME"] = "prior-visible-test"

    attempts = [
        client.get("/api/chatgpt/start", base_url=LOCAL),
        client.get("/api/chatgpt/start-new", base_url=LOCAL),
        client.post("/api/chatgpt/model", base_url=LOCAL, json={"model": "new-visible-test"}),
        client.get(
            "/api/chatgpt/callback?state=opaque-test-state&code=one-use-test-code",
            base_url=LOCAL, headers={"Sec-Fetch-Site": "cross-site"},
        ),
    ]
    for response in attempts:
        assert response.status_code == 409
        assert "simulación" in response.json["error"]
        assert response.headers["Cache-Control"] == "no-store"

    assert Config.LLM_MODEL_NAME == "prior-visible-test"
    assert os.environ["MIROFISH_PLAN_SELECTED_MODEL"] == "prior-visible-test"
    assert os.environ["LLM_MODEL_NAME"] == "prior-visible-test"


def test_inflight_plan_work_prevents_account_change_before_request_finishes(app_for_mode, monkeypatch):
    """A callback cannot interleave after the work gate but before LLM use."""

    monkeypatch.setattr(auth, "status", lambda: {"plan_usage_enabled": True})
    monkeypatch.setattr(Config, "MEMORY_BACKEND", "local")
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "gpt-visible-test")
    started = threading.Event()
    release = threading.Event()

    def work():
        started.set()
        assert release.wait(5)
        return "", 204

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Account methods must not run while work is active")

    monkeypatch.setattr(auth, "finish", forbidden)
    monkeypatch.setattr(auth, "select_model", forbidden)
    app = app_for_mode(True)
    monkeypatch.setattr(Config, "MEMORY_BACKEND", "local")
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "gpt-visible-test")
    app.add_url_rule("/api/graph/_active-plan-work", view_func=work, methods=["POST"])

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: app.test_client().post("/api/graph/_active-plan-work", base_url=LOCAL))
        try:
            assert started.wait(5)
            client = app.test_client()
            callback = client.get(
                "/api/chatgpt/callback?state=opaque&code=one-use",
                base_url=LOCAL, headers={"Sec-Fetch-Site": "cross-site"},
            )
            selected = client.post("/api/chatgpt/model", base_url=LOCAL, json={"model": "gpt-other-test"})
            assert callback.status_code == selected.status_code == 409
        finally:
            release.set()
        assert future.result(timeout=5).status_code == 204
    assert not plan_work_guard.has_work()


def test_async_plan_work_keeps_account_fixed_after_request_returns(app_for_mode, monkeypatch):
    """A worker's retained lease outlives its launching HTTP response."""

    monkeypatch.setattr(auth, "status", lambda: {"plan_usage_enabled": True})
    app = app_for_mode(True)
    monkeypatch.setattr(Config, "MEMORY_BACKEND", "local")
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "gpt-visible-test")
    release = threading.Event()
    worker_done = threading.Event()

    def launch():
        lease = retain_request_plan_work()

        def worker():
            try:
                release.wait(5)
            finally:
                release_plan_work(lease)
                worker_done.set()

        threading.Thread(target=worker, daemon=True).start()
        return "", 202

    monkeypatch.setattr(auth, "finish", lambda *_args: (_ for _ in ()).throw(AssertionError("busy")))
    app.add_url_rule("/api/report/_async-plan-work", view_func=launch, methods=["POST"])
    client = app.test_client()
    try:
        assert client.post("/api/report/_async-plan-work", base_url=LOCAL).status_code == 202
        assert plan_work_guard.has_work()
        callback = client.get(
            "/api/chatgpt/callback?state=opaque&code=one-use",
            base_url=LOCAL, headers={"Sec-Fetch-Site": "cross-site"},
        )
        assert callback.status_code == 409
    finally:
        release.set()
    assert worker_done.wait(5)
    assert not plan_work_guard.has_work()
