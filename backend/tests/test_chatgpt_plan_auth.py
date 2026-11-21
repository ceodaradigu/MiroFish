"""Offline checks for the local ChatGPT OAuth connection."""

import base64
import hashlib
import json
import os
import threading
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

import app.services.chatgpt_plan_auth as auth_module
from app.config import Config
from app.services.chatgpt_plan_auth import (
    ChatGPTPlanAuth,
    ChatGPTPlanAuthError,
    REGISTRATION_NAME,
    RESOURCE,
    SCOPES,
)


REDIRECT_URI = "http://127.0.0.1:5001/api/chatgpt/callback"


@pytest.fixture(autouse=True)
def preserve_process_model_selection(monkeypatch):
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", Config.LLM_MODEL_NAME)
    for key in ("MIROFISH_PLAN_SELECTED_MODEL", "LLM_MODEL_NAME"):
        if key in os.environ:
            monkeypatch.setenv(key, os.environ[key])
        else:
            monkeypatch.delenv(key, raising=False)


def authorize(auth):
    url = auth.begin(REDIRECT_URI)
    parsed = urlparse(url)
    return parsed, {key: values[0] for key, values in parse_qs(parsed.query).items()}


def token_response(
    *,
    access="synthetic-access",
    refresh="synthetic-refresh",
    scopes=SCOPES,
    id_token="synthetic-id",
):
    return {
        "access_token": access,
        "refresh_token": refresh,
        "id_token": id_token,
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": scopes,
    }


def test_begin_persists_only_stable_nonsecret_registration(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail("Unexpected HTTP request")))
    auth = ChatGPTPlanAuth(tmp_path, http_client=client)
    parsed, params = authorize(auth)

    assert parsed.scheme == "https"
    assert parsed.netloc == "auth.openai.com"
    assert parsed.path == "/api/accounts/authorize"
    assert params["client_id"] == "dynamic_agent_client"
    assert params["ext_agent_host_id"].startswith("urn:uuid:")
    assert params["redirect_uri"] == REDIRECT_URI
    assert params["response_type"] == "code"
    assert params["resource"] == RESOURCE
    assert params["scope"] == SCOPES
    assert params["agent_name_hint"] == "MiroFish"
    assert params["state"] and params["nonce"]
    assert params["code_challenge_method"] == "S256"
    assert len(base64.urlsafe_b64decode(params["code_challenge"] + "==")) == hashlib.sha256().digest_size
    assert params["code_challenge"] != params["state"]

    saved = json.loads((tmp_path / REGISTRATION_NAME).read_text(encoding="utf-8"))
    assert saved == {"host_id": params["ext_agent_host_id"]}
    second = ChatGPTPlanAuth(tmp_path, http_client=client)
    _, second_params = authorize(second)
    assert second_params["ext_agent_host_id"] == params["ext_agent_host_id"]
    assert second_params["state"] != params["state"]


def test_finish_verifies_identity_grant_and_keeps_tokens_in_memory(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "POST"
        assert str(request.url) == "https://auth.openai.com/api/accounts/oauth/token"
        form = parse_qs(request.content.decode())
        assert form["grant_type"] == ["authorization_code"]
        assert form["client_id"] == ["oaiapp_test"]
        assert form["code"] == ["synthetic-code"]
        assert form["redirect_uri"] == [REDIRECT_URI]
        assert form["resource"] == [RESOURCE]
        assert form["code_verifier"][0]
        return httpx.Response(200, json=token_response())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    verifications = []

    def verifier(token, client_id, nonce):
        verifications.append((token, client_id, nonce))
        return {"sub": "synthetic-subject"}

    auth = ChatGPTPlanAuth(tmp_path, http_client=client, verifier=verifier)
    _, params = authorize(auth)
    result = auth.finish({"state": params["state"], "client_id": "oaiapp_test", "code": "synthetic-code"}, REDIRECT_URI)

    assert result == {
        "registered": True, "identity_verified": True, "pending_registration": False,
        "connected": True, "plan_usage_enabled": True,
    }
    assert auth.get_access_token() == "synthetic-access"
    assert verifications == [("synthetic-id", "oaiapp_test", params["nonce"])]
    assert len(requests) == 1
    assert json.loads((tmp_path / REGISTRATION_NAME).read_text(encoding="utf-8")) == {
        "host_id": params["ext_agent_host_id"],
        "client_id": "oaiapp_test",
        "subject": "synthetic-subject",
    }
    assert "synthetic-access" not in (tmp_path / REGISTRATION_NAME).read_text(encoding="utf-8")
    auth.disconnect()
    assert auth.status() == {
        "registered": True, "identity_verified": True, "pending_registration": False,
        "connected": False, "plan_usage_enabled": False,
    }
    with pytest.raises(ChatGPTPlanAuthError):
        auth.get_access_token()


def test_wrong_state_rejected_before_token_exchange(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail("Unexpected HTTP request")))
    auth = ChatGPTPlanAuth(tmp_path, http_client=client)
    authorize(auth)
    with pytest.raises(ChatGPTPlanAuthError, match="no coincide"):
        auth.finish({"state": "wrong", "client_id": "oaiapp_test", "code": "synthetic-code"}, REDIRECT_URI)
    with pytest.raises(ChatGPTPlanAuthError, match="caducó"):
        auth.finish({"state": "wrong", "client_id": "oaiapp_test", "code": "synthetic-code"}, REDIRECT_URI)


def test_first_registration_keeps_issued_id_after_invalid_grant(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(400, json={"error": "invalid_grant"})
    ))
    auth = ChatGPTPlanAuth(tmp_path, http_client=client)
    _, initial = authorize(auth)
    with pytest.raises(ChatGPTPlanAuthError, match="intercambio"):
        auth.finish(
            {"state": initial["state"], "client_id": "oaiapp_issued", "code": "expired-test-code"},
            REDIRECT_URI,
        )

    saved = json.loads((tmp_path / REGISTRATION_NAME).read_text(encoding="utf-8"))
    assert saved == {"host_id": initial["ext_agent_host_id"], "client_id": "oaiapp_issued"}
    assert auth.status() == {
        "registered": True, "identity_verified": False, "pending_registration": False,
        "connected": False, "plan_usage_enabled": False,
    }
    _, resumed = authorize(auth)
    assert resumed["client_id"] == "oaiapp_issued"
    assert resumed["ext_agent_host_id"] == initial["ext_agent_host_id"]


def test_missing_plan_scope_retains_identity_without_plan_access(tmp_path):
    identity_only = token_response(scopes="openid profile email")
    identity_only.pop("refresh_token")
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=identity_only)
    ))
    auth = ChatGPTPlanAuth(tmp_path, http_client=client, verifier=lambda token, client, nonce: {"sub": "synthetic-subject"})
    _, params = authorize(auth)
    result = auth.finish({"state": params["state"], "client_id": "oaiapp_test", "code": "synthetic-code"}, REDIRECT_URI)
    assert result == {
        "registered": True, "identity_verified": True, "pending_registration": False,
        "connected": True, "plan_usage_enabled": False,
    }
    with pytest.raises(ChatGPTPlanAuthError):
        auth.get_access_token()


def test_serial_refresh_rotates_in_memory_token(tmp_path):
    now = [1000.0]
    posted = []

    def handler(request):
        form = parse_qs(request.content.decode())
        posted.append(form)
        if form["grant_type"] == ["authorization_code"]:
            return httpx.Response(200, json=token_response())
        assert form["grant_type"] == ["refresh_token"]
        assert form["client_id"] == ["oaiapp_test"]
        assert form["refresh_token"] == ["synthetic-refresh"]
        return httpx.Response(200, json=token_response(access="rotated-access", refresh="rotated-refresh"))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    auth = ChatGPTPlanAuth(
        tmp_path,
        http_client=client,
        verifier=lambda token, client, nonce: {"sub": "synthetic-subject"},
        clock=lambda: now[0],
    )
    _, params = authorize(auth)
    auth.finish({"state": params["state"], "client_id": "oaiapp_test", "code": "synthetic-code"}, REDIRECT_URI)
    assert auth.get_access_token() == "synthetic-access"
    now[0] += 3541
    assert auth.get_access_token() == "rotated-access"
    assert auth.get_access_token() == "rotated-access"
    assert len(posted) == 2
    saved = (tmp_path / REGISTRATION_NAME).read_text(encoding="utf-8")
    assert "rotated-access" not in saved and "rotated-refresh" not in saved


def test_reauthorization_rejects_different_account(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=token_response())))
    subject = ["first"]
    auth = ChatGPTPlanAuth(tmp_path, http_client=client, verifier=lambda token, client, nonce: {"sub": subject[0]})
    _, params = authorize(auth)
    auth.finish({"state": params["state"], "client_id": "oaiapp_test", "code": "synthetic-code"}, REDIRECT_URI)
    assert auth.get_access_token() == "synthetic-access"
    auth.disconnect()

    subject[0] = "other"
    _, params = authorize(auth)
    assert params["client_id"] == "oaiapp_test"
    with pytest.raises(ChatGPTPlanAuthError, match="no coincide"):
        auth.finish({"state": params["state"], "code": "second-code"}, REDIRECT_URI)
    assert auth.status()["connected"] is False


def test_legacy_unverified_registration_is_preserved_for_new_account(tmp_path):
    path = tmp_path / REGISTRATION_NAME
    legacy = '{"host_id":"urn:uuid:synthetic-host","client_id":"oaiapp_legacy"}'
    path.write_text(legacy, encoding="utf-8")
    client = httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail("Unexpected HTTP request")))

    auth = ChatGPTPlanAuth(tmp_path, http_client=client)
    assert path.read_text(encoding="utf-8") == legacy
    assert auth.status() == {
        "registered": True, "identity_verified": False, "pending_registration": False,
        "connected": False, "plan_usage_enabled": False,
    }
    _, existing_params = authorize(auth)
    assert existing_params["client_id"] == "oaiapp_legacy"
    new_url = auth.begin_new_registration(REDIRECT_URI)
    new_params = {key: values[0] for key, values in parse_qs(urlparse(new_url).query).items()}
    assert new_params["client_id"] == "dynamic_agent_client"
    assert new_params["ext_agent_host_id"] == existing_params["ext_agent_host_id"]
    assert new_params["agent_name_hint"] == "MiroFish"
    assert path.read_text(encoding="utf-8") == legacy


def test_new_registration_invalid_grant_keeps_both_ids_and_resumes_issued_id(tmp_path):
    path = tmp_path / REGISTRATION_NAME
    path.write_text(json.dumps({"host_id": "urn:uuid:synthetic-host", "client_id": "oaiapp_legacy"}), encoding="utf-8")
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(400, json={"error": "invalid_grant"})
    ))
    auth = ChatGPTPlanAuth(tmp_path, http_client=client)
    params = {key: values[0] for key, values in parse_qs(urlparse(auth.begin_new_registration(REDIRECT_URI)).query).items()}
    with pytest.raises(ChatGPTPlanAuthError, match="intercambio"):
        auth.finish({"state": params["state"], "client_id": "oaiapp_new", "code": "expired-test-code"}, REDIRECT_URI)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["host_id"] == "urn:uuid:synthetic-host"
    assert saved["registrations"] == [{"client_id": "oaiapp_legacy"}, {"client_id": "oaiapp_new"}]
    assert saved["active_client_id"] == "oaiapp_legacy"
    assert saved["pending_new_client_id"] == "oaiapp_new"
    assert auth.status()["pending_registration"] is True
    assert auth.status()["connected"] is False
    _, active_params = authorize(auth)
    assert active_params["client_id"] == "oaiapp_legacy"
    resumed = auth.begin_new_registration(REDIRECT_URI)
    resumed_params = {key: values[0] for key, values in parse_qs(urlparse(resumed).query).items()}
    assert resumed_params["client_id"] == "oaiapp_new"
    assert "agent_name_hint" not in resumed_params
    assert "synthetic-access" not in path.read_text(encoding="utf-8")


def test_new_account_stays_separate_until_verified_then_refreshes_only_active(tmp_path):
    path = tmp_path / REGISTRATION_NAME
    path.write_text(json.dumps({
        "host_id": "urn:uuid:synthetic-host", "client_id": "oaiapp_legacy", "subject": "legacy-subject",
    }), encoding="utf-8")
    now = [1000.0]
    reject_new_identity = [True]
    requests = []

    def handler(request):
        form = parse_qs(request.content.decode())
        requests.append(form)
        client_id = form["client_id"][0]
        if form["grant_type"] == ["refresh_token"]:
            assert client_id == "oaiapp_new"
            assert form["refresh_token"] == ["synthetic-refresh-oaiapp_new"]
            return httpx.Response(200, json=token_response(
                access="synthetic-new-rotated", refresh="synthetic-new-refresh-rotated",
            ))
        return httpx.Response(200, json=token_response(
            access=f"synthetic-access-{client_id}", refresh=f"synthetic-refresh-{client_id}",
        ))

    def verifier(_token, client_id, _nonce):
        if client_id == "oaiapp_new" and reject_new_identity[0]:
            raise ChatGPTPlanAuthError("Synthetic identity rejection")
        return {"sub": "legacy-subject" if client_id == "oaiapp_legacy" else "new-subject"}

    auth = ChatGPTPlanAuth(
        tmp_path, http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        verifier=verifier, clock=lambda: now[0],
    )
    _, old_params = authorize(auth)
    auth.finish({"state": old_params["state"], "code": "old-code"}, REDIRECT_URI)
    assert auth.get_access_token() == "synthetic-access-oaiapp_legacy"

    new_params = {key: values[0] for key, values in parse_qs(urlparse(auth.begin_new_registration(REDIRECT_URI)).query).items()}
    with pytest.raises(ChatGPTPlanAuthError, match="verificar"):
        auth.finish({"state": new_params["state"], "client_id": "oaiapp_new", "code": "new-code"}, REDIRECT_URI)
    assert auth.get_access_token() == "synthetic-access-oaiapp_legacy"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["active_client_id"] == "oaiapp_legacy"
    assert saved["registrations"] == [
        {"client_id": "oaiapp_legacy", "subject": "legacy-subject"},
        {"client_id": "oaiapp_new"},
    ]

    reject_new_identity[0] = False
    resumed_params = {key: values[0] for key, values in parse_qs(urlparse(auth.begin_new_registration(REDIRECT_URI)).query).items()}
    assert resumed_params["client_id"] == "oaiapp_new"
    assert "agent_name_hint" not in resumed_params
    result = auth.finish({"state": resumed_params["state"], "code": "new-retry-code"}, REDIRECT_URI)
    assert result["identity_verified"] and result["connected"] and result["plan_usage_enabled"]
    assert not result["pending_registration"]
    assert auth.get_access_token() == "synthetic-access-oaiapp_new"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["active_client_id"] == "oaiapp_new"
    assert saved["pending_new_client_id"] is None
    assert saved["registrations"] == [
        {"client_id": "oaiapp_legacy", "subject": "legacy-subject"},
        {"client_id": "oaiapp_new", "subject": "new-subject"},
    ]
    assert "synthetic-access" not in path.read_text(encoding="utf-8")
    assert "synthetic-refresh" not in path.read_text(encoding="utf-8")

    reloaded = ChatGPTPlanAuth(tmp_path, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    _, reauth_params = authorize(reloaded)
    assert reauth_params["client_id"] == "oaiapp_new"
    assert reloaded.status()["identity_verified"] is True
    assert reloaded.status()["connected"] is False
    now[0] += 3541
    assert auth.get_access_token() == "synthetic-new-rotated"
    assert requests[-1]["grant_type"] == ["refresh_token"]
    assert requests[-1]["client_id"] == ["oaiapp_new"]


def test_new_registration_rejects_an_existing_client_id_before_exchange(tmp_path):
    path = tmp_path / REGISTRATION_NAME
    legacy = '{"host_id":"urn:uuid:synthetic-host","client_id":"oaiapp_legacy","subject":"legacy-subject"}'
    path.write_text(legacy, encoding="utf-8")
    client = httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail("Unexpected HTTP request")))
    auth = ChatGPTPlanAuth(tmp_path, http_client=client)
    params = {key: values[0] for key, values in parse_qs(urlparse(auth.begin_new_registration(REDIRECT_URI)).query).items()}
    with pytest.raises(ChatGPTPlanAuthError, match="ya vinculado"):
        auth.finish({"state": params["state"], "client_id": "oaiapp_legacy", "code": "new-code"}, REDIRECT_URI)
    assert path.read_text(encoding="utf-8") == legacy
    assert auth.status()["identity_verified"] is True


def test_model_selection_cannot_restore_old_model_after_account_switch(tmp_path):
    path = tmp_path / REGISTRATION_NAME
    path.write_text(json.dumps({
        "host_id": "urn:uuid:synthetic-host", "client_id": "oaiapp_legacy", "subject": "subject-old",
    }), encoding="utf-8")
    model_lookup_started = threading.Event()
    release_model_lookup = threading.Event()

    def handler(request):
        if request.method == "GET":
            model_lookup_started.set()
            assert release_model_lookup.wait(5)
            return httpx.Response(200, json={"models": [
                {"slug": "old-visible-model", "visibility": "list"},
            ]})
        client_id = parse_qs(request.content.decode())["client_id"][0]
        return httpx.Response(200, json=token_response(
            access=f"synthetic-access-{client_id}", refresh=f"synthetic-refresh-{client_id}",
        ))

    auth = ChatGPTPlanAuth(
        tmp_path, http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        verifier=lambda _token, client_id, _nonce: {
            "sub": "subject-old" if client_id == "oaiapp_legacy" else "subject-new",
        },
    )
    _, old_params = authorize(auth)
    auth.finish({"state": old_params["state"], "code": "old-code"}, REDIRECT_URI)
    new_url = auth.begin_new_registration(REDIRECT_URI)
    new_params = {key: values[0] for key, values in parse_qs(urlparse(new_url).query).items()}

    outcomes = []

    def choose_old_model():
        outcomes.append(("model", auth.select_model("old-visible-model")))

    def switch_account():
        outcomes.append(("callback", auth.finish({
            "state": new_params["state"], "client_id": "oaiapp_new", "code": "new-code",
        }, REDIRECT_URI)))

    model_thread = threading.Thread(target=choose_old_model)
    callback_thread = threading.Thread(target=switch_account)
    try:
        model_thread.start()
        assert model_lookup_started.wait(5)
        callback_thread.start()
    finally:
        release_model_lookup.set()
        model_thread.join(timeout=5)
        callback_thread.join(timeout=5)

    assert not model_thread.is_alive() and not callback_thread.is_alive()
    assert outcomes[0] == ("model", True)
    assert outcomes[1][0] == "callback"
    assert outcomes[1][1]["identity_verified"] is True
    assert auth.get_access_token() == "synthetic-access-oaiapp_new"
    assert Config.LLM_MODEL_NAME == ""
    assert os.environ["MIROFISH_PLAN_SELECTED_MODEL"] == ""
    assert os.environ["LLM_MODEL_NAME"] == ""


def test_models_only_return_listed_models(tmp_path):
    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json=token_response())
        assert request.method == "GET"
        assert str(request.url) == f"{RESOURCE}/models"
        assert request.headers["Authorization"] == "Bearer synthetic-access"
        return httpx.Response(200, json={"models": [
            {"slug": "model-listed", "display_name": "Listed", "visibility": "list"},
            {"slug": "model-hidden", "display_name": "Hidden", "visibility": "hidden"},
            {"slug": "model-plain", "visibility": "list"},
        ]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    auth = ChatGPTPlanAuth(tmp_path, http_client=client, verifier=lambda token, client, nonce: {"sub": "synthetic-subject"})
    _, params = authorize(auth)
    auth.finish({"state": params["state"], "client_id": "oaiapp_test", "code": "synthetic-code"}, REDIRECT_URI)
    assert auth.list_models() == [
        {"slug": "model-listed", "display_name": "Listed"},
        {"slug": "model-plain", "display_name": "model-plain"},
    ]


@pytest.mark.parametrize("uri", [
    "https://127.0.0.1:5001/api/chatgpt/callback",
    "http://localhost:5001/api/chatgpt/callback",
    "http://127.0.0.1:5001/auth/callback",
    "http://127.0.0.1:5001/api/chatgpt/callback?extra=1",
    "http://127.0.0.1:invalid/api/chatgpt/callback",
])
def test_redirect_uri_must_be_local_callback(tmp_path, uri):
    client = httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail("Unexpected HTTP request")))
    auth = ChatGPTPlanAuth(tmp_path, http_client=client)
    with pytest.raises(ChatGPTPlanAuthError):
        auth.begin(uri)


@pytest.mark.parametrize("claim_change", [
    {},
    {"iss": "https://other.example"},
    {"aud": "other-client"},
    {"nonce": "other-nonce"},
    {"exp": 1},
])
def test_id_token_verifier_rejects_wrong_oidc_claims(monkeypatch, claim_change):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = private.public_key()

    class LocalJwks:
        def __init__(self, url):
            assert url == "https://auth.openai.com/.well-known/jwks.json"

        def get_signing_key_from_jwt(self, token):
            return type("SigningKey", (), {"key": public})()

    monkeypatch.setattr(auth_module.jwt, "PyJWKClient", LocalJwks)
    claims = {
        "sub": "synthetic-subject",
        "iss": "https://auth.openai.com",
        "aud": "oaiapp_test",
        "nonce": "expected-nonce",
        "iat": 1790000000,
        "exp": 4102444800,
    }
    claims.update(claim_change)
    token = jwt.encode(claims, private, algorithm="RS256")
    if claim_change:
        with pytest.raises((jwt.PyJWTError, ChatGPTPlanAuthError)):
            auth_module._verify_id_token(token, "oaiapp_test", "expected-nonce")
    else:
        assert auth_module._verify_id_token(token, "oaiapp_test", "expected-nonce")["sub"] == "synthetic-subject"
