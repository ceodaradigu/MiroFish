"""Offline checks for the local ChatGPT OAuth connection."""

import base64
import hashlib
import json
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

import app.services.chatgpt_plan_auth as auth_module
from app.services.chatgpt_plan_auth import (
    ChatGPTPlanAuth,
    ChatGPTPlanAuthError,
    REGISTRATION_NAME,
    RESOURCE,
    SCOPES,
)


REDIRECT_URI = "http://127.0.0.1:5001/api/chatgpt/callback"


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

    assert result == {"registered": True, "connected": True, "plan_usage_enabled": True}
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
    assert auth.status() == {"registered": True, "connected": False, "plan_usage_enabled": False}
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


def test_missing_plan_scope_retains_identity_without_plan_access(tmp_path):
    identity_only = token_response(scopes="openid profile email")
    identity_only.pop("refresh_token")
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=identity_only)
    ))
    auth = ChatGPTPlanAuth(tmp_path, http_client=client, verifier=lambda token, client, nonce: {"sub": "synthetic-subject"})
    _, params = authorize(auth)
    result = auth.finish({"state": params["state"], "client_id": "oaiapp_test", "code": "synthetic-code"}, REDIRECT_URI)
    assert result == {"registered": True, "connected": True, "plan_usage_enabled": False}
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
