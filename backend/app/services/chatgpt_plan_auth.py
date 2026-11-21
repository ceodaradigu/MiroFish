"""Local Sign in with ChatGPT plan authorization.

Only the host ID, issued client IDs and verified subjects are persisted. OAuth
tokens live in this process's memory and are never returned by public methods
other than ``get_access_token`` for the local inference bridge.

Implements the open-source direct flow documented at
https://developers.openai.com/siwc/token-sharing-open-source/sign-in .
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
import threading
import time
from typing import Any, Callable, Mapping
from urllib.parse import urlencode, urlparse
from uuid import uuid4

import httpx
import jwt


ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = f"{ISSUER}/api/accounts/authorize"
TOKEN_URL = f"{ISSUER}/api/accounts/oauth/token"
JWKS_URL = f"{ISSUER}/.well-known/jwks.json"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
REGISTRATION_NAME = "chatgpt-plan-registration.json"
_CLIENT_ID = re.compile(r"^[A-Za-z0-9_-]{3,128}$")


class ChatGPTPlanAuthError(RuntimeError):
    """A safe, user-facing authorization failure without credential details."""


def _verify_id_token(id_token: str, client_id: str, nonce: str | None) -> dict[str, Any]:
    """Validate OpenAI's signature and OIDC claims before trusting identity."""
    signing_key = jwt.PyJWKClient(JWKS_URL).get_signing_key_from_jwt(id_token)
    required = ["sub", "exp", "iat"]
    if nonce is not None:
        required.append("nonce")
    claims = jwt.decode(
        id_token,
        signing_key.key,
        algorithms=["RS256"],
        audience=client_id,
        issuer=ISSUER,
        leeway=5,
        options={"require": required},
    )
    if nonce is not None and not hmac.compare_digest(str(claims.get("nonce", "")), nonce):
        raise ChatGPTPlanAuthError("La identidad de ChatGPT no coincide con esta solicitud.")
    if not isinstance(claims.get("sub"), str) or not claims["sub"]:
        raise ChatGPTPlanAuthError("La identidad de ChatGPT no contiene un identificador válido.")
    return claims


class ChatGPTPlanAuth:
    """Keeps account/workspace registrations separate and one active token set."""

    def __init__(
        self,
        storage_dir: Path | None = None,
        *,
        http_client: httpx.Client | None = None,
        verifier: Callable[[str, str, str | None], Mapping[str, Any]] = _verify_id_token,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._storage_dir = storage_dir or Path(__file__).resolve().parents[2] / "uploads"
        self._registration_path = self._storage_dir / REGISTRATION_NAME
        self._http = http_client or httpx.Client(timeout=20.0, follow_redirects=False)
        self._verifier = verifier
        self._clock = clock
        self._lock = threading.RLock()
        self._pending: dict[str, Any] | None = None
        self._tokens: dict[str, Any] | None = None
        self._signed_in = False
        state = self._load_registration()
        self._host_id: str = state["host_id"]
        self._registrations: list[dict[str, str]] = state["registrations"]
        self._active_client_id: str | None = state["active_client_id"]
        self._pending_new_client_id: str | None = state["pending_new_client_id"]

    def _load_registration(self) -> dict[str, Any]:
        path = self._registration_path
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
                raise ChatGPTPlanAuthError("El registro local de ChatGPT no es un archivo normal.")
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ChatGPTPlanAuthError("No se puede leer el registro local de ChatGPT.") from exc
            if not isinstance(data, dict):
                raise ChatGPTPlanAuthError("El registro local de ChatGPT es inválido.")
            host_id = data.get("host_id")
            if not isinstance(host_id, str) or not host_id.startswith("urn:uuid:"):
                raise ChatGPTPlanAuthError("El registro local de ChatGPT es inválido.")
            if "registrations" not in data:
                # Legacy data is normalised in memory; never rewrite the real
                # file merely by starting the application.
                client_id = data.get("client_id")
                subject = data.get("subject")
                if client_id is not None and (not isinstance(client_id, str) or not _CLIENT_ID.fullmatch(client_id)):
                    raise ChatGPTPlanAuthError("El registro local de ChatGPT es inválido.")
                if subject is not None and (not isinstance(subject, str) or not subject or not client_id):
                    raise ChatGPTPlanAuthError("El registro local de ChatGPT es inválido.")
                record = {"client_id": client_id} if client_id else None
                if record is not None and subject:
                    record["subject"] = subject
                return {
                    "host_id": host_id,
                    "registrations": [record] if record else [],
                    "active_client_id": client_id,
                    "pending_new_client_id": None,
                }

            records = data["registrations"]
            active = data.get("active_client_id")
            pending_new = data.get("pending_new_client_id")
            if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
                raise ChatGPTPlanAuthError("El registro local de ChatGPT es inválido.")
            clean_records: list[dict[str, str]] = []
            seen: set[str] = set()
            for row in records:
                client_id, subject = row.get("client_id"), row.get("subject")
                if not isinstance(client_id, str) or not _CLIENT_ID.fullmatch(client_id) or client_id in seen:
                    raise ChatGPTPlanAuthError("El registro local de ChatGPT es inválido.")
                if subject is not None and (not isinstance(subject, str) or not subject):
                    raise ChatGPTPlanAuthError("El registro local de ChatGPT es inválido.")
                seen.add(client_id)
                clean_records.append({"client_id": client_id, **({"subject": subject} if subject else {})})
            if any(value is not None and not isinstance(value, str) for value in (active, pending_new)):
                raise ChatGPTPlanAuthError("El registro local de ChatGPT es inválido.")
            if active is not None and active not in seen:
                raise ChatGPTPlanAuthError("El registro local de ChatGPT es inválido.")
            if pending_new is not None and (
                pending_new not in seen or pending_new == active
                or any(row["client_id"] == pending_new and row.get("subject") for row in clean_records)
            ):
                raise ChatGPTPlanAuthError("El registro local de ChatGPT es inválido.")
            return {
                "host_id": host_id,
                "registrations": clean_records,
                "active_client_id": active,
                "pending_new_client_id": pending_new,
            }
        registration = {"host_id": f"urn:uuid:{uuid4()}"}
        self._save_registration(registration)
        return {"host_id": registration["host_id"], "registrations": [], "active_client_id": None, "pending_new_client_id": None}

    def _save_registration(self, data: Mapping[str, Any]) -> None:
        self._storage_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self._storage_dir.is_symlink():
            raise ChatGPTPlanAuthError("El directorio local de ChatGPT no es válido.")
        descriptor, temporary = tempfile.mkstemp(prefix=".chatgpt-plan-", dir=self._storage_dir)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(dict(data), file, separators=(",", ":"))
                file.flush()
                os.fsync(file.fileno())
            if os.name != "nt":
                os.chmod(temporary, 0o600)
            os.replace(temporary, self._registration_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _set_registration_state(
        self,
        records: list[dict[str, str]],
        active_client_id: str | None,
        pending_new_client_id: str | None,
    ) -> None:
        """Persist all registrations atomically before changing in-memory state."""

        if not pending_new_client_id and len(records) <= 1 and (
            not records or active_client_id == records[0]["client_id"]
        ):
            # Keep the historical one-account format when it still describes
            # the complete state; existing local installations remain valid.
            data: dict[str, Any] = {"host_id": self._host_id}
            if records:
                data.update(records[0])
        else:
            data = {
                "host_id": self._host_id,
                "registrations": records,
                "active_client_id": active_client_id,
                "pending_new_client_id": pending_new_client_id,
            }
        self._save_registration(data)
        self._registrations = records
        self._active_client_id = active_client_id
        self._pending_new_client_id = pending_new_client_id

    def _record_for(self, client_id: str | None) -> dict[str, str] | None:
        return next((row for row in self._registrations if row["client_id"] == client_id), None)

    @staticmethod
    def _check_redirect_uri(redirect_uri: str) -> None:
        parsed = urlparse(redirect_uri)
        try:
            port = parsed.port
        except ValueError:
            port = None
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or not port
            or parsed.path != "/api/chatgpt/callback"
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise ChatGPTPlanAuthError("La devolución de ChatGPT debe usar /api/chatgpt/callback en 127.0.0.1.")

    def begin(self, redirect_uri: str) -> str:
        """Reauthorize the active registration, or register the first account."""
        return self._begin(redirect_uri, new_account=False)

    def begin_new_registration(self, redirect_uri: str) -> str:
        """Register another account without replacing the active registration."""
        return self._begin(redirect_uri, new_account=True)

    def _begin(self, redirect_uri: str, *, new_account: bool) -> str:
        """Create a one-use PKCE transaction and return the authorization URL."""
        self._check_redirect_uri(redirect_uri)
        with self._lock:
            state = secrets.token_urlsafe(32)
            nonce = secrets.token_urlsafe(32)
            verifier = secrets.token_urlsafe(64)
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
            target = "new" if new_account or (not self._active_client_id and self._pending_new_client_id) else "active"
            client_id = (
                (self._pending_new_client_id if target == "new" else self._active_client_id)
                or "dynamic_agent_client"
            )
            self._pending = {
                "state": state,
                "nonce": nonce,
                "verifier": verifier,
                "client_id": client_id,
                "target": target,
                "redirect_uri": redirect_uri,
                "expires_at": self._clock() + 600,
            }
            params = {
                "client_id": client_id,
                "ext_agent_host_id": self._host_id,
                "response_type": "code",
                "redirect_uri": redirect_uri,
                "scope": SCOPES,
                "resource": RESOURCE,
                "state": state,
                "nonce": nonce,
                "code_challenge_method": "S256",
                "code_challenge": challenge,
            }
            if client_id == "dynamic_agent_client":
                params["agent_name_hint"] = "MiroFish"
            return f"{AUTHORIZE_URL}?{urlencode(params)}"

    @staticmethod
    def _token_fields(data: Any, *, refresh: bool = False) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ChatGPTPlanAuthError("ChatGPT devolvió credenciales incompletas.")
        fields = ("access_token", "token_type", "scope")
        if not refresh:
            fields += ("id_token",)
        if any(not isinstance(data.get(field), str) or not data[field] for field in fields):
            raise ChatGPTPlanAuthError("ChatGPT devolvió credenciales incompletas.")
        if data["token_type"].lower() != "bearer":
            raise ChatGPTPlanAuthError("ChatGPT devolvió un tipo de credencial no admitido.")
        try:
            expiry = int(data["expires_in"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ChatGPTPlanAuthError("ChatGPT devolvió una caducidad inválida.") from exc
        if expiry <= 0 or expiry > 86400:
            raise ChatGPTPlanAuthError("ChatGPT devolvió una caducidad inválida.")
        scopes = set(data["scope"].split())
        plan_granted = {"chatgpt.tokens.use.direct", "resource.invoke", "offline_access"}.issubset(scopes)
        refresh_token = data.get("refresh_token")
        if (refresh or plan_granted) and (not isinstance(refresh_token, str) or not refresh_token):
            raise ChatGPTPlanAuthError("ChatGPT devolvió credenciales incompletas.")
        return {
            "access_token": data["access_token"],
            "refresh_token": refresh_token,
            "id_token": data.get("id_token"),
            "expires_in": expiry,
            "scopes": scopes,
        }

    def _post_token(self, form: Mapping[str, str]) -> dict[str, Any]:
        try:
            response = self._http.post(TOKEN_URL, data=form, headers={"Accept": "application/json"})
            if response.status_code != 200:
                raise ChatGPTPlanAuthError("ChatGPT no completó el intercambio de credenciales.")
            return self._token_fields(response.json(), refresh=form["grant_type"] == "refresh_token")
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise ChatGPTPlanAuthError("ChatGPT no completó el intercambio de credenciales.") from None

    def finish(self, query: Mapping[str, str], redirect_uri: str) -> dict[str, bool]:
        """Complete one callback after checking state, client binding and ID token."""
        self._check_redirect_uri(redirect_uri)
        with self._lock:
            pending = self._pending
            self._pending = None
            if not pending or pending["redirect_uri"] != redirect_uri or self._clock() >= pending["expires_at"]:
                raise ChatGPTPlanAuthError("La solicitud de conexión de ChatGPT caducó.")
            state = query.get("state", "")
            if not isinstance(state, str) or not hmac.compare_digest(state, pending["state"]):
                raise ChatGPTPlanAuthError("La respuesta de ChatGPT no coincide con la solicitud.")
            if query.get("error"):
                raise ChatGPTPlanAuthError("No se autorizó el uso del plan ChatGPT.")
            code = query.get("code")
            if not isinstance(code, str) or not code:
                raise ChatGPTPlanAuthError("ChatGPT no devolvió el código de autorización.")

            first_registration = pending["client_id"] == "dynamic_agent_client"
            callback_client = query.get("client_id")
            target = pending["target"]
            if first_registration:
                if not isinstance(callback_client, str) or not _CLIENT_ID.fullmatch(callback_client) or callback_client == "dynamic_agent_client":
                    raise ChatGPTPlanAuthError("ChatGPT no devolvió un identificador de cliente válido.")
                client_id = callback_client
                if self._record_for(client_id) is not None:
                    raise ChatGPTPlanAuthError("ChatGPT devolvió un identificador ya vinculado a otra conexión.")
                if target == "active" and self._active_client_id is not None:
                    raise ChatGPTPlanAuthError("La conexión activa cambió durante el registro.")
                if target == "new" and self._pending_new_client_id is not None:
                    raise ChatGPTPlanAuthError("Ya existe un registro nuevo pendiente.")
                # Keep an issued ID for invalid_grant recovery, but never
                # replace the active account before identity verification.
                records = [*self._registrations, {"client_id": client_id}]
                self._set_registration_state(
                    records,
                    client_id if target == "active" else self._active_client_id,
                    client_id if target == "new" else self._pending_new_client_id,
                )
            else:
                client_id = pending["client_id"]
                if callback_client and callback_client != client_id:
                    raise ChatGPTPlanAuthError("ChatGPT devolvió un identificador de cliente distinto.")
                selected = self._active_client_id if target == "active" else self._pending_new_client_id
                if selected != client_id or self._record_for(client_id) is None:
                    raise ChatGPTPlanAuthError("El registro de ChatGPT cambió durante la solicitud.")

            tokens = self._post_token({
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": code,
                "code_verifier": pending["verifier"],
                "redirect_uri": redirect_uri,
                "resource": RESOURCE,
            })
            try:
                claims = self._verifier(tokens["id_token"], client_id, pending["nonce"])
            except (jwt.PyJWTError, ChatGPTPlanAuthError, ValueError) as exc:
                raise ChatGPTPlanAuthError("No se pudo verificar la identidad de ChatGPT.") from None
            subject = claims.get("sub")
            if not isinstance(subject, str) or not subject:
                raise ChatGPTPlanAuthError("ChatGPT no devolvió una identidad válida.")
            selected_record = self._record_for(client_id)
            if selected_record is None:
                raise ChatGPTPlanAuthError("El registro de ChatGPT no está disponible.")
            if selected_record.get("subject") and selected_record["subject"] != subject:
                raise ChatGPTPlanAuthError("La cuenta de ChatGPT no coincide con la conexión anterior.")
            records = [dict(row) for row in self._registrations]
            for row in records:
                if row["client_id"] == client_id:
                    row["subject"] = subject
                    break
            self._set_registration_state(
                records,
                client_id if target == "new" else self._active_client_id,
                None if target == "new" else self._pending_new_client_id,
            )
            # A verified callback may change the active account or its grants.
            # Reset the prior model while holding the same lock used to select it.
            self._set_selected_model("")
            self._signed_in = True
            if not {"chatgpt.tokens.use.direct", "resource.invoke", "offline_access"}.issubset(tokens["scopes"]):
                self._tokens = None
                return self.status()
            self._tokens = {
                **tokens,
                "expires_at": self._clock() + tokens["expires_in"],
            }
            return self.status()

    def get_access_token(self) -> str:
        """Return an in-memory token, refreshing it serially when near expiry."""
        with self._lock:
            tokens = self._tokens
            if not tokens:
                raise ChatGPTPlanAuthError("Conecta y autoriza primero tu plan ChatGPT.")
            if self._clock() < tokens["expires_at"] - 60:
                return tokens["access_token"]
            fresh = self._post_token({
                "grant_type": "refresh_token",
                "client_id": self._active_client_id,
                "refresh_token": tokens["refresh_token"],
                "resource": RESOURCE,
            })
            if not {"chatgpt.tokens.use.direct", "resource.invoke", "offline_access"}.issubset(fresh["scopes"]):
                self._tokens = None
                raise ChatGPTPlanAuthError("El permiso de uso del plan ChatGPT ya no está disponible.")
            if fresh.get("id_token"):
                try:
                    claims = self._verifier(fresh["id_token"], self._active_client_id, None)
                except (jwt.PyJWTError, ChatGPTPlanAuthError, ValueError):
                    self._tokens = None
                    raise ChatGPTPlanAuthError("No se pudo verificar la renovación de ChatGPT.") from None
                active_record = self._record_for(self._active_client_id)
                if active_record is None or claims.get("sub") != active_record.get("subject"):
                    self._tokens = None
                    raise ChatGPTPlanAuthError("La cuenta renovada de ChatGPT no coincide.")
            self._tokens = {
                **fresh,
                "expires_at": self._clock() + fresh["expires_in"],
            }
            return fresh["access_token"]

    def status(self) -> dict[str, bool]:
        """Safe status for a local UI; never include identity or credentials."""
        with self._lock:
            active = self._record_for(self._active_client_id)
            return {
                "registered": bool(self._registrations),
                "identity_verified": bool(active and active.get("subject")),
                "pending_registration": bool(self._pending_new_client_id),
                "connected": self._signed_in,
                "plan_usage_enabled": bool(self._tokens),
            }

    def list_models(self) -> list[dict[str, str]]:
        """List models visible to the selected account, without exposing tokens."""
        token = self.get_access_token()
        try:
            response = self._http.get(f"{RESOURCE}/models", headers={"Authorization": f"Bearer {token}"})
            if response.status_code != 200:
                raise ChatGPTPlanAuthError("No se pudieron consultar los modelos del plan ChatGPT.")
            data = response.json()
        except (httpx.HTTPError, ValueError):
            raise ChatGPTPlanAuthError("No se pudieron consultar los modelos del plan ChatGPT.") from None
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise ChatGPTPlanAuthError("ChatGPT devolvió una lista de modelos inválida.")
        return [
            {"slug": row["slug"], "display_name": row.get("display_name", row["slug"])}
            for row in data["models"]
            if isinstance(row, dict) and row.get("visibility") == "list"
            and isinstance(row.get("slug"), str) and row["slug"]
        ]

    @staticmethod
    def _set_selected_model(model: str) -> None:
        from ..config import Config

        Config.LLM_MODEL_NAME = model
        os.environ["MIROFISH_PLAN_SELECTED_MODEL"] = model
        os.environ["LLM_MODEL_NAME"] = model

    def select_model(self, model: str) -> bool:
        """Commit a visible model for the verified active session atomically."""
        with self._lock:
            active = self._record_for(self._active_client_id)
            if not active or not active.get("subject") or not self._tokens:
                raise ChatGPTPlanAuthError("Conecta y autoriza primero tu plan ChatGPT.")
            visible = {item["slug"] for item in self.list_models()}
            if model not in visible:
                return False
            self._set_selected_model(model)
            return True

    def disconnect(self) -> None:
        """Stop local use; retained registration permits later reauthorization."""
        with self._lock:
            self._tokens = None
            self._pending = None
            self._signed_in = False

    clear = disconnect


auth = ChatGPTPlanAuth()
