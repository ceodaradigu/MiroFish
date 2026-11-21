"""Loopback-only OpenAI Chat Completions compatibility route for plan usage."""

from __future__ import annotations

import hmac
import ipaddress
import os

from flask import Blueprint, Response, jsonify, request

from app.config import Config
from app.services.chatgpt_plan_proxy import (
    ProxyPlanUsageError,
    ProxyRequestError,
    ProxyUpstreamError,
    completion_sse,
    create_completion,
)


chatgpt_proxy_bp = Blueprint("chatgpt_proxy", __name__)
_MAX_REQUEST_BYTES = 2 * 1024 * 1024


def _is_loopback(address: str | None) -> bool:
    try:
        parsed = ipaddress.ip_address(address or "")
    except ValueError:
        return False
    if parsed.version == 6 and parsed.ipv4_mapped:
        return parsed.ipv4_mapped.is_loopback
    return parsed.is_loopback


def _authorized() -> bool:
    if not _is_loopback(request.remote_addr) or request.headers.get("Origin"):
        return False
    expected = os.environ.get("MIROFISH_PLAN_PROXY_TOKEN", "")
    supplied = request.headers.get("Authorization", "")
    return bool(expected) and supplied.startswith("Bearer ") and hmac.compare_digest(supplied[7:], expected)


@chatgpt_proxy_bp.post("/v1/chat/completions")
def chat_completions():
    if not _authorized():
        return jsonify({"error": {"message": "Local plan proxy access denied", "type": "authentication_error"}}), 403
    if request.content_length is not None and request.content_length > _MAX_REQUEST_BYTES:
        return jsonify({"error": {"message": "Request too large", "type": "invalid_request_error"}}), 413
    if not request.is_json:
        return jsonify({"error": {"message": "JSON request required", "type": "invalid_request_error"}}), 415
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": {"message": "Invalid JSON request", "type": "invalid_request_error"}}), 400
    selected_model = Config.LLM_MODEL_NAME
    if not selected_model or payload.get("model") != selected_model:
        return jsonify({"error": {"message": "Model is not the selected ChatGPT plan model", "type": "invalid_request_error"}}), 422
    try:
        completion, downstream_stream, ignored = create_completion(payload)
    except ProxyRequestError as exc:
        return jsonify({"error": {"message": str(exc), "type": "invalid_request_error"}}), 422
    except ProxyPlanUsageError as exc:
        error = {
            "message": exc.public_message,
            "type": "plan_usage_error",
            "code": exc.code,
        }
        if exc.http_status == 429:
            error["usage_url"] = "https://chatgpt.com/settings/usage"
        return jsonify({"error": error}), exc.http_status, {"Cache-Control": "no-store"}
    except ProxyUpstreamError:
        return jsonify({"error": {"message": "ChatGPT plan response unavailable", "type": "upstream_error"}}), 502
    headers = {"Cache-Control": "no-store"}
    if ignored:
        headers["X-MiroFish-Adapter-Omitted"] = ",".join(ignored)
    if downstream_stream:
        return Response(
            completion_sse(completion, include_usage=bool(payload.get("stream_options"))),
            mimetype="text/event-stream", headers=headers,
        )
    return jsonify(completion), 200, headers
