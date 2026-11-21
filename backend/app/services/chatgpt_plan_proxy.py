"""Local Chat Completions adapter for Sign in with ChatGPT plan usage.

The public Responses API is the only upstream. Each request is stateless,
streamed, and never stored. The adapter accepts the small Chat Completions
surface used by MiroFish and CAMEL, and rejects shapes it cannot translate.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


LOGGER = logging.getLogger(__name__)
TOOL_NAMESPACE = "mirofish_actions"
_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
_SAFE_ERROR_TYPES = {
    "invalid_request_error", "authentication_error", "permission_error",
    "rate_limit_error", "server_error", "api_error",
}
# Only these generic protocol words can appear in the optional diagnostic.
# Every other word (including names, credentials and request text) is replaced.
_ERROR_SKELETON_WORDS = {
    "allowed", "array", "content", "expected", "format", "input", "instructions",
    "invalid", "json", "length", "limit", "long", "message", "missing",
    "model", "must", "not", "permission", "policy", "required", "role",
    "safety", "string", "system", "token", "too", "type", "unsupported",
    "user", "workspace",
}
_RESPONSE_FAILURE_STATUSES = {
    "subscription_sharing_user_not_eligible": 403,
    "subscription_sharing_usage_limit_exceeded": 429,
    "subscription_sharing_usage_unavailable": 503,
    "subscription_sharing_unsupported_capability": 400,
    "subscription_sharing_route_not_supported": 403,
    "subscription_sharing_invalid_user": 401,
    "subscription_sharing_user_unavailable": 503,
    "chatpass_v2_scope_not_authorized": 403,
    "chatpass_v2_invalid_authorization_context": 403,
}
_SAFE_ERROR_CODES = set(_RESPONSE_FAILURE_STATUSES) | {
    "invalid_request_error", "rate_limit_exceeded", "server_error",
    "context_length_exceeded",
}
_SAFE_ERROR_PARAMS = {
    "input", "instructions", "model", "text.format", "tools",
    "tool_choice", "parallel_tool_calls", "response_format", "store", "stream",
}
_SAFE_BODY_KEYS = {"code", "error", "message", "param", "type"}
_DROPPABLE_FIELDS = {"temperature", "max_tokens", "max_completion_tokens"}
_ALLOWED_FIELDS = {
    "model", "messages", "stream", "stream_options", "tools", "tool_choice",
    "parallel_tool_calls", "response_format", *_DROPPABLE_FIELDS,
}


class ProxyRequestError(ValueError):
    """A Chat Completions request cannot be translated without data loss."""


class ProxyUpstreamError(RuntimeError):
    """The Responses API did not return a completed result."""

    def __init__(self, message: str, *, http_status: int = 502, code: str | None = None, param: str | None = None, request_id: str | None = None, provider_type: str | None = None, category: str | None = None):
        super().__init__(message)
        self.http_status = http_status
        self.code = code
        self.param = param
        self.request_id = request_id
        self.provider_type = provider_type
        self.category = category


class ProxyPlanUsageError(ProxyUpstreamError):
    """A recognized plan-use admission result, with fixed public recovery."""

    _RECOVERY = {
        "subscription_sharing_usage_limit_exceeded": (
            429,
            "Se alcanzó un límite de uso del plan ChatGPT para esta aplicación. Consulta Uso en ChatGPT antes de continuar.",
        ),
        "subscription_sharing_usage_unavailable": (
            503,
            "No se pudo comprobar la disponibilidad de uso del plan ChatGPT. Inténtalo más tarde.",
        ),
    }

    def __init__(self, code: str):
        if code not in self._RECOVERY:
            raise ValueError("Unrecognized plan usage error")
        self.code = code
        self.http_status, self.public_message = self._RECOVERY[code]
        super().__init__(self.public_message, http_status=self.http_status, code=code)


class _OneShotErrorMessageDiagnostic:
    """Process-local permission to emit one redacted provider-error skeleton."""

    def __init__(self, enabled: bool) -> None:
        self._lock = threading.Lock()
        self._available = enabled

    def take(self) -> bool:
        with self._lock:
            if not self._available:
                return False
            self._available = False
            return True


# This deliberately does not survive a restart or appear in the public API.
_error_message_diagnostic = _OneShotErrorMessageDiagnostic(
    os.environ.pop("MIROFISH_DIAG_ERROR_ONCE", "") == "1"
)


def _message_skeleton(message: str) -> str:
    """Keep protocol words in order, never arbitrary words from the input."""
    pieces: list[str] = []
    for token in re.findall(r"\w+|[^\w\s]", message, flags=re.UNICODE)[:48]:
        safe = token.lower() if token.lower() in _ERROR_SKELETON_WORDS else "[redacted]"
        if safe == "[redacted]" and pieces and pieces[-1] == safe:
            continue
        pieces.append(safe)
    return " ".join(pieces)[:240]


def _provider_message(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    detail = body.get("error")
    if isinstance(detail, dict):
        body = detail
    message = body.get("message")
    return message if isinstance(message, str) and message else None


def _message_category(message: Any) -> str:
    """Classify provider prose without writing or returning it."""
    if not isinstance(message, str):
        return "other"
    lower = message.lower()
    if any(term in lower for term in ("permission", "not eligible", "not allowed", "unauthorized", "access denied", "scope")):
        return "permission"
    if any(term in lower for term in ("usage", "quota", "rate limit", "credits", "usage limit")):
        return "usage"
    if any(term in lower for term in ("unsupported", "not supported", "does not support")):
        return "unsupported_capability"
    if any(term in lower for term in ("invalid input", "input is invalid", "malformed input", "input must", "input should", "input format")):
        return "invalid_input"
    return "other"


def _error_parts(body: Any) -> tuple[str | None, str | None, str | None, str, str]:
    """Return machine fields and only the *shape* of an upstream body."""
    if not isinstance(body, dict):
        return None, None, None, "other", type(body).__name__ if body is not None else "none"
    detail = body.get("error")
    if not isinstance(detail, dict):
        detail = body
    top_keys = ",".join(sorted(key for key in body if key in _SAFE_BODY_KEYS))
    error_keys = ",".join(sorted(key for key in detail if key in _SAFE_BODY_KEYS))
    shape = f"object[{top_keys}]/error[{error_keys}]"
    raw_type = detail.get("type")
    provider_type = raw_type if isinstance(raw_type, str) and raw_type in _SAFE_ERROR_TYPES else None
    return (
        detail.get("code") if isinstance(detail.get("code"), str) and detail["code"] in _SAFE_ERROR_CODES else None,
        detail.get("param") if isinstance(detail.get("param"), str) and detail["param"] in _SAFE_ERROR_PARAMS else None,
        provider_type,
        _message_category(detail.get("message")),
        shape,
    )


def _log_upstream_failure(stage: str, *, error: Exception | None = None, body: Any = None) -> tuple[int, str | None, str | None, str | None, str | None, str]:
    """Record enough to diagnose admission, transport and parsing safely."""
    source = getattr(error, "body", None) if error is not None else body
    code, param, provider_type, category, shape = _error_parts(source)
    raw_status = getattr(error, "status_code", None)
    status = raw_status if isinstance(raw_status, int) and 100 <= raw_status <= 599 else _RESPONSE_FAILURE_STATUSES.get(code, 502)
    # An upstream-controlled request ID is never copied to our logs or API.
    request_id = None
    LOGGER.error(
        "ChatGPT plan request failed: stage=%s class=%s status=%s code=%s param=%s provider_type=%s category=%s request_id=%s body_shape=%s",
        stage, type(error).__name__ if error is not None else "none", status,
        code or "none", param or "none", provider_type or "none", category,
        request_id or "none", shape,
    )
    if _error_message_diagnostic.take():
        message = _provider_message(source)
        if message is not None:
            LOGGER.error("ChatGPT plan one-shot error skeleton: %s", _message_skeleton(message))
    return status, code, param, request_id, provider_type, category


def _mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        result = value.model_dump(mode="json")
        if isinstance(result, dict):
            return result
    return dict(getattr(value, "__dict__", {}))


def _text_content(content: Any, *, allow_none: bool = False) -> Optional[str]:
    if content is None and allow_none:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if not isinstance(part, dict) or set(part) != {"type", "text"} or part["type"] != "text" or not isinstance(part["text"], str):
                raise ProxyRequestError("Only text message parts are supported")
            parts.append(part["text"])
        return "".join(parts)
    raise ProxyRequestError("Message content must be text")


def _convert_tools(tools: Any) -> List[Dict[str, Any]]:
    if not isinstance(tools, list) or not tools:
        raise ProxyRequestError("tools must be a nonempty list")
    converted = []
    names = set()
    for tool in tools:
        if not isinstance(tool, dict) or set(tool) != {"type", "function"} or tool.get("type") != "function":
            raise ProxyRequestError("Only Chat Completions function tools are supported")
        function = tool.get("function")
        if not isinstance(function, dict) or set(function) - {"name", "description", "parameters", "strict"}:
            raise ProxyRequestError("Unsupported function tool fields")
        name = function.get("name")
        if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name) or name in names:
            raise ProxyRequestError("Invalid or duplicate function tool name")
        names.add(name)
        schema = function.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(schema, dict):
            raise ProxyRequestError("Function parameters must be a JSON schema object")
        descriptor: Dict[str, Any] = {
            "type": "function", "name": name,
            "description": function.get("description") or "",
            "parameters": schema,
            # CAMEL action schemas are not necessarily strict-schema compatible.
            "strict": function.get("strict") if isinstance(function.get("strict"), bool) else False,
        }
        converted.append(descriptor)
    return [{
        "type": "namespace", "name": TOOL_NAMESPACE,
        "description": "Functions implemented locally by the MiroFish social simulation",
        "tools": converted,
    }]


def _convert_messages(messages: Any) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    if not isinstance(messages, list) or not messages:
        raise ProxyRequestError("messages must be a nonempty list")
    instructions: List[str] = []
    input_items: List[Dict[str, Any]] = []
    prior_calls: Dict[str, str] = {}
    seen_conversation = False
    for message in messages:
        if not isinstance(message, dict):
            raise ProxyRequestError("Each message must be an object")
        role = message.get("role")
        if role in ("system", "developer"):
            if seen_conversation or set(message) - {"role", "content"}:
                raise ProxyRequestError("System/developer messages must precede conversation messages")
            instructions.append(_text_content(message.get("content")) or "")
            continue

        seen_conversation = True
        if role in ("user", "assistant"):
            allowed = {"role", "content", "tool_calls"} if role == "assistant" else {"role", "content"}
            if set(message) - allowed:
                raise ProxyRequestError("Unsupported message fields")
            content = _text_content(message.get("content"), allow_none=role == "assistant")
            if content is not None and (content != "" or not message.get("tool_calls")):
                input_items.append({"role": role, "content": content})
            calls = message.get("tool_calls") if role == "assistant" else None
            if calls is not None:
                if not isinstance(calls, list) or not calls:
                    raise ProxyRequestError("tool_calls must be a nonempty list")
                for call in calls:
                    if not isinstance(call, dict) or set(call) != {"id", "type", "function"} or call.get("type") != "function":
                        raise ProxyRequestError("Only function tool call history is supported")
                    function = call.get("function")
                    if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
                        raise ProxyRequestError("Invalid function call history")
                    call_id, name, arguments = call.get("id"), function.get("name"), function.get("arguments")
                    if not isinstance(call_id, str) or not call_id or len(call_id) > 64 or call_id in prior_calls:
                        raise ProxyRequestError("Invalid or duplicate tool call id")
                    if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name) or not isinstance(arguments, str):
                        raise ProxyRequestError("Invalid function call history")
                    try:
                        json.loads(arguments)
                    except ValueError as exc:
                        raise ProxyRequestError("Function call arguments must be JSON") from exc
                    prior_calls[call_id] = name
                    input_items.append({
                        "type": "function_call", "call_id": call_id,
                        "name": name, "arguments": arguments,
                        "namespace": TOOL_NAMESPACE,
                    })
            elif role == "assistant" and content is None:
                raise ProxyRequestError("Assistant message has no content or tool calls")
        elif role == "tool":
            if set(message) - {"role", "tool_call_id", "name", "content"}:
                raise ProxyRequestError("Unsupported tool result fields")
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in prior_calls:
                raise ProxyRequestError("Tool result has no matching call")
            name = message.get("name")
            if name is not None and name != prior_calls[call_id]:
                raise ProxyRequestError("Tool result name does not match call")
            output = _text_content(message.get("content"))
            input_items.append({
                "type": "function_call_output", "call_id": call_id,
                "output": output, "namespace": TOOL_NAMESPACE,
            })
        else:
            raise ProxyRequestError("Unsupported message role")
    if not input_items:
        raise ProxyRequestError("At least one conversation message is required")
    return "\n\n".join(instructions) if instructions else None, input_items


def _ensure_json_instruction_in_input(input_items: List[Dict[str, Any]]) -> None:
    """JSON mode requires a JSON instruction in Responses ``input`` itself.

    Chat Completions system text is translated to ``instructions`` above, but
    the Responses API validates its ``input`` separately for JSON mode.
    """
    user_messages = [item for item in input_items if item.get("role") == "user"]
    if any("json" in item["content"].lower() for item in user_messages):
        return
    if user_messages:
        user_messages[-1]["content"] += "\n\nReturn a JSON object."
    else:
        input_items.append({"role": "user", "content": "Return a JSON object."})


def build_responses_request(payload: Any) -> Tuple[Dict[str, Any], bool, List[str]]:
    """Validate and translate one Chat Completions request.

    Returns (Responses kwargs, downstream stream flag, omitted compatibility
    fields). The only intentionally omitted fields are sampling/token limits,
    which this plan route explicitly forbids; callers receive their names.
    """
    if not isinstance(payload, dict):
        raise ProxyRequestError("Request body must be an object")
    unexpected = set(payload) - _ALLOWED_FIELDS
    if unexpected:
        raise ProxyRequestError("Unsupported Chat Completions fields: " + ", ".join(sorted(unexpected)))
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ProxyRequestError("model must be a nonempty string")
    downstream_stream = payload.get("stream", False)
    if not isinstance(downstream_stream, bool):
        raise ProxyRequestError("stream must be a boolean")
    options = payload.get("stream_options")
    if options is not None and (not isinstance(options, dict) or set(options) != {"include_usage"} or options["include_usage"] is not True or not downstream_stream):
        raise ProxyRequestError("Unsupported stream_options")
    for name in _DROPPABLE_FIELDS:
        if name in payload and (not isinstance(payload[name], (int, float)) or isinstance(payload[name], bool)):
            raise ProxyRequestError(name + " must be numeric")
    ignored = sorted(set(payload) & _DROPPABLE_FIELDS)
    if ignored:
        LOGGER.warning("ChatGPT plan adapter omitted unsupported request fields: %s", ", ".join(ignored))
    instructions, input_items = _convert_messages(payload.get("messages"))
    kwargs: Dict[str, Any] = {
        "model": model, "input": input_items, "store": False, "stream": True,
    }
    if instructions is not None:
        kwargs["instructions"] = instructions
    tools = payload.get("tools")
    if tools is not None:
        kwargs["tools"] = _convert_tools(tools)
    choice = payload.get("tool_choice")
    if choice is not None:
        if choice not in ("auto", "none", "required"):
            raise ProxyRequestError("Only auto, none, and required tool_choice are supported")
        if choice == "required" and tools is None:
            raise ProxyRequestError("required tool_choice needs tools")
        kwargs["tool_choice"] = choice
    parallel = payload.get("parallel_tool_calls")
    if parallel is not None:
        if not isinstance(parallel, bool) or tools is None:
            raise ProxyRequestError("parallel_tool_calls requires tools and a boolean value")
        kwargs["parallel_tool_calls"] = parallel
    response_format = payload.get("response_format")
    if response_format is not None:
        if response_format == {"type": "json_object"}:
            _ensure_json_instruction_in_input(input_items)
            kwargs["text"] = {"format": {"type": "json_object"}}
        elif response_format != {"type": "text"}:
            raise ProxyRequestError("Only text and json_object response formats are supported")
    return kwargs, downstream_stream, ignored


def _default_stream_factory(**kwargs: Any) -> Iterable[Any]:
    # Import on demand so unit tests never need a real account or token.
    from openai import OpenAI
    from app.services.chatgpt_plan_auth import auth

    access_token = auth.get_access_token()
    client = OpenAI(api_key=access_token, base_url="https://api.openai.com/v1", max_retries=0)
    return client.responses.create(**kwargs)


def _response_text_and_calls(response: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    text_parts: List[str] = []
    calls: List[Dict[str, Any]] = []
    output = response.get("output")
    if not isinstance(output, list):
        raise ProxyUpstreamError("Responses API returned no output")
    for raw_item in output:
        item = _mapping(raw_item)
        item_type = item.get("type")
        if item_type == "reasoning":
            continue
        if item_type == "message":
            if item.get("role") != "assistant" or not isinstance(item.get("content"), list):
                raise ProxyUpstreamError("Responses API returned an unsupported message")
            for raw_part in item["content"]:
                part = _mapping(raw_part)
                if part.get("type") != "output_text" or not isinstance(part.get("text"), str):
                    raise ProxyUpstreamError("Responses API returned unsupported content")
                text_parts.append(part["text"])
        elif item_type == "function_call":
            name, call_id, arguments = item.get("name"), item.get("call_id"), item.get("arguments")
            if item.get("namespace") not in (None, TOOL_NAMESPACE) or not isinstance(name, str) or not _TOOL_NAME.fullmatch(name) or not isinstance(call_id, str) or not call_id or not isinstance(arguments, str):
                raise ProxyUpstreamError("Responses API returned an unsupported function call")
            try:
                json.loads(arguments)
            except ValueError as exc:
                raise ProxyUpstreamError("Responses API returned invalid function arguments") from exc
            calls.append({
                "id": call_id, "type": "function",
                "function": {"name": name, "arguments": arguments},
            })
        else:
            raise ProxyUpstreamError("Responses API returned an unsupported output item")
    return "".join(text_parts), calls


def _usage(response: Dict[str, Any]) -> Optional[Dict[str, int]]:
    usage = _mapping(response.get("usage")) if response.get("usage") is not None else {}
    if not usage:
        return None
    prompt = usage.get("input_tokens")
    completion = usage.get("output_tokens")
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return None
    total = usage.get("total_tokens")
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total if isinstance(total, int) else prompt + completion}


def create_completion(payload: Dict[str, Any], *, stream_factory: Optional[Callable[..., Iterable[Any]]] = None) -> Tuple[Dict[str, Any], bool, List[str]]:
    """Consume the upstream SSE before returning any downstream result."""
    kwargs, downstream_stream, ignored = build_responses_request(payload)
    factory = stream_factory or _default_stream_factory
    stage = "request"
    try:
        stream = factory(**kwargs)
        stage = "stream"
        completed: Optional[Dict[str, Any]] = None
        try:
            for raw_event in stream:
                event = _mapping(raw_event)
                event_type = event.get("type")
                if event_type == "response.failed":
                    response = _mapping(event.get("response"))
                    error = _mapping(response.get("error"))
                    code = error.get("code")
                    status, safe_code, safe_param, _, provider_type, category = _log_upstream_failure("response.failed", body={"error": error})
                    if isinstance(code, str) and code in ProxyPlanUsageError._RECOVERY:
                        raise ProxyPlanUsageError(code)
                    raise ProxyUpstreamError(
                        "Responses API did not complete", http_status=status,
                        code=safe_code, param=safe_param,
                        provider_type=provider_type, category=category,
                    )
                if event_type in ("response.incomplete", "error"):
                    _log_upstream_failure(str(event_type), body=event)
                    raise ProxyUpstreamError("Responses API did not complete")
                if event_type == "response.completed":
                    completed = _mapping(event.get("response"))
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
    except ProxyUpstreamError:
        raise
    except Exception as exc:
        # Never put an upstream exception in an HTTP response: some SDK errors
        # include headers or request detail that may contain credentials.
        status, code, param, request_id, provider_type, category = _log_upstream_failure(stage, error=exc)
        raise ProxyUpstreamError(
            "Responses API request failed", http_status=status,
            code=code, param=param, request_id=request_id,
            provider_type=provider_type, category=category,
        ) from None
    if completed is None or completed.get("status") not in (None, "completed"):
        _log_upstream_failure("stream_end", body=completed)
        raise ProxyUpstreamError("Responses API ended without response.completed")
    try:
        text, calls = _response_text_and_calls(completed)
    except ProxyUpstreamError as exc:
        _log_upstream_failure("decode", error=exc)
        raise
    if not text and not calls:
        _log_upstream_failure("decode_empty", body=completed)
        raise ProxyUpstreamError("Responses API returned no assistant output")
    completion: Dict[str, Any] = {
        "id": completed.get("id") or "chatcmpl-local",
        "object": "chat.completion", "created": completed.get("created_at") or int(time.time()),
        "model": completed.get("model") or kwargs["model"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text or None, **({"tool_calls": calls} if calls else {})},
            "finish_reason": "tool_calls" if calls else "stop",
        }],
        "usage": _usage(completed),
    }
    return completion, downstream_stream, ignored


def completion_sse(completion: Dict[str, Any], *, include_usage: bool = False) -> Iterable[str]:
    """Emit SDK-compatible chunks after the completed upstream response."""
    common = {key: completion[key] for key in ("id", "created", "model")}
    common["object"] = "chat.completion.chunk"

    def emit(delta: Dict[str, Any], finish_reason: Optional[str] = None, usage: Optional[Dict[str, int]] = None) -> str:
        chunk = {**common, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
        if usage is not None:
            chunk["usage"] = usage
        return "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"

    message = completion["choices"][0]["message"]
    yield emit({"role": "assistant", "content": ""})
    if message.get("content"):
        yield emit({"content": message["content"]})
    if message.get("tool_calls"):
        yield emit({"tool_calls": [{"index": index, **call} for index, call in enumerate(message["tool_calls"])]})
    yield emit({}, completion["choices"][0]["finish_reason"])
    if include_usage:
        chunk = {**common, "choices": [], "usage": completion.get("usage")}
        yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
    yield "data: [DONE]\n\n"
