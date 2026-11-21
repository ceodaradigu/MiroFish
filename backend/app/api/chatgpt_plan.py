"""Loopback UI endpoints for the optional Sign in with ChatGPT plan flow."""

from __future__ import annotations

import os
import threading

from flask import Blueprint, jsonify, redirect, request

from ..config import Config
from ..services.chatgpt_plan_auth import ChatGPTPlanAuthError, auth
from ..services.chatgpt_plan_proxy import ProxyUpstreamError, create_completion
from ..services.plan_work_guard import PlanWorkBusyError, plan_work_guard
from ..services.simulation_runner import SimulationRunner


chatgpt_plan_bp = Blueprint('chatgpt_plan', __name__)
CALLBACK_URL = 'http://127.0.0.1:5001/api/chatgpt/callback'
FRONTEND_URL = 'http://127.0.0.1:3000/'


class _OneShotDiagnostic:
    """Opt-in, process-local permit for one minimal plan inference."""

    def __init__(self, enabled: bool) -> None:
        self._lock = threading.Lock()
        self._available = enabled

    def available(self) -> bool:
        with self._lock:
            return self._available

    def take(self) -> bool:
        with self._lock:
            if not self._available:
                return False
            self._available = False
            return True


# The process flag is consumed on import, then removed from its environment.
# It is never persisted in a project file or exposed to the browser.
_diagnostic_once = _OneShotDiagnostic(os.environ.pop('MIROFISH_DIAG_ONCE', '') == '1')


def _busy_response():
    if SimulationRunner.get_running_simulations():
        return jsonify({'error': 'Espera a que termine la simulación antes de cambiar la conexión o el modelo.'}), 409
    if plan_work_guard.has_work():
        return jsonify({'error': 'Espera a que termine el trabajo activo antes de cambiar la conexión o el modelo.'}), 409
    return None


@chatgpt_plan_bp.after_request
def no_cache(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response


@chatgpt_plan_bp.get('/status')
def status():
    state = auth.status()
    return jsonify({
        'planMode': True,
        'registered': state.get('registered', False),
        'identityVerified': state.get('identity_verified', False),
        'pendingRegistration': state.get('pending_registration', False),
        'connected': state['connected'],
        'planUsageEnabled': state['plan_usage_enabled'],
        'memoryBackend': Config.MEMORY_BACKEND,
        'memoryReady': Config.memory_backend_configured(),
        'zepConfigured': Config.zep_is_configured(),
        'selectedModel': Config.LLM_MODEL_NAME if state['plan_usage_enabled'] else '',
    })


@chatgpt_plan_bp.get('/start')
def start():
    busy = _busy_response()
    if busy:
        return busy
    try:
        return redirect(auth.begin(CALLBACK_URL), code=302)
    except ChatGPTPlanAuthError as exc:
        return jsonify({'error': str(exc)}), 400


@chatgpt_plan_bp.get('/start-new')
def start_new_registration():
    """Start a separate, user-consented registration for another account."""

    busy = _busy_response()
    if busy:
        return busy
    try:
        return redirect(auth.begin_new_registration(CALLBACK_URL), code=302)
    except ChatGPTPlanAuthError as exc:
        return jsonify({'error': str(exc)}), 400


@chatgpt_plan_bp.get('/callback')
def callback():
    try:
        with plan_work_guard.transition():
            if SimulationRunner.get_running_simulations():
                raise PlanWorkBusyError('Espera a que termine la simulación antes de cambiar la conexión.')
            auth.finish(request.args, CALLBACK_URL)
    except PlanWorkBusyError as exc:
        return jsonify({'error': str(exc)}), 409
    except ChatGPTPlanAuthError as exc:
        return jsonify({'error': str(exc)}), 400
    return redirect(FRONTEND_URL, code=303)


@chatgpt_plan_bp.get('/models')
def models():
    try:
        return jsonify({'models': auth.list_models()})
    except ChatGPTPlanAuthError as exc:
        return jsonify({'error': str(exc)}), 503


@chatgpt_plan_bp.post('/model')
def select_model():
    if request.content_length is not None and request.content_length > 4096:
        return jsonify({'error': 'La selección de modelo es demasiado grande.'}), 413
    payload = request.get_json(silent=True)
    model = payload.get('model') if isinstance(payload, dict) else None
    if not isinstance(model, str) or not model or len(model) > 128:
        return jsonify({'error': 'Selecciona un modelo válido.'}), 400
    try:
        with plan_work_guard.transition():
            if SimulationRunner.get_running_simulations():
                raise PlanWorkBusyError('Espera a que termine la simulación antes de cambiar el modelo.')
            selected = auth.select_model(model)
    except PlanWorkBusyError as exc:
        return jsonify({'error': str(exc)}), 409
    except ChatGPTPlanAuthError as exc:
        return jsonify({'error': str(exc)}), 503
    if not selected:
        return jsonify({'error': 'El modelo no está disponible para esta cuenta.'}), 400
    return jsonify({'selectedModel': model})


@chatgpt_plan_bp.post('/self-test')
def self_test():
    """Make one official minimal inference to isolate connection failures.

    Disabled by default. The response never includes model output, prompts,
    tokens, or provider prose. Existing loopback and origin checks still apply.
    """

    if not _diagnostic_once.available():
        return jsonify({'error': 'No disponible.'}), 404
    try:
        lease = plan_work_guard.claim()
    except PlanWorkBusyError:
        return jsonify({'error': 'Hay un trabajo activo.'}), 409
    try:
        if not auth.status()['plan_usage_enabled'] or not Config.LLM_MODEL_NAME:
            return jsonify({'error': 'Conecta ChatGPT y selecciona un modelo.'}), 503
        if not _diagnostic_once.take():
            return jsonify({'error': 'No disponible.'}), 404
        try:
            create_completion({
                'model': Config.LLM_MODEL_NAME,
                'messages': [{'role': 'user', 'content': 'Say exactly: Hello, world!'}],
                'stream': False,
            })
        except ProxyUpstreamError as exc:
            error = {'type': 'upstream_error'}
            if exc.code:
                error['code'] = exc.code
            if exc.param:
                error['param'] = exc.param
            if exc.request_id:
                error['request_id'] = exc.request_id
            if exc.provider_type:
                error['provider_type'] = exc.provider_type
            if exc.category:
                error['category'] = exc.category
            return jsonify({'ok': False, 'error': error}), exc.http_status
        return jsonify({'ok': True})
    finally:
        lease.release()
