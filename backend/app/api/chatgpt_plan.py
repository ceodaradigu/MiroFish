"""Loopback UI endpoints for the optional Sign in with ChatGPT plan flow."""

from __future__ import annotations

import os

from flask import Blueprint, jsonify, redirect, request

from ..config import Config
from ..services.chatgpt_plan_auth import ChatGPTPlanAuthError, auth
from ..services.simulation_runner import SimulationRunner


chatgpt_plan_bp = Blueprint('chatgpt_plan', __name__)
CALLBACK_URL = 'http://127.0.0.1:5001/api/chatgpt/callback'
FRONTEND_URL = 'http://127.0.0.1:3000/'


def _busy_response():
    if SimulationRunner.get_running_simulations():
        return jsonify({'error': 'Espera a que termine la simulación antes de cambiar la conexión o el modelo.'}), 409
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


@chatgpt_plan_bp.get('/callback')
def callback():
    busy = _busy_response()
    if busy:
        return busy
    try:
        auth.finish(request.args, CALLBACK_URL)
    except ChatGPTPlanAuthError as exc:
        return jsonify({'error': str(exc)}), 400
    # A renewed consent may expose a different model list; require fresh choice.
    Config.LLM_MODEL_NAME = ''
    os.environ['MIROFISH_PLAN_SELECTED_MODEL'] = ''
    os.environ['LLM_MODEL_NAME'] = ''
    return redirect(FRONTEND_URL, code=303)


@chatgpt_plan_bp.get('/models')
def models():
    try:
        return jsonify({'models': auth.list_models()})
    except ChatGPTPlanAuthError as exc:
        return jsonify({'error': str(exc)}), 503


@chatgpt_plan_bp.post('/model')
def select_model():
    busy = _busy_response()
    if busy:
        return busy
    if request.content_length is not None and request.content_length > 4096:
        return jsonify({'error': 'La selección de modelo es demasiado grande.'}), 413
    payload = request.get_json(silent=True)
    model = payload.get('model') if isinstance(payload, dict) else None
    if not isinstance(model, str) or not model or len(model) > 128:
        return jsonify({'error': 'Selecciona un modelo válido.'}), 400
    try:
        visible = {item['slug'] for item in auth.list_models()}
    except ChatGPTPlanAuthError as exc:
        return jsonify({'error': str(exc)}), 503
    if model not in visible:
        return jsonify({'error': 'El modelo no está disponible para esta cuenta.'}), 400
    Config.LLM_MODEL_NAME = model
    os.environ['MIROFISH_PLAN_SELECTED_MODEL'] = model
    os.environ['LLM_MODEL_NAME'] = model
    return jsonify({'selectedModel': model})
