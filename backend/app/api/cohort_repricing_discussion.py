"""Loopback-only peer discussion of a completed paired pricing test."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading

from flask import Blueprint, jsonify, request

from . import cohort as cohort_api
from . import cohort_discussion as discussion_api
from . import cohort_repricing as repricing_api
from ..config import Config
from ..services.cohort_repricing_discussion import (
    RepricingDiscussionError, choices_hash, generate_plan_discussion,
    validate_complete_repricing, validate_plan_discussion_batch,
)
from ..services.cohort_survey import BATCH_SIZE, COHORT_SIZE
from ..services.plan_work_guard import retain_request_plan_work, release_plan_work
from ..utils.llm_client import LLMClient


repricing_discussion_bp = Blueprint("cohort_repricing_discussion", __name__)
logger = logging.getLogger(__name__)


class PilotPause(Exception):
    """The first complete table is ready for review."""


@repricing_discussion_bp.before_request
def require_loopback():
    return cohort_api.require_loopback()


def _directory(cohort_id: str, run_id: str) -> Path:
    return repricing_api._directory(cohort_id, run_id) / "discussion"


def _read_state(cohort_id: str, run_id: str) -> dict | None:
    try:
        path = _directory(cohort_id, run_id) / "state.json"
    except ValueError:
        return None
    with cohort_api._storage_lock:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return state if isinstance(state, dict) else None


def _public_state(state: dict) -> dict:
    return {
        "cohort_id": state["cohort_id"],
        "run_id": state["run_id"],
        "status": state["status"],
        "synthetic": True,
        "participants_target": COHORT_SIZE,
        "replies_completed": len(state.get("replies", [])),
        "tables_completed": len(state.get("replies", [])) // BATCH_SIZE,
        "tables_target": COHORT_SIZE // BATCH_SIZE,
        "pilot": bool(state.get("pilot")),
        "model": state.get("model", ""),
        "error": state.get("error"),
    }


def _source(cohort_id: str, run_id: str) -> tuple[str, str, list[dict], list[dict], str]:
    state = repricing_api._read_state(cohort_id, run_id)
    if state is None or state.get("status") != "completed":
        raise RepricingDiscussionError("La comparación de precios debe estar completa.")
    try:
        scenario, rows = repricing_api._source(cohort_id, state)
        offer = state["offer_context"]
        rows, choices = validate_complete_repricing(
            scenario, offer, rows, state.get("choices"))
    except (ValueError, TypeError, KeyError) as exc:
        raise RepricingDiscussionError("La comparación guardada es inválida.") from exc
    return scenario, offer, rows, choices, choices_hash(choices)


def _work(cohort_id: str, run_id: str, lease) -> None:
    directory = _directory(cohort_id, run_id)
    state = _read_state(cohort_id, run_id)
    try:
        if state is None:
            raise RepricingDiscussionError("discussion state disappeared")
        scenario, offer, rows, choices, digest = _source(cohort_id, run_id)
        if digest != state.get("repricing_sha256"):
            raise RepricingDiscussionError("repricing changed")
        llm = LLMClient()

        def ask(messages: list[dict]) -> dict:
            raw = llm.chat(messages, temperature=0.6, max_tokens=4096,
                           response_format={"type": "json_object"})
            return json.loads(raw)

        def checkpoint(replies: list[dict]) -> None:
            state["replies"] = replies
            cohort_api._write_json(directory / "state.json", state)
            if state.get("pilot") and len(replies) == BATCH_SIZE:
                raise PilotPause()

        report = generate_plan_discussion(
            scenario, offer, rows, choices, ask,
            existing_replies=state.get("replies", []), on_table=checkpoint,
        )
        report.update({
            "cohort_id": cohort_id,
            "run_id": run_id,
            "model": state["model"],
        })
        cohort_api._write_json(directory / "report.json", report)
        state["status"] = "completed"
        state["error"] = None
        cohort_api._write_json(directory / "state.json", state)
    except PilotPause:
        state["status"] = "pilot_ready"
        state["error"] = None
        cohort_api._write_json(directory / "state.json", state)
    except Exception as exc:
        logger.warning("Repricing discussion %s stopped after %s replies (%s)",
                       run_id, len(state.get("replies", [])) if state else 0,
                       type(exc).__name__)
        if state is not None:
            state["status"] = "paused"
            state["error"] = (
                "La siguiente mesa no pasó la validación o el modelo dejó de estar disponible. "
                "Las mesas completas siguen guardadas; se puede reanudar."
            )
            cohort_api._write_json(directory / "state.json", state)
    finally:
        with cohort_api._active_lock:
            cohort_api._active_jobs.discard(f"repricing-discussion:{run_id}")
        release_plan_work(lease)


def _start(cohort_id: str, run_id: str, state: dict):
    with cohort_api._active_lock:
        if cohort_api._active_jobs:
            return jsonify({"error": "Ya hay un trabajo local generándose."}), 409
        cohort_api._active_jobs.add(f"repricing-discussion:{run_id}")
    lease = retain_request_plan_work()
    state["status"] = "running"
    state["error"] = None
    try:
        cohort_api._write_json(_directory(cohort_id, run_id) / "state.json", state)
        thread = threading.Thread(target=_work, args=(cohort_id, run_id, lease),
                                  daemon=True, name=f"mirofish-price-debate-{run_id[:8]}")
        thread.start()
    except Exception:
        with cohort_api._active_lock:
            cohort_api._active_jobs.discard(f"repricing-discussion:{run_id}")
        release_plan_work(lease)
        return jsonify({"error": "No se pudo iniciar el debate local."}), 500
    return jsonify({"success": True, "data": _public_state(state)}), 202


@repricing_discussion_bp.post("/<cohort_id>/repricing/<run_id>/discussion/run")
def start_discussion(cohort_id: str, run_id: str):
    payload = request.get_json(silent=True) or {}
    if (not isinstance(payload, dict) or not set(payload).issubset({"pilot"})
            or type(payload.get("pilot", False)) is not bool):
        return jsonify({"error": "Solo se acepta pilot:true o un objeto vacío."}), 400
    if _read_state(cohort_id, run_id) is not None:
        return jsonify({"error": "El debate ya existe; consulta su estado."}), 409
    try:
        _scenario, _offer, _rows, _choices, digest = _source(cohort_id, run_id)
    except RepricingDiscussionError as exc:
        return jsonify({"error": str(exc)}), 409
    if not discussion_api._model_available():
        return jsonify({"error": "Conecta ChatGPT y selecciona un modelo."}), 503
    state = {
        "cohort_id": cohort_id,
        "run_id": run_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "synthetic": True,
        "repricing_sha256": digest,
        "model": Config.LLM_MODEL_NAME,
        "pilot": payload.get("pilot", False),
        "replies": [],
        "error": None,
    }
    return _start(cohort_id, run_id, state)


@repricing_discussion_bp.get("/<cohort_id>/repricing/<run_id>/discussion")
def discussion_status(cohort_id: str, run_id: str):
    state = _read_state(cohort_id, run_id)
    if state is None:
        return jsonify({"error": "Debate no encontrado."}), 404
    return jsonify({"success": True, "data": _public_state(state)})


@repricing_discussion_bp.post("/<cohort_id>/repricing/<run_id>/discussion/resume")
def resume_discussion(cohort_id: str, run_id: str):
    state = _read_state(cohort_id, run_id)
    if state is None:
        return jsonify({"error": "Debate no encontrado."}), 404
    if state.get("status") not in ("paused", "pilot_ready", "running"):
        return jsonify({"error": "El debate no está pausado."}), 409
    if state.get("status") == "running":
        with cohort_api._active_lock:
            if f"repricing-discussion:{run_id}" in cohort_api._active_jobs:
                return jsonify({"error": "El debate sigue en curso."}), 409
    try:
        _source(cohort_id, run_id)
    except RepricingDiscussionError as exc:
        return jsonify({"error": str(exc)}), 409
    if state.get("model") != Config.LLM_MODEL_NAME:
        return jsonify({"error": "Selecciona el mismo modelo antes de reanudar."}), 409
    if not discussion_api._model_available():
        return jsonify({"error": "Conecta ChatGPT y selecciona un modelo."}), 503
    if state["status"] == "pilot_ready":
        state["pilot"] = False
    return _start(cohort_id, run_id, state)


@repricing_discussion_bp.get("/<cohort_id>/repricing/<run_id>/discussion/pilot")
def discussion_pilot(cohort_id: str, run_id: str):
    state = _read_state(cohort_id, run_id)
    if state is None or state.get("status") != "pilot_ready":
        return jsonify({"error": "No hay una mesa piloto lista."}), 409
    try:
        _scenario, _offer, rows, choices, digest = _source(cohort_id, run_id)
        if digest != state.get("repricing_sha256"):
            raise RepricingDiscussionError("repricing changed")
        replies = validate_plan_discussion_batch(
            {"replies": state.get("replies", [])}, choices[:BATCH_SIZE])
    except RepricingDiscussionError:
        return jsonify({"error": "La mesa piloto es inválida."}), 500
    return jsonify({"success": True, "data": {
        "synthetic": True,
        "participants": rows[:BATCH_SIZE],
        "initial_choices": choices[:BATCH_SIZE],
        "replies": replies,
        "note": "Debate ficticio sobre precios, no compradores reales.",
    }})


@repricing_discussion_bp.get("/<cohort_id>/repricing/<run_id>/discussion/report")
def discussion_report(cohort_id: str, run_id: str):
    state = _read_state(cohort_id, run_id)
    if state is None or state.get("status") != "completed":
        return jsonify({"error": "Aún no hay 1.000 réplicas validadas."}), 409
    try:
        scenario, offer, rows, choices, digest = _source(cohort_id, run_id)
        if digest != state.get("repricing_sha256"):
            raise RepricingDiscussionError("repricing changed")
        report = json.loads((_directory(cohort_id, run_id) / "report.json").read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report.get("replies") != state.get("replies"):
            raise RepricingDiscussionError("report differs from checkpoint")

        def no_model_calls(_messages):
            raise RepricingDiscussionError("incomplete discussion")

        verified = generate_plan_discussion(
            scenario, offer, rows, choices, no_model_calls,
            existing_replies=report.get("replies", []),
        )
        for key in ("cohort_sha256", "repricing_sha256", "participants", "tables",
                    "initial_counts", "final_counts", "transitions", "transition_examples",
                    "synthetic", "representative_sample"):
            if report.get(key) != verified[key]:
                raise RepricingDiscussionError("report differs from verified debate")
        if (report.get("cohort_id") != cohort_id or report.get("run_id") != run_id
                or report.get("model") != state.get("model")):
            raise RepricingDiscussionError("report source differs")
    except (OSError, ValueError, KeyError, TypeError):
        return jsonify({"error": "El informe del debate está incompleto."}), 500
    return jsonify({"success": True, "data": report})
