"""Local, checkpointed fictional discussion of a completed 1,000-cohort."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading

from flask import Blueprint, jsonify, request

from . import cohort as cohort_api
from ..config import Config
from ..services.cohort_discussion import (
    DiscussionValidationError, cohort_hash, generate_discussion,
    validate_complete_cohort, validate_discussion_batch,
)
from ..services.cohort_survey import BATCH_SIZE, COHORT_SIZE
from ..services.plan_work_guard import retain_request_plan_work, release_plan_work
from ..utils.llm_client import LLMClient


discussion_bp = Blueprint("cohort_discussion", __name__)
logger = logging.getLogger(__name__)


class PilotPause(Exception):
    """The first complete table is ready for review."""


@discussion_bp.before_request
def require_loopback():
    return cohort_api.require_loopback()


def _directory(job_id: str) -> Path:
    return cohort_api._directory(job_id) / "discussion"


def _read_state(job_id: str) -> dict | None:
    try:
        path = _directory(job_id) / "state.json"
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


def _source(job_id: str) -> tuple[str, list[dict], str]:
    source = cohort_api._read_state(job_id)
    if source is None or source.get("status") != "completed":
        raise DiscussionValidationError("La cohorte debe tener 1.000 perfiles completos.")
    try:
        scenario = source["scenario"]
        rows = validate_complete_cohort(scenario, source.get("rows"))
    except (KeyError, TypeError, DiscussionValidationError) as exc:
        raise DiscussionValidationError("La cohorte guardada es inválida.") from exc
    return scenario, rows, cohort_hash(rows)


def _model_available() -> bool:
    if Config.CHATGPT_PLAN_MODE:
        from ..services.chatgpt_plan_auth import auth
        return bool(auth.status()["plan_usage_enabled"] and Config.LLM_MODEL_NAME)
    return bool(Config.LLM_API_KEY and Config.LLM_MODEL_NAME)


def _work(job_id: str, lease) -> None:
    path = _directory(job_id)
    state = _read_state(job_id)
    try:
        if state is None:
            raise DiscussionValidationError("discussion state disappeared")
        scenario, rows, digest = _source(job_id)
        if digest != state.get("cohort_sha256"):
            raise DiscussionValidationError("cohort changed after discussion started")
        llm = LLMClient()

        def ask(messages: list[dict]) -> dict:
            raw = llm.chat(messages, temperature=0.6, max_tokens=4096,
                           response_format={"type": "json_object"})
            return json.loads(raw)

        def checkpoint(replies: list[dict]) -> None:
            state["replies"] = replies
            cohort_api._write_json(path / "state.json", state)
            if state.get("pilot") and len(replies) == BATCH_SIZE:
                raise PilotPause()

        report = generate_discussion(
            scenario, rows, ask, existing_replies=state.get("replies", []),
            on_table=checkpoint,
        )
        report["cohort_id"] = job_id
        report["model"] = state["model"]
        cohort_api._write_json(path / "report.json", report)
        state["status"] = "completed"
        state["error"] = None
        cohort_api._write_json(path / "state.json", state)
    except PilotPause:
        state["status"] = "pilot_ready"
        state["error"] = None
        cohort_api._write_json(path / "state.json", state)
    except Exception as exc:
        # Provider errors and prompts can contain private account material.
        logger.warning("Discussion %s stopped after %s replies (%s)",
                       job_id, len(state.get("replies", [])) if state else 0,
                       type(exc).__name__)
        if state is not None:
            state["status"] = "paused"
            state["error"] = (
                "La siguiente mesa no pasó la validación o el modelo dejó de estar disponible. "
                "Las mesas completas siguen guardadas; se puede reanudar."
            )
            cohort_api._write_json(path / "state.json", state)
    finally:
        with cohort_api._active_lock:
            cohort_api._active_jobs.discard(f"discussion:{job_id}")
        release_plan_work(lease)


def _start(job_id: str, state: dict):
    with cohort_api._active_lock:
        if cohort_api._active_jobs:
            return jsonify({"error": "Ya hay un trabajo local generándose."}), 409
        cohort_api._active_jobs.add(f"discussion:{job_id}")
    lease = retain_request_plan_work()
    state["status"] = "running"
    state["error"] = None
    try:
        cohort_api._write_json(_directory(job_id) / "state.json", state)
        thread = threading.Thread(target=_work, args=(job_id, lease), daemon=True,
                                  name=f"mirofish-discussion-{job_id[:8]}")
        thread.start()
    except Exception:
        with cohort_api._active_lock:
            cohort_api._active_jobs.discard(f"discussion:{job_id}")
        release_plan_work(lease)
        return jsonify({"error": "No se pudo iniciar el debate local."}), 500
    return jsonify({"success": True, "data": _public_state(state)}), 202


@discussion_bp.post("/<job_id>/discussion/run")
def start_discussion(job_id: str):
    payload = request.get_json(silent=True) or {}
    if (not isinstance(payload, dict) or not set(payload).issubset({"pilot"})
            or type(payload.get("pilot", False)) is not bool):
        return jsonify({"error": "Solo se acepta pilot:true o un objeto vacío."}), 400
    if _read_state(job_id) is not None:
        return jsonify({"error": "El debate ya existe; consulta su estado."}), 409
    try:
        _scenario, _rows, digest = _source(job_id)
    except DiscussionValidationError as exc:
        return jsonify({"error": str(exc)}), 409
    if not _model_available():
        return jsonify({"error": "Conecta ChatGPT y selecciona un modelo."}), 503
    state = {
        "cohort_id": job_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "synthetic": True,
        "cohort_sha256": digest,
        "model": Config.LLM_MODEL_NAME,
        "pilot": payload.get("pilot", False),
        "replies": [],
        "error": None,
    }
    return _start(job_id, state)


@discussion_bp.get("/<job_id>/discussion")
def discussion_status(job_id: str):
    state = _read_state(job_id)
    if state is None:
        return jsonify({"error": "Debate no encontrado."}), 404
    return jsonify({"success": True, "data": _public_state(state)})


@discussion_bp.post("/<job_id>/discussion/resume")
def resume_discussion(job_id: str):
    state = _read_state(job_id)
    if state is None:
        return jsonify({"error": "Debate no encontrado."}), 404
    if state.get("status") not in ("paused", "pilot_ready", "running"):
        return jsonify({"error": "El debate no está pausado."}), 409
    if state.get("status") == "running":
        # A process restart can leave a checkpoint marked running even though
        # its worker no longer exists. A live worker must never be duplicated.
        with cohort_api._active_lock:
            if f"discussion:{job_id}" in cohort_api._active_jobs:
                return jsonify({"error": "El debate sigue en curso."}), 409
    try:
        _scenario, _rows, digest = _source(job_id)
    except DiscussionValidationError as exc:
        return jsonify({"error": str(exc)}), 409
    if digest != state.get("cohort_sha256"):
        return jsonify({"error": "La cohorte cambió tras iniciar el debate."}), 409
    if state.get("model") != Config.LLM_MODEL_NAME:
        return jsonify({"error": "Selecciona el mismo modelo antes de reanudar."}), 409
    if not _model_available():
        return jsonify({"error": "Conecta ChatGPT y selecciona un modelo."}), 503
    if state["status"] == "pilot_ready":
        state["pilot"] = False
    return _start(job_id, state)


@discussion_bp.get("/<job_id>/discussion/pilot")
def discussion_pilot(job_id: str):
    state = _read_state(job_id)
    if state is None or state.get("status") != "pilot_ready":
        return jsonify({"error": "No hay una mesa piloto lista."}), 409
    try:
        _scenario, rows, digest = _source(job_id)
        if digest != state.get("cohort_sha256"):
            raise DiscussionValidationError("cohort changed")
        replies = validate_discussion_batch(
            {"replies": state.get("replies", [])}, rows[:BATCH_SIZE])
    except DiscussionValidationError:
        return jsonify({"error": "La mesa piloto guardada es inválida."}), 500
    return jsonify({"success": True, "data": {
        "synthetic": True,
        "participants": rows[:BATCH_SIZE],
        "replies": replies,
        "note": "Intervenciones ficticias para revisar formato, no compradores reales.",
    }})


@discussion_bp.get("/<job_id>/discussion/report")
def discussion_report(job_id: str):
    state = _read_state(job_id)
    if state is None or state.get("status") != "completed":
        return jsonify({"error": "Aún no hay 1.000 réplicas validadas."}), 409
    try:
        scenario, rows, digest = _source(job_id)
        report = json.loads((_directory(job_id) / "report.json").read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report.get("replies") != state.get("replies"):
            raise DiscussionValidationError("report differs from checkpoint")
        if digest != state.get("cohort_sha256"):
            raise DiscussionValidationError("cohort changed")

        def no_model_calls(_messages):
            raise DiscussionValidationError("incomplete discussion")

        verified = generate_discussion(
            scenario, rows, no_model_calls,
            existing_replies=report.get("replies", []),
        )
        for key in ("cohort_sha256", "participants", "tables", "initial_counts",
                    "initial_reason_counts", "final_counts", "changes",
                    "transition_examples", "synthetic", "representative_sample"):
            if report.get(key) != verified[key]:
                raise DiscussionValidationError("report differs from verified data")
        if report.get("cohort_id") != job_id or report.get("model") != state.get("model"):
            raise DiscussionValidationError("report source differs")
    except (OSError, ValueError, KeyError, TypeError):
        return jsonify({"error": "El informe del debate está incompleto."}), 500
    return jsonify({"success": True, "data": report})
