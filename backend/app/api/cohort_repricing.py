"""Loopback-only, checkpointed A/B offer reevaluation of an existing cohort."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
import threading
from uuid import uuid4

from flask import Blueprint, jsonify, request

from . import cohort as cohort_api
from . import cohort_discussion as discussion_api
from ..config import Config
from ..services.cohort_repricing import (
    RepricingValidationError, generate_repricing, validate_offer_context,
    validate_repricing_batch,
)
from ..services.cohort_survey import BATCH_SIZE, COHORT_SIZE
from ..services.plan_work_guard import retain_request_plan_work, release_plan_work
from ..utils.llm_client import LLMClient


repricing_bp = Blueprint("cohort_repricing", __name__)
logger = logging.getLogger(__name__)


class PilotPause(Exception):
    """One complete batch is ready for format review."""


@repricing_bp.before_request
def require_loopback():
    return cohort_api.require_loopback()


def _directory(cohort_id: str, run_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("invalid repricing ID")
    return cohort_api._directory(cohort_id) / "repricing" / run_id


def _read_state(cohort_id: str, run_id: str) -> dict | None:
    try:
        path = _directory(cohort_id, run_id) / "state.json"
    except ValueError:
        return None
    with cohort_api._storage_lock:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return value if isinstance(value, dict) else None


def _public_state(state: dict) -> dict:
    return {
        "cohort_id": state["cohort_id"],
        "run_id": state["run_id"],
        "status": state["status"],
        "synthetic": True,
        "participants_target": COHORT_SIZE,
        "choices_completed": len(state.get("choices", [])),
        "batches_completed": len(state.get("choices", [])) // BATCH_SIZE,
        "batches_target": COHORT_SIZE // BATCH_SIZE,
        "pilot": bool(state.get("pilot")),
        "model": state.get("model", ""),
        "offer_sha256": state.get("offer_sha256", ""),
        "error": state.get("error"),
    }


def _source(cohort_id: str, state: dict) -> tuple[str, list[dict]]:
    try:
        source_scenario, rows, digest = discussion_api._source(cohort_id)
    except ValueError as exc:
        raise RepricingValidationError("La cohorte debe tener 1.000 perfiles válidos.") from exc
    if digest != state.get("cohort_sha256"):
        raise RepricingValidationError("La cohorte original cambió.")
    offer = validate_offer_context(state.get("offer_context"))
    if sha256(offer.encode("utf-8")).hexdigest() != state.get("offer_sha256"):
        raise RepricingValidationError("La oferta guardada cambió.")
    return source_scenario, rows


def _work(cohort_id: str, run_id: str, lease) -> None:
    directory = _directory(cohort_id, run_id)
    state = _read_state(cohort_id, run_id)
    try:
        if state is None:
            raise RepricingValidationError("repricing state disappeared")
        source_scenario, rows = _source(cohort_id, state)
        llm = LLMClient()

        def ask(messages: list[dict]) -> dict:
            raw = llm.chat(messages, temperature=0.6, max_tokens=4096,
                           response_format={"type": "json_object"})
            return json.loads(raw)

        def checkpoint(choices: list[dict]) -> None:
            state["choices"] = choices
            cohort_api._write_json(directory / "state.json", state)
            if state.get("pilot") and len(choices) == BATCH_SIZE:
                raise PilotPause()

        report = generate_repricing(
            source_scenario, state["offer_context"], rows, ask,
            existing_choices=state.get("choices", []), on_batch=checkpoint,
        )
        report.update({
            "cohort_id": cohort_id,
            "run_id": run_id,
            "offer_context": state["offer_context"],
            "offer_sha256": state["offer_sha256"],
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
        # Provider errors can include private account material; log only type.
        logger.warning("Repricing %s stopped after %s rows (%s)",
                       run_id, len(state.get("choices", [])) if state else 0,
                       type(exc).__name__)
        if state is not None:
            state["status"] = "paused"
            state["error"] = (
                "El lote siguiente no pasó la validación o el modelo dejó de estar disponible. "
                "Se conservan solo los lotes completos; se puede reanudar."
            )
            cohort_api._write_json(directory / "state.json", state)
    finally:
        with cohort_api._active_lock:
            cohort_api._active_jobs.discard(f"repricing:{run_id}")
        release_plan_work(lease)


def _start(cohort_id: str, run_id: str, state: dict):
    with cohort_api._active_lock:
        if cohort_api._active_jobs:
            return jsonify({"error": "Ya hay un trabajo local generándose."}), 409
        cohort_api._active_jobs.add(f"repricing:{run_id}")
    lease = retain_request_plan_work()
    state["status"] = "running"
    state["error"] = None
    try:
        cohort_api._write_json(_directory(cohort_id, run_id) / "state.json", state)
        thread = threading.Thread(target=_work, args=(cohort_id, run_id, lease),
                                  daemon=True, name=f"mirofish-repricing-{run_id[:8]}")
        thread.start()
    except Exception:
        with cohort_api._active_lock:
            cohort_api._active_jobs.discard(f"repricing:{run_id}")
        release_plan_work(lease)
        return jsonify({"error": "No se pudo iniciar el ensayo local."}), 500
    return jsonify({"success": True, "data": _public_state(state)}), 202


@repricing_bp.post("/<cohort_id>/repricing/run")
def start_repricing(cohort_id: str):
    payload = request.get_json(silent=True)
    if (not isinstance(payload, dict)
            or set(payload) not in ({"offer_context"}, {"offer_context", "pilot"})
            or type(payload.get("pilot", False)) is not bool):
        return jsonify({"error": "Envía offer_context y, si quieres, pilot:true."}), 400
    try:
        offer = validate_offer_context(payload["offer_context"])
        _source_scenario, rows, digest = discussion_api._source(cohort_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not discussion_api._model_available():
        return jsonify({"error": "Conecta ChatGPT y selecciona un modelo."}), 503
    run_id = uuid4().hex
    state = {
        "cohort_id": cohort_id,
        "run_id": run_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "synthetic": True,
        "cohort_sha256": digest,
        "offer_context": offer,
        "offer_sha256": sha256(offer.encode("utf-8")).hexdigest(),
        "model": Config.LLM_MODEL_NAME,
        "pilot": payload.get("pilot", False),
        "choices": [],
        "error": None,
    }
    return _start(cohort_id, run_id, state)


@repricing_bp.get("/<cohort_id>/repricing/<run_id>")
def repricing_status(cohort_id: str, run_id: str):
    state = _read_state(cohort_id, run_id)
    if state is None:
        return jsonify({"error": "Ensayo de precios no encontrado."}), 404
    return jsonify({"success": True, "data": _public_state(state)})


@repricing_bp.post("/<cohort_id>/repricing/<run_id>/resume")
def resume_repricing(cohort_id: str, run_id: str):
    state = _read_state(cohort_id, run_id)
    if state is None:
        return jsonify({"error": "Ensayo de precios no encontrado."}), 404
    if state.get("status") not in ("paused", "pilot_ready", "running"):
        return jsonify({"error": "El ensayo no está pausado."}), 409
    if state.get("status") == "running":
        with cohort_api._active_lock:
            if f"repricing:{run_id}" in cohort_api._active_jobs:
                return jsonify({"error": "El ensayo sigue en curso."}), 409
    try:
        _source(cohort_id, state)
    except RepricingValidationError as exc:
        return jsonify({"error": str(exc)}), 409
    if state.get("model") != Config.LLM_MODEL_NAME:
        return jsonify({"error": "Selecciona el mismo modelo antes de reanudar."}), 409
    if not discussion_api._model_available():
        return jsonify({"error": "Conecta ChatGPT y selecciona un modelo."}), 503
    if state["status"] == "pilot_ready":
        state["pilot"] = False
    return _start(cohort_id, run_id, state)


@repricing_bp.get("/<cohort_id>/repricing/<run_id>/pilot")
def repricing_pilot(cohort_id: str, run_id: str):
    state = _read_state(cohort_id, run_id)
    if state is None or state.get("status") != "pilot_ready":
        return jsonify({"error": "No hay lote piloto listo."}), 409
    try:
        _scenario, rows = _source(cohort_id, state)
        choices = validate_repricing_batch(
            {"choices": state.get("choices", [])}, rows[:BATCH_SIZE])
    except RepricingValidationError:
        return jsonify({"error": "El lote piloto es inválido."}), 500
    return jsonify({"success": True, "data": {
        "synthetic": True,
        "participants": rows[:BATCH_SIZE],
        "choices": choices,
        "note": "Comparación ficticia A/B, no ventas reales.",
    }})


@repricing_bp.get("/<cohort_id>/repricing/<run_id>/report")
def repricing_report(cohort_id: str, run_id: str):
    state = _read_state(cohort_id, run_id)
    if state is None or state.get("status") != "completed":
        return jsonify({"error": "Aún no hay 1.000 decisiones emparejadas válidas."}), 409
    try:
        scenario, rows = _source(cohort_id, state)
        report = json.loads((_directory(cohort_id, run_id) / "report.json").read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report.get("choices") != state.get("choices"):
            raise RepricingValidationError("report differs from checkpoint")

        def no_model_calls(_messages):
            raise RepricingValidationError("incomplete repricing")

        verified = generate_repricing(
            scenario, state["offer_context"], rows, no_model_calls,
            existing_choices=report.get("choices", []),
        )
        for key in ("cohort_sha256", "participants", "batches", "previous_counts",
                    "annual_only_counts", "with_monthly_counts", "transitions", "transition_examples",
                    "synthetic", "representative_sample"):
            if report.get(key) != verified[key]:
                raise RepricingValidationError("report differs from verified choices")
        if (report.get("cohort_id") != cohort_id or report.get("run_id") != run_id
                or report.get("offer_context") != state.get("offer_context")
                or report.get("offer_sha256") != state.get("offer_sha256")
                or report.get("model") != state.get("model")):
            raise RepricingValidationError("report source differs")
    except (OSError, ValueError, KeyError, TypeError):
        return jsonify({"error": "El informe de precios está incompleto."}), 500
    return jsonify({"success": True, "data": report})
