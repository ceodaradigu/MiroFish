"""Local, checkpointed MiroFish cohort generation for synthetic purchase intent."""

from __future__ import annotations

from datetime import datetime, timezone
import ipaddress
import json
import logging
import os
from pathlib import Path
import re
import threading
from uuid import uuid4

from flask import Blueprint, jsonify, request

from ..config import Config
from ..services.cohort_survey import (
    BATCH_SIZE, COHORT_SIZE, CohortValidationError, generate_cohort,
    row_id, validate_batch, validate_scenario,
)
from ..services.plan_work_guard import retain_request_plan_work, release_plan_work
from ..utils.llm_client import LLMClient


cohort_bp = Blueprint("cohort", __name__)
logger = logging.getLogger(__name__)
_active_lock = threading.Lock()
_active_jobs: set[str] = set()
_storage_lock = threading.RLock()


class PilotPause(Exception):
    """The first validated batch is ready for human review."""


def _directory(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("invalid cohort ID")
    return Path(Config.UPLOAD_FOLDER) / "cohorts" / job_id


def _write_json(path: Path, value: dict) -> None:
    with _storage_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)


def _read_state(job_id: str) -> dict | None:
    try:
        path = _directory(job_id) / "state.json"
    except ValueError:
        return None
    with _storage_lock:
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return value if isinstance(value, dict) else None


def _public_state(state: dict) -> dict:
    """Never return the private scenario or model-generated rows during work."""
    return {
        "id": state["id"],
        "status": state["status"],
        "synthetic": True,
        "target": COHORT_SIZE,
        "completed": len(state.get("rows", [])),
        "batches_total": COHORT_SIZE // BATCH_SIZE,
        "batches_completed": len(state.get("rows", [])) // BATCH_SIZE,
        "pilot": bool(state.get("pilot")),
        "model": state.get("model", ""),
        "error": state.get("error"),
    }


@cohort_bp.before_request
def require_loopback():
    """Prevent a new model-consuming endpoint from being remotely callable."""
    try:
        local = ipaddress.ip_address(request.remote_addr or "").is_loopback
    except ValueError:
        local = False
    if not local:
        return jsonify({"error": "Local access only"}), 403
    origin = request.headers.get("Origin")
    if origin not in (None, "http://127.0.0.1:3000"):
        return jsonify({"error": "Origin not allowed"}), 403
    if request.headers.get("Sec-Fetch-Site") == "cross-site":
        return jsonify({"error": "Cross-site requests are not allowed"}), 403
    return None


def _work(job_id: str, lease) -> None:
    directory = _directory(job_id)
    state_path = directory / "state.json"
    state = _read_state(job_id)
    try:
        if state is None:
            raise CohortValidationError("cohort state disappeared")
        llm = LLMClient()

        def ask(messages: list[dict[str, str]]) -> dict:
            # One model request per batch. chat_json can negotiate JSON mode
            # with a second request; this route fails instead of doing that.
            raw = llm.chat(messages, temperature=0.7, max_tokens=4096,
                           response_format={"type": "json_object"})
            return json.loads(raw)

        def checkpoint(rows: list[dict]) -> None:
            state["rows"] = rows
            _write_json(state_path, state)
            if state.get("pilot") and len(rows) == BATCH_SIZE:
                raise PilotPause()

        report = generate_cohort(
            state["scenario"], ask,
            existing_rows=state.get("rows", []), on_batch=checkpoint,
        )
        report["model"] = state.get("model", "")
        report["scenario_sha256"] = state["scenario_sha256"]
        report["id"] = job_id
        _write_json(directory / "report.json", report)
        state["status"] = "completed"
        state["error"] = None
        _write_json(state_path, state)
    except PilotPause:
        state["status"] = "pilot_ready"
        state["error"] = None
        _write_json(state_path, state)
    except Exception as exc:
        # Never log provider errors or prompts: they can contain account data.
        logger.warning("Cohort %s stopped after %s rows (%s)",
                       job_id, len(state.get("rows", [])) if state else 0,
                       type(exc).__name__)
        if state is not None:
            state["status"] = "paused"
            state["error"] = (
                "El lote siguiente no pasó la validación o el modelo dejó de estar disponible. "
                "Se conservan solo los lotes completos; se puede reanudar."
            )
            _write_json(state_path, state)
    finally:
        with _active_lock:
            _active_jobs.discard(job_id)
        release_plan_work(lease)


def _start(job_id: str, state: dict):
    with _active_lock:
        if _active_jobs:
            return jsonify({"error": "Ya hay una cohorte local generándose."}), 409
        _active_jobs.add(job_id)
    lease = retain_request_plan_work()
    state["status"] = "running"
    state["error"] = None
    try:
        _write_json(_directory(job_id) / "state.json", state)
        thread = threading.Thread(target=_work, args=(job_id, lease), daemon=True,
                                  name=f"mirofish-cohort-{job_id[:8]}")
        thread.start()
    except Exception:
        with _active_lock:
            _active_jobs.discard(job_id)
        release_plan_work(lease)
        return jsonify({"error": "No se pudo iniciar el ensayo local."}), 500
    return jsonify({"success": True, "data": _public_state(state)}), 202


@cohort_bp.post("/run")
def start_cohort():
    data = request.get_json(silent=True)
    if (not isinstance(data, dict) or "scenario" not in data
            or not set(data).issubset({"scenario", "pilot"})
            or type(data.get("pilot", False)) is not bool):
        return jsonify({"error": "Envía el contexto y, si quieres, pilot:true."}), 400
    try:
        scenario = validate_scenario(data["scenario"])
    except CohortValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    if Config.CHATGPT_PLAN_MODE:
        from ..services.chatgpt_plan_auth import auth
        if not auth.status()["plan_usage_enabled"] or not Config.LLM_MODEL_NAME:
            return jsonify({"error": "Conecta ChatGPT y selecciona un modelo."}), 503
    elif not Config.LLM_API_KEY or not Config.LLM_MODEL_NAME:
        return jsonify({"error": "Configura el proveedor del modelo."}), 503
    import hashlib

    job_id = uuid4().hex
    state = {
        "id": job_id,
        "status": "queued",
        "synthetic": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "scenario_sha256": hashlib.sha256(scenario.encode("utf-8")).hexdigest(),
        "model": Config.LLM_MODEL_NAME,
        "pilot": data.get("pilot", False),
        "rows": [],
        "error": None,
    }
    return _start(job_id, state)


@cohort_bp.get("/<job_id>")
def cohort_status(job_id: str):
    state = _read_state(job_id)
    if state is None:
        return jsonify({"error": "Cohorte no encontrada."}), 404
    return jsonify({"success": True, "data": _public_state(state)})


@cohort_bp.post("/<job_id>/resume")
def resume_cohort(job_id: str):
    state = _read_state(job_id)
    if state is None:
        return jsonify({"error": "Cohorte no encontrada."}), 404
    if state.get("status") == "completed":
        return jsonify({"error": "La cohorte ya terminó."}), 409
    if state.get("model") != Config.LLM_MODEL_NAME:
        return jsonify({"error": "Selecciona el mismo modelo antes de reanudar."}), 409
    if state.get("status") == "pilot_ready":
        state["pilot"] = False
    return _start(job_id, state)


@cohort_bp.get("/<job_id>/pilot")
def cohort_pilot(job_id: str):
    state = _read_state(job_id)
    if state is None:
        return jsonify({"error": "Cohorte no encontrada."}), 404
    if state.get("status") != "pilot_ready":
        return jsonify({"error": "No hay un lote piloto listo para revisar."}), 409
    ids = [row_id(number) for number in range(1, BATCH_SIZE + 1)]
    try:
        rows = validate_batch({"rows": state.get("rows", [])}, ids)
    except CohortValidationError:
        return jsonify({"error": "El lote piloto guardado es inválido."}), 500
    return jsonify({"success": True, "data": {
        "synthetic": True,
        "representative_sample": False,
        "count": BATCH_SIZE,
        "rows": rows,
        "note": "Respuestas ficticias para revisar formato y diversidad; no son compradores reales.",
    }})


@cohort_bp.get("/<job_id>/report")
def cohort_report(job_id: str):
    state = _read_state(job_id)
    if state is None:
        return jsonify({"error": "Cohorte no encontrada."}), 404
    if state.get("status") != "completed":
        return jsonify({"error": "Aún no hay 1.000 respuestas validadas."}), 409
    try:
        report = json.loads((_directory(job_id) / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return jsonify({"error": "No se pudo leer el informe completo."}), 500
    def no_model_calls(_messages):
        raise CohortValidationError("incomplete stored report")

    try:
        verified = generate_cohort(state["scenario"], no_model_calls,
                                   existing_rows=report.get("rows", []))
    except (CohortValidationError, TypeError, KeyError):
        return jsonify({"error": "El informe guardado está incompleto."}), 500
    if (report.get("scenario_sha256") != state.get("scenario_sha256")
            or report.get("fictional_workshops") != COHORT_SIZE
            or report.get("synthetic") is not True
            or report.get("representative_sample") is not False
            or report.get("counts") != verified["counts"]
            or report.get("reason_counts") != verified["reason_counts"]
            or report.get("unique_profile_patterns") != verified["unique_profile_patterns"]):
        return jsonify({"error": "El informe guardado está incompleto."}), 500
    return jsonify({"success": True, "data": report})
