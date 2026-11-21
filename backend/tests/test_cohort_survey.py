"""The 1,000 synthetic answers must come from the model, never filler code."""

from __future__ import annotations

import json
import re
import time

import pytest
from flask import Flask

from app.services.cohort_survey import (
    COHORT_SIZE, BATCH_SIZE, CohortValidationError, generate_cohort,
    validate_batch,
)
from app.api import cohort as cohort_api
from app.config import Config


SCENARIO = "Propuesta hipotética de una licencia para talleres mexicanos. " * 3


def fake_model(messages):
    identifiers = re.findall(r"\bT\d{4}\b", messages[-1]["content"])
    return {"rows": [
        {
            "id": identifier,
            "workers": 1 + int(identifier[1:]) % 20,
            "current_process": "papel" if int(identifier[1:]) % 2 else "software",
            "main_need": "ordenar trabajos",
            "decision": "si" if int(identifier[1:]) % 2 else "no",
            "reason_code": "funciones" if int(identifier[1:]) % 2 else "precio",
            "reason": "Prueba generada por el doble del modelo",
        }
        for identifier in identifiers
    ]}


def test_exactly_1000_model_rows_and_40_calls():
    calls = []

    def ask(messages):
        calls.append(messages)
        return fake_model(messages)

    result = generate_cohort(SCENARIO, ask)
    assert len(calls) == COHORT_SIZE // BATCH_SIZE == 40
    assert result["fictional_workshops"] == COHORT_SIZE
    assert len(result["rows"]) == COHORT_SIZE
    assert result["counts"] == {"si": 500, "no": 500}
    assert result["unique_profile_patterns"] == 20  # IDs alone do not prove diverse profiles.
    assert result["synthetic"] is True
    assert result["representative_sample"] is False
    assert len({row["id"] for row in result["rows"]}) == COHORT_SIZE


def test_duplicate_or_missing_model_row_fails_without_filler():
    batch = fake_model([{"content": "T0001 T0002"}])
    batch["rows"][1]["id"] = "T0001"
    with pytest.raises(CohortValidationError, match="duplicate"):
        validate_batch(batch, ["T0001", "T0002"])
    with pytest.raises(CohortValidationError, match="wrong number"):
        validate_batch({"rows": [batch["rows"][0]]}, ["T0001", "T0002"])


def test_checkpoint_resumes_only_complete_validated_batches():
    first = fake_model([{"content": " ".join(f"T{n:04d}" for n in range(1, 26))}])["rows"]
    calls = []

    def ask(messages):
        calls.append(1)
        return fake_model(messages)

    result = generate_cohort(SCENARIO, ask, existing_rows=first)
    assert len(calls) == 39
    assert len(result["rows"]) == 1000
    with pytest.raises(CohortValidationError, match="complete batches"):
        generate_cohort(SCENARIO, ask, existing_rows=first[:24])


def test_local_endpoint_generates_exact_cohort_and_hides_partial_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "fake-model")
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-key")
    monkeypatch.setattr(Config, "CHATGPT_PLAN_MODE", False)

    class FakeLLM:
        def chat(self, messages, **_kwargs):
            return json.dumps(fake_model(messages))

    monkeypatch.setattr(cohort_api, "LLMClient", FakeLLM)
    app = Flask(__name__)
    app.register_blueprint(cohort_api.cohort_bp, url_prefix="/api/cohort")
    client = app.test_client()
    denied = client.post("/api/cohort/run", json={"scenario": SCENARIO},
                         environ_base={"REMOTE_ADDR": "198.51.100.2"})
    assert denied.status_code == 403
    cross_origin = client.post("/api/cohort/run", json={"scenario": SCENARIO},
                               headers={"Origin": "https://evil.example"})
    assert cross_origin.status_code == 403
    started = client.post("/api/cohort/run", json={"scenario": SCENARIO})
    assert started.status_code == 202
    job_id = started.json["data"]["id"]
    for _ in range(200):
        status = client.get(f"/api/cohort/{job_id}").json["data"]
        assert "rows" not in status and "scenario" not in status
        if status["status"] == "completed":
            break
        time.sleep(0.01)
    assert status["status"] == "completed"
    response = client.get(f"/api/cohort/{job_id}/report")
    assert response.status_code == 200
    assert response.json["data"]["counts"] == {"si": 500, "no": 500}
    assert len(response.json["data"]["rows"]) == 1000


def test_pilot_stops_after_25_and_resume_keeps_first_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "fake-model")
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-key")
    monkeypatch.setattr(Config, "CHATGPT_PLAN_MODE", False)
    calls = 0

    class FakeLLM:
        def chat(self, messages, **_kwargs):
            nonlocal calls
            calls += 1
            return json.dumps(fake_model(messages))

    monkeypatch.setattr(cohort_api, "LLMClient", FakeLLM)
    app = Flask(__name__)
    app.register_blueprint(cohort_api.cohort_bp, url_prefix="/api/cohort")
    client = app.test_client()
    started = client.post("/api/cohort/run", json={"scenario": SCENARIO, "pilot": True})
    assert started.status_code == 202
    job_id = started.json["data"]["id"]
    for _ in range(200):
        status = client.get(f"/api/cohort/{job_id}").json["data"]
        if status["status"] == "pilot_ready":
            break
        time.sleep(0.01)
    assert status["status"] == "pilot_ready"
    assert status["completed"] == 25
    assert status["pilot"] is True
    assert calls == 1
    assert client.get(f"/api/cohort/{job_id}/report").status_code == 409
    pilot_response = client.get(f"/api/cohort/{job_id}/pilot")
    assert pilot_response.status_code == 200
    assert pilot_response.json["data"]["count"] == 25
    assert pilot_response.json["data"]["synthetic"] is True
    state_path = tmp_path / "cohorts" / job_id / "state.json"
    first_rows = json.loads(state_path.read_text(encoding="utf-8"))["rows"]
    assert len(first_rows) == 25
    assert pilot_response.json["data"]["rows"] == first_rows

    resumed = client.post(f"/api/cohort/{job_id}/resume")
    assert resumed.status_code == 202
    for _ in range(200):
        status = client.get(f"/api/cohort/{job_id}").json["data"]
        if status["status"] == "completed":
            break
        time.sleep(0.01)
    assert status["status"] == "completed"
    assert calls == 40  # 1 pilot call + exactly 39 remaining calls.
    report = client.get(f"/api/cohort/{job_id}/report").json["data"]
    assert report["rows"][:25] == first_rows
    assert len(report["rows"]) == 1000


def test_local_endpoint_pauses_invalid_batch_without_claiming_1000(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "fake-model")
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-key")
    monkeypatch.setattr(Config, "CHATGPT_PLAN_MODE", False)
    calls = 0

    class BadLLM:
        def chat(self, messages, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                return json.dumps({"rows": []})
            return json.dumps(fake_model(messages))

    monkeypatch.setattr(cohort_api, "LLMClient", BadLLM)
    app = Flask(__name__)
    app.register_blueprint(cohort_api.cohort_bp, url_prefix="/api/cohort")
    client = app.test_client()
    started = client.post("/api/cohort/run", json={"scenario": SCENARIO})
    assert started.status_code == 202
    job_id = started.json["data"]["id"]
    for _ in range(200):
        status = client.get(f"/api/cohort/{job_id}").json["data"]
        if status["status"] == "paused":
            break
        time.sleep(0.01)
    assert status["status"] == "paused"
    assert status["completed"] == 25
    assert client.get(f"/api/cohort/{job_id}/report").status_code == 409
    assert not (tmp_path / "cohorts" / job_id / "report.json").exists()
    resumed = client.post(f"/api/cohort/{job_id}/resume")
    assert resumed.status_code == 202
    for _ in range(200):
        status = client.get(f"/api/cohort/{job_id}").json["data"]
        if status["status"] == "completed":
            break
        time.sleep(0.01)
    assert status["status"] == "completed"
    assert client.get(f"/api/cohort/{job_id}/report").json["data"]["fictional_workshops"] == 1000
