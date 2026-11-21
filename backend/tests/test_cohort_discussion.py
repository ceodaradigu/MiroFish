"""Exactly the original 1,000 fictional workshops must participate in debate."""

from __future__ import annotations

import json
import time

import pytest
from flask import Flask

from app.api import cohort_discussion as discussion_api
from app.config import Config
from app.services.cohort_discussion import (
    DiscussionValidationError, generate_discussion, reply_targets,
    validate_discussion_batch,
)
from app.services.cohort_survey import COHORT_SIZE


SCENARIO = "Oferta hipotética de Tallerna a talleres mexicanos por 6900 MXN anuales. " * 3
JOB_ID = "a" * 32


def cohort_rows():
    return [{
        "id": f"T{number:04d}",
        "workers": 1 + number % 20,
        "current_process": "papel" if number % 2 else "software",
        "main_need": "coordinar órdenes",
        "decision": "si" if number % 2 else "no",
        "reason_code": "funciones" if number % 2 else "precio",
        "reason": f"Motivo inicial del taller {number} sobre el flujo de trabajo",
    } for number in range(1, COHORT_SIZE + 1)]


def fake_debate(messages):
    participants = json.loads(messages[-1]["content"].split("Participantes:\n", 1)[1])
    return {"replies": [{
        "id": row["id"],
        "responds_to": row["responds_to"],
        "reply": f"Respondo a {row['responds_to']}: conviene revisar la orden y el precio.",
        "final_decision": (
            "no" if row["initial_decision"] == "si" else "si"
        ) if int(row["id"][1:]) % 10 == 0 else row["initial_decision"],
        "final_reason": "Mantengo o ajusto mi postura tras oír el argumento del otro taller.",
    } for row in participants]}


def test_exactly_1000_model_replies_in_40_tables_with_changed_counts():
    calls = []

    def ask(messages):
        calls.append(messages)
        return fake_debate(messages)

    report = generate_discussion(SCENARIO, cohort_rows(), ask)
    assert len(calls) == 40
    assert report["participants"] == 1000
    assert report["tables"] == 40
    assert report["initial_counts"] == {"si": 500, "no": 500}
    assert report["final_counts"] == {"si": 600, "no": 400}
    assert report["changes"]["no_a_si"] == 100
    assert len(report["replies"]) == 1000
    assert len({item["id"] for item in report["replies"]}) == 1000
    assert sum(report["initial_reason_counts"]["no"].values()) == 500
    assert report["transition_examples"]["no_a_si"]
    assert report["synthetic"] is True
    assert report["representative_sample"] is False


def test_wrong_target_duplicate_or_incomplete_table_fails():
    group = cohort_rows()[:25]
    payload = fake_debate([{"content": "Participantes:\n" + json.dumps([
        {"id": row["id"], "initial_decision": row["decision"],
         "responds_to": reply_targets(group)[row["id"]]}
        for row in group
    ])}])
    payload["replies"][0]["responds_to"] = "T0025"
    with pytest.raises(DiscussionValidationError, match="target"):
        validate_discussion_batch(payload, group)
    payload = fake_debate([{"content": "Participantes:\n" + json.dumps([
        {"id": row["id"], "initial_decision": row["decision"],
         "responds_to": reply_targets(group)[row["id"]]}
        for row in group
    ])}])
    payload["replies"][1]["id"] = payload["replies"][0]["id"]
    with pytest.raises(DiscussionValidationError, match="duplicate"):
        validate_discussion_batch(payload, group)
    with pytest.raises(DiscussionValidationError, match="25 replies"):
        validate_discussion_batch({"replies": payload["replies"][:24]}, group)
    payload = fake_debate([{"content": "Participantes:\n" + json.dumps([
        {"id": row["id"], "initial_decision": row["decision"],
         "responds_to": reply_targets(group)[row["id"]]}
        for row in group
    ])}])
    payload["replies"][0]["reply"] = "Estoy de acuerdo con la idea general, pero revisaría las órdenes."
    with pytest.raises(DiscussionValidationError, match="address"):
        validate_discussion_batch(payload, group)


def test_discussion_checkpoint_requires_complete_source_and_tables():
    rows = cohort_rows()
    group = rows[:25]
    first = validate_discussion_batch(fake_debate([{"content":
        "Participantes:\n" + json.dumps([
            {"id": row["id"], "initial_decision": row["decision"],
             "responds_to": reply_targets(group)[row["id"]]}
            for row in group
        ])}]), group)
    calls = []

    def ask(messages):
        calls.append(1)
        return fake_debate(messages)

    report = generate_discussion(SCENARIO, rows, ask, existing_replies=first)
    assert len(calls) == 39
    assert len(report["replies"]) == 1000
    with pytest.raises(DiscussionValidationError, match="complete tables"):
        generate_discussion(SCENARIO, rows, ask, existing_replies=first[:24])
    with pytest.raises(DiscussionValidationError, match="incomplete"):
        generate_discussion(SCENARIO, rows[:999], ask)


def test_local_discussion_pilot_resume_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "fake-model")
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-key")
    monkeypatch.setattr(Config, "CHATGPT_PLAN_MODE", False)
    cohort_dir = tmp_path / "cohorts" / JOB_ID
    cohort_dir.mkdir(parents=True)
    (cohort_dir / "state.json").write_text(json.dumps({
        "id": JOB_ID, "status": "completed", "scenario": SCENARIO,
        "rows": cohort_rows(),
    }), encoding="utf-8")
    calls = 0

    class FakeLLM:
        def chat(self, messages, **_kwargs):
            nonlocal calls
            calls += 1
            return json.dumps(fake_debate(messages))

    monkeypatch.setattr(discussion_api, "LLMClient", FakeLLM)
    app = Flask(__name__)
    app.register_blueprint(discussion_api.discussion_bp, url_prefix="/api/cohort")
    client = app.test_client()
    url = f"/api/cohort/{JOB_ID}/discussion"
    assert client.post(url + "/run", json={"pilot": True},
                       environ_base={"REMOTE_ADDR": "198.51.100.1"}).status_code == 403
    assert client.post(url + "/run", json={"pilot": True},
                       headers={"Origin": "https://evil.example"}).status_code == 403
    started = client.post(url + "/run", json={"pilot": True})
    assert started.status_code == 202
    for _ in range(200):
        status = client.get(url).json["data"]
        assert "replies" not in status
        if status["status"] == "pilot_ready":
            break
        time.sleep(0.01)
    assert status["status"] == "pilot_ready"
    assert status["replies_completed"] == 25
    assert calls == 1
    assert client.get(url + "/report").status_code == 409
    pilot = client.get(url + "/pilot")
    assert pilot.status_code == 200
    assert len(pilot.json["data"]["replies"]) == 25
    assert client.post(url + "/resume").status_code == 202
    for _ in range(300):
        status = client.get(url).json["data"]
        if status["status"] == "completed":
            break
        time.sleep(0.01)
    assert status["status"] == "completed"
    assert calls == 40
    report = client.get(url + "/report")
    assert report.status_code == 200
    assert report.json["data"]["participants"] == 1000
    assert report.json["data"]["final_counts"] == {"si": 600, "no": 400}
    assert client.post(url + "/resume").status_code == 409


def test_rejects_incomplete_cohort_and_pauses_invalid_table(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "fake-model")
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-key")
    monkeypatch.setattr(Config, "CHATGPT_PLAN_MODE", False)
    cohort_dir = tmp_path / "cohorts" / JOB_ID
    cohort_dir.mkdir(parents=True)
    source_path = cohort_dir / "state.json"
    source = {"id": JOB_ID, "status": "running", "scenario": SCENARIO,
              "rows": cohort_rows()[:975]}
    source_path.write_text(json.dumps(source), encoding="utf-8")
    calls = 0

    class SometimesBadLLM:
        def chat(self, messages, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                return json.dumps({"replies": []})
            return json.dumps(fake_debate(messages))

    monkeypatch.setattr(discussion_api, "LLMClient", SometimesBadLLM)
    app = Flask(__name__)
    app.register_blueprint(discussion_api.discussion_bp, url_prefix="/api/cohort")
    client = app.test_client()
    url = f"/api/cohort/{JOB_ID}/discussion"
    assert client.post(url + "/run", json={}).status_code == 409
    source["status"] = "completed"
    source["rows"] = cohort_rows()
    source_path.write_text(json.dumps(source), encoding="utf-8")
    assert client.post(url + "/run", json={}).status_code == 202
    for _ in range(200):
        status = client.get(url).json["data"]
        if status["status"] == "paused":
            break
        time.sleep(0.01)
    assert status["status"] == "paused"
    assert status["replies_completed"] == 25
    assert client.get(url + "/report").status_code == 409
    assert client.post(url + "/resume").status_code == 202
    for _ in range(300):
        status = client.get(url).json["data"]
        if status["status"] == "completed":
            break
        time.sleep(0.01)
    assert status["status"] == "completed"
    assert client.get(url + "/report").status_code == 200
