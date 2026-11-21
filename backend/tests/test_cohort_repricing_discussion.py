"""All 1,000 fixed profiles must actually speak in the revised offer debate."""

from __future__ import annotations

from hashlib import sha256
import json
import time

import pytest
from flask import Flask

from app.api import cohort_repricing_discussion as api
from app.config import Config
from app.services.cohort_discussion import cohort_hash
from app.services.cohort_repricing_discussion import (
    RepricingDiscussionError, choices_hash, generate_plan_discussion,
    plan_discussion_prompt, validate_plan_discussion_batch,
)


SOURCE_SCENARIO = "Escenario original de los mismos negocios ficticios. " * 3
OFFER_CONTEXT = (
    "Oferta revisada de un servicio ficticio: anual de 1.200 unidades totales "
    "por doce meses, con impuestos aplicables incluidos y sin renovación automática. "
    "Mensual de 139 unidades totales por mes, con impuestos aplicables incluidos, "
    "sin permanencia y con renovación manual. La oferta no promete facturación integrada. "
    "No se ha comprobado la restauración externa ni abierto cobros."
)
COHORT_ID = "b" * 32
RUN_ID = "c" * 32


def cohort_rows():
    return [{
        "id": f"T{number:04d}",
        "workers": 1 + number % 20,
        "current_process": "papel" if number % 2 else "software",
        "main_need": "coordinar órdenes",
        "decision": "si" if number % 2 else "no",
        "reason_code": "funciones" if number % 2 else "precio",
        "reason": f"Motivo histórico del perfil {number}",
    } for number in range(1, 1001)]


def repricing_choices():
    return [{
        "id": f"T{number:04d}",
        "annual_only_decision": "si" if number % 2 else "no",
        "annual_only_reason": "Comparación anual con precio total claro.",
        "with_monthly_choice": (
            "annual" if number % 4 == 1 else
            "monthly" if number % 4 == 2 else "none"
        ),
        "with_monthly_reason": f"Elección inicial de plan del perfil {number}.",
    } for number in range(1, 1001)]


def fake_debate(messages):
    participants = json.loads(messages[-1]["content"].split("Participantes:\n", 1)[1])
    replies = []
    for row in participants:
        number = int(row["id"][1:])
        choice = row["initial_choice"]
        if number % 10 == 0:
            choice = "monthly" if choice == "none" else "none"
        replies.append({
            "id": row["id"],
            "responds_to": row["responds_to"],
            "reply": f"{row['responds_to']}, entiendo tu argumento; revisaré el precio y el proceso.",
            "final_choice": choice,
            "final_reason": "Postura final tras escuchar al otro perfil ficticio.",
        })
    return {"replies": replies}


def test_1000_original_ids_reply_once_in_40_complete_tables():
    calls = []

    def ask(messages):
        calls.append(messages)
        return fake_debate(messages)

    report = generate_plan_discussion(
        SOURCE_SCENARIO, OFFER_CONTEXT, cohort_rows(), repricing_choices(), ask,
    )
    assert len(calls) == 40
    assert "se pudieran contratar hoy" in calls[0][0]["content"]
    assert "no es por sí solo motivo" in calls[0][0]["content"]
    assert report["participants"] == 1000
    assert report["tables"] == 40
    assert len(report["replies"]) == 1000
    assert len({row["id"] for row in report["replies"]}) == 1000
    assert report["initial_counts"] == {"annual": 250, "monthly": 250, "none": 500}
    assert report["final_counts"] == {"annual": 250, "monthly": 250, "none": 500}
    assert report["transitions"]["monthly_a_none"] == 50
    assert report["transitions"]["none_a_monthly"] == 50
    assert sum(report["transitions"].values()) == 1000
    assert report["synthetic"] is True
    assert report["representative_sample"] is False


def test_invalid_target_duplicate_or_partial_table_cannot_be_counted():
    rows = cohort_rows()[:25]
    choices = repricing_choices()[:25]
    prompt = plan_discussion_prompt(OFFER_CONTEXT, rows, choices, 1)
    good = fake_debate(prompt)
    assert len(validate_plan_discussion_batch(good, choices)) == 25

    wrong_target = {"replies": [dict(item) for item in good["replies"]]}
    wrong_target["replies"][0]["responds_to"] = "T0025"
    with pytest.raises(RepricingDiscussionError):
        validate_plan_discussion_batch(wrong_target, choices)

    duplicate = {"replies": [dict(item) for item in good["replies"]]}
    duplicate["replies"][1]["id"] = duplicate["replies"][0]["id"]
    with pytest.raises(RepricingDiscussionError):
        validate_plan_discussion_batch(duplicate, choices)

    with pytest.raises(RepricingDiscussionError):
        validate_plan_discussion_batch({"replies": good["replies"][:24]}, choices)
    with pytest.raises(RepricingDiscussionError):
        generate_plan_discussion(
            SOURCE_SCENARIO, OFFER_CONTEXT, cohort_rows(), repricing_choices(),
            lambda messages: fake_debate(messages),
            existing_replies=good["replies"][:24],
        )


def test_loopback_api_pilot_resume_and_verified_complete_report(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "fake-model")
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-key")
    monkeypatch.setattr(Config, "CHATGPT_PLAN_MODE", False)
    cohort_dir = tmp_path / "cohorts" / COHORT_ID
    cohort_dir.mkdir(parents=True)
    rows = cohort_rows()
    (cohort_dir / "state.json").write_text(json.dumps({
        "id": COHORT_ID, "status": "completed",
        "scenario": SOURCE_SCENARIO, "rows": rows,
    }), encoding="utf-8")
    choices = repricing_choices()
    offer = OFFER_CONTEXT.strip()
    repricing_dir = cohort_dir / "repricing" / RUN_ID
    repricing_dir.mkdir(parents=True)
    (repricing_dir / "state.json").write_text(json.dumps({
        "cohort_id": COHORT_ID, "run_id": RUN_ID, "status": "completed",
        "cohort_sha256": cohort_hash(rows),
        "offer_context": offer,
        "offer_sha256": sha256(offer.encode("utf-8")).hexdigest(),
        "model": "fake-model", "choices": choices,
    }), encoding="utf-8")
    calls = 0

    class FakeLLM:
        def chat(self, messages, **_kwargs):
            nonlocal calls
            calls += 1
            return json.dumps(fake_debate(messages))

    monkeypatch.setattr(api, "LLMClient", FakeLLM)
    app = Flask(__name__)
    app.register_blueprint(api.repricing_discussion_bp, url_prefix="/api/cohort")
    client = app.test_client()
    url = f"/api/cohort/{COHORT_ID}/repricing/{RUN_ID}/discussion"
    assert client.post(url + "/run", json={"pilot": True},
                       environ_base={"REMOTE_ADDR": "198.51.100.1"}).status_code == 403
    assert client.post(url + "/run", json={"pilot": True},
                       headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post(url + "/run", json={"pilot": True}).status_code == 202
    for _ in range(200):
        status = client.get(url).json["data"]
        assert "replies" not in status
        if status["status"] == "pilot_ready":
            break
        time.sleep(0.01)
    assert status["status"] == "pilot_ready"
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
    assert len(report.json["data"]["replies"]) == 1000
    assert client.post(url + "/resume").status_code == 409
