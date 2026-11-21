"""The corrected monthly offer is scored by the model for the SAME 1,000 IDs."""

from __future__ import annotations

import json
import time

import pytest
from flask import Flask

from app.api import cohort_repricing as repricing_api
from app.config import Config
from app.services.cohort_repricing import (
    RepricingValidationError, generate_repricing, validate_repricing_batch,
)
from app.services.cohort_survey import COHORT_SIZE


SOURCE_SCENARIO = "Oferta original hipotética para negocios ficticios. " * 3
OFFER_CONTEXT = (
    "Oferta hipotética corregida de un servicio ficticio. Plan anual: "
    "1.200 unidades totales por 12 meses, "
    "con impuestos aplicables incluidos, sin renovación automática. "
    "Plan mensual: 139 unidades totales por mes, con impuestos aplicables incluidos, "
    "renovación manual, sin permanencia. Cobros desactivados."
)


def cohort_rows():
    return [{
        "id": f"T{number:04d}",
        "workers": 1 + number % 20,
        "current_process": "papel" if number % 2 else "software",
        "main_need": "coordinar órdenes",
        "decision": "si" if number % 2 else "no",
        "reason_code": "funciones" if number % 2 else "precio",
        "reason": f"Motivo original del perfil {number}",
    } for number in range(1, COHORT_SIZE + 1)]


def fake_model(messages):
    participants = json.loads(messages[-1]["content"].split("Participantes:\n", 1)[1])
    return {"choices": [{
        "id": row["id"],
        "annual_only_decision": "si" if int(row["id"][1:]) % 2 else "no",
        "annual_only_reason": f"Evaluación corregida del anual para {row['id']}",
        "with_monthly_choice": (
            "annual" if int(row["id"][1:]) % 4 == 1 else
            "monthly" if int(row["id"][1:]) % 4 == 2 else "none"
        ),
        "with_monthly_reason": f"Evaluación del anual y mensual para {row['id']}",
    } for row in participants]}


def test_same_1000_model_decisions_and_paired_comparison():
    calls = []

    def ask(messages):
        calls.append(messages)
        return fake_model(messages)

    report = generate_repricing(SOURCE_SCENARIO, OFFER_CONTEXT, cohort_rows(), ask)
    assert len(calls) == 40
    assert report["participants"] == 1000
    assert report["previous_counts"] == {"si": 500, "no": 500}
    assert report["annual_only_counts"] == {"si": 500, "no": 500}
    assert report["with_monthly_counts"] == {
        "annual": 250, "monthly": 250, "none": 500,
    }
    assert report["transitions"] == {
        "si_a_annual": 250, "si_a_monthly": 0, "si_a_none": 250,
        "no_a_annual": 0, "no_a_monthly": 250, "no_a_none": 250,
    }
    assert len(report["choices"]) == 1000
    assert [choice["id"] for choice in report["choices"]] == [
        row["id"] for row in cohort_rows()
    ]
    assert report["synthetic"] is True
    assert report["representative_sample"] is False
    assert "139" in calls[0][-1]["content"]
    assert "1.200" in calls[0][-1]["content"]
    assert "supón que los planes descritos se pudieran contratar hoy" in calls[0][0]["content"]
    assert "desactivados no es" in calls[0][0]["content"]
    first_participants = json.loads(
        calls[0][-1]["content"].split("Participantes:\n", 1)[1]
    )
    assert len(first_participants) == 25
    assert {"id", "workers", "current_process", "main_need"}.issubset(
        first_participants[0]
    )
    assert "decision" not in first_participants[0]
    assert "reason" not in first_participants[0]


def test_missing_duplicate_or_unknown_id_never_counts_as_complete_batch():
    group = cohort_rows()[:25]
    good = {"choices": [{
        "id": row["id"],
        "annual_only_decision": "no",
        "annual_only_reason": "El desembolso anual me pesa.",
        "with_monthly_choice": "monthly",
        "with_monthly_reason": "Prefiero pagar por mes.",
    } for row in group]}
    assert len(validate_repricing_batch(good, group)) == 25

    incomplete = {"choices": good["choices"][:24]}
    with pytest.raises(RepricingValidationError):
        validate_repricing_batch(incomplete, group)

    duplicate = {"choices": [dict(item) for item in good["choices"]]}
    duplicate["choices"][1]["id"] = duplicate["choices"][0]["id"]
    with pytest.raises(RepricingValidationError):
        validate_repricing_batch(duplicate, group)

    unknown = {"choices": [dict(item) for item in good["choices"]]}
    unknown["choices"][0]["id"] = "T0999"
    with pytest.raises(RepricingValidationError):
        validate_repricing_batch(unknown, group)

    invalid_choice = {"choices": [dict(item) for item in good["choices"]]}
    invalid_choice["choices"][0]["with_monthly_choice"] = "maybe"
    with pytest.raises(RepricingValidationError):
        validate_repricing_batch(invalid_choice, group)


def test_checkpoint_resumes_after_first_complete_batch_without_replacing_profiles():
    rows = cohort_rows()
    first = validate_repricing_batch(fake_model([{
        "content": "Participantes:\n" + json.dumps([
            {"id": row["id"]} for row in rows[:25]
        ])
    }]), rows[:25])
    calls = []

    def ask(messages):
        calls.append(messages)
        return fake_model(messages)

    report = generate_repricing(
        SOURCE_SCENARIO, OFFER_CONTEXT, rows, ask, existing_choices=first,
    )
    assert len(calls) == 39
    assert report["choices"][:25] == first
    with pytest.raises(RepricingValidationError):
        generate_repricing(
            SOURCE_SCENARIO, OFFER_CONTEXT, rows, ask,
            existing_choices=first[:24],
        )
    with pytest.raises(RepricingValidationError):
        generate_repricing(SOURCE_SCENARIO, OFFER_CONTEXT, rows[:999], ask)


def test_local_api_pilot_resume_and_complete_paired_report(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "fake-model")
    monkeypatch.setattr(Config, "LLM_API_KEY", "fake-key")
    monkeypatch.setattr(Config, "CHATGPT_PLAN_MODE", False)
    job_id = "b" * 32
    cohort_dir = tmp_path / "cohorts" / job_id
    cohort_dir.mkdir(parents=True)
    (cohort_dir / "state.json").write_text(json.dumps({
        "id": job_id, "status": "completed", "scenario": SOURCE_SCENARIO,
        "rows": cohort_rows(),
    }), encoding="utf-8")
    calls = 0

    class FakeLLM:
        def chat(self, messages, **_kwargs):
            nonlocal calls
            calls += 1
            return json.dumps(fake_model(messages))

    monkeypatch.setattr(repricing_api, "LLMClient", FakeLLM)
    app = Flask(__name__)
    app.register_blueprint(repricing_api.repricing_bp, url_prefix="/api/cohort")
    client = app.test_client()
    url = f"/api/cohort/{job_id}/repricing"
    request = {"offer_context": OFFER_CONTEXT, "pilot": True}

    assert client.post(url + "/run", json=request,
                       environ_base={"REMOTE_ADDR": "198.51.100.1"}).status_code == 403
    assert client.post(url + "/run", json=request,
                       headers={"Origin": "https://evil.example"}).status_code == 403
    started = client.post(url + "/run", json=request)
    assert started.status_code == 202
    run_id = started.json["data"]["run_id"]
    run_url = url + "/" + run_id
    for _ in range(200):
        status = client.get(run_url).json["data"]
        assert "offer_context" not in status and "choices" not in status
        if status["status"] == "pilot_ready":
            break
        time.sleep(0.01)
    assert status["status"] == "pilot_ready"
    assert calls == 1
    assert client.get(run_url + "/report").status_code == 409
    pilot = client.get(run_url + "/pilot")
    assert pilot.status_code == 200
    assert len(pilot.json["data"]["choices"]) == 25

    assert client.post(run_url + "/resume").status_code == 202
    for _ in range(300):
        status = client.get(run_url).json["data"]
        if status["status"] == "completed":
            break
        time.sleep(0.01)
    assert status["status"] == "completed"
    assert calls == 40
    report = client.get(run_url + "/report")
    assert report.status_code == 200
    assert report.json["data"]["participants"] == 1000
    assert report.json["data"]["with_monthly_counts"] == {
        "annual": 250, "monthly": 250, "none": 500,
    }
    assert len(report.json["data"]["choices"]) == 1000
    assert client.post(run_url + "/resume").status_code == 409
