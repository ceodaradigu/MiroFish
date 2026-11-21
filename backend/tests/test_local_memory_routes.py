"""Credential-free HTTP checks for the local memory option."""

from __future__ import annotations

import io
import time

import pytest

from app import create_app
from app.api import graph as graph_api
from app.config import Config
from app.models.project import ProjectManager, ProjectStatus
from app.services.simulation_manager import SimulationManager
from app.services.chatgpt_plan_auth import auth
from app.utils.local_graph import clear_local_zep_clients, get_local_zep_client

from test_local_memory import FictionalExtractor, SCHEMA


LOCAL_ORIGIN = "http://127.0.0.1:5001"


class FixedOntology:
    def generate(self, **_kwargs):
        return {**SCHEMA, "analysis_summary": "Fictional test ontology."}


class NoEntitiesExtractor:
    def chat_json(self, messages, **_kwargs):
        return {"entities": [], "relationships": []}


@pytest.fixture
def local_app(tmp_path, monkeypatch):
    clear_local_zep_clients()
    monkeypatch.setattr(Config, "CHATGPT_PLAN_MODE", True)
    monkeypatch.setattr(Config, "MEMORY_BACKEND", "local")
    monkeypatch.setattr(Config, "ZEP_API_KEY", None)
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "synthetic-model")
    monkeypatch.setattr(Config, "MEMORY_EMBEDDINGS", False)
    monkeypatch.setattr(Config, "LOCAL_GRAPH_DB", str(tmp_path / "graph.sqlite"))
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    monkeypatch.setattr(ProjectManager, "PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(tmp_path / "simulations"))
    monkeypatch.setattr(Config, "OASIS_SIMULATION_DATA_DIR", str(tmp_path / "simulations"))
    monkeypatch.setattr(graph_api, "OntologyGenerator", FixedOntology)
    monkeypatch.setattr(auth, "status", lambda: {
        "registered": True, "connected": True, "plan_usage_enabled": True,
    })
    get_local_zep_client()._llm_client = FictionalExtractor()
    app = create_app()
    app.config.update(TESTING=True)
    yield app, tmp_path
    clear_local_zep_clients()


def _new_project(client):
    response = client.post(
        "/api/graph/ontology/generate", base_url=LOCAL_ORIGIN,
        data={
            "simulation_requirement": "Discuss a fictional workshop service.",
            "files": (io.BytesIO(b"Nora considers Workshop Alpha. RELATION_CONFIRMED"), "fiction.md"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    assert response.json["success"] is True
    return response.json["data"]["project_id"]


def _wait_for_task(client, task_id):
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        response = client.get(f"/api/graph/task/{task_id}", base_url=LOCAL_ORIGIN)
        assert response.status_code == 200
        task = response.json["data"]
        if task["status"] in ("completed", "failed"):
            return task
        time.sleep(0.03)
    pytest.fail("local graph build did not finish")


def test_plan_gate_allows_local_memory_without_zep_key(local_app):
    app, _root = local_app
    app.add_url_rule("/api/graph/_local-gate-check", view_func=lambda: ("", 204), methods=["POST"])
    client = app.test_client()
    state = client.get("/api/chatgpt/status", base_url=LOCAL_ORIGIN)
    assert state.status_code == 200
    assert state.json["memoryBackend"] == "local"
    assert state.json["memoryReady"] is True
    assert state.json["zepConfigured"] is False
    work = client.post("/api/graph/_local-gate-check", base_url=LOCAL_ORIGIN, json={})
    assert work.status_code == 204


def test_fictional_file_builds_local_graph_and_simulation_record(local_app):
    app, _root = local_app
    client = app.test_client()
    project_id = _new_project(client)
    started = client.post("/api/graph/build", base_url=LOCAL_ORIGIN, json={"project_id": project_id})
    assert started.status_code == 200, started.get_data(as_text=True)
    task = _wait_for_task(client, started.json["data"]["task_id"])
    assert task["status"] == "completed", task.get("error")
    project = ProjectManager.get_project(project_id)
    assert project.status == ProjectStatus.GRAPH_COMPLETED
    graph = client.get(f"/api/graph/data/{project.graph_id}", base_url=LOCAL_ORIGIN)
    assert graph.status_code == 200
    assert graph.json["data"]["node_count"] >= 2
    assert graph.json["data"]["edge_count"] == 1
    simulation = client.post(
        "/api/simulation/create", base_url=LOCAL_ORIGIN, json={"project_id": project_id},
    )
    assert simulation.status_code == 200
    assert simulation.json["data"]["graph_id"] == project.graph_id


def test_empty_extraction_never_completes_local_graph(local_app):
    app, _root = local_app
    get_local_zep_client()._llm_client = NoEntitiesExtractor()
    client = app.test_client()
    project_id = _new_project(client)
    started = client.post("/api/graph/build", base_url=LOCAL_ORIGIN, json={"project_id": project_id})
    assert started.status_code == 200
    task = _wait_for_task(client, started.json["data"]["task_id"])
    assert task["status"] == "failed"
    project = ProjectManager.get_project(project_id)
    assert project.status == ProjectStatus.FAILED
