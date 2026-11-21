"""A legacy Zep project cannot be read or deleted through local memory."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app import create_app
from app.api import graph as graph_api
from app.config import Config
from app.models.project import ProjectManager, ProjectStatus
from app.services.simulation_manager import SimulationManager
from app.services.simulation_runner import RunnerStatus, SimulationRunner
from app.services.zep_graph_memory_updater import ZepGraphMemoryManager


@pytest.fixture
def legacy_project(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "CHATGPT_PLAN_MODE", False)
    monkeypatch.setattr(Config, "MEMORY_BACKEND", "zep")
    monkeypatch.setattr(ProjectManager, "PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(tmp_path / "simulations"))
    monkeypatch.setattr(Config, "OASIS_SIMULATION_DATA_DIR", str(tmp_path / "simulations"))
    project = ProjectManager.create_project("Synthetic legacy project")
    project.status = ProjectStatus.GRAPH_COMPLETED
    project.graph_id = "synthetic-zep-graph"
    ProjectManager.save_project(project)

    # Existing installations saved projects before the backend field existed.
    path = ProjectManager._get_project_meta_path(project.project_id)
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    data.pop("memory_backend")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file)
    assert ProjectManager.get_project(project.project_id).memory_backend == "zep"

    monkeypatch.setattr(Config, "MEMORY_BACKEND", "local")
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client(), project


def _assert_backend_conflict(response):
    assert response.status_code == 409, response.get_data(as_text=True)
    assert response.json["success"] is False
    assert "MEMORY_BACKEND=zep" in response.json["error"]


def test_legacy_zep_graph_is_not_read_or_deleted_as_local(legacy_project, monkeypatch):
    client, project = legacy_project
    monkeypatch.setattr(
        graph_api,
        "_delete_graph_if_present",
        lambda graph_id: pytest.fail("A different backend's graph was touched"),
    )
    graph_id = project.graph_id

    _assert_backend_conflict(client.get(f"/api/graph/data/{graph_id}"))
    _assert_backend_conflict(client.get(f"/api/simulation/entities/{graph_id}"))
    _assert_backend_conflict(client.post("/api/report/tools/search", json={"graph_id": graph_id, "query": "test"}))
    _assert_backend_conflict(client.post("/api/report/tools/statistics", json={"graph_id": graph_id}))
    _assert_backend_conflict(client.delete(f"/api/graph/delete/{graph_id}"))
    _assert_backend_conflict(client.delete(f"/api/graph/project/{project.project_id}"))
    _assert_backend_conflict(client.post(f"/api/graph/project/{project.project_id}/reset"))
    _assert_backend_conflict(client.post("/api/graph/build", json={"project_id": project.project_id}))
    _assert_backend_conflict(client.post("/api/simulation/create", json={"project_id": project.project_id}))

    retained = ProjectManager.get_project(project.project_id)
    assert retained.graph_id == graph_id
    assert retained.memory_backend == "zep"


def test_old_simulation_and_report_cannot_use_switched_backend(legacy_project, monkeypatch):
    client, project = legacy_project
    state = SimulationManager().create_simulation(project.project_id, project.graph_id)
    _assert_backend_conflict(client.post("/api/simulation/prepare", json={"simulation_id": state.simulation_id}))
    _assert_backend_conflict(client.post("/api/simulation/start", json={"simulation_id": state.simulation_id}))
    _assert_backend_conflict(client.post("/api/report/chat", json={"simulation_id": state.simulation_id, "message": "Synthetic question"}))

    monkeypatch.setattr(
        SimulationRunner,
        "get_run_state",
        lambda simulation_id: SimpleNamespace(runner_status=RunnerStatus.COMPLETED),
    )
    monkeypatch.setattr(ZepGraphMemoryManager, "get_updater", lambda simulation_id: None)
    _assert_backend_conflict(client.post("/api/report/generate", json={"simulation_id": state.simulation_id}))


def test_new_project_records_local_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "MEMORY_BACKEND", "local")
    monkeypatch.setattr(ProjectManager, "PROJECTS_DIR", str(tmp_path / "projects"))
    project = ProjectManager.create_project("Synthetic local project")
    assert project.memory_backend == "local"
    assert ProjectManager.get_project(project.project_id).memory_backend == "local"
