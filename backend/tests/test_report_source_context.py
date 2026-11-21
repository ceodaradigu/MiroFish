"""A report must not lose uploaded facts when a local graph has no edges."""

import json

from app.api import report as report_api
from app.services.report_agent import ReportAgent, ReportOutline, ReportSection
from app.utils.locale import set_locale


class FakeLLM:
    def __init__(self):
        self.messages = []

    def chat_json(self, *, messages, temperature):
        self.messages.append(messages)
        return {
            "title": "Ensayo ficticio",
            "summary": "Dudas para revisar",
            "sections": [
                {"title": "Hechos y voces", "description": "Separar fuentes"},
                {"title": "Límites", "description": "Qué falta probar"},
            ],
        }

    def chat(self, *, messages, temperature, max_tokens):
        self.messages.append(messages)
        return "Final Answer: HECHO QA aportado; HIPÓTESIS de personaje ficticio."


class EmptyGraph:
    def get_simulation_context(self, **_kwargs):
        return {
            "graph_statistics": {"total_nodes": 55, "total_edges": 0},
            "related_facts": [],
        }


def test_spanish_report_uses_seed_and_saved_actions_with_empty_graph():
    set_locale("es")
    try:
        llm = FakeLLM()
        agent = ReportAgent(
            graph_id="graph-local",
            simulation_id="sim-local",
            simulation_requirement="Revisar una orden ficticia",
            llm_client=llm,
            zep_tools=EmptyGraph(),
            source_text="HECHO QA: se guardó el cambio de Puesto A a Puesto B.",
            simulated_actions=[{"agent_name": "Recepción ficticia", "text": "Revisaría la autorización."}],
        )

        outline = agent.plan_outline()
        assert len(outline.sections) == 2
        planning_messages = llm.messages[0]
        assert "Escribe íntegramente en español" in planning_messages[0]["content"]
        assert "HECHO QA: se guardó" in planning_messages[1]["content"]
        assert "Recepción ficticia" in planning_messages[1]["content"]
        assert "relaciones: 0" in planning_messages[1]["content"]

        section = agent._generate_section_react(
            ReportSection("Hechos y voces"),
            ReportOutline("Ensayo ficticio", "Dudas para revisar", []),
            [],
        )
        assert section.startswith("HECHO QA")
        section_messages = llm.messages[1]
        assert "Escribe un capítulo" in section_messages[0]["content"]
        assert "Puesto A a Puesto B" not in section_messages[0]["content"]
        assert "Puesto A a Puesto B" in section_messages[1]["content"]
        assert "Recepción ficticia" in section_messages[1]["content"]
    finally:
        set_locale("zh")


def test_report_source_context_marks_truncation():
    agent = ReportAgent(
        graph_id="graph-local",
        simulation_id="sim-local",
        simulation_requirement="test",
        llm_client=FakeLLM(),
        zep_tools=EmptyGraph(),
        source_text="A" * 12001,
        simulated_actions=[{"text": "x"}] * 31,
    )
    context = agent._source_context()
    assert "Documento truncado" in context
    assert "31 acciones en total" in context


def test_report_reads_only_bounded_recent_action_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(report_api.SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    folder = tmp_path / "sim-local" / "twitter"
    folder.mkdir(parents=True)
    rows = [
        {"agent_id": index, "timestamp": f"2026-11-21T00:00:{index:02d}", "action_args": {"text": "ficticio"}}
        for index in range(40)
    ]
    rows.append({"event_type": "simulation_end", "timestamp": "2026-11-21T23:59:59"})
    (folder / "actions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )

    sample = report_api._recent_simulated_actions("sim-local")
    assert len(sample) == 30
    assert sample[0]["agent_id"] == 39
    assert sample[-1]["agent_id"] == 10
    assert all(row["platform"] == "twitter" for row in sample)
    assert report_api._recent_simulated_actions("../elsewhere") == []
