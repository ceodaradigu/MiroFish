"""Offline acceptance tests for the optional local graph memory.

The source document, graph extraction, and model responses here are fictional.
No Zep or OpenAI endpoint is contacted. The test covers MiroFish's public
GraphBuilder contract instead of relying on a network service.
"""

from __future__ import annotations

import sqlite3

import pytest
from zep_cloud import NotFoundError

from app.config import Config
from app.services.graph_builder import GraphBuilderService
from app.utils.local_graph import LocalZep


SCHEMA = {
    "entity_types": [
        {"name": "Person", "description": "A fictional person", "attributes": []},
        {"name": "Workshop", "description": "A fictional workshop", "attributes": []},
    ],
    "edge_types": [
        {
            "name": "CONSIDERS",
            "description": "A person explicitly considers a workshop",
            "source_targets": [{"source": "Person", "target": "Workshop"}],
            "attributes": [],
        }
    ],
}


class FictionalExtractor:
    """Fixed model answers to test storage, not model quality."""

    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def chat_json(self, messages, **_kwargs):
        self.calls += 1
        text = messages[-1]["content"]
        if self.fail:
            raise ValueError("synthetic extraction failure")
        if "NO_RELATION" in text:
            return {
                "entities": [
                    {"name": "Nora", "type": "Person", "summary": "Nora is mentioned."},
                    {"name": "Workshop Beta", "type": "Workshop", "summary": "Workshop Beta is mentioned."},
                ],
                "relationships": [],
            }
        return {
            "entities": [
                {"name": "Nora", "type": "Person", "summary": "Nora considers Workshop Alpha."},
                {"name": "Workshop Alpha", "type": "Workshop", "summary": "Workshop Alpha is mentioned."},
            ],
            "relationships": [
                {
                    "type": "CONSIDERS", "source": "Nora", "target": "Workshop Alpha",
                    "fact": "Nora considers Workshop Alpha.",
                }
            ],
        }


def _builder(client: LocalZep) -> GraphBuilderService:
    # Supply only the actual storage dependency; the normal builder workflow
    # still creates the graph, sets ontology, submits batches, and reads it.
    builder = object.__new__(GraphBuilderService)
    builder.client = client
    builder.api_key = None
    return builder


def _ingest(client: LocalZep, chunks: list[str], graph_id: str = "fictional_graph"):
    builder = _builder(client)
    builder.create_graph("Fictional scenario", graph_id=graph_id)
    builder.set_ontology(graph_id, SCHEMA)
    submission = builder.add_text_batches(graph_id, chunks, batch_size=2)
    assert client.wait_idle(10), "local extraction did not finish"
    return builder, submission


def test_local_graph_persists_own_labels_and_only_asserted_relationships(tmp_path):
    database = tmp_path / "fictional-memory.sqlite"
    extractor = FictionalExtractor()
    client = LocalZep(str(database), llm_client=extractor, max_workers=1, embedder=None)
    try:
        builder, submission = _ingest(
            client,
            [
                "Nora considers Workshop Alpha. RELATION_CONFIRMED",
                "Nora and Workshop Beta were named separately. NO_RELATION",
            ],
        )
        assert len(builder._wait_for_batch(submission, timeout=5)) == 2
        assert client.batch.get(batch_id=submission.batch_id).status == "succeeded"
        assert extractor.calls == 2
        graph = builder.get_graph_data("fictional_graph")
        assert {node["name"] for node in graph["nodes"]} == {
            "Nora", "Workshop Alpha", "Workshop Beta",
        }
        assert {label for node in graph["nodes"] for label in node["labels"]} >= {
            "Person", "Workshop",
        }
        assert [edge["name"] for edge in graph["edges"]] == ["CONSIDERS"]
        assert graph["edges"][0]["fact"] == "Nora considers Workshop Alpha."
        assert len(submission.episode_uuids) == 2
        assert all(client.graph.episode.get(uuid_=item).processed for item in submission.episode_uuids)
    finally:
        client.close()

    # A new process would construct a new client from the same local file.
    reopened = LocalZep(str(database), llm_client=FictionalExtractor(), max_workers=1, embedder=None)
    try:
        assert _builder(reopened).get_graph_data("fictional_graph")["node_count"] == 3
        found_nodes = reopened.graph.search(
            graph_id="fictional_graph", query="Workshop Beta", scope="nodes",
        ).nodes
        assert any(node.name == "Workshop Beta" for node in found_nodes)
        found_edges = reopened.graph.search(
            graph_id="fictional_graph", query="Nora considers Workshop Alpha", scope="edges",
        ).edges
        assert any(edge.name == "CONSIDERS" for edge in found_edges)
    finally:
        reopened.close()


def test_extraction_failure_marks_batch_failed_instead_of_claiming_a_graph(tmp_path):
    client = LocalZep(
        str(tmp_path / "failed-memory.sqlite"),
        llm_client=FictionalExtractor(fail=True), max_workers=1, embedder=None,
    )
    try:
        builder, submission = _ingest(client, ["This extraction deliberately fails."])
        assert client.batch.get(batch_id=submission.batch_id).status == "failed"
        with pytest.raises(RuntimeError):
            builder._wait_for_batch(submission, timeout=5)
        assert builder.get_graph_data("fictional_graph")["node_count"] == 0
    finally:
        client.close()


def test_simulation_activity_is_searchable_without_another_llm_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "MEMORY_EXTRACT_ACTIVITY", False)
    extractor = FictionalExtractor()
    client = LocalZep(
        str(tmp_path / "activity-memory.sqlite"),
        llm_client=extractor,
        extract_activity=Config.MEMORY_EXTRACT_ACTIVITY,
        max_workers=1,
        embedder=None,
    )
    try:
        builder, submission = _ingest(client, ["Nora considers Workshop Alpha."])
        assert len(builder._wait_for_batch(submission, timeout=5)) == 1
        assert extractor.calls == 1

        activity = "Round 1: Nora rejected the delivery fee for Workshop Alpha."
        episode = client.graph.add(
            graph_id="fictional_graph",
            type="text",
            data=activity,
            created_at="2026-01-01T12:00:00Z",
            source_description="MiroFish simulation activity batch",
            metadata={
                "source": "mirofish_simulation",
                "simulation_id": "fictional_simulation",
                "platform": "twitter",
                "activity_count": 1,
                "first_round": 1,
                "last_round": 1,
                "agent_ids": "1",
                "action_types": "post",
            },
        )
        stored_episode = client.graph.episode.get(uuid_=episode.uuid_)
        assert stored_episode.processed is True
        assert stored_episode.source_description == "MiroFish simulation activity batch"
        assert stored_episode.metadata["source"] == "mirofish_simulation"
        matches = client.graph.search(
            graph_id="fictional_graph", query="rejected delivery fee", scope="edges",
        ).edges
        assert any(edge.name == "SIMULATION_ACTIVITY" and edge.fact == activity for edge in matches)
        assert extractor.calls == 1
    finally:
        client.close()


def test_deleting_graph_removes_its_batches_and_cached_extractions(tmp_path):
    database = tmp_path / "delete-memory.sqlite"
    client = LocalZep(str(database), llm_client=FictionalExtractor(), max_workers=1, embedder=None)
    try:
        builder, submission = _ingest(client, ["Nora considers Workshop Alpha."])
        assert len(builder._wait_for_batch(submission, timeout=5)) == 1
        node_uuid = builder.get_graph_data("fictional_graph")["nodes"][0]["uuid"]
        episode_uuid = submission.episode_uuids[0]
        client.graph.delete("fictional_graph")
        with pytest.raises(NotFoundError):
            client.graph.get("fictional_graph")
        with pytest.raises(NotFoundError):
            client.graph.node.get(uuid_=node_uuid)
        with pytest.raises(NotFoundError):
            client.graph.episode.get(uuid_=episode_uuid)
        with pytest.raises(NotFoundError):
            client.batch.get(batch_id=submission.batch_id)
        with pytest.raises(NotFoundError):
            client.graph.search(graph_id="fictional_graph", query="Nora", scope="nodes")
    finally:
        client.close()

    # This asserts logical retention in the database, beyond endpoint 404s.
    with sqlite3.connect(database) as connection:
        for table in ("graphs", "nodes", "edges", "episodes", "batch_items", "batches", "extraction_cache"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
