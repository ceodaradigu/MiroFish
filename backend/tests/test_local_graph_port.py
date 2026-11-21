"""Focused local-memory checks for Spanish search and deletion retention."""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from zep_cloud import NotFoundError

from app.utils.local_graph import LocalZep, bm25_rank, tokenize


class FixedExtractor:
    def __init__(self, result, started=None, release=None):
        self.result = result
        self.started = started
        self.release = release

    def chat_json(self, **_kwargs):
        if self.started:
            self.started.set()
        if self.release:
            assert self.release.wait(5)
        return self.result


def _add_batch(client, graph_id, content):
    batch = client.batch.create(metadata={"operation_id": graph_id})
    client.batch.add(
        batch.batch_id,
        items=[SimpleNamespace(type="graph_episode", graph_id=graph_id, data=content)],
    )
    return batch.batch_id


def test_spanish_words_with_accents_match_unaccented_search():
    assert tokenize("México: mecánicos, reparación") == [
        "mexico", "mecanicos", "reparacion",
    ]
    assert bm25_rank("mecanicos reparacion", ["Mecánicos de reparación", "Otro taller"])[0][0] == 0


def test_empty_extraction_fails_and_missing_fact_does_not_create_edge(tmp_path):
    client = LocalZep(
        str(tmp_path / "empty.sqlite"),
        llm_client=FixedExtractor({"entities": [], "relationships": []}),
        max_workers=1,
    )
    try:
        client.graph.create(graph_id="empty")
        batch_id = _add_batch(client, "empty", "Un texto sin extracción")
        client.batch.process(batch_id)
        assert client.wait_idle(5)
        assert client.batch.get(batch_id).status == "failed"
        assert client._store.one("SELECT COUNT(*) FROM nodes WHERE graph_id = ?", ("empty",))[0] == 0
    finally:
        client.close()

    client = LocalZep(
        str(tmp_path / "only-missing-fact.sqlite"),
        llm_client=FixedExtractor({
            "entities": [],
            "relationships": [{"source": "Ana", "target": "Taller B", "type": "LIKES"}],
        }),
        max_workers=1,
    )
    try:
        client.graph.create(graph_id="only-missing-fact")
        batch_id = _add_batch(client, "only-missing-fact", "Relación sin prueba")
        client.batch.process(batch_id)
        assert client.wait_idle(5)
        assert client.batch.get(batch_id).status == "failed"
        assert client._store.one("SELECT COUNT(*) FROM nodes")[0] == 0
    finally:
        client.close()

    client = LocalZep(
        str(tmp_path / "missing-fact.sqlite"),
        llm_client=FixedExtractor({
            "entities": [{"name": "Ana", "type": "Person"}, {"name": "Taller B", "type": "Workshop"}],
            "relationships": [{"source": "Ana", "target": "Taller B", "type": "LIKES"}],
        }),
        max_workers=1,
    )
    try:
        client.graph.create(graph_id="missing-fact")
        batch_id = _add_batch(client, "missing-fact", "Ana y Taller B")
        client.batch.process(batch_id)
        assert client.wait_idle(5)
        assert client.batch.get(batch_id).status == "succeeded"
        assert client._store.one("SELECT COUNT(*) FROM nodes WHERE graph_id = ?", ("missing-fact",))[0] == 2
        assert client._store.one("SELECT COUNT(*) FROM edges WHERE graph_id = ?", ("missing-fact",))[0] == 0
    finally:
        client.close()


def test_delete_preserves_shared_batch_and_clears_global_caches(tmp_path):
    database = tmp_path / "shared.sqlite"
    client = LocalZep(str(database), llm_client=FixedExtractor({}), max_workers=1)
    try:
        client.graph.create(graph_id="one")
        client.graph.create(graph_id="two")
        batch = client.batch.create(metadata={"operation_id": "shared"})
        unrelated_empty_batch = client.batch.create(metadata={"operation_id": "unrelated"})
        deleted_empty_batch = client.batch.create(metadata={"graph_id": "one"})
        client.batch.add(batch.batch_id, items=[
            SimpleNamespace(type="graph_episode", graph_id="one", data="Contenido privado uno"),
            SimpleNamespace(type="graph_episode", graph_id="two", data="Contenido privado dos"),
        ])
        client._store.execute(
            "INSERT INTO extraction_cache(cache_key, result, created_at) VALUES (?, ?, ?)",
            ("private", '{"text":"Contenido privado uno"}', "2026-01-01"),
        )
        client._store.execute(
            "INSERT INTO embeddings(model, text_hash, dim, vector) VALUES (?, ?, ?, ?)",
            ("test", "private", 1, b"private-vector"),
        )
        assert client._store.one("PRAGMA secure_delete")[0] == 1

        client.graph.delete("one")
        assert client.batch.get(batch.batch_id).item_count == 1
        assert client.batch.get(unrelated_empty_batch.batch_id).status == "draft"
        with pytest.raises(NotFoundError):
            client.batch.get(deleted_empty_batch.batch_id)
        assert client.batch.list_items(batch.batch_id).items[0].graph_id == "two"
        assert client._store.one("SELECT COUNT(*) FROM episodes WHERE graph_id = 'one'")[0] == 0
        assert client._store.one("SELECT COUNT(*) FROM extraction_cache")[0] == 0
        assert client._store.one("SELECT COUNT(*) FROM embeddings")[0] == 0

        client.graph.delete("two")
        assert client.batch.get(unrelated_empty_batch.batch_id).status == "draft"
        client.batch.delete(unrelated_empty_batch.batch_id)
        assert client._store.one("SELECT COUNT(*) FROM batches")[0] == 0
        assert client._store.one("SELECT COUNT(*) FROM batch_items")[0] == 0
    finally:
        client.close()
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM graphs").fetchone()[0] == 0


def test_delete_while_extraction_runs_does_not_restore_cache(tmp_path):
    started, release = threading.Event(), threading.Event()
    client = LocalZep(
        str(tmp_path / "inflight.sqlite"),
        llm_client=FixedExtractor(
            {"entities": [{"name": "Ana", "type": "Person"}], "relationships": []},
            started,
            release,
        ),
        max_workers=1,
    )
    try:
        client.graph.create(graph_id="gone")
        batch_id = _add_batch(client, "gone", "Ana")
        client.batch.process(batch_id)
        assert started.wait(5)
        client.graph.delete("gone")
        release.set()
        assert client.wait_idle(5)
        for table in ("graphs", "episodes", "nodes", "edges", "batch_items", "batches", "extraction_cache"):
            assert client._store.one(f"SELECT COUNT(*) FROM {table}")[0] == 0
    finally:
        release.set()
        client.close()


def test_recreated_graph_id_does_not_receive_old_extraction(tmp_path):
    started, release = threading.Event(), threading.Event()
    client = LocalZep(
        str(tmp_path / "recreated.sqlite"),
        llm_client=FixedExtractor(
            {"entities": [{"name": "Dato anterior", "type": "Entity"}], "relationships": []},
            started,
            release,
        ),
        max_workers=1,
    )
    try:
        client.graph.create(graph_id="reused")
        batch_id = _add_batch(client, "reused", "Dato anterior")
        client.batch.process(batch_id)
        assert started.wait(5)
        client.graph.delete("reused")
        client.graph.create(graph_id="reused")
        release.set()
        assert client.wait_idle(5)
        assert client._store.one("SELECT COUNT(*) FROM nodes WHERE graph_id = 'reused'")[0] == 0
        assert client._store.one("SELECT COUNT(*) FROM extraction_cache")[0] == 0
    finally:
        release.set()
        client.close()


def test_delete_while_embedding_runs_does_not_restore_vector(tmp_path):
    started, release = threading.Event(), threading.Event()

    class BlockingEmbedder:
        available = True
        model_name = "test"

        def embed(self, texts, *, query=False):
            started.set()
            assert release.wait(5)
            return [np.array([1.0, 0.0], dtype=np.float32) for _ in texts]

    client = LocalZep(
        str(tmp_path / "inflight-vector.sqlite"),
        llm_client=FixedExtractor({}),
        embedder=BlockingEmbedder(),
        max_workers=1,
    )
    try:
        client.graph.create(graph_id="gone")
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(client._vectors_for, ["Private workshop note"], graph_id="gone")
            assert started.wait(5)
            client.graph.delete("gone")
            release.set()
            assert result.result(timeout=5) == [None]
        assert client._store.one("SELECT COUNT(*) FROM embeddings")[0] == 0
        assert client._vector_cache == {}
    finally:
        release.set()
        client.close()
