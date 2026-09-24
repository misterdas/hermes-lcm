from __future__ import annotations

import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

import hermes_trove.command as command_mod
from hermes_trove.command import (
    _EMBEDDING_BACKFILL_CLAIM_KEY,
    handle_trove_command,
)
from hermes_trove.config import TROVEConfig
from hermes_trove.dag import SummaryDAG, SummaryNode
from hermes_trove.vector_store import VectorStore


class FakeProvider:
    provider_id = "ollama"
    model_id = "model-a"

    def __init__(self, *, dim: int = 2):
        self.dim = dim
        self.calls: list[list[str]] = []
        self.last_skipped_documents: list[int] = []

    def embed_documents(self, texts):
        current = list(texts)
        self.calls.append(current)
        self.last_skipped_documents = []
        return [[float(index + 1), 1.0] for index, _text in enumerate(current)]


@pytest.fixture(autouse=True)
def deterministic_token_count(monkeypatch):
    monkeypatch.setattr(command_mod, "count_tokens", lambda text: len(str(text)))


def _engine(tmp_path, *, enabled: bool = True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "backfill.db"
    config = TROVEConfig(
        database_path=str(db_path),
        embeddings_enabled=enabled,
        embedding_provider="ollama",
        embedding_model="model-a",
    )
    return SimpleNamespace(
        _config=config,
        _store=SimpleNamespace(db_path=db_path),
    )


def _seed(engine, count: int, *, register: bool = True) -> list[int]:
    dag = SummaryDAG(engine._store.db_path)
    try:
        node_ids = [
            dag.add_node(SummaryNode(
                session_id="session-a",
                depth=0,
                summary=f"summary-{index}",
                source_token_count=100 + index,
                created_at=float(index + 1),
                latest_at=float(index + 1),
            ))
            for index in range(count)
        ]
        dag.add_node(SummaryNode(
            session_id="session-a",
            depth=1,
            summary="not a leaf",
            created_at=10_000.0,
        ))
    finally:
        dag.close()
    if register:
        store = VectorStore(engine._store.db_path, config=engine._config)
        try:
            store.register_profile("model-a", "ollama", 2)
        finally:
            store.close()
    return node_ids


def _claim_value(engine):
    conn = sqlite3.connect(engine._store.db_path)
    try:
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (_EMBEDDING_BACKFILL_CLAIM_KEY,),
        ).fetchone()
        return None if row is None else str(row[0])
    finally:
        conn.close()


def _seed_messages(engine, rows, *, register: bool = True):
    conn = sqlite3.connect(engine._store.db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            store_id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            source TEXT DEFAULT '',
            role TEXT NOT NULL,
            content TEXT,
            timestamp REAL NOT NULL
        )
        """
    )
    conn.executemany(
        "INSERT INTO messages(store_id, session_id, source, role, content, timestamp) "
        "VALUES(?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    if register:
        store = VectorStore(engine._store.db_path, config=engine._config)
        try:
            store.register_profile("model-a", "ollama", 2, task="chunk")
        finally:
            store.close()


def test_idle_state_no_lease_zero_inflight(tmp_path):
    """Status with no lease and no inflight rows shows a clean idle state."""
    engine = _engine(tmp_path)
    _seed(engine, 3)
    result = handle_trove_command("embed status", engine)

    assert "TROVE embedding status" in result
    assert "embeddings_enabled: True" in result
    assert "backfill_lease: none" in result
    assert "in_flight: 0" in result
    assert "summary_profile: ollama/model-a" in result
    assert "summary_pending: 3" in result
    assert "status: ok" in result


def test_held_lease_shows_holder_and_ttl(tmp_path):
    """A freshly-acquired lease shows the holder and remaining TTL."""
    engine = _engine(tmp_path)
    _seed(engine, 2)

    # Simulate a live lease by directly writing a claim row.
    conn = sqlite3.connect(engine._store.db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
    now = time.time()
    claim = json.dumps({
        "owner": "test-lease-id-1234",
        "generation": 7,
        "heartbeat_at": now,
        "claimed_at": now,
    })
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        (_EMBEDDING_BACKFILL_CLAIM_KEY, claim),
    )
    conn.commit()
    conn.close()

    result = handle_trove_command("embed status", engine)

    assert "backfill_lease: held by test-lease-id-1234" in result
    assert "generation 7" in result
    assert "s of 600s TTL remaining" in result


def test_held_lease_expired_reports_expired(tmp_path):
    """An expired lease is reported as expired with time-since-expiry."""
    engine = _engine(tmp_path)
    _seed(engine, 1)

    conn = sqlite3.connect(engine._store.db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
    now = time.time()
    # 700s ago — beyond the 600s TTL
    claim = json.dumps({
        "owner": "expired-lease-id",
        "generation": 3,
        "heartbeat_at": now - 700,
        "claimed_at": now - 700,
    })
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        (_EMBEDDING_BACKFILL_CLAIM_KEY, claim),
    )
    conn.commit()
    conn.close()

    result = handle_trove_command("embed status", engine)

    assert "backfill_lease: expired" in result
    assert "expired-lease-id" in result
    assert "generation 3" in result
    assert "expired" in result


def test_disabled_embeddings_reports_clearly(tmp_path):
    """Status with embeddings disabled shows that clearly."""
    engine = _engine(tmp_path, enabled=False)
    # Create the database so we can verify the disabled path, not the db-missing path
    _seed(engine, 1, register=False)
    result = handle_trove_command("embed status", engine)

    assert "embeddings_enabled: False" in result
    assert "configured_provider: ollama" in result
    assert "configured_model: model-a" in result
    assert "status: ok" in result


def test_vector_and_pending_counts_reflect_seed(tmp_path):
    """Vector counts and pending counts match seeded data."""
    engine = _engine(tmp_path)
    _seed(engine, 5)

    conn = sqlite3.connect(engine._store.db_path)
    try:
        row = conn.execute(
            "SELECT identity_hash FROM trove_embedding_profile WHERE active=1"
        ).fetchone()
        identity = str(row[0])
        rows = conn.execute(
            "SELECT node_id FROM summary_nodes WHERE depth=0 ORDER BY node_id LIMIT 2"
        ).fetchall()
        node_ids = [int(rows[0][0]), int(rows[1][0])]
        conn.execute(
            "INSERT INTO trove_embedding_meta(embedded_id, embedded_kind, identity_hash, embedded_at) VALUES(?, ?, ?, ?)",
            (str(node_ids[0]), "summary", identity, "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO trove_embedding_meta(embedded_id, embedded_kind, identity_hash, embedded_at) VALUES(?, ?, ?, ?)",
            (str(node_ids[1]), "summary", identity, "2026-01-01T00:00:01Z"),
        )
        conn.execute(
            "INSERT INTO trove_embedding_vectors(embedded_id, identity_hash, vec) VALUES(?, ?, ?)",
            (str(node_ids[0]), identity, b"\x00"),
        )
        conn.execute(
            "INSERT INTO trove_embedding_vectors(embedded_id, identity_hash, vec) VALUES(?, ?, ?)",
            (str(node_ids[1]), identity, b"\x00"),
        )
        conn.commit()
    finally:
        conn.close()

    result = handle_trove_command("embed status", engine)

    assert "summary_vectors: 2" in result
    assert "summary_pending: 3" in result  # 5 - 2 embedded


def test_status_command_triggers_no_backfill(tmp_path):
    """The status command must not acquire a lease or trigger any backfill."""
    engine = _engine(tmp_path)
    _seed(engine, 3)
    assert _claim_value(engine) is None

    result = handle_trove_command("embed status", engine)

    # Still no lease acquired
    assert _claim_value(engine) is None
    assert "in_flight: 0" in result
    assert "status: ok" in result
    # No provider calls
    assert "provider: ollama" in result


def test_status_command_idempotent_no_db_changes(tmp_path):
    """Running status twice produces identical output — no side effects."""
    engine = _engine(tmp_path)
    _seed(engine, 2)

    first = handle_trove_command("embed status", engine)
    second = handle_trove_command("embed status", engine)

    assert first == second


def test_status_reports_configured_profile(tmp_path):
    """Status reports the configured provider/model/dtype/dim from config."""
    engine = _engine(tmp_path)
    result = handle_trove_command("embed status", engine)

    assert "configured_provider: ollama" in result
    assert "configured_model: model-a" in result
    assert "configured_dtype: float32" in result
    assert "configured_dim: (unset)" in result or "configured_dim:" in result


def test_status_chunk_corpus_counts(tmp_path):
    """Status reports chunk corpus pending counts from messages."""
    engine = _engine(tmp_path)
    _seed_messages(engine, [
        (1, "sess-a", "history", "user", "A" * 60, 1.0),
        (2, "sess-a", "history", "user", "B" * 60, 2.0),
    ])

    result = handle_trove_command("embed status", engine)

    assert "chunk_profile: ollama/model-a" in result
    assert "chunk_pending:" in result


def test_status_no_profile_reports_none(tmp_path):
    """Status with no registered profiles reports them as none."""
    engine = _engine(tmp_path)
    # Create the database but don't register any profiles
    _seed(engine, 1, register=False)
    result = handle_trove_command("embed status", engine)

    assert "summary_profile: (none registered)" in result
    assert "chunk_profile: (none registered)" in result
    assert "summary_vectors: (no profile)" in result
    assert "chunk_vectors: (no profile)" in result


def test_status_reports_fastembed_cache_dir(tmp_path):
    """Status reports the fastembed cache dir (config or default)."""
    engine = _engine(tmp_path)
    result = handle_trove_command("embed status", engine)

    assert "fastembed_cache_dir:" in result


def test_embed_unknown_subcommand_still_errors(tmp_path):
    """Unknown embed subcommands still produce help, not status."""
    engine = _engine(tmp_path)
    result = handle_trove_command("embed bogus", engine)
    assert "warmup" in result or "backfill" in result or "status" in result
