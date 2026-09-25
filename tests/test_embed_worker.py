"""Tests for the auto-embedding background worker (F1).

Covers the debounce coalescing, the no-op paths (nothing pending, lease
held, feature disabled), a single-batch run when pending + lease-free,
and safe shutdown behavior.
"""

from __future__ import annotations

import json
import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import hermes_trove.command as command_mod
import hermes_trove.embed_worker as worker_mod
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


def _engine(tmp_path, *, enabled: bool = True, auto_backfill: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "backfill.db"
    config = TROVEConfig(
        database_path=str(db_path),
        embeddings_enabled=enabled,
        embedding_provider="ollama",
        embedding_model="model-a",
        embed_auto_backfill_enabled=auto_backfill,
        embed_auto_backfill_debounce_s=300.0 if auto_backfill else 30.0,
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
    finally:
        dag.close()
    if register:
        store = VectorStore(engine._store.db_path, config=engine._config)
        try:
            store.register_profile("model-a", "ollama", 2)
        finally:
            store.close()
    return node_ids


def _meta_ids(engine) -> list[str]:
    conn = sqlite3.connect(engine._store.db_path)
    try:
        return [
            str(row[0])
            for row in conn.execute(
                "SELECT embedded_id FROM trove_embedding_meta ORDER BY CAST(embedded_id AS INTEGER)"
            ).fetchall()
        ]
    finally:
        conn.close()


def _claim_value(engine):
    conn = sqlite3.connect(engine._store.db_path)
    try:
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (command_mod._EMBEDDING_BACKFILL_CLAIM_KEY,),
        ).fetchone()
        return None if row is None else str(row[0])
    finally:
        conn.close()


def _fresh_scheduler():
    """Create a fresh scheduler for tests so we don't leak state."""
    return worker_mod._EmbedAutoBackfillScheduler()


def test_noop_when_nothing_pending(tmp_path, monkeypatch):
    """With no pending embeddings, the worker is a cheap no-op."""
    engine = _engine(tmp_path, auto_backfill=True)
    _seed(engine, 0)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **kw: provider)

    scheduler = _fresh_scheduler()
    monkeypatch.setattr(worker_mod, "_EMBED_AUTO_BACKFILL_SCHEDULER", scheduler)

    scheduler._do_auto_backfill(engine)
    assert provider.calls == []
    assert _meta_ids(engine) == []


def test_noop_when_lease_held(tmp_path, monkeypatch):
    """When the lease is held by another worker, skip without erroring."""
    engine = _engine(tmp_path, auto_backfill=True)
    _seed(engine, 5)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **kw: provider)

    # Insert a fresh claim into the metadata table
    conn = sqlite3.connect(engine._store.db_path)
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES(?, ?)",
        (
            command_mod._EMBEDDING_BACKFILL_CLAIM_KEY,
            json.dumps({"owner": "someone-else", "claimed_at": time.time()}),
        ),
    )
    conn.commit()
    conn.close()

    scheduler = _fresh_scheduler()
    monkeypatch.setattr(worker_mod, "_EMBED_AUTO_BACKFILL_SCHEDULER", scheduler)

    scheduler._do_auto_backfill(engine)
    assert provider.calls == []
    assert _meta_ids(engine) == []


def test_runs_one_batch_when_pending_and_lease_free(tmp_path, monkeypatch):
    """With pending work and a free lease, run exactly one batch."""
    engine = _engine(tmp_path, auto_backfill=True)
    node_ids = _seed(engine, 5)
    provider = FakeProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **kw: provider)

    scheduler = _fresh_scheduler()
    monkeypatch.setattr(worker_mod, "_EMBED_AUTO_BACKFILL_SCHEDULER", scheduler)

    scheduler._do_auto_backfill(engine)
    assert len(provider.calls) == 1
    assert len(provider.calls[0]) == 5
    assert _meta_ids(engine) == [str(nid) for nid in node_ids]
    # The lease should be released after the run completes
    assert _claim_value(engine) is None


def test_disable_flag_disables_it(tmp_path, monkeypatch):
    """When embed_auto_backfill_enabled=False, schedule_auto_backfill is a no-op."""
    engine = _engine(tmp_path, enabled=True, auto_backfill=False)
    _seed(engine, 5)

    scheduler = _fresh_scheduler()
    # Embed a spy on _do_auto_backfill to ensure it's never called
    scheduler._do_auto_backfill = MagicMock()
    monkeypatch.setattr(worker_mod, "_EMBED_AUTO_BACKFILL_SCHEDULER", scheduler)

    worker_mod.schedule_auto_backfill(engine)
    # If the flag were ignored, the timer would be started. With it disabled,
    # nothing should happen.
    scheduler._do_auto_backfill.assert_not_called()
    # Timer should not have been scheduled
    assert scheduler._timer is None


def test_debounce_coalesces_burst(tmp_path, monkeypatch):
    """A burst of ingests coalesces into a single timer."""
    engine = _engine(tmp_path, enabled=True, auto_backfill=True)

    scheduler = _fresh_scheduler()
    monkeypatch.setattr(worker_mod, "_EMBED_AUTO_BACKFILL_SCHEDULER", scheduler)

    # Fake Timer so we don't actually start real timers
    timer_started = []

    class FakeTimer:
        def __init__(self, interval, function, args=()):
            self.interval = interval
            self.function = function
            self.args = args
            self.daemon = True
            self._canceled = False
            timer_started.append(self)

        def start(self):
            pass

        def cancel(self):
            self._canceled = True

    monkeypatch.setattr(worker_mod.threading, "Timer", FakeTimer)

    # Schedule multiple times rapidly
    for _ in range(5):
        scheduler.schedule_auto_backfill(engine)

    # All 5 calls should coalesce: exactly one timer is still active
    # (canceled timers are tracked separately; only the last one survives)
    active_timers = [t for t in timer_started if not t._canceled]
    assert len(active_timers) == 1
    # 5 timers were created but 4 were canceled on re-schedule
    assert len(timer_started) == 5
    canceled_timers = [t for t in timer_started if t._canceled]
    assert len(canceled_timers) == 4


def test_debounce_does_not_cancel_another_engines_pending_work(tmp_path, monkeypatch):
    """Each engine gets its own debounce timer instead of last-writer-wins."""
    engine_a = _engine(tmp_path / "a", enabled=True, auto_backfill=True)
    engine_b = _engine(tmp_path / "b", enabled=True, auto_backfill=True)
    scheduler = _fresh_scheduler()
    timer_started = []

    class FakeTimer:
        def __init__(self, interval, function, args=()):
            self.interval = interval
            self.function = function
            self.args = args
            self.daemon = True
            self._canceled = False
            timer_started.append(self)

        def start(self):
            pass

        def cancel(self):
            self._canceled = True

    monkeypatch.setattr(worker_mod.threading, "Timer", FakeTimer)
    scheduler.schedule_auto_backfill(engine_a)
    scheduler.schedule_auto_backfill(engine_b)

    assert len(timer_started) == 2
    assert not timer_started[0]._canceled
    assert not timer_started[1]._canceled
    assert timer_started[0].args == (engine_a,)
    assert timer_started[1].args == (engine_b,)


def test_shutdown_aborts_in_flight_safely(tmp_path, monkeypatch):
    """After shutdown, no new runs are scheduled; in-flight ones abandon cleanly."""
    engine = _engine(tmp_path, enabled=True, auto_backfill=True)
    _seed(engine, 5)

    scheduler = _fresh_scheduler()
    monkeypatch.setattr(worker_mod, "_EMBED_AUTO_BACKFILL_SCHEDULER", scheduler)

    # Manually mark in_flight to simulate a run in progress
    scheduler._in_flight = True

    # Shutdown should not raise and should set the shutdown flag
    scheduler.shutdown()
    assert scheduler._shutdown is True

    # After shutdown, scheduling should be a no-op
    worker_mod.schedule_auto_backfill(engine)
    assert scheduler._timer is None

    # The in-flight run can still call _run_auto_backfill, which bails early
    # because shutdown is True. It should not raise.
    scheduler._run_auto_backfill(engine)
    # _in_flight stays True (the run that was in progress never cleared it),
    # but that's fine because no new runs will be scheduled.
    assert scheduler._in_flight is True


def test_shutdown_no_op_when_no_timer(tmp_path, monkeypatch):
    """Shutdown is safe even if no timer has ever been created."""
    scheduler = _fresh_scheduler()
    scheduler.shutdown()
    assert scheduler._shutdown is True
    assert scheduler._timer is None


def test_noop_when_embeddings_disabled(tmp_path, monkeypatch):
    """When embeddings are disabled, schedule_auto_backfill is a no-op."""
    engine = _engine(tmp_path, enabled=False, auto_backfill=True)

    scheduler = _fresh_scheduler()
    monkeypatch.setattr(worker_mod, "_EMBED_AUTO_BACKFILL_SCHEDULER", scheduler)

    worker_mod.schedule_auto_backfill(engine)
    assert scheduler._timer is None


def test_provider_not_configured_noop(tmp_path, monkeypatch):
    """When resolve_provider returns None, the worker is a no-op."""
    engine = _engine(tmp_path, auto_backfill=True)
    _seed(engine, 5)
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **kw: None)

    scheduler = _fresh_scheduler()
    monkeypatch.setattr(worker_mod, "_EMBED_AUTO_BACKFILL_SCHEDULER", scheduler)

    scheduler._do_auto_backfill(engine)
    assert _meta_ids(engine) == []
