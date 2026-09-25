import logging
import sqlite3
import threading
import time

import pytest

from hermes_trove.config import TROVEConfig
from hermes_trove.db_bootstrap import log_sqlite_failure, sqlite_failure_diagnostic
from hermes_trove.dag import SummaryDAG
from hermes_trove.engine import TROVEEngine
from hermes_trove.lifecycle_state import LifecycleStateStore
from hermes_trove.query_view_store import QueryViewStore
from hermes_trove.rollup_store import RollupStore
from hermes_trove.store import MessageStore


def test_sqlite_failure_diagnostic_is_structured_and_does_not_include_exception_text():
    error = sqlite3.OperationalError("database is locked: secret conversation text")
    diagnostic = sqlite_failure_diagnostic(
        error,
        store="MessageStore",
        operation="append",
        db_path="/tmp/trove.db",
    )
    assert diagnostic == {
        "store": "MessageStore",
        "operation": "append",
        "db_path": "/tmp/trove.db",
        "error_type": "OperationalError",
        "sqlite_errorcode": None,
        "retryable": True,
    }
    assert "secret conversation text" not in repr(diagnostic)


def test_log_sqlite_failure_emits_metadata_without_exception_text(caplog):
    error = sqlite3.OperationalError("database is locked: private prompt")
    with caplog.at_level(logging.WARNING):
        diagnostic = log_sqlite_failure(
            error,
            store="VectorStore",
            operation="publish",
            db_path="/tmp/trove.db",
        )
    assert diagnostic["retryable"] is True
    record = caplog.records[-1]
    assert "VectorStore" in record.getMessage()
    assert "publish" in record.getMessage()
    assert "private prompt" not in record.getMessage()


def test_query_view_store_transaction_is_reentrant(tmp_path):
    db_path = tmp_path / "query-view.db"
    source_store = MessageStore(db_path)
    source_store.close()
    store = QueryViewStore(db_path)
    try:
        with store._write_transaction():
            store._conn.execute("CREATE TABLE nested_test (id INTEGER PRIMARY KEY)")
            with store._write_transaction():
                store._conn.execute("INSERT INTO nested_test(id) VALUES (1)")
        assert store._conn.execute("SELECT id FROM nested_test").fetchone()[0] == 1
    finally:
        store.close()


def test_query_view_store_nested_transaction_failure_rolls_back_only_inner_work(tmp_path):
    db_path = tmp_path / "query-view-rollback.db"
    source_store = MessageStore(db_path)
    source_store.close()
    store = QueryViewStore(db_path)
    try:
        with store._write_transaction():
            store._conn.execute("CREATE TABLE nested_rollback_test (id INTEGER PRIMARY KEY)")
            store._conn.execute("INSERT INTO nested_rollback_test(id) VALUES (1)")
            with pytest.raises(RuntimeError, match="inner failure"):
                with store._write_transaction():
                    store._conn.execute("INSERT INTO nested_rollback_test(id) VALUES (2)")
                    raise RuntimeError("inner failure")
        rows = store._conn.execute("SELECT id FROM nested_rollback_test ORDER BY id").fetchall()
        assert [row[0] for row in rows] == [1]
    finally:
        store.close()


def test_engine_shutdown_is_idempotent(tmp_path):
    engine = TROVEEngine(
        config=TROVEConfig(database_path=str(tmp_path / "lifecycle.db")),
        hermes_home=str(tmp_path / "home"),
    )
    engine.shutdown()
    engine.shutdown()
    assert engine._lifecycle_state == "shutdown"


@pytest.mark.parametrize(
    ("factory", "lock_name"),
    [
        (lambda path: MessageStore(path), "_write_lock"),
        (lambda path: SummaryDAG(path), "_db_lock"),
        (lambda path: LifecycleStateStore(path), "_lock"),
        (lambda path: RollupStore(path), "_write_lock"),
    ],
)
def test_store_close_waits_for_active_write_lock(tmp_path, factory, lock_name):
    store = factory(tmp_path / "close-race.db")
    lock = getattr(store, lock_name)
    thread = None
    try:
        with lock:
            closed = threading.Event()

            def close_store():
                store.close()
                closed.set()

            thread = threading.Thread(target=close_store)
            thread.start()
            time.sleep(0.02)

            assert not closed.is_set()
    finally:
        store.close()
    thread.join(timeout=2)
    assert closed.is_set()
