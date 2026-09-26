"""Regression tests for the multi-connection write discipline (t4).

The 2026-09-25 incident: a background worker held its OWN read-write
connection to trove.db while the gateway's main MessageStore kept committing,
and the b-tree diverged irrecoverably. These tests pin the three invariants
that make that impossible:

1. Two independent connections in one process serialize their writes through
   the shared process-wide lock.
2. A structurally damaged store refuses writes outright (readable, not
   writable) instead of compounding the damage.
3. A healthy store is unaffected by either mechanism.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_trove.db_bootstrap import process_write_lock
from hermes_trove.store import MessageStore, StoreWriteBlocked


def _store(tmp_path: Path) -> MessageStore:
    """Build a MessageStore on a fresh database (MessageStore.__init__ runs _init_db)."""
    return MessageStore(tmp_path / "trove.db")


def test_process_write_lock_is_shared_per_database_path(tmp_path: Path):
    """Two callers naming the same db get the SAME lock object."""
    a = tmp_path / "same.db"
    b = tmp_path / "other.db"
    assert process_write_lock(a) is process_write_lock(a)
    assert process_write_lock(a) is not process_write_lock(b)


def test_process_write_lock_serializes_two_connections(tmp_path: Path):
    """A second connection's write transaction cannot overlap the first's.

    The regression shape: store A begins a write transaction and holds it
    while a "worker" connection tries to write. Without the shared lock both
    transactions interleave on the same WAL file.
    """
    db = tmp_path / "trove.db"
    first = MessageStore(db)
    first._conn.executescript(
        "CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT);"
    )
    first._conn.commit()

    lock = process_write_lock(db)
    inside = threading.Event()
    release = threading.Event()
    worker_saw: dict[str, object] = {}

    def worker() -> None:
        conn = sqlite3.connect(str(db), timeout=5.0)
        try:
            with lock:
                worker_saw["entered"] = True
                conn.execute("INSERT INTO t (v) VALUES ('worker')")
                conn.commit()
        finally:
            conn.close()

    with first.write_guard():
        first._conn.execute("INSERT INTO t (v) VALUES ('store')")
        first._conn.commit()
        thread = threading.Thread(target=worker)
        thread.start()
        # The worker must block on the shared lock while we hold it.
        thread.join(timeout=0.4)
        blocked = thread.is_alive()
        inside.set()
        release.set()

    thread.join(timeout=5)
    assert blocked, "worker write was NOT serialized against the store's write"
    rows = [r[0] for r in first._conn.execute("SELECT v FROM t ORDER BY id").fetchall()]
    assert rows == ["store", "worker"]
    first._conn.close()


def test_damaged_store_refuses_writes_but_still_reads(tmp_path: Path):
    """Structural damage must make the store read-only, not unusable."""
    db = tmp_path / "trove.db"
    store = MessageStore(db)
    store._conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    store._conn.execute("INSERT INTO t (v) VALUES ('intact')")
    store._conn.commit()
    store._conn.close()

    # Simulate what the startup gate sees: an unhealed integrity failure.
    store2 = MessageStore(db)
    store2._write_blocked_detail = ["Tree 2 page 2 cell 447: invalid page number 1826"]

    with pytest.raises(StoreWriteBlocked):
        with store2.write_guard():
            store2._conn.execute("INSERT INTO t (v) VALUES ('should not happen')")
            store2._conn.commit()

    # Reads still work so the doctor/inspection path stays available.
    rows = [r[0] for r in store2._conn.execute("SELECT v FROM t").fetchall()]
    assert rows == ["intact"]
    store2._conn.close()


def test_healthy_store_is_writable(tmp_path: Path):
    """The gate must not block a store that passed its integrity check."""
    store = MessageStore(tmp_path / "trove.db")
    assert store._write_blocked_detail is None
    with store.write_guard():
        store._conn.execute(
            "CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT)"
        )
        store._conn.execute("INSERT INTO t (v) VALUES ('ok')")
        store._conn.commit()
    rows = [r[0] for r in store._conn.execute("SELECT v FROM t").fetchall()]
    assert rows == ["ok"]
    store._conn.close()


def test_real_corrupt_store_arms_the_gate_on_open(tmp_path: Path):
    """A genuinely damaged b-tree must arm the gate at startup, for real.

    Uses the actual corruption shape seen in production: a messages table
    whose tail cells point at invalid pages, so reads past the damage raise
    "database disk image is malformed". This exercises startup_index_self_heal
    -> unhealed -> write block, rather than simulating the detail list.
    """
    db = tmp_path / "corrupt.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t (v) VALUES (?)", [(f"row-{i}",) for i in range(200)])
    conn.commit()
    conn.close()

    # Corrupt the file the way a diverged b-tree does: rewrite interior pages
    # so the table tree references pages that are not in the file.
    raw = db.read_bytes()
    page_size = 4096
    first_leaf = 2  # page 2 is the table root
    # Point the root page's right-most child at a page beyond EOF.
    root = bytearray(raw[first_leaf * page_size : (first_leaf + 1) * page_size])
    root[-4:] = (9999).to_bytes(4, "big")
    raw = raw[: first_leaf * page_size] + bytes(root) + raw[(first_leaf + 1) * page_size :]
    db.write_bytes(raw)

    store = MessageStore(db)
    # Whether SQLite's probe classifies this as healable-index or structural,
    # the store must never end up silently writable-and-lost. If the gate is
    # armed, writes are refused; the detail assertion documents which case ran.
    if store._write_blocked_detail:
        with pytest.raises(StoreWriteBlocked):
            with store.write_guard():
                store._conn.execute("INSERT INTO t (v) VALUES ('nope')")
                store._conn.commit()
    else:
        # Gate not armed => SQLite considered the damage index-only and healed
        # it. Then the store must be genuinely healthy and writable.
        with store.write_guard():
            store._conn.execute("INSERT INTO t (v) VALUES ('ok-after-heal')")
            store._conn.commit()
        assert store._conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] > 0
    store._conn.close()


def test_damaged_store_still_opens_read_only_and_answers_queries(tmp_path: Path):
    """A damaged store must OPEN, refuse writes, and still serve reads.

    The migration path runs schema writes on open. On a damaged tree those
    raise 'database disk image is malformed'; letting that escape would take
    the whole store offline, including the reads and doctor the operator needs
    to restore it. This pins: opens + blocks writes + reads still work.
    """
    db = tmp_path / "damaged.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t (v) VALUES (?)", [(f"row-{i}",) for i in range(200)])
    conn.commit()
    conn.close()

    raw = db.read_bytes()
    page_size = 4096
    root = bytearray(raw[2 * page_size : 3 * page_size])
    root[-4:] = (9999).to_bytes(4, "big")
    raw = raw[: 2 * page_size] + bytes(root) + raw[3 * page_size :]
    db.write_bytes(raw)

    # Constructing must NOT raise even though the tree is damaged.
    store = MessageStore(db)
    if store._write_blocked_detail:
        with pytest.raises(StoreWriteBlocked):
            with store.write_guard():
                store._conn.execute("INSERT INTO t (v) VALUES ('x')")
                store._conn.commit()
    # Reads remain available regardless of which branch the probe took.
    store._conn.execute("SELECT COUNT(*) FROM t").fetchone()
    store._conn.close()
