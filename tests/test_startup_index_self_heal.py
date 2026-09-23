"""Tests for the startup index self-heal, stale -shm cleanup, and transient
ingest retry (multi-process WAL split-brain hardening, Sep 2026).

Production fault being covered: ``sqlite_autoindex_metadata_1`` loses a row
entry while the row survives. ``PRAGMA quick_check`` misses it;
``PRAGMA integrity_check`` reports ``row N missing from index`` /
``wrong # of entries in index``. The store must heal that in place with
REINDEX on open — never rename or move the database file.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from hermes_trove.sqlite_util import (
    _is_index_corruption_detail,
    _remove_stale_shm_sidecar,
    startup_index_self_heal,
)


PROD_DETAILS = [
    "row 5 missing from index sqlite_autoindex_metadata_1",
    "wrong # of entries in index sqlite_autoindex_metadata_1",
]


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _StubConn:
    """Replay canned integrity_check responses; count REINDEX calls."""

    def __init__(self, script):
        self.script = list(script)
        self.reindex_calls = 0

    def execute(self, sql):
        op = sql.strip().upper()
        assert op in ("PRAGMA INTEGRITY_CHECK", "REINDEX"), sql
        if op == "REINDEX":
            self.reindex_calls += 1
            return _Rows([])
        return _Rows([(d,) for d in self.script.pop(0)])


class TestIndexCorruptionDetailMatcher:
    def test_matches_production_strings(self):
        for detail in PROD_DETAILS:
            assert _is_index_corruption_detail(detail), detail

    def test_rejects_non_index_damage(self):
        assert not _is_index_corruption_detail("ok")
        assert not _is_index_corruption_detail("Page 12 is never used")
        assert not _is_index_corruption_detail("*** in database main ***")
        assert not _is_index_corruption_detail("")


class TestStartupIndexSelfHeal:
    def test_healthy_db_is_noop(self, tmp_path: Path):
        db = tmp_path / "healthy.db"
        conn = sqlite3.connect(str(db))
        try:
            conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO metadata VALUES ('a','1'),('b','2')")
            conn.commit()
            assert startup_index_self_heal(conn) == {"healed": False, "details": []}
        finally:
            conn.close()

    def test_production_details_heal_and_reindex_once(self):
        conn = _StubConn([PROD_DETAILS, ["ok"]])
        assert startup_index_self_heal(conn) == {"healed": True, "details": []}
        assert conn.reindex_calls == 1

    def test_persistent_damage_reports_without_heal(self):
        conn = _StubConn([PROD_DETAILS, PROD_DETAILS])
        result = startup_index_self_heal(conn)
        assert result == {"healed": False, "details": PROD_DETAILS}
        assert conn.reindex_calls == 1

    def test_non_index_damage_never_reindexes(self):
        conn = _StubConn([["*** in database main ***", "Page 12 is never used"]])
        result = startup_index_self_heal(conn)
        assert result["healed"] is False
        assert conn.reindex_calls == 0


class TestStaleShmCleanup:
    def _shm(self, db: Path) -> Path:
        return db.with_name(db.name + "-shm")

    def _wal(self, db: Path) -> Path:
        return db.with_name(db.name + "-wal")

    def test_removes_shm_when_wal_absent(self, tmp_path: Path):
        db = tmp_path / "t.db"
        db.write_bytes(b"SQLite format 3\x00")
        self._shm(db).write_bytes(b"x" * 32768)
        assert _remove_stale_shm_sidecar(db) is True
        assert not self._shm(db).exists()

    def test_noop_when_no_shm(self, tmp_path: Path):
        db = tmp_path / "t.db"
        db.write_bytes(b"SQLite format 3\x00")
        assert _remove_stale_shm_sidecar(db) is False

    def test_keeps_shm_when_wal_nonempty(self, tmp_path: Path):
        db = tmp_path / "t.db"
        db.write_bytes(b"SQLite format 3\x00")
        self._shm(db).write_bytes(b"x" * 32768)
        self._wal(db).write_bytes(b"y" * 100)
        assert _remove_stale_shm_sidecar(db) is False
        assert self._shm(db).exists()

    def test_removes_shm_when_wal_empty(self, tmp_path: Path):
        db = tmp_path / "t.db"
        db.write_bytes(b"SQLite format 3\x00")
        self._shm(db).write_bytes(b"x" * 32768)
        self._wal(db).write_bytes(b"")
        assert _remove_stale_shm_sidecar(db) is True
        assert not self._shm(db).exists()


class TestRetryWorthyIngestError:
    """Predicate contract: lock contention retries; corruption never does."""

    @staticmethod
    def _engine_predicate():
        from hermes_trove import engine as engine_module

        # Unbound method: call as pred(None, exc) — self unused by predicate.
        unbound = engine_module.TROVEEngine._is_retry_worthy_ingest_error
        assert callable(unbound)
        return lambda exc: unbound(None, exc)

    def test_locked_errors_retry(self):
        pred = self._engine_predicate()
        assert pred(sqlite3.OperationalError("database is locked")) is True
        assert pred(sqlite3.OperationalError("database table is locked")) is True

    def test_corruption_never_retries(self):
        pred = self._engine_predicate()
        assert pred(sqlite3.DatabaseError("database disk image is malformed")) is False
        assert pred(sqlite3.OperationalError("disk I/O error")) is False
        assert pred(sqlite3.OperationalError("no such table: foo")) is False
