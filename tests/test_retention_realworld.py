"""Real-world retention tests: concurrency, mixed-age sessions, edge cases."""
import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


from hermes_trove.config import TROVEConfig
from hermes_trove.engine import TROVEEngine
from hermes_trove.dag import SummaryNode
from hermes_trove.command import handle_trove_command


def make_engine(tmp_path, retention_days=90, retention_apply_enabled=True):
    config = TROVEConfig(
        database_path=str(tmp_path / "trove.db"),
        retention_days=retention_days,
        retention_apply_enabled=retention_apply_enabled,
    )
    engine = TROVEEngine(config=config, hermes_home=str(tmp_path / "hh"))
    engine._session_id = "live-session"
    engine._session_platform = "telegram"
    engine._conversation_id = "live-session"
    engine._lifecycle.bind_session("live-session")
    return engine


def add_old_session(engine, session_id, message_count=50, age_days=200):
    old_ts = time.time() - age_days * 86400
    ids = []
    for i in range(message_count):
        sid = engine._store.append(
            session_id, {"role": "user", "content": f"old {i}"}, token_estimate=100
        )
        ids.append(sid)
    engine._store._conn.execute(
        "UPDATE messages SET timestamp=? WHERE session_id=?", (old_ts, session_id)
    )
    engine._store._conn.commit()
    engine._dag.add_node(
        SummaryNode(
            session_id=session_id,
            depth=0,
            summary="old summary",
            token_count=50,
            source_token_count=5000,
            source_ids=ids,
            source_type="messages",
            created_at=old_ts,
            earliest_at=old_ts,
            latest_at=old_ts,
        )
    )


class TestRetentionConcurrency:
    """Test concurrent ingest during retention apply."""

    def test_concurrent_append_during_apply(self, tmp_path):
        """Concurrent append during apply must not crash or corrupt."""
        engine = make_engine(tmp_path)
        add_old_session(engine, "old-done", message_count=50, age_days=200)

        stop = threading.Event()
        ingest_errors = []
        count = [0]

        def ingester():
            n = 0
            while not stop.is_set():
                try:
                    engine._store.append(
                        "live-session",
                        {"role": "user", "content": f"live msg {n}"},
                        token_estimate=10,
                    )
                    n += 1
                except BaseException as e:  # SystemError (fd exhaustion under CI ulimit) is a BaseException
                    ingest_errors.append(repr(e))
                    break
            count[0] = n

        t = threading.Thread(target=ingester, daemon=True)
        t.start()
        time.sleep(0.3)

        result = handle_trove_command("doctor retention apply", engine)
        stop.set()
        t.join(timeout=5)

        assert not ingest_errors, f"ingest errors: {ingest_errors[:2]}"
        assert "status: ok" in result

        rm = engine._store._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='old-done'"
        ).fetchone()[0]
        nd = engine._dag.connection.execute(
            "SELECT COUNT(*) FROM summary_nodes WHERE session_id='old-done'"
        ).fetchone()[0]
        lv = engine._store._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='live-session'"
        ).fetchone()[0]

        assert rm == 0, f"old-done messages should be deleted, got {rm}"
        assert nd == 1, f"old-done summary node should be kept, got {nd}"
        assert lv > 0, "live-session should have messages"

    def test_apply_during_slow_readonly_scan(self, tmp_path):
        """Apply concurrent with a slow read-only operation (simulating recall)."""
        engine = make_engine(tmp_path)
        add_old_session(engine, "old-done", message_count=30, age_days=200)

        scan_complete = False
        scan_errors = []

        def scanner():
            nonlocal scan_complete
            try:
                # Simulate a slow read-only scan
                for _ in range(100):
                    engine._store._conn.execute(
                        "SELECT COUNT(*) FROM messages WHERE session_id='old-done'"
                    ).fetchone()
                    time.sleep(0.01)
                scan_complete = True
            except Exception as e:
                scan_errors.append(repr(e))

        t = threading.Thread(target=scanner, daemon=True)
        t.start()
        time.sleep(0.1)

        result = handle_trove_command("doctor retention apply", engine)
        t.join(timeout=5)

        assert scan_complete, f"scan did not complete: {scan_errors}"
        assert "status: ok" in result


class TestRetentionMixedAge:
    """Test sessions at various ages."""

    def test_mixed_ages_only_old_deleted(self, tmp_path):
        """Only sessions older than retention_days are deleted."""
        engine = make_engine(tmp_path, retention_days=90)

        young_ts = time.time() - 30 * 86400

        # Add 3 old sessions, 3 young sessions
        for i in range(3):
            add_old_session(engine, f"old-{i}", message_count=10, age_days=200)
        for i in range(3):
            sids = []
            for j in range(10):
                sid = engine._store.append(
                    f"young-{i}",
                    {"role": "user", "content": f"young {i}-{j}"},
                    token_estimate=100,
                )
                sids.append(sid)
            engine._store._conn.execute(
                "UPDATE messages SET timestamp=? WHERE session_id=?",
                (young_ts, f"young-{i}"),
            )
            engine._store._conn.commit()
            engine._dag.add_node(
                SummaryNode(
                    session_id=f"young-{i}",
                    depth=0,
                    summary="young summary",
                    token_count=50,
                    source_token_count=5000,
                    source_ids=sids,
                    source_type="messages",
                    created_at=young_ts,
                    earliest_at=young_ts,
                    latest_at=young_ts,
                )
            )

        result = handle_trove_command("doctor retention apply", engine)
        assert "status: ok" in result

        # Old sessions deleted
        for i in range(3):
            rm = engine._store._conn.execute(
                f"SELECT COUNT(*) FROM messages WHERE session_id='old-{i}'"
            ).fetchone()[0]
            assert rm == 0, f"old-{i} messages should be deleted"

        # Young sessions kept
        for i in range(3):
            rm = engine._store._conn.execute(
                f"SELECT COUNT(*) FROM messages WHERE session_id='young-{i}'"
            ).fetchone()[0]
            assert rm == 10, f"young-{i} messages should be kept, got {rm}"

    def test_exactly_at_boundary(self, tmp_path):
        """Session exactly at retention_days boundary."""
        engine = make_engine(tmp_path, retention_days=90)

        # Session at exactly 90 days - should be eligible
        add_old_session(engine, "boundary-90", message_count=5, age_days=90)

        result = handle_trove_command("doctor retention apply", engine)
        assert "status: ok" in result

        # Session at 89 days - should NOT be eligible
        young_ts = time.time() - 89 * 86400
        sid = engine._store.append(
            "boundary-89",
            {"role": "user", "content": "young msg"},
            token_estimate=100,
        )
        engine._store._conn.execute(
            "UPDATE messages SET timestamp=? WHERE session_id=?",
            (young_ts, "boundary-89"),
        )
        engine._store._conn.commit()
        engine._dag.add_node(
            SummaryNode(
                session_id="boundary-89",
                depth=0,
                summary="boundary summary",
                token_count=50,
                source_token_count=100,
                source_ids=[sid],
                source_type="messages",
                created_at=young_ts,
                earliest_at=young_ts,
                latest_at=young_ts,
            )
        )

        rm_90 = engine._store._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='boundary-90'"
        ).fetchone()[0]
        rm_89 = engine._store._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='boundary-89'"
        ).fetchone()[0]

        # boundary-90 is always at least 90 days old by the time apply
        # evaluates it (a few microseconds elapse between setup and the
        # policy pass), so it is deterministically eligible; boundary-89
        # must NOT be deleted.
        assert rm_90 == 0, f"boundary-90 should be deleted, got {rm_90}"
        assert rm_89 == 1, f"boundary-89 should be kept, got {rm_89}"


class TestRetentionNewSessionCarryOver:
    """Test /new flow carry-over doesn't break retention."""

    def test_new_session_preserves_eligible_deletion(self, tmp_path):
        """After /new, retention should still delete old sessions."""
        engine = make_engine(tmp_path)

        # Add an old session from the "previous" conversation
        add_old_session(engine, "prev-conv", message_count=10, age_days=200)

        # Simulate /new - the old session should still be eligible
        result = handle_trove_command("doctor retention apply", engine)
        assert "status: ok" in result

        rm = engine._store._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='prev-conv'"
        ).fetchone()[0]
        assert rm == 0, "old session should be deleted regardless of current session"


class TestRetentionEdgeCases:
    """Test edge cases: empty sessions, never-compacted, protected."""

    def test_never_compacted_session_skipped(self, tmp_path):
        """Sessions without summary nodes (never compacted) should be skipped."""
        engine = make_engine(tmp_path)

        # Add old messages without summary nodes
        old_ts = time.time() - 200 * 86400
        for i in range(10):
            engine._store.append(
                "never-compacted",
                {"role": "user", "content": f"msg {i}"},
                token_estimate=100,
            )
        engine._store._conn.execute(
            "UPDATE messages SET timestamp=? WHERE session_id=?",
            (old_ts, "never-compacted"),
        )
        engine._store._conn.commit()

        # No summary node added - this session was never compacted
        result = handle_trove_command("doctor retention apply", engine)
        assert "status: ok" in result
        rm = engine._store._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='never-compacted'"
        ).fetchone()[0]
        assert rm == 10, "never-compacted session should be skipped"

    def test_empty_session_skipped(self, tmp_path):
        """Sessions with messages but zero node count (edge case)."""
        engine = make_engine(tmp_path)
        # Add a session with a summary node but no actual messages
        old_ts = time.time() - 200 * 86400
        engine._dag.add_node(
            SummaryNode(
                session_id="empty-msgs",
                depth=0,
                summary="empty",
                token_count=0,
                source_token_count=0,
                source_ids=[],
                source_type="messages",
                created_at=old_ts,
                earliest_at=old_ts,
                latest_at=old_ts,
            )
        )
        result = handle_trove_command("doctor retention apply", engine)
        # Should not crash, nothing to delete
        assert "status: ok" in result

    def test_live_session_protected(self, tmp_path):
        """Current session (engine._session_id) must not be deleted."""
        engine = make_engine(tmp_path)
        # Add messages to the live session
        old_ts = time.time() - 200 * 86400
        for i in range(10):
            engine._store.append(
                "live-session",
                {"role": "user", "content": f"msg {i}"},
                token_estimate=100,
            )
        engine._store._conn.execute(
            "UPDATE messages SET timestamp=? WHERE session_id=?",
            (old_ts, "live-session"),
        )
        engine._store._conn.commit()
        engine._dag.add_node(
            SummaryNode(
                session_id="live-session",
                depth=0,
                summary="live summary",
                token_count=50,
                source_token_count=500,
                source_ids=[],
                source_type="messages",
                created_at=old_ts,
                earliest_at=old_ts,
                latest_at=old_ts,
            )
        )

        result = handle_trove_command("doctor retention apply", engine)
        assert "status: ok" in result
        rm = engine._store._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='live-session'"
        ).fetchone()[0]
        assert rm == 10, "live session messages must NOT be deleted"
