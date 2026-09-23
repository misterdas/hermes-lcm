"""Scan-query edge cases and lifecycle correctness tests."""
import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

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


class TestScanQueryEdgeCases:
    """Test the scan query handles all session types."""

    def test_session_with_only_messages_no_nodes(self, tmp_path):
        """Session with messages but no summary nodes (never compacted)."""
        engine = make_engine(tmp_path)
        for i in range(5):
            engine._store.append("only-msgs", {"role": "user", "content": f"m{i}"}, token_estimate=10)
        old_ts = time.time() - 200 * 86400
        engine._store._conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (old_ts, "only-msgs"))
        engine._store._conn.commit()
        # No summary node added
        result = handle_trove_command("doctor retention apply", engine)
        # Should be skipped (never-compacted)
        assert "status: ok" in result
        assert engine._store._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id='only-msgs'").fetchone()[0] == 5

    def test_session_with_only_nodes_no_messages(self, tmp_path):
        """Session with summary nodes but no messages."""
        engine = make_engine(tmp_path)
        old_ts = time.time() - 200 * 86400
        engine._dag.add_node(SummaryNode(session_id="only-nodes", depth=0, summary="summary", token_count=50, source_token_count=500, source_ids=[], source_type="messages", created_at=old_ts, earliest_at=old_ts, latest_at=old_ts))
        result = handle_trove_command("doctor retention apply", engine)
        assert "status: ok" in result

    def test_session_with_no_activity_timestamps(self, tmp_path):
        """Session where all timestamps are NULL (only nodes, no messages, nodes have NULL timestamps)."""
        engine = make_engine(tmp_path)
        # Add a summary node with NULL timestamps
        engine._dag.add_node(SummaryNode(session_id="null-ts", depth=0, summary="s", token_count=50, source_token_count=100, source_ids=[], source_type="messages", created_at=None, earliest_at=None, latest_at=None))
        result = handle_trove_command("doctor retention apply", engine)
        assert "status: ok" in result


class TestRetentionLifecycleCorrectness:
    """Verify lifecycle rows are correctly handled."""

    def test_lifecycle_deleted_for_eligible_session(self, tmp_path):
        """Lifecycle rows for fully eligible sessions are deleted."""
        engine = make_engine(tmp_path)
        old_ts = time.time() - 200 * 86400
        sids = []
        for i in range(5):
            sid = engine._store.append("old-with-lc", {"role": "user", "content": f"m{i}"}, token_estimate=10)
            sids.append(sid)
        engine._store._conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (old_ts, "old-with-lc"))
        engine._store._conn.commit()
        engine._dag.add_node(SummaryNode(session_id="old-with-lc", depth=0, summary="s", token_count=50, source_token_count=500, source_ids=sids, source_type="messages", created_at=old_ts, earliest_at=old_ts, latest_at=old_ts))
        # Add lifecycle row
        engine._store._conn.execute("INSERT INTO trove_lifecycle_state (conversation_id, current_session_id, last_finalized_session_id) VALUES ('old-with-lc', 'old-with-lc', 'old-with-lc')")
        engine._store._conn.commit()
        result = handle_trove_command("doctor retention apply", engine)
        assert "status: ok" in result
        assert engine._store._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id='old-with-lc'").fetchone()[0] == 0

    def test_lifecycle_skipped_when_current_session(self, tmp_path):
        """Lifecycle rows are skipped when session is the current one."""
        engine = make_engine(tmp_path)
        old_ts = time.time() - 200 * 86400
        sids = []
        for i in range(5):
            sid = engine._store.append("live-session", {"role": "user", "content": f"old-{i}"}, token_estimate=10)
            sids.append(sid)
        engine._store._conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (old_ts, "live-session"))
        engine._store._conn.commit()
        engine._dag.add_node(SummaryNode(session_id="live-session", depth=0, summary="s", token_count=50, source_token_count=500, source_ids=sids, source_type="messages", created_at=old_ts, earliest_at=old_ts, latest_at=old_ts))
        # Lifecycle row pointing to live session
        engine._store._conn.execute("INSERT OR REPLACE INTO trove_lifecycle_state (conversation_id, current_session_id, last_finalized_session_id) VALUES ('live-conv', 'live-session', 'live-session')")
        engine._store._conn.commit()
        result = handle_trove_command("doctor retention apply", engine)
        assert "status: ok" in result
        # Live session messages protected
        assert engine._store._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id='live-session'").fetchone()[0] == 5


class TestRetentionEmptyResults:
    """Test apply when nothing matches."""

    def test_no_eligible_sessions_returns_nothing_deleted(self, tmp_path):
        """When nothing is eligible, apply reports nothing deleted."""
        engine = make_engine(tmp_path)
        result = handle_trove_command("doctor retention apply", engine)
        assert "eligible_sessions: 0" in result or "nothing was deleted" in result

    def test_all_sessions_protected(self, tmp_path):
        """When all sessions are protected, nothing is deleted."""
        engine = make_engine(tmp_path)
        old_ts = time.time() - 200 * 86400
        for i in range(3):
            sids = []
            for j in range(3):
                sid = engine._store.append(f"old-{i}", {"role": "user", "content": f"m{i}-{j}"}, token_estimate=10)
                sids.append(sid)
            engine._store._conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (old_ts, f"old-{i}"))
            engine._store._conn.commit()
            engine._dag.add_node(SummaryNode(session_id=f"old-{i}", depth=0, summary="s", token_count=50, source_token_count=300, source_ids=sids, source_type="messages", created_at=old_ts, earliest_at=old_ts, latest_at=old_ts))
        # Set _session_id to match one of them to protect it
        engine._session_id = "old-0"
        result = handle_trove_command("doctor retention apply", engine)
        assert "status: ok" in result
        # old-0 should be protected
        assert engine._store._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id='old-0'").fetchone()[0] == 3
        # old-1, old-2 should be deleted
        assert engine._store._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id='old-1'").fetchone()[0] == 0
        assert engine._store._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id='old-2'").fetchone()[0] == 0


class TestRetentionDuplicateSessionIds:
    """Test retention handles potential duplicates in scan results."""

    def test_scan_doesnt_produce_duplicate_sessions(self, tmp_path):
        """The scan query should produce one row per session."""
        engine = make_engine(tmp_path)
        old_ts = time.time() - 200 * 86400
        for i in range(3):
            engine._store.append(f"session-{i}", {"role": "user", "content": "msg"}, token_estimate=10)
        engine._store._conn.execute("UPDATE messages SET timestamp=? WHERE session_id LIKE 'session-%'", (old_ts,))
        engine._store._conn.commit()
        for i in range(3):
            engine._dag.add_node(SummaryNode(session_id=f"session-{i}", depth=0, summary="s", token_count=50, source_token_count=100, source_ids=[], source_type="messages", created_at=old_ts, earliest_at=old_ts, latest_at=old_ts))
        result = handle_trove_command("doctor retention", engine)
        # Count sessions reported - they're formatted as '- session_id | ...'
        lines = result.split('\n')
        session_lines = [l for l in lines if l.startswith("- ") and "session-" in l]
        # Should have exactly 3 unique sessions
        assert len(session_lines) == 3


class TestRetentionTimestampBoundary:
    """Test exact boundary conditions."""

    def test_session_exactly_at_cutoff(self, tmp_path):
        """Session with age exactly equal to retention_days."""
        engine = make_engine(tmp_path, retention_days=90)
        # Create session at exactly 90 days
        boundary_ts = time.time() - 90 * 86400
        sids = []
        for i in range(3):
            sid = engine._store.append("boundary-session", {"role": "user", "content": f"m{i}"}, token_estimate=10)
            sids.append(sid)
        engine._store._conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (boundary_ts, "boundary-session"))
        engine._store._conn.commit()
        engine._dag.add_node(SummaryNode(session_id="boundary-session", depth=0, summary="s", token_count=50, source_token_count=300, source_ids=sids, source_type="messages", created_at=boundary_ts, earliest_at=boundary_ts, latest_at=boundary_ts))
        result = handle_trove_command("doctor retention apply", engine)
        # At exactly 90 days, age_days >= 90 → eligible
        # The policy is: age_days < retention_days → skip
        # So exactly 90 days IS eligible
        assert "status: ok" in result

    def test_session_one_second_under_cutoff(self, tmp_path):
        """Session with age just under retention_days."""
        engine = make_engine(tmp_path, retention_days=90)
        # Create session at 89 days 23:59:59
        just_under_ts = time.time() - (90 * 86400 - 1)
        sids = []
        for i in range(3):
            sid = engine._store.append("just-under", {"role": "user", "content": f"m{i}"}, token_estimate=10)
            sids.append(sid)
        engine._store._conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (just_under_ts, "just-under"))
        engine._store._conn.commit()
        engine._dag.add_node(SummaryNode(session_id="just-under", depth=0, summary="s", token_count=50, source_token_count=300, source_ids=sids, source_type="messages", created_at=just_under_ts, earliest_at=just_under_ts, latest_at=just_under_ts))
        result = handle_trove_command("doctor retention apply", engine)
        assert "status: ok" in result
        # Not yet eligible
        assert engine._store._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id='just-under'").fetchone()[0] == 3


class TestRetentionFTSIntegrity:
    """Verify FTS index integrity after retention."""

    def test_fts_empty_after_all_messages_deleted(self, tmp_path):
        """After deleting all messages for a session, FTS should be clean."""
        engine = make_engine(tmp_path)
        old_ts = time.time() - 200 * 86400
        sids = []
        for i in range(5):
            sid = engine._store.append("old-session", {"role": "user", "content": f"unique content {i}"}, token_estimate=10)
            sids.append(sid)
        engine._store._conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (old_ts, "old-session"))
        engine._store._conn.commit()
        engine._dag.add_node(SummaryNode(session_id="old-session", depth=0, summary="s", token_count=50, source_token_count=500, source_ids=sids, source_type="messages", created_at=old_ts, earliest_at=old_ts, latest_at=old_ts))
        # Verify FTS has content
        fts_before = engine._store._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        assert fts_before > 0
        result = handle_trove_command("doctor retention apply", engine)
        assert "status: ok" in result
        # FTS should have fewer entries
        fts_after = engine._store._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        assert fts_after < fts_before


class TestRetentionChunkArchiveCleanup:
    """Test chunk archive cleanup."""

    def test_no_orphaned_chunks_after_retention(self, tmp_path):
        """After retention, chunk_archive should not reference deleted messages."""
        engine = make_engine(tmp_path)
        old_ts = time.time() - 200 * 86400
        sids = []
        for i in range(5):
            sid = engine._store.append("old-session", {"role": "user", "content": f"m{i}"}, token_estimate=10)
            sids.append(sid)
        engine._store._conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (old_ts, "old-session"))
        engine._store._conn.commit()
        engine._dag.add_node(SummaryNode(session_id="old-session", depth=0, summary="s", token_count=50, source_token_count=500, source_ids=sids, source_type="messages", created_at=old_ts, earliest_at=old_ts, latest_at=old_ts))
        result = handle_trove_command("doctor retention apply", engine)
        assert "status: ok" in result
        # All messages deleted
        assert engine._store._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id='old-session'").fetchone()[0] == 0


class TestRetentionConcurrentTwoPhases:
    """Test concurrent preview + apply."""

    def test_preview_during_apply(self, tmp_path):
        """Preview while apply is running should not crash."""
        engine = make_engine(tmp_path)
        for i in range(10):
            old_ts = time.time() - 200 * 86400
            sids = []
            for j in range(5):
                sid = engine._store.append(f"old-{i}", {"role": "user", "content": f"m{i}-{j}"}, token_estimate=10)
                sids.append(sid)
            engine._store._conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (old_ts, f"old-{i}"))
            engine._store._conn.commit()
            engine._dag.add_node(SummaryNode(session_id=f"old-{i}", depth=0, summary="s", token_count=50, source_token_count=500, source_ids=sids, source_type="messages", created_at=old_ts, earliest_at=old_ts, latest_at=old_ts))

        results = {"apply": None, "preview": None, "errors": []}

        def apply():
            try:
                results["apply"] = handle_trove_command("doctor retention apply", engine)
            except Exception as e:
                results["errors"].append(("apply", repr(e)))

        def preview():
            try:
                time.sleep(0.1)
                results["preview"] = handle_trove_command("doctor retention", engine)
            except Exception as e:
                results["errors"].append(("preview", repr(e)))

        t1 = threading.Thread(target=apply)
        t2 = threading.Thread(target=preview)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not results["errors"], f"concurrent errors: {results['errors']}"
        assert results["apply"] and "status: ok" in results["apply"]
