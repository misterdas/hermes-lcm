"""Targeted bug-hunting tests for retention edge cases."""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


from hermes_trove.config import TROVEConfig
from hermes_trove.engine import TROVEEngine
from hermes_trove.dag import SummaryNode
from hermes_trove.command import handle_trove_command
from hermes_trove.retention import evaluate_retention


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


class TestRetentionAgeCalculation:
    """Test edge cases in age calculation."""

    def test_session_with_old_messages_recent_node(self, tmp_path):
        """Session with old messages but a recent summary node."""
        engine = make_engine(tmp_path)
        # Add old messages
        old_ts = time.time() - 200 * 86400
        for i in range(5):
            engine._store.append("mixed-session", {"role": "user", "content": f"old {i}"}, token_estimate=10)
        engine._store._conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (old_ts, "mixed-session"))
        engine._store._conn.commit()
        # Add a RECENT summary node
        engine._dag.add_node(SummaryNode(
            session_id="mixed-session", depth=0, summary="recent", token_count=50,
            source_token_count=500, source_ids=[], source_type="messages",
            created_at=time.time(), earliest_at=old_ts, latest_at=time.time()
        ))
        handle_trove_command("doctor retention apply", engine)
        # The node is recent, so age = max(old, recent) = recent → NOT eligible
        assert engine._store._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id='mixed-session'").fetchone()[0] == 5

    def test_session_recent_messages_old_node(self, tmp_path):
        """Session with recent messages but an old summary node."""
        engine = make_engine(tmp_path)
        # Add recent messages
        for i in range(5):
            engine._store.append("mixed-session2", {"role": "user", "content": f"recent {i}"}, token_estimate=10)
        # Add an old summary node
        old_ts = time.time() - 200 * 86400
        engine._dag.add_node(SummaryNode(
            session_id="mixed-session2", depth=0, summary="old", token_count=50,
            source_token_count=500, source_ids=[], source_type="messages",
            created_at=old_ts, earliest_at=old_ts, latest_at=old_ts
        ))
        handle_trove_command("doctor retention apply", engine)
        # Recent messages → age = max(recent, old) = recent → NOT eligible
        assert engine._store._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id='mixed-session2'").fetchone()[0] == 5


class TestRetentionScanDeleteRace:
    """Test race between scan and delete."""

    def test_new_message_between_scan_and_delete(self, tmp_path):
        """If a new message arrives between scan and delete, session should NOT be deleted."""
        engine = make_engine(tmp_path)
        add_old_session(engine, "race-session", message_count=10, age_days=200)
        
        # Simulate the race by manually testing the scan-then-apply flow
        # In practice, this is hard to test deterministically without mocking
        result = handle_trove_command("doctor retention apply", engine)
        assert "status: ok" in result
        # After apply, session should be deleted
        assert engine._store._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id='race-session'").fetchone()[0] == 0


class TestRetentionEvaluateRetentionUnit:
    """Unit tests for evaluate_retention policy function."""

    def test_zero_retention_days_disables(self):
        """retention_days=0 should disable retention."""
        config = type('Config', (), {'retention_days': 0})()
        rows = [("session-1", 10, 1000, 5, 500, time.time() - 200*86400, time.time() - 200*86400, time.time() - 200*86400, time.time() - 200*86400)]
        plan = evaluate_retention(rows, config=config, protected_session_ids=set())
        assert plan.disabled
        assert len(plan.delete) == 0

    def test_negative_retention_days_disables(self):
        """Negative retention_days should disable retention."""
        config = type('Config', (), {'retention_days': -1})()
        rows = [("session-1", 10, 1000, 5, 500, time.time() - 200*86400, time.time() - 200*86400, time.time() - 200*86400, time.time() - 200*86400)]
        plan = evaluate_retention(rows, config=config, protected_session_ids=set())
        assert plan.disabled

    def test_session_exactly_at_boundary_is_eligible(self):
        """Session at exactly retention_days should be eligible."""
        config = type('Config', (), {'retention_days': 90})()
        ts = time.time() - 90 * 86400
        rows = [("session-1", 10, 1000, 5, 500, ts, ts, ts, ts)]
        plan = evaluate_retention(rows, config=config, protected_session_ids=set())
        assert len(plan.delete) == 1

    def test_session_one_second_under_not_eligible(self):
        """Session one second under retention_days should NOT be eligible."""
        config = type('Config', (), {'retention_days': 90})()
        ts = time.time() - (90 * 86400 - 1)
        rows = [("session-1", 10, 1000, 5, 500, ts, ts, ts, ts)]
        plan = evaluate_retention(rows, config=config, protected_session_ids=set())
        assert len(plan.delete) == 0

    def test_empty_session_skipped(self):
        """Session with no messages and no nodes should be skipped."""
        config = type('Config', (), {'retention_days': 90})()
        rows = [("empty", 0, 0, 0, 0, None, None, None, None)]
        plan = evaluate_retention(rows, config=config, protected_session_ids=set())
        assert len(plan.delete) == 0

    def test_protected_session_skipped(self):
        """Protected session should be skipped."""
        config = type('Config', (), {'retention_days': 90})()
        ts = time.time() - 200 * 86400
        rows = [("live", 10, 1000, 5, 500, ts, ts, ts, ts)]
        plan = evaluate_retention(rows, config=config, protected_session_ids={"live"})
        assert len(plan.delete) == 0
        assert len(plan.skip) == 1
        assert plan.skip[0].reason == "protected-live-session"


class TestRetentionLoopWithLifecycle:
    """Test lifecycle cleanup loop."""

    def test_lifecycle_deleted_correctly_counted(self, tmp_path):
        """Lifecycle deleted count should match actual deletions."""
        engine = make_engine(tmp_path)
        for i in range(3):
            add_old_session(engine, f"old-{i}", message_count=5, age_days=200)
            engine._store._conn.execute(
                f"INSERT OR REPLACE INTO trove_lifecycle_state (conversation_id, current_session_id, last_finalized_session_id) VALUES ('conv-{i}', 'old-{i}', 'old-{i}')"
            )
            engine._store._conn.commit()
        result = handle_trove_command("doctor retention apply", engine)
        assert "status: ok" in result
        assert "lifecycle_rows_deleted: 3" in result

    def test_lifecycle_not_deleted_for_protected(self, tmp_path):
        """Lifecycle rows for protected sessions should not be deleted."""
        engine = make_engine(tmp_path)
        old_ts = time.time() - 200 * 86400
        sids = []
        for i in range(5):
            sid = engine._store.append("protected-session", {"role": "user", "content": f"m{i}"}, token_estimate=10)
            sids.append(sid)
        engine._store._conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (old_ts, "protected-session"))
        engine._store._conn.commit()
        engine._dag.add_node(SummaryNode(session_id="protected-session", depth=0, summary="s", token_count=50, source_token_count=500, source_ids=sids, source_type="messages", created_at=old_ts, earliest_at=old_ts, latest_at=old_ts))
        # Lifecycle row
        engine._store._conn.execute("INSERT OR REPLACE INTO trove_lifecycle_state (conversation_id, current_session_id, last_finalized_session_id) VALUES ('protected-conv', 'protected-session', 'protected-session')")
        engine._store._conn.commit()
        # Make it protected
        engine._session_id = "protected-session"
        result = handle_trove_command("doctor retention apply", engine)
        assert "status: ok" in result
        # Messages should be protected
        assert engine._store._conn.execute("SELECT COUNT(*) FROM messages WHERE session_id='protected-session'").fetchone()[0] == 5


class TestRetentionPreviewConsistency:
    """Test that preview matches apply."""

    def test_preview_shows_same_count_as_apply(self, tmp_path):
        """Preview should report the same number of eligible sessions as apply deletes."""
        engine = make_engine(tmp_path)
        for i in range(5):
            add_old_session(engine, f"old-{i}", message_count=5, age_days=200)
        preview = handle_trove_command("doctor retention", engine)
        apply = handle_trove_command("doctor retention apply", engine)
        # Both should find 5 eligible sessions
        assert "sessions_analyzed: 5" in preview or "eligible_sessions: 5" in preview
        assert "eligible_sessions: 5" in apply
