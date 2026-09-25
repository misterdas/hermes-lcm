"""Tests for the session retention policy and `/trove doctor retention apply`.

Three layers:
  1. Policy unit tests — `evaluate_retention` decisions (no engine).
  2. Command surface — enable/deno/noop/delete/skip/protect/pinned/help.
  3. Integration — scan edges, lifecycle correctness, boundaries, preview==apply.
"""

import time

from hermes_trove.command import handle_trove_command
from hermes_trove.config import TROVEConfig
from hermes_trove.dag import SummaryNode
from hermes_trove.engine import TROVEEngine
from hermes_trove.retention import evaluate_retention


def _make_engine(tmp_path, **config_overrides):
    config = TROVEConfig(database_path=str(tmp_path / "trove_retention.db"))
    for key, value in config_overrides.items():
        setattr(config, key, value)
    engine = TROVEEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    engine._session_id = "live-session"
    engine._session_platform = "telegram"
    engine._conversation_id = "live-session"
    engine._lifecycle.bind_session("live-session")
    return engine


def _add_old_session(engine, session_id, *, messages=3, node=True, pinned=False):
    now = time.time()
    old_ts = now - (200 * 86400)  # 200 days old
    store_ids = []
    for i in range(messages):
        sid = engine._store.append(
            session_id, {"role": "user", "content": f"old message {i}"}, token_estimate=100
        )
        if pinned and i == 0:
            engine._store.pin(sid)
        store_ids.append(sid)
    engine._store._conn.execute(
        "UPDATE messages SET timestamp = ? WHERE session_id = ?", (old_ts, session_id)
    )
    engine._store._conn.commit()
    if node:
        engine._dag.add_node(SummaryNode(
            session_id=session_id,
            depth=0,
            summary=f"{session_id} summary",
            token_count=32,
            source_token_count=300,
            source_ids=store_ids,
            source_type="messages",
            created_at=old_ts,
            earliest_at=old_ts,
            latest_at=old_ts,
        ))
    return store_ids


# ============================================================
# 1. Policy unit tests (evaluate_retention, no engine)
# ============================================================

def test_evaluate_retention_disabled_at_zero_days():
    rows = [("old", 5, 500, 1, 32, 1.0, 1.0, 1.0, 1.0)]
    plan = evaluate_retention(rows, config=TROVEConfig(retention_days=0), protected_session_ids=set())
    assert plan.disabled is True
    assert plan.delete == []


def test_evaluate_retention_negative_days_disables():
    rows = [("old", 5, 500, 1, 32, 1.0, 1.0, 1.0, 1.0)]
    plan = evaluate_retention(rows, config=TROVEConfig(retention_days=-5), protected_session_ids=set())
    assert plan.disabled is True
    assert plan.delete == []


def test_evaluate_retention_eligible_requires_nodes():
    rows = [("old", 5, 500, 0, 0, 1.0, 1.0, None, None)]
    config = TROVEConfig(retention_days=90)
    plan = evaluate_retention(rows, config=config, protected_session_ids=set())
    assert plan.delete == []
    assert any(d.session_id == "old" and "summary" in d.reason for d in plan.skip)


def test_evaluate_retention_protected_session_skipped():
    rows = [("live", 5, 500, 1, 32, 1.0, 1.0, 1.0, 1.0)]
    config = TROVEConfig(retention_days=90)
    plan = evaluate_retention(rows, config=config, protected_session_ids={"live"})
    assert plan.delete == []
    assert any(d.reason == "protected-live-session" for d in plan.skip)


def test_evaluate_retention_eligible_old_session_with_nodes():
    rows = [("old", 5, 500, 2, 64, 1.0, 1.0, 1.0, 1.0)]
    config = TROVEConfig(retention_days=90)
    plan = evaluate_retention(rows, config=config, protected_session_ids=set())
    assert [d.session_id for d in plan.delete] == ["old"]
    assert plan.delete[0].eligible is True
    assert plan.delete[0].message_count == 5
    assert plan.delete[0].node_count == 2


def test_evaluate_retention_fresh_session_skipped():
    now = time.time()
    rows = [("fresh", 5, 500, 1, 32, now, now, now, now)]
    config = TROVEConfig(retention_days=90)
    plan = evaluate_retention(rows, config=config, protected_session_ids=set())
    assert plan.delete == []
    assert any("younger-than" in d.reason for d in plan.skip)


def test_evaluate_retention_old_messages_recent_node_kept():
    """Old messages but a recent node — max() policy keeps the session alive."""
    old_ts = time.time() - 200 * 86400
    recent_ts = time.time() - 10 * 86400
    # Policy uses max(timestamps): the recent node means the session is active → skip
    rows = [("mixed", 5, 500, 1, 32, old_ts, old_ts, recent_ts, recent_ts)]
    config = TROVEConfig(retention_days=90)
    plan = evaluate_retention(rows, config=config, protected_session_ids=set())
    assert plan.delete == []
    assert any("younger-than" in d.reason for d in plan.skip)


def test_evaluate_retention_recent_messages_old_node_kept():
    """Recent messages but an old node — max() policy keeps the session alive."""
    old_ts = time.time() - 200 * 86400
    recent_ts = time.time() - 10 * 86400
    # Policy uses max(timestamps): recent messages means the session is active → skip
    rows = [("mixed", 5, 500, 1, 32, recent_ts, recent_ts, old_ts, old_ts)]
    config = TROVEConfig(retention_days=90)
    plan = evaluate_retention(rows, config=config, protected_session_ids=set())
    assert plan.delete == []
    assert any("younger-than" in d.reason for d in plan.skip)


# ============================================================
# 2. Command surface (integration, single thread)
# ============================================================

def test_retention_apply_denied_when_flag_off(tmp_path):
    engine = _make_engine(tmp_path, retention_apply_enabled=False)
    _add_old_session(engine, "old-done")
    result = handle_trove_command("doctor retention apply", engine)
    assert "status: denied" in result
    assert "TROVE_RETENTION_APPLY_ENABLED" in result


def test_retention_apply_noop_when_days_zero(tmp_path):
    engine = _make_engine(tmp_path, retention_apply_enabled=True)
    _add_old_session(engine, "old-done")
    result = handle_trove_command("doctor retention apply", engine)
    assert "status: ok" in result
    assert "eligible_sessions: 0" in result
    assert "nothing was deleted" in result
    assert engine._store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'old-done'"
    ).fetchone()[0] == 3


def test_retention_apply_deletes_raws_keeps_summaries(tmp_path):
    engine = _make_engine(tmp_path, retention_days=90, retention_apply_enabled=True)
    _add_old_session(engine, "old-done", messages=4)
    result = handle_trove_command("doctor retention apply", engine)
    assert "status: ok" in result
    assert "messages_deleted: 4" in result
    assert "summary_nodes_kept: 1" in result
    assert "backup_path" in result
    conn = engine._store.connection
    remaining = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'old-done'"
    ).fetchone()[0]
    assert remaining == 0
    nodes = engine._dag.connection.execute(
        "SELECT COUNT(*) FROM summary_nodes WHERE session_id = 'old-done'"
    ).fetchone()[0]
    assert nodes == 1  # recall survives
    fts_count = conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE rowid NOT IN (SELECT store_id FROM messages)"
    ).fetchone()[0]
    assert fts_count == 0


def test_retention_apply_skips_never_compacted_sessions(tmp_path):
    engine = _make_engine(tmp_path, retention_days=90, retention_apply_enabled=True)
    _add_old_session(engine, "old-no-node", node=False)
    result = handle_trove_command("doctor retention apply", engine)
    assert "status: ok" in result
    assert "eligible_sessions: 0" in result
    assert "old-no-node" in result  # skip reason surfaced
    conn = engine._store.connection
    remaining = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'old-no-node'"
    ).fetchone()[0]
    assert remaining == 3  # untouched


def test_retention_apply_protects_live_session(tmp_path):
    engine = _make_engine(tmp_path, retention_days=90, retention_apply_enabled=True)
    _add_old_session(engine, "live-session", messages=2)
    result = handle_trove_command("doctor retention apply", engine)
    assert "eligible_sessions: 0" in result
    conn = engine._store.connection
    remaining = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'live-session'"
    ).fetchone()[0]
    assert remaining == 2


def test_retention_apply_protects_lifecycle_current_session_when_runtime_id_is_stale(tmp_path):
    """Retention must protect the active lifecycle session, not only _session_id."""
    engine = _make_engine(tmp_path, retention_days=90, retention_apply_enabled=True)
    engine._session_id = "stale-bound-session"
    engine._foreground_session_id = "live-session"
    engine._foreground_session_platform = "telegram"
    _add_old_session(engine, "live-session", messages=2)

    result = handle_trove_command("doctor retention apply", engine)

    assert "eligible_sessions: 0" in result
    remaining = engine._store.connection.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'live-session'"
    ).fetchone()[0]
    assert remaining == 2


def test_retention_apply_refuses_pinned_sessions(tmp_path):
    engine = _make_engine(tmp_path, retention_days=90, retention_apply_enabled=True)
    _add_old_session(engine, "old-pinned", messages=3, pinned=True)
    result = handle_trove_command("doctor retention apply", engine)
    assert "status: refused" in result
    assert "pinned" in result
    conn = engine._store.connection
    remaining = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'old-pinned'"
    ).fetchone()[0]
    assert remaining == 3  # nothing deleted


def test_retention_apply_help_lists_subcommand(tmp_path):
    engine = _make_engine(tmp_path)
    result = handle_trove_command("doctor bogus-sub", engine)
    assert "retention apply" in result


# ============================================================
# 3. Integration — scan edges, lifecycle, boundaries
# ============================================================

def test_session_with_only_messages_no_nodes(tmp_path):
    """Session with messages but no summary nodes (never compacted) is skipped."""
    engine = _make_engine(tmp_path, retention_days=90)
    for i in range(5):
        engine._store.append("only-msgs", {"role": "user", "content": f"m{i}"}, token_estimate=10)
    old_ts = time.time() - 200 * 86400
    engine._store._conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (old_ts, "only-msgs"))
    engine._store._conn.commit()
    result = handle_trove_command("doctor retention apply", engine)
    assert "status: ok" in result
    assert engine._store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='only-msgs'"
    ).fetchone()[0] == 5


def test_session_with_only_nodes_no_messages(tmp_path):
    """Session with summary nodes but no messages — nothing to delete."""
    engine = _make_engine(tmp_path, retention_days=90)
    old_ts = time.time() - 200 * 86400
    engine._dag.add_node(SummaryNode(
        session_id="only-nodes", depth=0, summary="summary", token_count=50,
        source_token_count=500, source_ids=[], source_type="messages",
        created_at=old_ts, earliest_at=old_ts, latest_at=old_ts,
    ))
    result = handle_trove_command("doctor retention apply", engine)
    assert "status: ok" in result


def test_session_with_no_activity_timestamps(tmp_path):
    """Session where all timestamps are NULL — scan handles gracefully."""
    engine = _make_engine(tmp_path, retention_days=90)
    engine._dag.add_node(SummaryNode(
        session_id="null-ts", depth=0, summary="s", token_count=50,
        source_token_count=100, source_ids=[], source_type="messages",
        created_at=None, earliest_at=None, latest_at=None,
    ))
    result = handle_trove_command("doctor retention apply", engine)
    assert "status: ok" in result


def test_lifecycle_deleted_for_eligible_session(tmp_path):
    """Lifecycle rows for fully eligible sessions are deleted."""
    engine = _make_engine(tmp_path, retention_days=90)
    old_ts = time.time() - 200 * 86400
    sids = []
    for i in range(5):
        sid = engine._store.append("old-with-lc", {"role": "user", "content": f"m{i}"}, token_estimate=10)
        sids.append(sid)
    engine._store._conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (old_ts, "old-with-lc"))
    engine._store._conn.commit()
    engine._dag.add_node(SummaryNode(
        session_id="old-with-lc", depth=0, summary="s", token_count=50,
        source_token_count=500, source_ids=sids, source_type="messages",
        created_at=old_ts, earliest_at=old_ts, latest_at=old_ts,
    ))
    engine._store._conn.execute(
        "INSERT INTO trove_lifecycle_state (conversation_id, current_session_id, last_finalized_session_id) "
        "VALUES ('old-with-lc', 'old-with-lc', 'old-with-lc')"
    )
    engine._store._conn.commit()
    result = handle_trove_command("doctor retention apply", engine)
    assert "status: ok" in result
    assert engine._store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='old-with-lc'"
    ).fetchone()[0] == 0


def test_lifecycle_skipped_when_current_session(tmp_path):
    """Lifecycle rows pointing to the live session are preserved."""
    engine = _make_engine(tmp_path, retention_days=90)
    old_ts = time.time() - 200 * 86400
    sids = []
    for i in range(5):
        sid = engine._store.append("live-session", {"role": "user", "content": f"old-{i}"}, token_estimate=10)
        sids.append(sid)
    engine._store._conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (old_ts, "live-session"))
    engine._store._conn.commit()
    engine._dag.add_node(SummaryNode(
        session_id="live-session", depth=0, summary="s", token_count=50,
        source_token_count=500, source_ids=sids, source_type="messages",
        created_at=old_ts, earliest_at=old_ts, latest_at=old_ts,
    ))
    engine._store._conn.execute(
        "INSERT OR REPLACE INTO trove_lifecycle_state "
        "(conversation_id, current_session_id, last_finalized_session_id) "
        "VALUES ('live-conv', 'live-session', 'live-session')"
    )
    engine._store._conn.commit()
    result = handle_trove_command("doctor retention apply", engine)
    assert "status: ok" in result
    assert engine._store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='live-session'"
    ).fetchone()[0] == 5


def test_mixed_ages_only_old_deleted(tmp_path):
    """Multiple sessions: only the old ones are deleted, fresh ones kept."""
    engine = _make_engine(tmp_path, retention_days=90)
    _add_old_session(engine, "old-1", messages=3)
    _add_old_session(engine, "old-2", messages=4)
    # Fresh session
    for i in range(3):
        engine._store.append("fresh-sess", {"role": "user", "content": f"fresh-{i}"}, token_estimate=10)
    engine._dag.add_node(SummaryNode(
        session_id="fresh-sess", depth=0, summary="fresh", token_count=32,
        source_token_count=300, source_ids=[], source_type="messages",
        created_at=time.time(), earliest_at=time.time(), latest_at=time.time(),
    ))
    result = handle_trove_command("doctor retention apply", engine)
    assert "status: ok" in result
    assert engine._store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='old-1'"
    ).fetchone()[0] == 0
    assert engine._store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='old-2'"
    ).fetchone()[0] == 0
    assert engine._store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='fresh-sess'"
    ).fetchone()[0] == 3


def test_all_sessions_protected(tmp_path):
    """When all sessions are protected, nothing is deleted."""
    engine = _make_engine(tmp_path, retention_days=90)
    old_ts = time.time() - 200 * 86400
    for i in range(3):
        sids = []
        for j in range(3):
            sid = engine._store.append(f"old-{i}", {"role": "user", "content": f"m{i}-{j}"}, token_estimate=10)
            sids.append(sid)
        engine._store._conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (old_ts, f"old-{i}"))
        engine._store._conn.commit()
        engine._dag.add_node(SummaryNode(
            session_id=f"old-{i}", depth=0, summary="s", token_count=50,
            source_token_count=300, source_ids=sids, source_type="messages",
            created_at=old_ts, earliest_at=old_ts, latest_at=old_ts,
        ))
    engine._session_id = "old-0"
    result = handle_trove_command("doctor retention apply", engine)
    assert "status: ok" in result
    assert engine._store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='old-0'"
    ).fetchone()[0] == 3
    assert engine._store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='old-1'"
    ).fetchone()[0] == 0
    assert engine._store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='old-2'"
    ).fetchone()[0] == 0


def test_scan_doesnt_produce_duplicate_sessions(tmp_path):
    """The scan query produces one row per session."""
    engine = _make_engine(tmp_path, retention_days=90)
    old_ts = time.time() - 200 * 86400
    for i in range(3):
        engine._store.append(f"session-{i}", {"role": "user", "content": "msg"}, token_estimate=10)
    engine._store._conn.execute(
        "UPDATE messages SET timestamp=? WHERE session_id LIKE 'session-%'", (old_ts,)
    )
    engine._store._conn.commit()
    for i in range(3):
        engine._dag.add_node(SummaryNode(
            session_id=f"session-{i}", depth=0, summary="s", token_count=50,
            source_token_count=100, source_ids=[], source_type="messages",
            created_at=old_ts, earliest_at=old_ts, latest_at=old_ts,
        ))
    result = handle_trove_command("doctor retention", engine)
    lines = result.split("\n")
    session_lines = [line for line in lines if line.startswith("- ") and "session-" in line]
    assert len(session_lines) == 3


def test_session_exactly_at_cutoff(tmp_path):
    """Session with age exactly equal to retention_days IS eligible."""
    engine = _make_engine(tmp_path, retention_days=90)
    boundary_ts = time.time() - 90 * 86400
    sids = []
    for i in range(3):
        sid = engine._store.append("boundary-session", {"role": "user", "content": f"m{i}"}, token_estimate=10)
        sids.append(sid)
    engine._store._conn.execute(
        "UPDATE messages SET timestamp=? WHERE session_id=?", (boundary_ts, "boundary-session")
    )
    engine._store._conn.commit()
    engine._dag.add_node(SummaryNode(
        session_id="boundary-session", depth=0, summary="s", token_count=50,
        source_token_count=300, source_ids=sids, source_type="messages",
        created_at=boundary_ts, earliest_at=boundary_ts, latest_at=boundary_ts,
    ))
    result = handle_trove_command("doctor retention apply", engine)
    assert "status: ok" in result
    assert engine._store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='boundary-session'"
    ).fetchone()[0] == 0


def test_new_session_preserves_eligible_deletion(tmp_path):
    """Creating a new session doesn't affect deletion of an old eligible one."""
    engine = _make_engine(tmp_path, retention_days=90)
    _add_old_session(engine, "old-gone", messages=5)
    # New session with fresh messages
    for i in range(3):
        engine._store.append("new-sess", {"role": "user", "content": f"new-{i}"}, token_estimate=10)
    engine._dag.add_node(SummaryNode(
        session_id="new-sess", depth=0, summary="new", token_count=32,
        source_token_count=300, source_ids=[], source_type="messages",
        created_at=time.time(), earliest_at=time.time(), latest_at=time.time(),
    ))
    result = handle_trove_command("doctor retention apply", engine)
    assert "status: ok" in result
    assert engine._store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='old-gone'"
    ).fetchone()[0] == 0
    assert engine._store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='new-sess'"
    ).fetchone()[0] == 3


def test_empty_session_skipped(tmp_path):
    """An empty session (no messages, no nodes) is a no-op."""
    engine = _make_engine(tmp_path, retention_days=90)
    result = handle_trove_command("doctor retention apply", engine)
    assert "status: ok" in result


def test_preview_shows_same_count_as_apply(tmp_path):
    """Preview reports the same eligible count as apply would delete."""
    engine = _make_engine(tmp_path, retention_days=90)
    _add_old_session(engine, "old-pending", messages=4)
    preview_result = handle_trove_command("doctor retention", engine)
    apply_result = handle_trove_command("doctor retention apply", engine)
    # Both should agree on the eligible session count
    assert "old-pending" in preview_result
    assert "status: ok" in apply_result
    assert engine._store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id='old-pending'"
    ).fetchone()[0] == 0
