"""Tests for the session retention policy and `/trove doctor retention apply`."""

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


# --- policy unit tests -------------------------------------------------------


def test_evaluate_retention_disabled_at_zero_days():
    rows = [("old", 5, 500, 1, 32, 1.0, 1.0, 1.0, 1.0)]
    plan = evaluate_retention(rows, config=TROVEConfig(retention_days=0), protected_session_ids=set())
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


# --- command surface ---------------------------------------------------------


def test_retention_apply_denied_when_flag_off(tmp_path):
    engine = _make_engine(tmp_path, retention_apply_enabled=False)
    _add_old_session(engine, "old-done")
    result = handle_trove_command("doctor retention apply", engine)
    assert "status: denied" in result
    assert "TROVE_RETENTION_APPLY_ENABLED" in result


def test_retention_apply_noop_when_days_zero(tmp_path):
    engine = _make_engine(tmp_path, retention_apply_enabled=True)  # flag is True by default now
    _add_old_session(engine, "old-done")
    result = handle_trove_command("doctor retention apply", engine)
    assert "status: ok" in result
    assert "eligible_sessions: 0" in result
    assert "nothing was deleted" in result
    # Raw messages untouched
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
    # FTS must not hold dead rows
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
    # Age the LIVE session artificially — it must still be protected.
    _add_old_session(engine, "live-session", messages=2)
    result = handle_trove_command("doctor retention apply", engine)
    assert "eligible_sessions: 0" in result
    conn = engine._store.connection
    remaining = conn.execute(
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
