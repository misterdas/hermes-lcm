"""Tests for TROVE guardrail C2: ingest-failure counters surfaced as
first-class error flags in trove_status, trove_doctor, and the /trove status
text surface.

Read-only observability — does NOT alter retry semantics in
_ingest_with_transient_retry.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from hermes_trove.command import handle_trove_command
from hermes_trove.config import TROVEConfig
from hermes_trove.engine import TROVEEngine


@pytest.fixture
def engine(tmp_path):
    config = TROVEConfig()
    config.fresh_tail_count = 4
    config.leaf_chunk_tokens = 100
    config.database_path = str(tmp_path / "trove_test.db")
    e = TROVEEngine(config=config)
    e._session_id = "test-session"
    e.context_length = 200_000
    e.threshold_tokens = int(200_000 * config.context_threshold)
    try:
        yield e
    finally:
        e.shutdown()


def _ingest_fail_n(engine: TROVEEngine, n: int) -> None:
    """Force ``n`` consecutive ingest failures via the engine path (not bypassing
    retry semantics — uses a non-retry-worthy OperationalError so the failure
    is recorded immediately by _record_ingest_failure)."""

    def boom(_messages):
        # disk I/O error is NOT in the retry-worthy set, so _record_ingest_failure
        # is invoked directly after the first attempt (no 0.05s retry sleep).
        raise sqlite3.OperationalError("disk I/O error")

    engine._ingest_messages = boom
    msgs = [{"role": "user", "content": "turn that cannot be persisted"}]
    for _ in range(n):
        engine.ingest(msgs)
    del engine._ingest_messages


# --- trove_status JSON surface -------------------------------------------------


def test_trove_status_healthy_engine_has_no_ingest_warning(engine):
    """Baseline: a healthy engine reports ingest_health with no warning field."""
    status = json.loads(engine.handle_tool_call("trove_status", {}))

    assert status["ingest_health"]["consecutive_failures"] == 0
    assert status["ingest_health"]["total_failures"] == 0
    assert "warning" not in status["ingest_health"]


def test_trove_status_ingest_warning_appears_at_three_consecutive_failures(engine):
    """Guardrail C2: once consecutive_failures >= 3, trove_status emits a
    first-class warning flag that makes a sustained fault operator-visible
    without requiring the operator to read logs."""
    _ingest_fail_n(engine, 3)

    status = json.loads(engine.handle_tool_call("trove_status", {}))

    assert status["ingest_health"]["consecutive_failures"] == 3
    assert status["ingest_health"]["total_failures"] == 3
    assert "warning" in status["ingest_health"]
    warning = status["ingest_health"]["warning"]
    assert "ingest is failing" in warning
    assert "messages may be LOST" in warning
    assert "OperationalError" in warning


def test_trove_status_ingest_warning_absent_below_threshold(engine):
    """1–2 consecutive failures do NOT set the ERROR warning (only the
    counters surface); the escalation threshold is >= 3."""
    _ingest_fail_n(engine, 2)

    status = json.loads(engine.handle_tool_call("trove_status", {}))

    assert status["ingest_health"]["consecutive_failures"] == 2
    assert status["ingest_health"]["total_failures"] == 2
    assert "warning" not in status["ingest_health"]


def test_trove_status_ingest_last_error_and_timestamp_present(engine):
    """last_error text + timestamp are surfaced so the operator can triage."""
    _ingest_fail_n(engine, 4)

    status = json.loads(engine.handle_tool_call("trove_status", {}))
    ingest = status["ingest_health"]

    assert "OperationalError" in ingest["last_error"]
    assert ingest["last_error_time"] > 0


# --- trove_doctor surface ------------------------------------------------------


def test_trove_doctor_healthy_engine_passes_ingest_health(engine):
    payload = json.loads(engine.handle_tool_call("trove_doctor", {}))
    ingest_check = next(c for c in payload["checks"] if c["check"] == "ingest_health")

    assert ingest_check["status"] == "pass"


def test_trove_doctor_consecutive_failures_flagged_fail(engine):
    _ingest_fail_n(engine, 3)

    payload = json.loads(engine.handle_tool_call("trove_doctor", {}))
    ingest_check = next(c for c in payload["checks"] if c["check"] == "ingest_health")

    assert ingest_check["status"] == "fail"
    assert ingest_check["detail"]["consecutive_failures"] == 3
    assert "OperationalError" in ingest_check["detail"]["last_error"]


def test_trove_doctor_guidance_for_failing_ingest(engine):
    """Guidance entry appears with actionable LOST-message triage text."""
    _ingest_fail_n(engine, 3)

    payload = json.loads(engine.handle_tool_call("trove_doctor", {}))
    guidance = payload["guidance"]

    ingest_guidance = [g for g in guidance if g["check"] == "ingest_health"]
    assert ingest_guidance, "expected ingest_health guidance entry"
    msg = ingest_guidance[0]["operator_action"]
    assert "messages may be LOST" in msg


# --- /trove status text surface ------------------------------------------------


def test_slash_status_text_healthy_engine_ingest_section(engine):
    """Text status includes an ingest section; healthy engine shows counts=0."""
    text = handle_trove_command("status", engine)

    assert "ingest_total_failures: 0" in text
    assert "ingest_consecutive_failures: 0" in text
    assert "ingest_health: ERROR" not in text


def test_slash_status_text_three_consecutive_shows_error(engine):
    _ingest_fail_n(engine, 3)

    text = handle_trove_command("status", engine)

    assert "ingest_consecutive_failures: 3" in text
    assert "ingest_health: ERROR" in text
    assert "messages may be LOST" in text
    assert "ingest_last_error:" in text


def test_slash_status_text_two_consecutive_no_error(engine):
    _ingest_fail_n(engine, 2)

    text = handle_trove_command("status", engine)

    assert "ingest_consecutive_failures: 2" in text
    assert "ingest_health: ERROR" not in text
    assert "ingest_health: WARN" in text
