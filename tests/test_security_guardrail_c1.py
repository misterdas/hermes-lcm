"""Tests for the PII-redaction-plus-backup security guardrail (C1).

The guardrail warns when BOTH conditions hold:
  1. ``sensitive_patterns_enabled`` is False (PII redaction is OFF).
  2. The active ``trove.db`` lives inside ``HERMES_HOME`` — the tree the
     Hermes backup channel ships to a remote repo.

It is read-only evidence: it never flips the redaction default and never
changes ingest behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

from hermes_trove import tools as trove_tools
from hermes_trove.command import handle_trove_command
from hermes_trove.config import TROVEConfig
from hermes_trove.diagnostics import (
    db_path_lives_under_hermes_home,
    security_backup_guardrail,
)
from hermes_trove.engine import TROVEEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(
    tmp_path: Path,
    *,
    sensitive_patterns_enabled: bool = False,
    database_under_home: bool = True,
    hermes_home: str | None = None,
) -> TROVEEngine:
    """Build an engine with controlled redaction + db-path placement.

    When ``database_under_home`` is True the trove.db is placed at
    ``{hermes_home}/trove.db`` (the default deployment layout that the
    Hermes backup channel captures). When False it is placed at
    ``{tmp_path}/trove.db`` — outside HERMES_HOME.
    """
    home = Path(hermes_home) if hermes_home else (tmp_path / "hermes_home")
    home.mkdir(parents=True, exist_ok=True)
    if database_under_home:
        db_path = home / "trove.db"
        cfg = TROVEConfig(sensitive_patterns_enabled=sensitive_patterns_enabled)
    else:
        db_path = tmp_path / "trove_outside.db"
        cfg = TROVEConfig(
            database_path=str(db_path),
            sensitive_patterns_enabled=sensitive_patterns_enabled,
        )
    engine = TROVEEngine(config=cfg, hermes_home=str(home))
    engine._session_id = "sec-test-session"
    engine._session_platform = "telegram"
    return engine


# ---------------------------------------------------------------------------
# Unit tests for the pure helper functions
# ---------------------------------------------------------------------------

class TestDbPathLivesUnderHermesHome:
    def test_true_when_db_is_under_home(self, tmp_path):
        engine = _make_engine(tmp_path, database_under_home=True)
        try:
            assert db_path_lives_under_hermes_home(engine) is True
        finally:
            engine.shutdown()

    def test_false_when_db_is_outside_home(self, tmp_path):
        engine = _make_engine(tmp_path, database_under_home=False)
        try:
            assert db_path_lives_under_hermes_home(engine) is False
        finally:
            engine.shutdown()

    def test_false_when_hermes_home_is_empty(self, tmp_path):
        cfg = TROVEConfig(database_path=str(tmp_path / "orphan.db"))
        engine = TROVEEngine(config=cfg, hermes_home="")
        try:
            assert db_path_lives_under_hermes_home(engine) is False
        finally:
            engine.shutdown()

    def test_false_when_store_missing(self):
        # No _store attribute at all — must not crash.
        engine = object.__new__(TROVEEngine)
        engine.__dict__["_hermes_home"] = ""
        engine.__dict__["_store"] = None
        assert db_path_lives_under_hermes_home(engine) is False


class TestSecurityBackupGuardrail:
    def test_warning_when_redaction_off_and_db_in_home(self, tmp_path):
        engine = _make_engine(
            tmp_path,
            sensitive_patterns_enabled=False,
            database_under_home=True,
        )
        try:
            result = security_backup_guardrail(engine)
            assert result["check"] == "pii_redaction_and_backup_guardrail"
            assert result["status"] == "warn"
            assert result["detail"]["redaction_off"] is True
            assert result["detail"]["db_in_hermes_home"] is True
            assert "warning" in result["detail"]
            assert "recommendation" in result["detail"]
            assert "TROVE_SENSITIVE_PATTERNS_ENABLED" in result["detail"]["recommendation"]
            assert "TROVE_DATABASE_PATH" in result["detail"]["recommendation"]
        finally:
            engine.shutdown()

    def test_pass_when_redaction_enabled(self, tmp_path):
        engine = _make_engine(
            tmp_path,
            sensitive_patterns_enabled=True,
            database_under_home=True,
        )
        try:
            result = security_backup_guardrail(engine)
            assert result["status"] == "pass"
            assert result["detail"]["redaction_off"] is False
            assert result["detail"]["db_in_hermes_home"] is True
        finally:
            engine.shutdown()

    def test_pass_when_db_outside_home(self, tmp_path):
        engine = _make_engine(
            tmp_path,
            sensitive_patterns_enabled=False,
            database_under_home=False,
        )
        try:
            result = security_backup_guardrail(engine)
            assert result["status"] == "pass"
            assert result["detail"]["redaction_off"] is True
            assert result["detail"]["db_in_hermes_home"] is False
        finally:
            engine.shutdown()

    def test_pass_when_both_mitigated(self, tmp_path):
        engine = _make_engine(
            tmp_path,
            sensitive_patterns_enabled=True,
            database_under_home=False,
        )
        try:
            result = security_backup_guardrail(engine)
            assert result["status"] == "pass"
        finally:
            engine.shutdown()

    def test_no_crash_when_engine_minimal(self):
        # Minimal engine-like object — must not raise.
        engine = object.__new__(TROVEEngine)
        engine.__dict__["_hermes_home"] = ""
        engine.__dict__["_store"] = None
        engine.__dict__["_config"] = None
        result = security_backup_guardrail(engine)
        assert result["status"] == "pass"

    def test_warns_when_config_is_none_redaction_off_and_db_in_home(self, tmp_path):
        # When config is None/missing, sensitive_patterns_enabled defaults to
        # False (redaction OFF), so with db inside home the guardrail warns.
        engine = _make_engine(tmp_path, database_under_home=True)
        try:
            engine._config = None
            result = security_backup_guardrail(engine)
            assert result["status"] == "warn"
            assert result["detail"]["redaction_off"] is True
            assert result["detail"]["db_in_hermes_home"] is True
        finally:
            engine.shutdown()


# ---------------------------------------------------------------------------
# Integration tests: guardrail surfaces in trove_doctor / /trove doctor
# ---------------------------------------------------------------------------

class TestGuardrailInDoctor:
    def test_tool_warns_when_redaction_off_and_db_in_home(self, tmp_path):
        engine = _make_engine(
            tmp_path,
            sensitive_patterns_enabled=False,
            database_under_home=True,
        )
        try:
            doctor = json.loads(trove_tools.trove_doctor({}, engine=engine))
            guardrail = next(
                c for c in doctor["checks"]
                if c["check"] == "pii_redaction_and_backup_guardrail"
            )
            assert guardrail["status"] == "warn"
            assert guardrail["detail"]["redaction_off"] is True
            assert guardrail["detail"]["db_in_hermes_home"] is True
            # Overall must reflect the warning.
            assert doctor["overall"] == "warnings"
            # Guidance must include the guardrail.
            guidance = doctor["guidance"]
            assert any(
                item["check"] == "pii_redaction_and_backup_guardrail"
                for item in guidance
            )
        finally:
            engine.shutdown()

    def test_tool_passes_when_redaction_enabled(self, tmp_path):
        engine = _make_engine(
            tmp_path,
            sensitive_patterns_enabled=True,
            database_under_home=True,
        )
        try:
            doctor = json.loads(trove_tools.trove_doctor({}, engine=engine))
            guardrail = next(
                c for c in doctor["checks"]
                if c["check"] == "pii_redaction_and_backup_guardrail"
            )
            assert guardrail["status"] == "pass"
        finally:
            engine.shutdown()

    def test_tool_passes_when_db_outside_home(self, tmp_path):
        engine = _make_engine(
            tmp_path,
            sensitive_patterns_enabled=False,
            database_under_home=False,
        )
        try:
            doctor = json.loads(trove_tools.trove_doctor({}, engine=engine))
            guardrail = next(
                c for c in doctor["checks"]
                if c["check"] == "pii_redaction_and_backup_guardrail"
            )
            assert guardrail["status"] == "pass"
        finally:
            engine.shutdown()

    def test_text_warns_when_redaction_off_and_db_in_home(self, tmp_path):
        engine = _make_engine(
            tmp_path,
            sensitive_patterns_enabled=False,
            database_under_home=True,
        )
        try:
            result = handle_trove_command("doctor", engine)
            assert "TROVE doctor" in result
            # The guardrail warning surfaces in triage_guidance.
            assert "pii_redaction_and_backup_guardrail" in result
            assert "TROVE_SENSITIVE_PATTERNS_ENABLED" in result
        finally:
            engine.shutdown()

    def test_text_no_warning_when_redaction_enabled(self, tmp_path):
        engine = _make_engine(
            tmp_path,
            sensitive_patterns_enabled=True,
            database_under_home=True,
        )
        try:
            result = handle_trove_command("doctor", engine)
            assert "TROVE doctor" in result
            # Guardrail must NOT appear in triage_guidance when passing.
            assert "pii_redaction_and_backup_guardrail" not in result
        finally:
            engine.shutdown()

    def test_text_no_warning_when_db_outside_home(self, tmp_path):
        engine = _make_engine(
            tmp_path,
            sensitive_patterns_enabled=False,
            database_under_home=False,
        )
        try:
            result = handle_trove_command("doctor", engine)
            assert "TROVE doctor" in result
            assert "pii_redaction_and_backup_guardrail" not in result
        finally:
            engine.shutdown()

    def test_no_crash_when_nothing_configured(self, tmp_path):
        # Default engine: redaction OFF but db outside home — guardrail pass.
        engine = _make_engine(
            tmp_path,
            sensitive_patterns_enabled=False,
            database_under_home=False,
        )
        try:
            doctor = json.loads(trove_tools.trove_doctor({}, engine=engine))
            assert "overall" in doctor
            assert "checks" in doctor
        finally:
            engine.shutdown()
