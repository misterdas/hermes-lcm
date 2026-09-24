"""Shared read-only diagnostic helpers for TROVE tools and commands."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DOCTOR_ACTION_SAFE_IGNORE = "safe/ignore"
DOCTOR_ACTION_INSPECT = "inspect"
DOCTOR_ACTION_BACKUP_FIRST_CLEANUP = "backup-first cleanup"


def _enforce_state_db_containment(path: Path, *, description: str) -> Path:
    resolved = path.expanduser().resolve()
    env_base = os.environ.get("TROVE_HERMES_BASE_DIR")
    if env_base:
        allowed_base = Path(env_base).expanduser().resolve()
        try:
            resolved.relative_to(allowed_base)
        except ValueError:
            raise ValueError(
                f"{description} resolves to {resolved} which is not within allowed base {allowed_base}"
            )
    return resolved


def state_db_path_for_engine(engine: Any) -> Path:
    """Return the Hermes state database path for an TROVE engine.

    The path is read-only diagnostic input. When ``TROVE_HERMES_BASE_DIR`` is
    configured, enforce the same containment guard for all diagnostic surfaces.
    """
    hermes_home = getattr(engine, "_hermes_home", "") or ""
    if hermes_home:
        return _enforce_state_db_containment(
            Path(hermes_home) / "state.db",
            description=f"hermes_home {hermes_home}",
        )
    db_path = Path(getattr(engine._store, "db_path", Path.home() / ".hermes" / "trove.db"))
    return _enforce_state_db_containment(
        db_path.parent / "state.db",
        description=f"state database fallback from TROVE database {db_path}",
    )


def has_lifecycle_fragmentation(stats: dict[str, Any]) -> bool:
    """Return whether lifecycle diagnostics should be treated as warning evidence.

    Retained-history drift is intentionally read-only diagnostic context. Keep the
    doctor warning for concrete operator action (empty lifecycle rows that the
    explicit backup-first cleanup path can prune) or diagnostic unreadability, but
    do not make overall health unhealthy solely because historical TROVE/state
    indexes no longer agree.
    """
    empty_lifecycle_rows = int(stats.get("empty_lifecycle_rows", 0) or 0)
    return empty_lifecycle_rows > 0 or (
        bool(stats.get("state_db_checked")) and bool(stats.get("state_db_error"))
    )


def doctor_guidance_for_check(check: dict[str, Any]) -> dict[str, Any] | None:
    """Return operator triage guidance for one trove_doctor check.

    Guidance is deliberately conservative: most warning classes are inspect-only
    evidence, and any mutation path is framed as preview/backup/apply rather than
    implied automatic cleanup.
    """
    status = str(check.get("status") or "")
    if status not in {"warn", "fail"}:
        return None

    name = str(check.get("check") or "unknown")
    detail = check.get("detail")
    action = DOCTOR_ACTION_INSPECT
    command = "inspect the reported detail and confirm the active HERMES_HOME/TROVE_DATABASE_PATH"
    warning_only = False
    rationale = "operator review required before changing persisted TROVE state"

    if name == "database_integrity":
        command = "stop and inspect the SQLite database path; restore from backup if integrity_check is not ok"
    elif name == "schema_core_tables":
        command = "verify HERMES_HOME/TROVE_DATABASE_PATH points at the intended TROVE database before repair or restore"
    elif name in {"messages_fts_integrity", "nodes_fts_integrity", "fts_index_sync"}:
        if status == "warn" and isinstance(detail, dict) and detail.get("status") == "unchecked":
            action = DOCTOR_ACTION_INSPECT
            command = "rerun `/trove doctor` with read-write SQLite access if a deep FTS integrity result is needed"
            warning_only = True
            rationale = "the deep FTS check could not run, but this is not evidence that the index is corrupt"
        else:
            action = DOCTOR_ACTION_BACKUP_FIRST_CLEANUP
            command = "run `/trove doctor repair` first; if it still recommends repair, run `/trove backup` before `/trove doctor repair apply`"
            rationale = "FTS repair is rebuildable, but it still mutates SQLite indexes"
    elif name == "sqlite_storage":
        command = "inspect journal/quick_check output and database/WAL size; restore from backup if SQLite reports corruption"
    elif name == "payload_storage":
        missing_refs = 0
        heartbeat_rows = 0
        suspicious_rows = 0
        if isinstance(detail, dict):
            missing_refs = int(detail.get("externalized_payload_refs_missing", 0) or 0)
            heartbeat_rows = len(detail.get("heartbeat_noise_rows") or [])
            suspicious_rows = sum(
                len(detail.get(key) or [])
                for key in (
                    "suspicious_data_uri_content_rows",
                    "suspicious_data_uri_tool_calls_rows",
                    "suspicious_base64_like_rows",
                    "suspicious_repetitive_assistant_rows",
                )
            )
        if status == "warn" and heartbeat_rows and not missing_refs and not suspicious_rows:
            action = DOCTOR_ACTION_SAFE_IGNORE
            command = "safe to ignore unless heartbeat/progress noise is crowding useful recall; consider message/session filters for future rows"
            rationale = "heartbeat rows are read-only noise diagnostics, not corruption"
        else:
            command = "inspect payload rows/refs; restore missing externalized payload files from backup before deleting or rewriting anything"
            if status == "warn":
                warning_only = True
                rationale = "payload warnings may represent preserved user/tool data"
            else:
                rationale = "payload diagnostic failures mean doctor could not read storage risk state reliably"
    elif name == "sensitive_pattern_handling":
        command = "inspect TROVE_SENSITIVE_PATTERNS settings; remove unknown names or configure supported catalog entries"
    elif name == "orphaned_dag_nodes":
        command = "inspect affected DAG/source IDs; do not auto-delete summaries without confirming recall impact"
        if status == "warn":
            warning_only = True
        else:
            rationale = "DAG diagnostic failures mean doctor could not read summary/source state reliably"
    elif name == "summary_quality":
        command = "inspect worst_nodes and retrieval behavior; treat as summary quality evidence, not cleanup input"
        if status == "warn":
            warning_only = True
        else:
            rationale = "summary-quality diagnostic failures mean doctor could not read DAG quality state reliably"
    elif name == "config_validation":
        command = "inspect TROVE_* environment/config values and adjust only intentional operator overrides"
    elif name == "source_lineage_hygiene" and status == "warn":
        action = DOCTOR_ACTION_SAFE_IGNORE
        command = "safe to ignore legacy blank-source observations; use `/trove doctor source` only when you intentionally want backup-first normalization"
        rationale = "legacy blank sources are normalized to unknown for compatibility"
    elif name == "source_lineage_hygiene":
        command = "inspect source-lineage diagnostics and SQLite read errors before running any source normalization workflow"
        rationale = "source-lineage failures indicate the doctor could not read attribution state reliably"
    elif name == "lifecycle_fragmentation":
        command = "inspect lifecycle categories; only use explicit backup-first lifecycle cleanup for empty lifecycle rows"
        if status == "warn":
            warning_only = True
            rationale = "not every lifecycle/state mismatch is harmful or safe to mutate"
        else:
            rationale = "lifecycle diagnostic failures mean doctor could not read session lifecycle state reliably"
    elif name == "context_pressure":
        action = DOCTOR_ACTION_SAFE_IGNORE
        command = "safe to ignore if compaction proceeds normally; inspect trove_status only if pressure stays high or compaction loops"
        warning_only = True
        rationale = "context pressure is an operating state, not persisted-state corruption"
    elif name == "ingest_health":
        last_error = ""
        if isinstance(detail, dict):
            last_error = str(detail.get("last_error", "") or "")
        if status == "fail":
            command = "ingest is failing — messages may be LOST; inspect disk/storage and last_error above; restore trove.db from backup if storage is healthy but TROVE still cannot persist"
            rationale = "consecutive ingest failures mean the lossless guarantee is actively breaking"
            warning_only = False
        else:
            command = "past ingest failures recovered; verify no messages were lost and monitor trove_status"
            action = DOCTOR_ACTION_SAFE_IGNORE
            warning_only = True
            rationale = "ingest recovered; failures were transient"
    elif name == "cleanup_candidates":
        action = DOCTOR_ACTION_BACKUP_FIRST_CLEANUP
        command = "run `/trove doctor clean` first; if candidates are expected junk/noise, run `/trove backup` before `/trove doctor clean apply`"
        rationale = "candidate cleanup deletes rows and must stay preview-and-backup gated"
    elif name == "pii_redaction_and_backup_guardrail":
        action = DOCTOR_ACTION_INSPECT
        command = "enable TROVE_SENSITIVE_PATTERNS_ENABLED=1, OR move trove.db outside HERMES_HOME via TROVE_DATABASE_PATH, OR exclude trove.db from the Hermes backup channel"
        warning_only = True
        rationale = "PII redaction is OFF and trove.db lives inside HERMES_HOME, which the Hermes backup channel ships to a remote repo — unredacted secrets may leak"

    return {
        "check": name,
        "status": status,
        "action": action,
        "operator_action": command,
        "warning_only": warning_only,
        "rationale": rationale,
    }


def _resolve_db_path_for_engine(engine: Any) -> Path:
    """Return the resolved trove.db path the engine's store is bound to.

    Read-only diagnostic input — same containment guard as the state-db path
    helper so a deployment that pins ``TROVE_HERMES_BASE_DIR`` cannot have its
    database escape into an arbitrary path via a symbolic link or ``~/``
    expansion.
    """
    db_path = Path(getattr(getattr(engine, "_store", None), "db_path", ""))
    if not db_path or str(db_path) == ":memory:":
        return Path()
    return _enforce_state_db_containment(db_path, description=f"TROVE database {db_path}")


def _resolve_hermes_home_for_engine(engine: Any) -> Path:
    """Return the resolved HERMES_HOME the engine is bound to."""
    hermes_home = getattr(engine, "_hermes_home", "") or ""
    if not hermes_home:
        return Path()
    return _enforce_state_db_containment(
        Path(hermes_home),
        description=f"hermes_home {hermes_home}",
    )


def db_path_lives_under_hermes_home(engine: Any) -> bool:
    """Return whether the active trove.db resides inside HERMES_HOME.

    This is the structural precondition for the Hermes backup channel (which
    ships ``~/.hermes`` to a remote repo) to capture unredacted secrets from
    the trove database. When the operator pins ``TROVE_DATABASE_PATH`` to a
    path outside HERMES_HOME, the precondition is false and the guardrail is
    not applicable.
    """
    db_path = _resolve_db_path_for_engine(engine)
    hermes_home = _resolve_hermes_home_for_engine(engine)
    if not db_path or not hermes_home:
        return False
    try:
        db_path.relative_to(hermes_home)
        return True
    except ValueError:
        return False


def security_backup_guardrail(engine: Any) -> dict[str, Any]:
    """Build the read-only PII-plus-backup guardrail check.

    The guardrail fires only when BOTH of the following hold:
      1. ``sensitive_patterns_enabled`` is False (PII redaction is OFF).
      2. The active ``trove.db`` lives inside ``HERMES_HOME`` — the tree the
         Hermes backup channel ships to a remote repo.

    It is deliberately conservative: the default deployment stores
    ``trove.db`` at ``~/.hermes/trove.db`` with redaction OFF, which means
    the operator is one ``hermes backup`` away from pushing unredacted API
    keys, bearer tokens, and private keys to a (possibly public) remote.

    The finding is read-only evidence — it never flips the redaction default
    and never changes ingest behavior.
    """
    config = getattr(engine, "_config", None)
    redaction_off = not bool(getattr(config, "sensitive_patterns_enabled", False))
    db_in_home = db_path_lives_under_hermes_home(engine)
    hermes_home = str(_resolve_hermes_home_for_engine(engine) or "")
    db_path = str(_resolve_db_path_for_engine(engine) or "")

    if redaction_off and db_in_home:
        status = "warn"
        detail = {
            "redaction_off": True,
            "db_in_hermes_home": True,
            "hermes_home": hermes_home,
            "database_path": db_path,
            "warning": (
                "PII redaction is OFF and trove.db lives inside HERMES_HOME "
                "({hermes_home}), which the Hermes backup channel ships to a "
                "remote repo. Unredacted secrets (API keys, bearer tokens, "
                "passwords, private keys) may leak publicly via backup."
            ),
            "recommendation": (
                "Enable TROVE_SENSITIVE_PATTERNS_ENABLED=1, OR move trove.db "
                "outside HERMES_HOME via TROVE_DATABASE_PATH, OR exclude "
                "trove.db from the Hermes backup channel."
            ),
        }
    else:
        status = "pass"
        detail = {
            "redaction_off": redaction_off,
            "db_in_hermes_home": db_in_home,
            "hermes_home": hermes_home,
            "database_path": db_path,
        }

    return {
        "check": "pii_redaction_and_backup_guardrail",
        "status": status,
        "detail": detail,
    }


def doctor_guidance_for_checks(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return actionable guidance for warning/failing trove_doctor checks."""
    guidance = []
    for check in checks:
        item = doctor_guidance_for_check(check)
        if item is not None:
            guidance.append(item)
    return guidance


# Backward-compatible private aliases for existing command/tool internals and tests.
_state_db_path_for_engine = state_db_path_for_engine
_has_lifecycle_fragmentation = has_lifecycle_fragmentation
