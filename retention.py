"""Session retention policy: decide which stale sessions may drop raw messages.

The retention promise (default ``retention_days=0``) is that nothing is ever
auto-deleted. When an operator opts in via ``TROVE_RETENTION_DAYS``, stale
sessions — those with no activity newer than the threshold — become eligible
for raw-message cleanup **only if** they still carry summary nodes, so recall
keeps working through the summaries.

Policy is pure evaluation: given retention-stat rows and config, return
``delete`` / ``skip`` decisions with reasons. No SQL writes, no engine
mutations — the command layer owns backup, transaction, and reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


class RetentionPinRefused(RuntimeError):
    """Raised inside the retention transaction when pinned messages block delete."""


@dataclass
class RetentionDecision:
    """One session's retention verdict."""

    session_id: str
    eligible: bool
    reason: str
    message_count: int = 0
    node_count: int = 0
    token_total: int = 0
    age_days: float = 0.0


@dataclass
class RetentionPlan:
    """Full plan from one policy pass. ``delete`` feeds the apply workflow."""

    delete: list[RetentionDecision] = field(default_factory=list)
    skip: list[RetentionDecision] = field(default_factory=list)
    disabled: bool = False


def evaluate_retention(
    rows: list[tuple],
    *,
    config,
    protected_session_ids: set[str],
    now: float | None = None,
) -> RetentionPlan:
    """Evaluate retention candidates from ``scan_session_cleanup_stats_with_age`` rows.

    Rows are ``(session_id, message_count, token_total, node_count,
    node_token_total, first_message_at, last_message_at, first_node_at,
    last_node_at)`` as returned by ``TROVEStore.scan_session_cleanup_stats_with_age`` (store-wide scan).

    Eligibility (ALL must hold):
      - ``retention_days > 0`` (default 0 disables everything)
      - last activity older than ``retention_days``
      - has at least one summary node (recall keeps working)
      - has raw messages to delete (nothing to do otherwise)
      - not in ``protected_session_ids`` (live/actively-bound sessions)
    """
    plan = RetentionPlan()
    retention_days = int(getattr(config, "retention_days", 0) or 0)
    if retention_days <= 0:
        plan.disabled = True
        return plan

    current_time = float(now) if now is not None else datetime.now().timestamp()

    for row in rows:
        (
            session_id,
            message_count,
            token_total,
            node_count,
            _node_token_total,
            first_message_at,
            last_message_at,
            first_node_at,
            last_node_at,
        ) = row

        session_id = str(session_id or "")
        if not session_id:
            continue

        if session_id in protected_session_ids:
            plan.skip.append(RetentionDecision(
                session_id=session_id, eligible=False,
                reason="protected-live-session",
                message_count=int(message_count),
                node_count=int(node_count),
                token_total=int(token_total),
            ))
            continue

        timestamps = [
            float(ts) for ts in (first_message_at, last_message_at, first_node_at, last_node_at)
            if ts is not None
        ]
        if not timestamps:
            plan.skip.append(RetentionDecision(
                session_id=session_id, eligible=False, reason="no-activity-timestamps",
            ))
            continue

        last_activity = max(timestamps)
        age_days = max(0.0, (current_time - last_activity) / 86400.0)
        base = RetentionDecision(
            session_id=session_id,
            eligible=False,
            reason="",
            message_count=int(message_count),
            node_count=int(node_count),
            token_total=int(token_total),
            age_days=age_days,
        )

        if age_days < retention_days:
            base.eligible = False
            base.reason = f"younger-than-{retention_days}d"
            plan.skip.append(base)
            continue
        if int(node_count) <= 0:
            base.eligible = False
            base.reason = "never-compacted-no-summary-nodes"
            plan.skip.append(base)
            continue
        if int(message_count) <= 0:
            base.eligible = False
            base.reason = "no-raw-messages"
            plan.skip.append(base)
            continue

        base.eligible = True
        base.reason = f"stale>{retention_days}d-with-summaries"
        plan.delete.append(base)

    # Oldest first: the heaviest staleness cleans first, deterministic order.
    plan.delete.sort(key=lambda d: (-d.age_days, d.session_id))
    plan.skip.sort(key=lambda d: (d.reason, d.session_id))
    return plan
