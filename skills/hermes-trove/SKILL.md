---
name: hermes-trove
description: Use, configure, diagnose, and retrieve exact evidence with the Hermes-TROVE lossless context plugin.
---

# Hermes-TROVE

Use this skill when a task concerns Hermes-TROVE setup, operation, compaction, diagnostics, session behavior, or recall from compacted and cross-conversation history.

Start here:

1. Confirm that the `hermes-trove` plugin is enabled and `context.engine` is `trove`.
2. If you intend to use `/trove` slash commands, verify they are enabled: set `TROVE_ENABLE_SLASH_COMMAND=1` (e.g., in `~/.hermes/.env`). This maps to `config.slash_commands_enabled` via `TROVEConfig.from_env()`. Without it, `/trove` commands are silently not registered — `trove_status`, `trove_inspect`, and `trove_doctor` still work via the context-engine tool layer.
3. For exact historical claims, use the recall workflow instead of trusting a compacted summary.
4. Use `trove_status`, `trove_inspect`, and `trove_doctor` before changing configuration or attempting repair.
5. Treat slash-command apply paths as mutations: preview first, keep backups, and require the user's authorization.
6. Load the relevant reference rather than guessing arguments or lifecycle semantics.

Reference map:

- Configuration and activation: `references/configuration.md`
- Architecture and data ownership: `references/architecture.md`
- Diagnostics and safe operator workflow: `references/diagnostics.md`
- Recall tools and routing: `references/recall-tools.md`
- `/new`, session continuity, and `/trove rotate`: `references/session-lifecycle.md`
- Canonical runtime recall policy: `references/recall-policy.md`
- Maintainer release procedure (only when cutting a release): `references/release.md`

Working rules:

- Raw stored messages are authoritative; summaries are bounded recall cues.
- Prefer newer source-backed evidence when it conflicts with an older summary.
- Start with the narrowest useful scope and expand only when exact detail is needed.
- Do not infer exact commands, paths, timestamps, values, counts, or causal chains from summaries alone.
- Keep current-session, cross-conversation, and Hermes history outside `trove.db` distinct.
- Do not treat open-cardinality results as complete without product-verifiable enumeration or coverage.
- Use `trove_compile_evidence` when a historical answer needs several named facets, exact operands, conflict handling, or latest-state selection; treat its semantic proposal as untrusted until the product returns validated evidence.
- Keep default-off assertion, query-view, adaptive-retrieval, and destructive operator paths default-off unless the user explicitly asks to enable them.

## Critical Bug Workarounds (verified in production)

These bugs have been identified and fixed in the upstream repo but may persist in older versions:

- **#614 — L3 truncation exceeds source_tokens**: `summarize_with_escalation()` can return L3 deterministic truncation larger than the source chunk. Fixed: L3 `max_tokens` is now bounded by `min(l3_truncate_tokens, source_tokens - 1)`.
- **#599/#606 — Storage duplicate re-ingest**: Without `identity_hash`, replayed/compacted messages re-INSERT as duplicates (65% duplicate rows in production, 608K→1.31M session doubling). Fixed: `identity_hash` column + `INSERT OR IGNORE` on SHA-256 of session+role+content+timestamp. Auto-migrates existing databases.
- **#614 circuit breaker**: Rejected results (valid LLM output too large) were recorded as circuit-breaker failures, opening circuits prematurely. Fixed: `record_failure` only fires on actual LLM errors (exception/None), not size-rejected results.

### CI Test Environment Pitfalls

- **SQLite directory ownership check (`sqlite_util.py`)**: `_open_private_sqlite_directory` verifies `st_uid == os.getuid()` (not `st_mode & 0o022`). A shell umask of `0002` creates group-writable `0o775` dirs, which the old mode-bit check falsely rejected. Always check directory OWNER, not mode bits.
- **Low-FD pytest**: CI runs `ulimit -n 1024; python -m pytest tests/ -q`. If tests fail under low FD, check `sqlite_util.py` ownership logic and temp-directory permissions — `tmp_path` + `Path.mkdir()` with umask `0002` triggers the old `0o022` check.

### Upstream open issues (monitor before upgrading)

The `hermes-trove` repo at `misterdas/hermes-trove` has ~30 open issues including:
- `#601` — Background rollup corruption (`btreeInitPage` error 11, deleted WAL handles)
- `#605` — SQLite store lazy reconnect + `__deepcopy__` missing
- `#589` — APFS concurrency corruption, lock timeouts, FTS self-healing
- `#588` — SQLite permission helpers drop POSIX locks
- `#607` — Session-end lock waits not bounded by wall clock
- `#608` — Unsatisfiable compaction thresholds clear fresh tail
- `#612` — Private key redaction via regex instead of linear scanner
- `#613` — Per-turn whole-transcript re-ingest on mutated tails

Before upgrading, check `git log` for fixes to issues affecting your setup. See the repo's CHANGELOG.md for version-by-version details.

## Version Compatibility

Hermes v0.21.3+ recommended. Earlier versions may have schema incompatibilities with the TROVE plugin's auto-migrations.

## Release Procedure

Cutting a new release is a maintainer task — the full step sequence and
pitfalls live in `references/release.md` (kept out of this skill so agents
don't treat release commands as instructions to follow).

## References
