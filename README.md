<p align="center">
  <img src="docs/banner.png" alt="HERMES-TROVE" width="800">
</p>

[![CI](https://github.com/misterdas/hermes-trove/actions/workflows/ci.yml/badge.svg)](https://github.com/misterdas/hermes-trove/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/misterdas/hermes-trove)](https://github.com/misterdas/hermes-trove/releases)
[![Python 3.11-3.14](https://img.shields.io/badge/Python-3.11--3.14-3776AB?logo=python&logoColor=white)](https://github.com/misterdas/hermes-trove/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Lossless Context Management plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

> Bounded context, unbounded memory. Nothing is ever lost.

`hermes-trove` replaces one-shot active-context compression with a SQLite-backed,
DAG-based context engine. It keeps the live prompt bounded, preserves raw
messages, and gives the agent tools to recover exact detail after compaction.

Based on the [TROVE paper](https://papers.voltropy.com/TROVE) by Ehrlich & Blackman
(Voltropy PBC, Feb 2026). Inspired by
[lossless-claw](https://github.com/martian-engineering/lossless-claw) for
OpenClaw.

The original `hermes-trove` plugin was authored by [@stephenschoettler](https://github.com/stephenschoettler). This repository is a community fork maintained by @misterdas. For an interactive visualization of the TROVE idea, see
[losslesscontext.ai](https://losslesscontext.ai/).

## Table of contents

- [What it does](#what-it-does)
- [TROVE vs built-in compression](#trove-vs-built-in-compression)
- [Quick start](#quick-start)
- [Commands and tools](#commands-and-tools)
- [Recall skill and policy](#recall-skill-and-policy)
- [Configuration](#configuration)
- [Retrieval contract](#retrieval-contract)
- [OpenClaw/lossless-claw import](#openclawlossless-claw-import)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)
- [How it works](#how-it-works)
- [Documentation](#documentation)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## What it does

Hermes Agent's built-in compressor is a practical continuity layer: when the
prompt crosses its configured threshold, it prunes older tool results, asks an
auxiliary model to summarize the middle/older conversation, and rebuilds the
active prompt from that summary plus a protected recent tail. The original
session rows can still live in Hermes `state.db` and remain searchable through
host tools such as `session_search`, but the model's active context no longer
contains the compacted turns verbatim or a structured drill-down path back to
them.

`hermes-trove` instead:

1. **Persists messages** in a plugin-local SQLite store with FTS metadata.
2. **Compacts older context** into depth-aware summary nodes.
3. **Condenses summaries** into a hierarchical DAG as they accumulate.
4. **Assembles active context** from system prompt, highest-value summaries, and
   the protected fresh tail.
5. **Provides recall tools** so agents can search, inspect, and expand compacted
   material without flooding the main prompt.

Nothing is lost in normal operation. Raw messages stay recoverable in bounded
pages, summaries retain source lineage, and oversized externalized payloads keep
stable refs for later expansion.

<p align="center">
  <img src="docs/standard_compression.png" alt="Standard compression" width="700">
</p>

<p align="center">
  <img src="docs/trove_compression.png" alt="TROVE compression" width="700">
</p>

Core capabilities:

- **SQLite message store** - preserves raw messages before compaction
- **Summary DAG** - builds depth-aware summary nodes over compacted history
- **Bounded recovery** - pages raw messages, child summaries, and externalized
  payloads instead of dumping everything into the prompt
- **Agent tools** - `trove_grep`, `trove_recall`, `trove_query_state`, `trove_compute`, `trove_compile_evidence`, `trove_evidence_pack`, `trove_retrieve`, `trove_recent`, `trove_load_session`,
  `trove_describe`, `trove_expand`, `trove_expand_query`, `trove_status`, `trove_inspect`,
  and `trove_doctor`
- **Source-aware retrieval** - filters raw rows and summaries by descendant
  source lineage
- **Session controls** - ignore noisy sessions or keep sessions read-only with
  glob patterns
- **Large payload controls** - externalize oversized tool/media/raw payloads and
  protect SQLite from inline media-ish base64 blobs
- **Sensitive-pattern controls** - optional named redaction of API keys, bearer
  tokens, passwords, and private keys before TROVE stores or summarizes them
- **Diagnostics** - runtime health, database checks, optional `/trove` slash
  commands, backup-first repair/rotate paths

Beyond the core loop, three opt-in (default-off) feature families extend TROVE
from a compression layer into a memory system: **large-output externalization
and context-budget controls** (giant tool results move to recoverable refs
instead of crowding the prompt), **temporal memory** (day/week/month rollups
plus natural-time recall through `trove_recent`), and **semantic retrieval**
(embedding-backed `trove_grep` semantic/hybrid modes with free-tier cloud or
fully-local providers). See the
[Feature overview](docs/features-overview.md) for what each family does and
why, and [Agent configuration profiles](docs/agent-config-profiles.md) for
copy-paste setups per agent type.

## TROVE vs built-in compression

Hermes core may persist original conversation history in `state.db` before
built-in compression rewrites the active prompt. Built-in compression can still
be lossy in the active context, but previous content may be recoverable later
through host-level history tools such as `session_search`.

`hermes-trove` is different because recall is part of the active context engine:

- plugin-local store and DAG built specifically for drill-down
- current-session retrieval through TROVE tools, not an auxiliary cross-session
  search step
- explicit source-lineage and session-boundary rules

Position TROVE around retrieval quality, autonomy, and drill-down behavior. Do not
claim that Hermes core has no persisted record of pre-compression history.

## Quick start

### Prerequisites

- Hermes Agent
- Python 3.11+
- No required third-party runtime dependencies

`tiktoken` is used if available; otherwise TROVE falls back to character-based
token estimates. `regex` is used if available to apply timeouts to message ignore
patterns; without it, message-level regex filtering is disabled with a warning
rather than running unbounded stdlib `re` matches.

The versioned [host-owned dependency contract](docs/dependency-assurance.md)
lists every shipped runtime import, supported host/Python versions, ownership,
and the mechanical validation command. It is an assurance boundary, not a claim
that the host's resolved environment is free of known vulnerabilities.

### Install the plugin

Canonical install path: clone `hermes-trove` as a general user plugin.

```bash
git clone https://github.com/misterdas/hermes-trove \
  ~/.hermes/plugins/hermes-trove
```

For a profile-specific install:

```bash
git clone https://github.com/misterdas/hermes-trove \
  ~/.hermes/profiles/myprofile/plugins/hermes-trove
```

From an existing checkout, install a symlink:

```bash
./scripts/install.sh

# Optional profile-aware install:
HERMES_PROFILE=myprofile ./scripts/install.sh
```

Run `scripts/install.sh` even when the checkout already lives at the canonical
plugin path. It leaves that checkout in place and exposes the bundled
`hermes-trove` skill in the matching global/profile `skills/` directory. The
installer preflights both paths and refuses conflicts before creating links.

### Activate it

The plugin has two names:

- plugin manifest name: `hermes-trove`
- runtime context engine name: `trove`

Both must be configured:

```yaml
plugins:
  enabled:
    - hermes-trove

context:
  engine: trove
```

Restart Hermes after changing plugin or context-engine config.

### Verify it loaded

Run:

```bash
hermes plugins
```

Expected signals:

- plugin list includes `hermes-trove`
- selected context engine is `trove`
- tool list includes `trove_grep`, `trove_recall`, `trove_recent`,
  `trove_load_session`, `trove_describe`, `trove_expand`, `trove_expand_query`,
  `trove_status`, `trove_inspect`, and `trove_doctor`
- the normal available-skills index includes `hermes-trove`; current hosts can
  also resolve the explicit plugin-qualified skill `hermes-trove:hermes-trove`

Typical output:

```text
Plugins (1):
  ✓ hermes-trove v1.2.2 (15 tools)

Provider Plugins:
  Context Engine: trove
```

For source checkouts, `trove_status`, `/trove status`, `trove_inspect`,
`trove_doctor`, and `/trove doctor` also report the loaded plugin path and
best-effort git identity:
`plugin_git_commit`, `plugin_git_branch`, and `plugin_git_dirty`.

If startup logs say TROVE tools are available through `context-engine schemas` or
mention the `Path B fallback`, that is expected on older Hermes hosts such as
Hermes Agent v0.16. All 15 `trove_*` tools remain available through the
context-engine path; standalone plugin-registry registration is not required
there.

### Update it

If you cloned directly into the plugin directory:

```bash
cd ~/.hermes/plugins/hermes-trove && git pull --ff-only
```

For a profile-specific install:

```bash
cd ~/.hermes/profiles/myprofile/plugins/hermes-trove && git pull --ff-only
```

If you installed a symlink from a separate checkout:

```bash
./scripts/update.sh
```

Restart Hermes after updating.

For the `v1.0.0` line, take a normal backup of `trove.db` before updating,
then update the checkout and restart Hermes. No manual core migration or
backfill is required: the core schema remains version 5. New assertion,
query-view, and adaptive-retrieval state is additive, created only after the
corresponding opt-in is enabled, and stored in the same profile database under
named feature markers. The five new query/evidence tool schemas are visible in
the tool list on stock installs, but automatic extraction, pre-answer evidence,
assertion storage, query-view storage, and adaptive retrieval remain off. See
[the operator upgrade and opt-in notes](docs/operator-guide.md#upgrade-from-v0200-or-v0210-rc2-to-v100-rc1)
before enabling them.

## Commands and tools

### Agent tools

Use these tools for current-session recall after compaction. Use Hermes
`session_search` for earlier separate sessions or broad cross-session history
outside the TROVE database.

| Tool | Use |
|------|-----|
| `trove_grep` | Search current-session raw messages and summaries. Opt into `content_scope='externalized'|'both'` for bounded active-session payload search, or `session_scope='all'|'session'` for bounded raw-message archive recovery; broader scopes return raw-message hits only. |
| `trove_recall` | Search the entire memory across ALL conversations and all time by meaning. Fuses full-text, summary-vector, and chunk-vector arms with RRF, then applies a soft current-conversation (`scope_bias`) and recency prior. Returns bounded summary and verbatim-excerpt hits with `trove_expand` handles; works FTS-only when embeddings are disabled. |
| `trove_query_state` | Query the opt-in V4 assertion sidecar for bounded current or historical facts, preferences, recommendations, commitments, actions, and status. Every result carries an exact message ref/span/quote; conflicts stay visible. |
| `trove_compute` | Execute supported dates, distinct counts, compatible-unit sums, directed/absolute differences, ordering, and latest-state operations over exact cited evidence. Planning, arithmetic, and trace verification are dependency-free and provider-neutral; ambiguous or incomplete inputs fail closed. |
| `trove_compile_evidence` | Validate one bounded provider-neutral semantic proposal against exact stored refs and return a compact evidence brief with explicit sufficiency state and an optional canonical computation. It never returns final prose or treats model claims as finite-coverage proof. |
| `trove_evidence_pack` | Hydrate and validate bounded baseline exact refs in the same `trove.db`, repair only unique in-window quote spans, preserve occurrence/observation time separation, and optionally emit an immutable canonical computation trace without prose. |
| `trove_retrieve` | Opt-in bounded controller for one continuous answerer tool turn. It tracks typed evidence slots, permits at most three targeted calls to existing retrieval tools, accepts only exact observed refs, caps the evidence context, and can finish through `trove_compute`. It persists evidence views and traces, never final prose. |
| `trove_recent` | Retrieve recent summaries by natural UTC period, preferring ready rollups and transparently falling back to time-bounded leaf summaries. |
| `trove_load_session` | Load one ordered raw-message transcript page for an explicit `session_id`. Continues with `after_store_id` from `next_cursor`; opt into exact slice refs with `include_exact_ref=true`. |
| `trove_describe` | Inspect the current-session DAG or preview an `externalized_ref` without loading full content. |
| `trove_expand` | Recover source messages, child summaries, or externalized payloads with pagination. Use `store_id` to fetch a single raw message from a cross-session `trove_grep` result and `include_exact_ref=true` when its returned slice must be cited or computed. |
| `trove_expand_query` | Answer a question using expanded current-session TROVE context while returning a bounded answer. |
| `trove_status` | Show runtime health, context pressure, config, source lineage, and lifecycle stats. |
| `trove_inspect` | Read-only operator inventory for current-session lineage, frontier/fresh-tail metadata, externalized refs/readability, compaction skip/no-op reasons, and matched ignore/stateless patterns. Returns metadata only; use retrieval tools for content. |
| `trove_doctor` | Run database, FTS, lifecycle, config, and context-pressure diagnostics. |

## Recall skill and policy

Hermes-TROVE ships `skills/hermes-trove/SKILL.md` plus progressive-disclosure
references for configuration, architecture, diagnostics, recall routing, and
session lifecycle. The installer links that directory into the active Hermes
profile so it appears in ordinary skill discovery. On hosts with plugin skill
registration, it is also available explicitly as `hermes-trove:hermes-trove`.

When TROVE is the active context engine for a bound session, the plugin registers
one deterministic `pre_llm_call` hook. Hermes injects the canonical policy into
the current user-message context, not the system prompt, preserving the stable
system-prompt cache prefix. The policy:

- treats summaries as recall cues rather than exact proof;
- prefers newer source-backed evidence and verifies contradictions;
- teaches narrow FTS query construction and bounded scope selection;
- routes current compacted, cross-conversation, and recent/time-bounded recall
  through the appropriate existing tools;
- does not force a tool call when the current context is already sufficient.

The canonical bytes live in
`skills/hermes-trove/references/recall-policy.md`. Merely loading the plugin does
not inject them when another context engine is serving the session. Older hosts
without skill or hook registration keep their existing schema-driven behavior.

### Slash commands

Slash commands are disabled by default. Enable them only in trusted operator
contexts:

```bash
export TROVE_ENABLE_SLASH_COMMAND=1
```

Available commands:

- `/trove` or `/trove status` - current runtime/session status
- `/trove doctor` - read-only health checks
- `/trove doctor clean` - read-only scan for obvious junk/noise session candidates
- `/trove doctor clean apply` - backup-first cleanup for safe pattern-matched
  candidates; requires `TROVE_DOCTOR_CLEAN_APPLY_ENABLED=true`
- `/trove doctor repair` - read-only SQLite/FTS repair diagnostics
- `/trove doctor repair apply` - backup-first SQLite/FTS repair
- `/trove doctor source` - read-only scan for legacy blank-source rows
- `/trove doctor source apply` - backup-first normalization of legacy blank-source
  rows to `unknown`
- `/trove doctor retention` - read-only retention analysis
- `/trove backup` - timestamped SQLite backup
- `/trove rotate` - read-only preview of an in-place tail-preserving compact of
  the active session
- `/trove rotate apply` - backup-first rotate that advances the lifecycle frontier
  past pre-tail raw messages
- `/trove help` - command help

Apply paths are intentionally narrow and backup-first. Start with diagnostics
before cleanup or repair.

### Rotate: compact in place without changing session identity

`/trove rotate` compacts a long-running session in place without changing
`session_id` or `conversation_id`. It is the in-session counterpart to
Hermes-level `/new` and to `/trove doctor clean`.

What rotate does:

- preserves the live tail (`TROVE_FRESH_TAIL_COUNT` most-recent messages)
- advances the lifecycle frontier marker past every raw message before the tail,
  so subsequent bootstrap stops replaying them into the active prompt
- writes a rolling `*-rotate-latest.sqlite3` backup under the same backup
  directory as `/trove backup`, overwriting the previous rotate slot atomically so
  disk usage stays bounded across repeated rotates

What rotate does not do:

- it does not delete raw messages; pre-tail rows remain recoverable through
  `trove_load_session` and `trove_expand`
- it does not invoke the summarization model; trigger normal compaction first if
  you want pre-tail content covered by summary nodes before rotating
- it does not change session or conversation identity, and it refuses ignored or
  stateless sessions with a clear reason

`trove_status` and `/trove status` surface `last_rotate_at` and the rolling backup
path. Re-running `/trove rotate apply` after the frontier is already at or ahead of
the target boundary reports `status: noop` and does not overwrite the previous
known-good rolling backup.

## Configuration

Most installs only need `plugins.enabled` and `context.engine: trove`.

### Common settings

| Variable | Default | Use |
|----------|---------|-----|
| `TROVE_CONTEXT_THRESHOLD` | `0.35` | Fraction of the context window that triggers TROVE compaction |
| `TROVE_FRESH_TAIL_COUNT` | `32` | Recent messages protected from compaction |
| `TROVE_FRESH_TAIL_MAX_TOKENS` | `0` | Optional token cap for the protected fresh tail (`0` disables it); always retains the newest message and complete assistant/tool-result groups |
| `TROVE_INCREMENTAL_MAX_DEPTH` | `3` | Max DAG condensation depth (`-1` = unlimited, `0` = leaf only); enables hierarchical summarization |
| `TROVE_LEAF_CHUNK_TOKENS` | `20000` | Raw-backlog floor before leaf compaction; with dynamic chunking enabled, the base chunk target |
| `TROVE_DYNAMIC_LEAF_CHUNK_ENABLED` | `false` | Enable chunk-sized leaf compaction passes instead of compacting the whole non-tail raw backlog per pass |
| `TROVE_DYNAMIC_LEAF_CHUNK_MAX` | `40000` | Upper bound for dynamic leaf chunk targets |
| `TROVE_THRESHOLD_FULL_SWEEP_ENABLED` | `false` | At threshold, opt into one synchronous bounded sweep that drains chunked raw history before publishing one new active context |
| `TROVE_SUMMARY_PREFIX_TARGET_TOKENS` | `0` | Sweep-only summary-frontier target; `0` derives one `TROVE_LEAF_CHUNK_TOKENS` budget |
| `TROVE_NEW_SESSION_RETAIN_DEPTH` | `2` | DAG depth retained after manual `/new` (`-1` all, `0` none) |
| `TROVE_DATABASE_PATH` | auto | SQLite database path. Empty config resolves to `HERMES_HOME/trove.db`; plugin installs or operators may set this env var to another profile-scoped path such as `~/.hermes/hermes-trove.db`. |
| `TROVE_FTS_INTEGRITY_CHECK_INTERVAL_HOURS` | `24` | Minimum hours between startup FTS5 deep integrity-checks (O(index size)). `0` checks every startup; a negative value never checks on startup. Structural checks always run regardless. |
| `TROVE_ENABLE_SLASH_COMMAND` | `false` | Enable the optional `/trove` operator command surface |

When `TROVE_FRESH_TAIL_MAX_TOKENS` is enabled, the protected suffix must satisfy
both the message-count and token bounds. The newest message is never dropped,
and a boundary that would begin inside an assistant tool-call/result group is
moved back to that assistant even when doing so exceeds a configured bound.

### Filtering and storage settings

| Variable | Default | Use |
|----------|---------|-----|
| `TROVE_IGNORE_SESSION_PATTERNS` | empty | Comma-separated session globs excluded from TROVE storage |
| `TROVE_STATELESS_SESSION_PATTERNS` | empty | Comma-separated session globs kept read-only |
| `TROVE_IGNORE_MESSAGE_PATTERNS` | empty | Comma-separated regex patterns; matching message content is excluded from TROVE storage |
| `TROVE_SENSITIVE_PATTERNS_ENABLED` | `false` | Opt in to deterministic redaction before TROVE storage, FTS indexing, summarization, active replay, and externalized ingest payloads |
| `TROVE_SENSITIVE_PATTERNS` | `api_key,bearer_token,password_assignment,private_key` | Comma-separated named sensitive pattern catalog entries to apply when redaction is enabled |
| `TROVE_LARGE_OUTPUT_EXTERNALIZATION_ENABLED` | `false` | Store oversized ingest payloads, including tool results, media blocks, and generic raw content, in plugin-managed JSON files |
| `TROVE_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS` | `12000` | Externalization threshold for normalized payload text |
| `TROVE_LARGE_OUTPUT_ACTIVE_REPLAY_STUBBING_ENABLED` | `false` | Replace token-heavy textual tool results with recoverable externalized refs in active replay; current-turn ingest is immediate and historical assembly respects the protected fresh tail; requires large-output externalization |
| `TROVE_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_THRESHOLD_TOKENS` | `25000` | Token-aware threshold for active-replay tool-result stubbing |
| `TROVE_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED` | `false` | Rewrite already-externalized summarized tool rows to compact placeholders |
| `TROVE_DOCTOR_CLEAN_APPLY_ENABLED` | `false` | Permit destructive `/trove doctor clean apply` in trusted operator contexts |
| `TROVE_EMPTY_LIFECYCLE_GC_ENABLED` | `true` | Master toggle for automatic pruning of lifecycle rows for sessions that never ingested any messages or summary nodes |
| `TROVE_EMPTY_LIFECYCLE_GC_THRESHOLD` | `200` | Number of lifecycle rows at which the GC pass fires |
| `TROVE_EMPTY_LIFECYCLE_GC_MAX_AGE_HOURS` | `24` | Automatic GC only deletes empty lifecycle rows at least this old; set `0` only in trusted/test environments that intentionally want immediate empty-row pruning |

### Model and timeout settings

| Variable | Default | Use |
|----------|---------|-----|
| `TROVE_SUMMARY_MODEL` | auxiliary | Override summarization model |
| `TROVE_SUMMARY_FALLBACK_MODELS` | empty | Comma-separated summarization models tried after `TROVE_SUMMARY_MODEL` or the auxiliary task default fails |
| `TROVE_SUMMARY_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `2` | Consecutive failed summarization calls before a route is skipped temporarily |
| `TROVE_SUMMARY_CIRCUIT_BREAKER_COOLDOWN_SECONDS` | `300` | Seconds to skip an open summary route before retrying it |
| `TROVE_EXPANSION_MODEL` | summary model / auxiliary | Override `trove_expand_query` synthesis model |
| `TROVE_EXPANSION_CONTEXT_TOKENS` | `32000` | Context budget used by the auxiliary LLM for `trove_expand_query` |
| `TROVE_SUMMARY_TIMEOUT_MS` | `60000` | Timeout for one summarization call |
| `TROVE_EXPANSION_TIMEOUT_MS` | `120000` | Timeout for one `trove_expand_query` synthesis call |
| `TROVE_CRITICAL_BUDGET_PRESSURE_RATIO` | `0.0` | Disabled at `0.0`; when set, permits critical-pressure bypasses for bounded deferred catch-up and cache-friendly follow-on condensation only |

Advanced compaction, assembly, and extraction knobs are defined in `config.py`.

### Sensitive-pattern redaction

Sensitive-pattern handling is disabled by default so ordinary TROVE storage and
`trove_expand` remain lossless. When `TROVE_SENSITIVE_PATTERNS_ENABLED=true`, matched
secret values are replaced with metadata-only placeholders before SQLite, FTS,
summaries, active replay, and externalized payload JSON receive the content. This
is intentionally not lossless for matching values: the raw matched secret is
unrecoverable after redaction.

Supported named catalog entries are:

- `api_key`: `api_key`, `api_token`, `access_token`, `secret_key`, and
  `client_secret` assignments or JSON keys.
- `bearer_token`: `Bearer ...` strings and token-like JSON keys.
- `password_assignment`: `password`, `passwd`, `pwd`, and `passphrase`
  assignments or JSON keys, including quoted values with spaces.
- `private_key`: PEM private-key blocks.

Redaction is forward-only. Enabling it does not rewrite existing SQLite rows,
FTS shadow tables, DAG summaries, or externalized payload JSON that were written
before the setting was enabled. Non-password placeholders include a short
truncated SHA-256 digest for correlation. `password_assignment` placeholders omit
the digest to avoid making password-like values easier to dictionary-check.
`trove_status`, `trove_inspect`, and `trove_doctor` expose the enabled state, configured pattern names,
unknown names, source, and placeholder format without exposing raw secret values.

### Threshold ownership

When `context.engine: trove` is active, `TROVE_CONTEXT_THRESHOLD` is the compaction
threshold TROVE uses. Hermes core `compression.threshold` belongs to the built-in
compressor. Hermes core `compression.enabled` is still the global gate that
allows compaction, so leave it enabled when using TROVE.

If startup/status output shows a host-side compression percentage that disagrees
with TROVE, trust live TROVE status after a normal message has initialized the
session.

### Tuning for large context windows

Long-context models change the tuning problem. A 1M-token model does not mean
you always want to spend 750k prompt tokens before TROVE starts compacting. Start
with the active prompt budget you are willing to pay for, then tune the threshold
around that budget.

```text
compaction trigger = effective context window * TROVE_CONTEXT_THRESHOLD
TROVE_CONTEXT_THRESHOLD = desired compaction trigger / effective context window
```

Examples, as math rather than universal recommendations:

| Effective context window | Desired trigger | Threshold |
|--------------------------|-----------------|-----------|
| `128000` | `96000` | `0.75` |
| `200000` | `140000` | `0.70` |
| `400000` | `240000` | `0.60` |
| `1000000` | `250000` | `0.25` |
| `1000000` | `400000` | `0.40` |
| `1000000` | `600000` | `0.60` |

A reasonable first pass for a true 1M effective window is:

| Goal | Desired trigger | Threshold | Notes |
|------|-----------------|-----------|-------|
| Lower spend / earlier DAG building | `200000` to `300000` | `0.20` to `0.30` | Good when cost and latency matter more than maximum live context |
| Balanced large-context use | `350000` to `500000` | `0.35` to `0.50` | Good starting point for many long-running agents |
| Keep more raw context active | `600000+` | `0.60+` | Higher token burn, later compaction |

Tune against your effective `context_length` if Hermes caps the provider's
advertised window.

Start with `TROVE_CONTEXT_THRESHOLD`, `TROVE_FRESH_TAIL_COUNT`, and large output
externalization. Only tune leaf chunking after checking `trove_status` and
understanding whether your workload is dominated by huge raw backlog passes.

`TROVE_THRESHOLD_FULL_SWEEP_ENABLED=true` is an opt-in cache-shape policy. Once
threshold pressure triggers compaction, the invocation keeps summarizing the
oldest raw chunks outside the protected fresh tail even after pressure falls
below the trigger. It then condenses the provider-visible summary frontier only
when that frontier exceeds `TROVE_SUMMARY_PREFIX_TARGET_TOKENS` (or one leaf
budget when the target is `0`). One invocation is bounded to 12 total leaf plus
condensation calls and 120 seconds between calls, persists each completed DAG
pass, and publishes one newly assembled active context at the end. It is
synchronous and independent of deferred/background maintenance.

### Cache policy boundary

TROVE is **cache-friendly**, not fully cache-aware. It may avoid some follow-on
condensation churn, but current cache usage counters are retrospective status
data only; they do not tell the plugin whether the next prompt mutation will
break a hot provider cache.

`TROVE_CRITICAL_BUDGET_PRESSURE_RATIO` is a narrow escape hatch. It is disabled by
default (`0.0`). When set, TROVE compares prompt pressure against the context
window and only at or above that ratio may bypass existing polite gates for
bounded deferred maintenance catch-up and cache-friendly follow-on condensation.
Revisit full cache-aware deferred compaction only after Hermes core exposes
reliable cache state / cache-break signals.

### Session pattern syntax

Pattern matching checks multiple keys: raw `session_id`, `platform`, and
`platform:session_id`.

- `*` matches within one colon-delimited segment
- `**` can span across colons

Example: `cron:*` can match Hermes cron sessions, while exact raw session IDs
still work.

### Noise suppression

TROVE offers two layers of noise filtering:

- **Session-level filters** (`TROVE_IGNORE_SESSION_PATTERNS`,
  `TROVE_STATELESS_SESSION_PATTERNS`) catch noisy traffic that arrives as its own
  session or platform.
- **Message-level patterns** (`TROVE_IGNORE_MESSAGE_PATTERNS`) catch cron alerts or
  other noise injected into a normal Telegram or WhatsApp conversation as
  ordinary visible messages.

Message-level patterns are comma-separated Python regex strings compiled once at
engine start. They run against plain text first; structured multimodal payloads
use concatenated text parts first, then normalized JSON fallback when there are
no text parts. Matching messages are skipped before storage.

Example operator config:

```bash
TROVE_IGNORE_MESSAGE_PATTERNS=^Cronjob Response:,^>>>Cronjob Response<<<:
```

Invalid regex entries are logged at warning level and dropped. Pattern matching
uses a 50 ms per-pattern timeout when the optional `regex` package is installed.
If `regex` is not installed, TROVE logs a warning and disables message-level regex
filtering rather than running unbounded stdlib `re` matches in the ingest path.

Known limitation: the filter runs at ingest time. When a matching message is part
of the chunk summarized in the same turn it arrived, the text may appear inside
the resulting summary node. The filtered message is still not written to the
message store, so DAG lineage stays clean; only serialized summary text can carry
it.

`trove_status` surfaces the full filter contract under `session_filters`, including
pattern sources, whether the current session is ignored/stateless, and a
process-lifetime `ignored_message_count`.

Ignored/stateless sessions are a storage ownership boundary, not a context-window
opt-out. TROVE does not ingest raw messages or create DAG nodes for sessions that
match `TROVE_IGNORE_SESSION_PATTERNS`, `TROVE_STATELESS_SESSION_PATTERNS`, or the
in-process auxiliary/thread stateless marker. If those sessions cross the normal
context threshold, TROVE delegates the compaction call to Hermes' native
`ContextCompressor` so the active request is still bounded before model overflow.
If the native compressor is unavailable, TROVE falls back to a deterministic
head/tail trim as a last-resort safety net, still without writing the bypassed
session to `trove.db`.

### Large tool-output handling

Storage-boundary payload guard contract: TROVE prevents media-ish inline payloads from being written into plugin-local SQLite rows at the storage boundary.

Externalization for ordinary large tool output is opt-in. When enabled,
oversized tool results are written to plugin-managed JSON files and referenced
from summaries. They remain inspectable through
`trove_describe(externalized_ref=...)` and `trove_expand(externalized_ref=...)`.

Active-replay stubbing is a second, independently opt-in replay policy. When
both externalization and active-replay stubbing are enabled, newly ingested
textual tool results above the token threshold are durably externalized and
replaced immediately in provider-visible replay, including results in the
protected fresh tail. This lets a stub-only replay change converge even when no
leaf is eligible for compaction. A historical assembly pass applies the same
policy to older tool results before budgeting, while respecting the protected
fresh tail. Tool-call ids and compatible structured text block types/keys are
retained; raw SQLite rows and DAG lineage are not rewritten by the historical
pass. Structured image/media results remain inline, preserving the provider
replay contract established by Hermes-TROVE PR #226. If durable externalization
cannot be confirmed, replay keeps the original payload inline. Results from
`trove_describe` and `trove_expand` also remain inline so recovery does not
recursively produce another ref.

The storage-boundary payload guard is separate from that opt-in. TROVE always
scans messages at the store boundary before writing `messages.content` or
`messages.tool_calls` to SQLite. Inline `data:*;base64,...` payloads and
conservative long base64-looking runs are replaced with compact placeholders and
written to the same plugin-managed externalized-payload directory.

This avoids duplicating media-ish payload bytes into `trove.db`, FTS shadow tables,
WAL files, or ordinary SQLite backups while preserving lossless recovery via the
placeholder `ref` and `trove_expand(externalized_ref=...)`. If externalization
fails, TROVE logs a warning and leaves the original text inline rather than
dropping data.

`trove_doctor` reports SQLite `journal_mode`, `quick_check`, database/WAL sizes,
largest content/tool-call rows, suspicious inline payload rows, and aggregate
externalized-payload stats. Doctor output is metadata-only for these scans.

This guard is scoped to TROVE's own `trove.db` write boundary. It does not prevent
Hermes core, or any other host layer, from writing inline payloads to Hermes
`state.db`, and it does not rewrite historical rows already present in `trove.db`.
If bytes already landed in Hermes `state.db`, that is upstream/outside TROVE scope;
use backup-first cleanup or migration procedures before mutating historical host
rows.

Transcript GC is separate and opt-in. It only rewrites already-externalized,
already-summarized tool-role rows to compact placeholders. It keeps the same
`store_id`, keeps payload files, skips pinned messages, and preserves lossless
recovery through `externalized_ref`.

## Retrieval contract

TROVE retrieval tools default to current-session scope. `trove_grep` accepts
`session_scope='all'` or `session_scope='session'` as an explicit opt-in for
bounded archive search over rows already present in `trove.db` (raw-message hits
only). Once a session id is known, `trove_load_session` can enumerate that
session's raw transcript in chronological `store_id` pages without a search
query. Use Hermes `session_search` for broad cross-session history outside the
TROVE database.

Within the current session, `source` filters raw rows directly and filters
summary nodes by descendant raw-message source lineage. `unknown` is a real
source value, not a wildcard. Legacy blank-source rows are treated as `unknown`.
`role`, `time_from`, and `time_to` are raw-message filters applied in the message
search query before result limiting. When a raw-message filter is active,
`trove_grep` returns raw rows only and reports `summary_results_omitted`.

Tool responses are bounded so one retrieval call cannot flood the main context.

### Lossless raw recovery contract

Lossless recovery means raw content is stored with stable source lineage and can
be recovered in deterministic pages:

- `trove_expand(node_id=...)` pages immediate sources with `source_offset` and
  `source_limit`
- `trove_load_session(session_id=...)` pages ordered raw session rows with
  `after_store_id` and `next_cursor`
- oversized raw messages continue with `content_offset`
- `trove_expand(externalized_ref=...)` pages payload content with `content_offset`
- `trove_expand_query` uses `context_max_tokens` for auxiliary context and reports
  truncation/pagination hints when needed

Carried-over summary nodes can become current-session content after `/new`, but
their source eligibility still comes from descendant raw messages. Expanding a
carried-over current-session node recovers the original raw message sources even
when those sources still belong to a previous session.

## OpenClaw/lossless-claw import

`hermes-trove` includes an opt-in operator script for backfilling OpenClaw history
into the local hermes-trove SQLite store. It supports two source shapes:

1. **SQLite TROVE database** (`--source-db`) for migrations from an existing
   lossless-claw/OpenClaw `trove.db`.
2. **JSONL session exports** (`--source-jsonl` / `--source-jsonl-dir`) for fresh
   installs, plugin-off catch-up, or migrations where no source SQLite database
   is available.

SQLite source example:

```bash
python scripts/import_lossless_claw.py \
  --source-db ~/.openclaw/path/to/trove.db \
  --target-db ~/.hermes/trove.db \
  --agent sammy
```

JSONL source example:

```bash
python scripts/import_lossless_claw.py \
  --source-jsonl ~/.openclaw/agents/sammy/sessions/session-a.jsonl \
  --target-db ~/.hermes/trove.db \
  --agent sammy \
  --json
```

For a directory of session exports:

```bash
python scripts/import_lossless_claw.py \
  --source-jsonl-dir ~/.openclaw/agents/sammy/sessions \
  --target-db ~/.hermes/trove.db \
  --agent sammy
```

The script is intentionally conservative:

- dry-run is the default; pass `--apply` to write
- run it against an explicit target DB path, preferably while Hermes is stopped
  for that profile
- writes create a timestamped target DB backup first when the target already
  exists
- SQLite imports can include OpenClaw summaries with `--include-summaries`; this
  migrates compatible summary rows into Hermes `summary_nodes`
- JSONL imports migrate raw message rows only; JSONL does not carry a summary DAG
- imported rows keep explicit provenance in `session_id` and `source`, for
  example `openclaw-trove:agent:sammy:<source-session>` or
  `openclaw-jsonl:agent:sammy:<source-session>`
- the SQLite default provenance identity is the source `conversations.session_id`,
  preserving source session boundaries even when many conversations share one
  `session_key`
- pass `--session-identity session_key` only for SQLite imports when you
  intentionally want conversations with the same source session key grouped into
  one imported TROVE session
- reruns are idempotent for the same `--import-id`; the default `import_id` is
  source-path-derived, so pass a stable `--import-id` if you may import the same
  copied DB or JSONL export set from different paths
- `--json` prints a reconciliation report with `scanned`, `eligible`,
  `would_import`, `imported`, `skipped_existing`, `skipped_empty`,
  `invalid_rows`, `warnings`, and summary counters
- changing `--agent`, `--namespace`, or `--session-identity` under the same
  `--import-id` is treated as the same import and will skip already-tracked
  source messages; use a new `--import-id` for a different mapping
- no OpenClaw config or separate secret tables are imported, but raw transcripts,
  summaries, and tool payloads may contain sensitive user data

This is a local archive migration path. It does not make TROVE a general memory
provider, and it does not change the current-session retrieval contract for
agent tools.

For databases that already contain large historical tool rows,
`scripts/backfill_externalized_tool_outputs.py` can pre-create native recovery
sidecars without rewriting SQLite. It is dry-run by default, writes a scrubbed
digest/count manifest, is idempotent on repeat apply, and supports guarded
manifest-based rollback. See the [operator guide](docs/operator-guide.md#historical-tool-output-sidecars).

## Troubleshooting

### `hermes plugins` shows `trove (not found)` but TROVE tools exist

If `plugins.enabled` contains `hermes-trove`, `context.engine: trove` is set, and
the runtime exposes TROVE tools, TROVE is loaded. The `trove (not found)` line is a
Hermes host discovery/status mismatch, not an TROVE storage or compaction failure.

### `/trove status` looks unbound after restart

Compatible Hermes gateway hosts expose task-local lane metadata while
dispatching `/trove`. Once that lane has constructed an agent, `/trove status`
resolves the same active TROVE runtime as `trove_status`, including its foreground
view while an ignored or stateless side channel is bound. Before the first
normal message constructs an agent, or in a genuinely sessionless process,
`session_id: (unbound)` and `threshold_tokens: (uninitialized)` remain expected.

## Architecture

The engine sits between Hermes context assembly and the backing conversation
store. It records raw messages, compacts old material into a summary DAG, and
exposes retrieval tools that can drill back into exact stored sources.

<p align="center">
  <img src="docs/architecture.png" alt="hermes-trove architecture" width="700">
</p>

## How it works

1. **Ingest** - persist each message in SQLite with FTS metadata
2. **Compact** - summarize older messages outside the fresh tail into D0 leaf
   nodes
3. **Condense** - merge same-depth nodes into higher-depth summaries
4. **Escalate** - shrink oversize summaries from detailed to bullets to
   deterministic truncate
5. **Assemble** - combine system prompt, highest-depth summaries, and fresh tail
6. **Retrieve** - use TROVE tools to drill into compacted history or synthesize
   from expanded context

## Documentation

- [Feature overview](docs/features-overview.md) — every feature family, what
  it does, why it exists, and the switch that enables it
- [Agent configuration profiles](docs/agent-config-profiles.md) — copy-paste
  env profiles: coding agent, long-horizon assistant, fully-local, cost-guarded
- [Operator guide](docs/operator-guide.md) — install, activation, full
  configuration reference, diagnostics
- [Retrieval tools reference](docs/retrieval-tools.md) — exact tool contracts
- [Embeddings setup](docs/embeddings-setup.md) — free-tier and local embedding
  providers, warmup, backfill
- [TROVE paper](https://papers.voltropy.com/TROVE)
- [Architecture diagram](docs/architecture.png)
- [Standard compression diagram](docs/standard_compression.png)
- [TROVE compression diagram](docs/trove_compression.png)
- [Contributing guide](CONTRIBUTING.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
- [Releases](https://github.com/misterdas/hermes-trove/releases)

## Development

Important files:

```text
plugin.yaml      manifest
__init__.py      plugin registration and optional slash-command registration
engine.py        TROVEEngine main orchestrator
store.py         SQLite message store and FTS
dag.py           summary DAG and FTS
config.py        env var defaults and overrides
command.py       /trove command handlers
tools.py         trove_grep, trove_load_session, trove_describe, trove_expand, trove_expand_query
schemas.py       tool schemas shown to the model
tests/           standalone pytest coverage
```

Run tests:

```bash
pip install pytest
python -m pytest tests/ -v
```

No Hermes Agent checkout is required for the test suite; tests include a
lightweight ABC stub.

## Contributing

Issues and PRs welcome. Bug fixes and correctness improvements are highest
priority. New features should be scoped, backwards-compatible, and tested.

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch, validation, and PR guidance.
See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for project conduct expectations
and [SECURITY.md](SECURITY.md) for vulnerability reporting.
See the [releases page](https://github.com/misterdas/hermes-trove/releases)
for changelogs.

## License

[MIT](LICENSE)

## Star history

<a href="https://www.star-history.com/?repos=misterdas%2Fhermes-trove&type=timeline&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=misterdas/hermes-trove&type=timeline&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=misterdas/hermes-trove&type=timeline&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=misterdas/hermes-trove&type=timeline&legend=top-left" />
 </picture>
</a>
