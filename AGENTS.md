# AGENTS.md — Hermes Developer Guide: `hermes-trove`

> Lossless Context Management plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).
> Bounded context, unbounded memory. Nothing is ever lost.
>
> This file is the working manual for AI coding agents (and humans) modifying this repo.
> Read it before writing code. The repo carries its own conventions in `pyproject.toml`
> (ruff) and `.github/workflows/ci.yml` (CI contract) — those two are the source of truth
> for "what passes"; this file is the source of truth for "how we work here".

## 1. What this plugin is

`hermes-trove` is a **context engine** (not just a memory store) that plugs into Hermes via
the `agent.context_engine` slot. It:

- **Ingests** every message into a lossless SQLite store (`messages`), organized in a
  **DAG of summary nodes** (`summary_nodes` in `dag.py` / `store.py`) — leaf chunks roll up
  into higher-depth summaries; raw messages are never destroyed, only compressed.
- **Compacts** the live prompt boundedly (`compaction.py`, `fresh_tail.py`,
  `placeholder_ledger.py`) so the model sees a small, cache-friendly window while every
  dropped detail stays recoverable.
- **Recalls** on demand through two tools:
  - `trove_grep` — single-arm query: full-text (FTS5) OR semantic (vector KNN) per request,
    with **explicit, reported degradation** between the two.
  - `trove_recall` — three-arm hybrid: FTS arm + summary-vector arm + chunk-vector arm,
    fused with RRF (`retrieval_core.py`), optional Voyage rerank, then hydrated back to
    message/summary rows.
- **Optionally** runs semantic side-features, all default-off: proactive recall injection
  at prompt assembly, pre-answer evidence, extraction, assertions, query views.

Design invariant, repeated in code comments: **degrade, never silently fail.** Any
retrieval arm that cannot run (disabled, unconfigured provider, timeout, empty corpus)
must surface that in the tool result (`coverage`, `degraded_reasons`,
`degraded_to_fts`) so an LLM caller can tell "I searched everything" from "only the
keyword arm ran".

## 2. Repo layout (read these before touching anything)

| Path | Role | Notes |
|---|---|---|
| `engine.py` | `TROVEEngine` — the ContextEngine implementation (~6.9k LOC, mixin-based) | Mixins: `CompactionMixin`, `ResetStateMixin`, `ReconcileMixin`, `AuxiliarySessionMixin`, `PlaceholderLedgerMixin`, `BypassMixin`. Do not split without an issue. |
| `tools.py` | Tool implementations: `trove_grep`, `trove_recall`, `trove_inspect`, `trove_recent`, `trove_rotate` + every retrieval arm (~6.9k LOC) | The degradation contract lives here. |
| `retrieval_core.py` | Pooled `VectorStore` handling + `run_knn`/`run_chunk_knn` + RRF fusion + hydration | Connection pooling with deadline guards. |
| `vector_store.py` | Brute-force/two-stage KNN over `trove_embedding_profile` tables; int8 + sign-bit prescreen (SPEC C1) | Identity-hash isolation: vectors from different embedding models never mix. |
| `embedding_provider.py` | `resolve_provider()` → voyage / ollama / fastembed; spend guard + circuit breaker | Backfill and query paths have **different** budgets — keep them different. |
| `store.py`, `db_bootstrap.py`, `schemas.py` | Raw message store, WAL/JournalMode, FTS5 index, schema migration steps | Schema changes MUST go through a migration step (`mark_migration_step_complete`); never ALTER live DB. |
| `trajectory_store.py`, `compaction.py`, `rollup_*.py` | History trajectories, compaction math, rollup builders/stores | |
| `retention.py` | Session retention (v1.2.0): delete stale raw messages, keep summary nodes; `TROVE_RETENTION_DAYS=0` = keep forever | Single-gate model; pin guard is transactional. |
| `embed_worker.py` | Background embedding backfill worker (F1) + `embed status` command (F2) | |
| `config.py` | `TROVEConfig` dataclass + `ENV_FIELD_SPECS` env-driven loading | **All** config is env-var-driven (`TROVE_*`); `hermes config.yaml` keys are intentionally ignored (see `ignored_config_yaml_trove_keys`). |
| `tests/` | ~101 test files, pytest | `conftest.py` isolates `HERMES_HOME` + stubs `hermes_trove` package registration. |
| `benchmarks/`, `benchmarking/` | LongMemEval-style harnesses, replay fixtures, QA bridge | Slow; not in default CI. |
| `scripts/install.sh` / `uninstall.sh` / `update.sh` | Catalog install lifecycle | Writes `TROVE_ENABLE_SLASH_COMMAND=1` + `TROVE_RETENTION_DAYS=0` into `~/.hermes/.env`. |

## 3. Ground rules for changes

1. **The degradation contract is load-bearing.** Every change in `tools.py` retrieval
   paths must preserve: (a) the arm that dies reports *why* via `coverage`/
   `degraded_reasons`; (b) hybrid never falls below its best arm; (c) FTS stays
   default-on value. If you add a new gate, add its reason string and a test asserting
   the payload carries it.
2. **Deadlines, not sleeps.** Every blocking operation runs under an absolute
   `deadline` (monotonic) with progress-handler guards on SQLite connections
   (`set_progress_handler`). Never introduce `time.sleep` in hot paths; never drop a
   deadline check when refactoring.
3. **Config is env-only.** New knobs go in `ENV_FIELD_SPECS` with a `TROVE_`-prefixed
   env var and a dataclass field in `TROVEConfig` with an explicit default. Default-off
   is the norm for anything semantic/cost-bearing. Document the default in
   `docs/operator-guide.md` table.
4. **Schema migrations only.** Any new table/column: add a migration step
   (`ensure_*` + `mark_migration_step_complete` in `db_bootstrap.py`), keep it
   idempotent, and test the fresh-DB + upgrade paths. WAL sidecar files (`-wal`,
   `-shm`) are never deleted on size alone — the t3 incident (cf4ce2961f): an open
   idle handle at a 0-byte WAL boundary caused split-brain "file is not a database".
   Cleanup must prove no attachment via `/proc/*/fd` before unlink.
5. **Embedding identity isolation.** Vectors are keyed by `(provider, model, dim, dtype,
   revision)` identity hash (`vector_store.py`). A query embedded with model A must
   never score against model B's vectors. Test any change that touches identity with
   both an identity-mixed store and a single-identity store.
6. **Tests first, and real.** Tests use the `conftest.py` isolation fixture (fresh
   `HERMES_HOME`, stub `agent.context_engine` ABC). For retrieval work, the repo's
   LongMemEval harness (`benchmarking/longmemeval.py`) is the recall guarantee — when
   recall quality matters, measure it, don't eyeball it.
7. **CI contract:** `ruff check .` (rule set in `pyproject.toml`: E4/E7/E9/F, no E501,
   no isort — historical decision, don't reformat wholesale) on Python 3.11, pytest on
   3.11 + 3.14, actionlint on workflows. Locally:
   ```bash
   python3.11 -m venv .venv-dev && .venv-dev/bin/pip install pytest numpy ruff==0.15.13
   # CI also stubs agent/context_engine.py — needed locally if importing engine directly
   .venv-dev/bin/python -m pytest -q -p no:cacheprovider
   .venv-dev/bin/ruff check .
   ```
8. **No secrets in code.** Provider keys come from the operator's environment. CI
   must not require any API key: local providers (fastembed) are the default test path;
   Voyage tests run against fakes.

## 4. Known sharp edges (bite marks from incident history)

- **`TROVE_EMBEDDINGS_ENABLED` defaults to `false`** (config.py:673). A stock install is
  **FTS-only**: every semantic arm reports `coverage: disabled` and results degrade to
  full-text. This is *by design* (default-off = zero-cost, byte-identical assembly) but
  it is the #1 "recall is degraded" complaint. Full semantic operation requires the
  operator to set in their env:
  ```bash
  TROVE_EMBEDDINGS_ENABLED=true
  TROVE_EMBEDDING_PROVIDER=fastembed        # or voyage / ollama
  TROVE_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
  ```
  Then run the backfill (`trove embed backfill --apply` / embed worker F1) so the
  corpus has vectors. `install.sh` deliberately does NOT write these — changing that
  default is a release decision, not a code change.
- **`trove_grep` semantic mode degrades to FTS on:** broader session scope, time filters,
  conversation filters, `role=` filter, `content_scope != history`, disabled embeddings,
  unconfigured provider, query-embed failure, deadline exhaustion, or `coverage=none`
  (no vectors stored). Each has a distinct `degraded_reason` — grep for
  `degraded(` in `tools.py` for the full list before adding gates.
- **Spend guard:** query-path embeds run under a sliding-window
  `EmbeddingSpendGuard` (default 600 calls/60s, `max_calls=0` disables). The backfill
  path uses `max_calls=0`. Don't unify them.
- **int8 two-stage KNN** (`vector_store.py`, SPEC C1): binary Hamming prescreen →
  int8 rescore. Prescreen is *approximate* by construction (recall@M guarantee, not
  exact top-k) — `knn_prescreen_multiplier` controls stage-1 breadth.
- **Retention** (v1.2.0): `TROVE_RETENTION_DAYS=0` is a **no-op** (keep forever).
  Deletion holds the store write lock in both paths; pinned sessions are guarded
  transactionally.
- **`plugin.yaml` declares `python_runtime: external`** — the Hermes package manager
  must NOT uv-lock this repo's dependencies. The plugin's `__init__.py` re-exports the
  `hermes-trove` PyPI package.

## 5. Working style in this repo

- Branch naming: `feat/<topic>`, `fix/<topic>`; release work lands on a
  `feat/v<version>-maintainer-notes` branch, merges feature PRs, then the release.
- Commit messages are type-prefixed and specific: `fix:`, `feat:`, `lint:`, `ci:`,
  `chore:`, `config:`, `security:` — with the affected spec/issue tag when one exists
  (`F1`, `F2`, `C1`, `C4` are the doctor-guardrail / embed-feature identifiers from the
  catalog review).
- Before shipping: `ruff check .`, full pytest on both Python legs, and for any
  retrieval change the LongMemEval fastembed metrics JSON under `benchmarks/results/`.
- When in doubt, read the code comment above the gate you're about to change — the
  comments in this repo are unusually deliberate (they record the *why* of every
  degradation and deadline). Treat them as spec, not prose.
