# Embeddings setup — free and local options

Semantic and hybrid retrieval are **opt-in** and need an embedding provider. You have three good
options, two of which cost nothing. If you configure nothing, nothing changes — retrieval stays
FTS-only exactly as before.

## TL;DR

| Option | Cost | Signup | Install | Best for |
|---|---|---|---|---|
| Voyage AI | free tier (200M tokens for the voyage-4 group), then ~$0.02–0.12 per million tokens | yes (no credit card) | none | best quality, zero local footprint |
| fastembed | $0 | no | `pip install fastembed` (ONNX, no torch) | local default — no accounts, no daemon |
| Ollama | $0 | no | Ollama app/daemon | you already run Ollama |

All providers feed the same store. Switching provider/model is one config change plus a backfill;
each full identity keeps its own vectors, so switching **back** to a
previously-registered provider reactivates its vectors with no re-backfill (see *Switching or
removing providers*).

## Option 1 — Voyage AI (free tier)

The Voyage-4 generation (`voyage-4`, `voyage-4-large`, `voyage-4-lite`) carries **200 million free
tokens for that model group**, and signup does not require a credit card. For a personal TROVE corpus
(thousands of summaries), that free allotment covers initial backfill and years of queries; past it,
embedding costs $0.02/M (`voyage-4-lite`), $0.06/M (`voyage-4`), or $0.12/M (`voyage-4-large`). A
query embeds exactly one short vector, so a semantic or hybrid query costs a fraction of a cent.
Voyage documents its allotments and rate tiers on its
[pricing](https://docs.voyageai.com/docs/pricing) and
[rate-limits](https://docs.voyageai.com/docs/rate-limits) pages — treat those as the source of
truth; the numbers above were verified 2026-07.

```bash
export VOYAGE_API_KEY=...           # from dash.voyageai.com
export TROVE_EMBEDDINGS_ENABLED=true
export TROVE_EMBEDDING_PROVIDER=voyage
export TROVE_EMBEDDING_MODEL=voyage-4-lite   # or voyage-4 / voyage-4-large
/trove embed warmup                   # probes the API, registers model + dimensions
/trove embed backfill                 # dry run: shows counts + estimated tokens, writes nothing
/trove embed backfill --apply         # embeds your history in bounded batches
```

Notes: requests are batched under Voyage's caps — both the token budget and the 1000-item
per-request cap; over-length documents are skipped and reported, never silently truncated;
rate-limit responses are honored with bounded waits under one absolute per-operation deadline.
That deadline starts before document conversion/token counting and covers every split, retry, and
backoff, plus response decoding and validation. Automatic HTTP resend is deliberately narrower:
Voyage `429` is an authoritative rejection and may be retried within the remaining deadline, but a
timeout/network failure or `5xx` after transport starts may follow remote acceptance and is never
automatically resent. A deadline that expires after durable dispatch marking but before transport
is a typed `not started` outcome, so backfill can safely clear those exact rows. Individual
token-count calls run in bounded workers; an overrun returns at the deadline without dispatching a
later request (the timed-out worker may finish in the background while holding one of the fixed
worker slots).

## Option 2 — fastembed (local, no signup, recommended local default)

[fastembed](https://github.com/qdrant/fastembed) runs ONNX models on CPU with no PyTorch and no
external service — just a pip package.

```bash
pip install fastembed
export TROVE_EMBEDDINGS_ENABLED=true
export TROVE_EMBEDDING_PROVIDER=fastembed
export TROVE_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5   # 384-dim, compact and quick on CPU
/trove embed warmup     # downloads the model ONCE, explicitly (a few hundred MB incl. onnxruntime)
/trove embed backfill --apply
```

The model download happens **only** during `warmup` — never lazily during a query or an agent turn.
If you skip warmup, semantic search simply stays off and the tools tell you why. Queries use the
model's query-specific encoding (distinct from document encoding) so query/passage asymmetry is
preserved. When `TROVE_EMBEDDINGS_ENABLED=false`, `warmup` is inert: it does not resolve a provider,
download a model, create embedding tables, or create the configured database.

## Option 3 — Ollama (local daemon)

If you already run [Ollama](https://ollama.com), use its embeddings endpoint:

```bash
ollama pull nomic-embed-text        # 768-dim; mxbai-embed-large and bge-m3 also work
export TROVE_EMBEDDINGS_ENABLED=true
export TROVE_EMBEDDING_PROVIDER=ollama
export TROVE_EMBEDDING_MODEL=nomic-embed-text
# TROVE_OLLAMA_BASE_URL defaults to http://localhost:11434
/trove embed warmup && /trove embed backfill --apply
```

Ollama requests set `truncate: false`, so an input that exceeds the model's context fails loudly
rather than being silently truncated to a misleading embedding. As with Voyage, an Ollama
timeout/network failure after transport starts is acceptance-ambiguous and is not automatically
resent.

Bulk document embedding uses `TROVE_EMBEDDING_BACKFILL_TIMEOUT_S` as its
per-provider-operation deadline (120 seconds by default) for Voyage, Ollama,
and fastembed. This is intentionally separate from the latency-sensitive
`TROVE_EMBEDDING_QUERY_TIMEOUT_S` (3 seconds by default), so a normal document
batch or local model load is not aborted by the interactive query policy. The
optional `TROVE_EMBEDDING_BACKFILL_BUDGET_S` still caps the whole apply run
between batches (`0`, the default, means no whole-run cap); the lease and
post-call ownership CAS remain authoritative independently of both timeouts.

## What you get

`trove_grep` gains two modes on top of the existing ones:

- `semantic` — paraphrase-tolerant vector search; local providers make this $0
- `hybrid` — keyword ∪ semantic, fused with reciprocal-rank fusion (RRF); the best
  "have we discussed X?" mode. (Fusion is RRF only — there is no external reranker.)

Semantic and hybrid requests use one absolute deadline beginning at `trove_grep` entry. It includes
provider resolution, query embedding, optional NumPy import, bounded KNN, result hydration, any FTS
fallback, both hybrid arms, and fusion. A semantic failure can degrade to full-text with
`degraded_to_fts`; the fallback uses separate read-only SQLite connections with progress
interruption. If hybrid already computed FTS results before its semantic arm times out, it may
return that existing payload without starting new I/O. If no usable result exists when time runs
out, the request returns an explicit `timeout` error and starts no later fallback/arm. Provider
authentication errors also remain operator-visible instead of degrading. Explicit `full_text`
mode itself is unchanged and byte-for-byte identical to prior behavior.

Role, time, conversation, and broader-session filters degrade to raw FTS before provider work,
because summaries cannot prove those raw-message dimensions. Source is different: SQL first selects
a bounded candidate window, then verifies descendant source lineage inside that window. The lineage
walk is itself capped; a missing legacy `source` column or an over-budget lineage graph fails closed
with `unverifiable_provenance`, so it never becomes an allow-all. A source-filtered semantic result
therefore reports bounded coverage rather than claiming universal pre-bound source coverage.

## Performance & footprint

- Vector math is dependency-free by default; installing **numpy** (optional) accelerates large
  corpora — the top-k scan is milliseconds warm once numpy is loaded.
- Metadata/id resolution uses a temp-table join rather than a giant `IN (...)` list, so it scales
  past the SQLite host-parameter limit that previously failed near ~32k ids (validated to 40k).
- Without numpy, search scans the most recent `TROVE_EMBEDDING_BOUNDED_SCAN_ROWS` vectors (default
  2,000) and reports `coverage: bounded`. The candidate enumeration is bounded at the SQL layer
  (`ORDER BY` recency `+ LIMIT`), so a large corpus never materializes every id in host memory.
- With numpy, the cache is still only for that bounded candidate set and is keyed by canonical
  identity, transactional `data_version`, and candidate ids; it is not a corpus-sized matrix.

### Small hosts: cap the CPU a backfill takes

With a local provider (fastembed, or Ollama on the same box) a backfill is real CPU work, and it
runs in a background thread **while your agent is answering you**. By default the ONNX runtime
takes every core for the duration of a batch, so a 2-core host has nothing left for the
interactive turn. Set:

```bash
TROVE_EMBEDDING_THREADS=1
```

Measured on a 2-core host, one config per process (the ONNX thread pool is global, so measuring
both in one process would contaminate the second reading):

| `TROVE_EMBEDDING_THREADS` | 64 texts, wall | 64 texts, CPU | cores in use |
|---|---|---|---|
| unset (`0`) | 4.222s | 8.292s | ~1.96 |
| `1` | 7.967s | 7.954s | ~1.00 |

The trade is **wall-clock, not total CPU** — same CPU-seconds, about 1.9x slower per batch, and
the backfill yields half the machine. A backlog therefore drains in more, shorter passes; the
worker re-arms itself while work remains, so it still completes unattended. `0` (the default)
omits the argument entirely rather than passing `0`, so behaviour on hosts that do not set it is
byte-identical. Ignored when the provider is not fastembed — voyage and ollama take no such
argument.

The cap is applied when the model session is constructed, not per call, because fastembed builds
the ONNX session once and reuses it. Change it and restart the process.

### When the auto-backfill does not run

The background worker arms a debounce timer after each completed turn, and a new turn re-arms it.
During a rapid back-and-forth the timer keeps being reset, so the run only starts once messages
stop for `TROVE_EMBED_AUTO_BACKFILL_DEBOUNCE_S` (default 30). On an active host lower it:

```bash
TROVE_EMBED_AUTO_BACKFILL_DEBOUNCE_S=5
```

A run already in progress is never cancelled by a new message — it queues one follow-up pass
instead — and a run that finds more work re-arms itself, so once it starts the backlog drains
without further messages. A manual `/trove embed backfill --apply` and the worker share one
lease, so they exclude each other rather than double-embedding; if you run one while the other
holds the lease you get `refused: another embedding backfill holds the lease`, which clears when
that run exits or its `TROVE_EMBEDDING_BACKFILL_LEASE_TTL_S` expires.

A backfill is a background best-effort job, not a queue: there is no persistence across a
process restart, so if you need coverage guaranteed at a point in time, run
`/trove embed backfill --corpus both --apply` yourself. It is resumable and idempotent, and stops
cleanly between batches at `TROVE_EMBEDDING_BACKFILL_BUDGET_S`, so repeating it is safe.

## Switching or removing providers

Change provider/model → run `/trove embed warmup` (registers the new profile as the current identity)
→ `/trove embed backfill --apply` (embeds under the new identity; the previous model's vectors are
kept separate and never mixed). Every vector is published under the exact identity that produced it
— the identity is captured at provider-resolution time and carried through the write, so switching
the active provider A→B mid-backfill can never rebind an A-vector onto B. If A becomes inactive
after its request was accepted but before publication, the exact A request is atomically moved to
`uncertain`, the still-owned backfill lease is released, and the run stops before another dispatch.
Because each identity
`(provider, model, revision, dim, dtype, byteorder, task)` owns its own vectors, switching **back**
to a previously-registered provider reactivates it with its existing vectors — no re-backfill
needed. The stored representation is currently restricted to `float32` / `little` / `summary`;
unsupported identity variants are rejected rather than normalized onto another profile.

Backfill records each actual remote dispatch durably. Every accepted sub-batch is published
immediately under the captured identity and current lease CAS. If remote acceptance is ambiguous or
local publication fails after acceptance, those rows become `uncertain` and normal discovery will
not bill them again. Recovery is deliberately operator-authorized:

```bash
/trove embed backfill --apply --retry-uncertain --limit 32
```

The authorization is bound to the exact oldest uncertain rows selected by that invocation, up to
`--limit`; the risky recovery run does not mix in ordinary pending rows. Their durable uncertainty
markers are not cleared before discovery or dispatch, and any row not successfully published
(including a skipped row, definitive rejection, budget stop, or lease loss) remains `uncertain` for
another explicit decision. The command reports the uncertain count and warning because retrying may
rebill. Disable everything with `TROVE_EMBEDDINGS_ENABLED=false` — data stays, behavior reverts to
FTS-only instantly.

## The chunk corpus — raw verbatim text, and the consent gate

`/trove embed backfill` has two corpora, selected with `--corpus`:

- `summary` (default) — embeds the generated **summaries** of your history.
- `chunks` — embeds **raw, verbatim message text**, chunked by `--policy`
  (`conversational` | `heads` | `full`), for verbatim/chunk-KNN recall.
- `both` — runs the summary backfill, then the chunk backfill, in one command.

The distinction matters for privacy. The summary corpus sends only model-generated summaries to the
embedding provider. **The chunk corpus sends the raw message bytes** — including tool-result output
and error/traceback content (the `heads`/`full` policies specifically target error signatures) —
which is exactly the content most likely to carry secrets. When the provider is a **cloud** provider
(e.g. Voyage), that raw text leaves this machine.

Because of this, `--corpus chunks --apply` and `--corpus both --apply` **refuse** on a cloud provider
unless you pass an explicit acknowledgment:

```bash
/trove embed backfill --corpus chunks --apply --confirm-raw-text
```

Local providers (**fastembed**, **ollama**) never transmit text off the machine, so the gate is
waived for them. Dry runs (no `--apply`) never send anything and never require the flag.

> **Redaction caveat.** `TROVE_SENSITIVE_PATTERNS_ENABLED` redaction runs at **ingest** time, so it
> only affects text stored *after* it was enabled. Turning it on does **not** retro-redact history
> already in the store — that older raw text is still what gets sent to the provider during a chunk
> backfill. Prefer a local provider for the chunk corpus if the history may contain secrets.
