"""TROVE auto-embedding background worker (F1).

After a burst of ingests, waits N seconds of quiescence before running
one bounded batch of pending SUMMARY-CORPUS embeddings on a daemon
thread. This makes newly ingested conversations searchable by semantic
retrieval without the operator manually running ``/trove embed backfill``
for the summary corpus. The chunk corpus is intentionally left to the
manual, consent-gated backfill command.

The worker is deliberately simple: it reuses the existing lease,
inflight, provider, and publish machinery from ``command.py``. It only
adds a debounce timer and a single-batch dispatch. All the hard
correctness work (lease CAS, identity validation, per-row savepoints,
provider batching) is already done there.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from typing import Any, Optional

from . import command as command_mod
from .command import (
    _EMBEDDING_BACKFILL_BATCH_SIZE,
    _EMBEDDING_BACKFILL_CLAIM_KEY,
    _acquire_embedding_backfill_lease,
    _embedding_backfill_heartbeat_s,
    _embedding_backfill_lease_ttl_s,
    _embedding_current_profile,
    _embedding_pending_rows,
    _embedding_read_connection,
    _ensure_inflight_table,
    _mark_dispatched,
    _mark_inflight,
    _prepare_inflight_for_lease,
    _provider_document_batches,
    count_tokens,
)
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class _EmbedAutoBackfillScheduler:
    """Process-wide debounced auto-embedding scheduler.

    Each :meth:`schedule_auto_backfill` call resets a debounce timer, so a
    burst of ingests coalesces into a single background run. The run
    executes on a daemon thread and never blocks the LLM turn.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # One debounce timer per engine prevents a busy profile from cancelling
        # another profile's pending auto-backfill. The worker remains serialized
        # by _in_flight and the existing database lease.
        self._timers: dict[int, threading.Timer] = {}
        self._timer: Optional[threading.Timer] = None
        self._in_flight = False
        self._shutdown = False

    def schedule_auto_backfill(self, engine: Any) -> None:
        """Schedule (or reschedule) the auto-backfill timer.

        Called from the ``post_llm_call`` hook. Each call resets the
        timer, so a burst of ingests coalesces into a single backfill
        run. Returns immediately; the timer fires on a daemon thread.
        """
        config = getattr(engine, "_config", None)
        if config is None:
            return
        if not bool(getattr(config, "embed_auto_backfill_enabled", True)):
            return
        if not bool(getattr(config, "embeddings_enabled", False)):
            return

        debounce_s = float(getattr(config, "embed_auto_backfill_debounce_s", 30.0))
        if debounce_s <= 0:
            debounce_s = 30.0

        engine_key = id(engine)
        with self._lock:
            if self._shutdown:
                return
            previous = self._timers.get(engine_key)
            if previous is not None:
                previous.cancel()
            timer = threading.Timer(
                debounce_s, self._run_auto_backfill, args=(engine,)
            )
            self._timers[engine_key] = timer
            # Keep the legacy single-timer attribute for diagnostics/tests; it
            # represents the most recently scheduled engine, not the queue.
            self._timer = timer
            timer.daemon = True
            timer.start()

    def _run_auto_backfill(self, engine: Any) -> None:
        """Run one bounded batch of pending embeddings."""
        engine_key = id(engine)
        with self._lock:
            if self._shutdown:
                return
            if self._timers.get(engine_key) is not None:
                self._timers.pop(engine_key, None)
            if self._in_flight:
                # A previous run is still going; reschedule this engine briefly
                # so its remaining work is not lost behind the other profile.
                timer = threading.Timer(
                    5.0, self._run_auto_backfill, args=(engine,)
                )
                self._timers[engine_key] = timer
                self._timer = timer
                timer.daemon = True
                timer.start()
                return
            self._in_flight = True

        has_more = False
        try:
            has_more = self._do_auto_backfill(engine)
        except Exception as exc:  # noqa: BLE001 — background worker, log and move on
            logger.debug("TROVE auto-backfill error: %s", exc)
        finally:
            with self._lock:
                self._in_flight = False
        if has_more:
            self.schedule_auto_backfill(engine)

    def _do_auto_backfill(self, engine: Any) -> bool | None:
        """Run one bounded batch; return whether more work remains."""
        config = engine._config
        db_path = engine._store.db_path
        has_more = False

        # --- Fast path: check pending count (read-only, cheap, no lock) ---
        try:
            read_conn = _embedding_read_connection(db_path)
        except sqlite3.Error as exc:
            logger.debug("TROVE auto-backfill: cannot open db: %s", exc)
            return

        try:
            profile = _embedding_current_profile(read_conn)
            if profile is None:
                logger.debug("TROVE auto-backfill: no active profile")
                return
            identity = str(profile["identity_hash"])
            model = str(profile["model_name"])
            provider_name = str(profile["provider"])
            pending, rows = _embedding_pending_rows(
                read_conn, identity, _EMBEDDING_BACKFILL_BATCH_SIZE
            )
        finally:
            read_conn.close()

        if pending == 0:
            logger.debug("TROVE auto-backfill: nothing pending")
            return

        # --- Slow path: open read-write connection and acquire lease ---
        store = VectorStore(db_path, config=config)
        rw_conn = store.connection
        if rw_conn is None:
            logger.debug("TROVE auto-backfill: store connection is None")
            store.close()
            return
        try:
            _ensure_inflight_table(rw_conn)

            ttl_s = _embedding_backfill_lease_ttl_s()
            heartbeat_s = _embedding_backfill_heartbeat_s()
            lease = _acquire_embedding_backfill_lease(
                rw_conn, ttl_s=ttl_s, heartbeat_s=heartbeat_s
            )
            if lease is None:
                logger.debug("TROVE auto-backfill: lease held, skipping")
                return

            try:
                _prepare_inflight_for_lease(rw_conn, identity, lease)
                captured_identity = store.capture_identity(model, provider=provider_name)
                if captured_identity.identity_hash != identity:
                    logger.debug("TROVE auto-backfill: identity changed, aborting")
                    return

                # Re-query pending rows under the lease so the batch reflects
                # the state at claim time, not a stale pre-claim snapshot.
                pending, rows = _embedding_pending_rows(
                    rw_conn, identity, _EMBEDDING_BACKFILL_BATCH_SIZE
                )
                if pending == 0:
                    logger.debug("TROVE auto-backfill: nothing pending after lease")
                    return

                documents = [
                    (str(row["node_id"]), str(row["summary"]), count_tokens(row["summary"]))
                    for row in rows
                ]

                provider = command_mod.resolve_provider(config, for_backfill=True)
                if provider is None:
                    logger.debug("TROVE auto-backfill: provider not configured")
                    return
                if (
                    provider.model_id != model
                    or str(provider.provider_id).lower() != provider_name.lower()
                ):
                    logger.debug("TROVE auto-backfill: provider mismatch")
                    return

                # Run one bounded batch through the existing machinery.
                batch = documents[:_EMBEDDING_BACKFILL_BATCH_SIZE]
                _mark_inflight(rw_conn, identity, lease, [item[0] for item in batch])

                request_by_index: dict[int, str] = {}
                latest_request_id: Optional[str] = None

                def before_dispatch(indexes: tuple[int, ...]) -> None:
                    nonlocal latest_request_id
                    normalized = tuple(int(index) for index in indexes)
                    if (
                        not normalized
                        or len(set(normalized)) != len(normalized)
                        or any(index < 0 or index >= len(batch) for index in normalized)
                        or any(index in request_by_index for index in normalized)
                    ):
                        raise RuntimeError("provider dispatched invalid indexes")
                    request_id = uuid.uuid4().hex
                    ids = [batch[index][0] for index in normalized]
                    if _mark_dispatched(rw_conn, identity, lease, ids, request_id) != len(ids):
                        raise RuntimeError("lease lost before dispatch")
                    for index in normalized:
                        request_by_index[index] = request_id
                    latest_request_id = request_id

                accepted_indexes: set[int] = set()
                for accepted_batch in _provider_document_batches(
                    provider, [item[1] for item in batch], before_dispatch=before_dispatch
                ):
                    if len(accepted_batch.indexes) != len(accepted_batch.vectors):
                        raise RuntimeError("provider returned mismatched indexes/vectors")
                    publish_rows = []
                    for batch_index, vector in zip(
                        accepted_batch.indexes, accepted_batch.vectors
                    ):
                        index = int(batch_index)
                        if index < 0 or index >= len(batch) or index in accepted_indexes:
                            raise RuntimeError("provider returned invalid index")
                        accepted_indexes.add(index)
                        item = batch[index]
                        request_id = request_by_index.get(index)
                        if request_id is None:
                            raise RuntimeError("provider returned embedding before dispatch")
                        publish_rows.append(
                            (item[0], "summary", model, tuple(vector), request_id)
                        )

                    store.publish_embedding_batch_under_lease(
                        publish_rows,
                        identity=captured_identity,
                        claim_key=_EMBEDDING_BACKFILL_CLAIM_KEY,
                        lease_id=lease.lease_id,
                        generation=lease.generation,
                    )

                has_more = pending > len(accepted_indexes)
                logger.debug(
                    "TROVE auto-backfill: embedded %d documents (pending: %d)",
                    len(accepted_indexes),
                    pending,
                )
            finally:
                lease.release()
        finally:
            store.close()
        return has_more

    def shutdown(self) -> None:
        """Stop accepting new debounces.

        Any in-flight batch is allowed to finish or abandon its lease
        (the lease TTL guarantees another worker can take over). The
        timer is cancelled so no new runs are scheduled.
        """
        with self._lock:
            self._shutdown = True
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()
            self._timer = None


# Process-wide scheduler instance. There is one per Hermes process, shared
# by all TROVE engines, exactly like _ROLLUP_MAINTENANCE_SCHEDULER.
_EMBED_AUTO_BACKFILL_SCHEDULER = _EmbedAutoBackfillScheduler()


def schedule_auto_backfill(engine: Any) -> None:
    """Schedule (or reschedule) the auto-backfill timer.

    Called from the ``post_llm_call`` hook. Returns immediately.
    """
    _EMBED_AUTO_BACKFILL_SCHEDULER.schedule_auto_backfill(engine)


def shutdown_auto_backfill() -> None:
    """Shutdown the process-wide auto-backfill scheduler."""
    _EMBED_AUTO_BACKFILL_SCHEDULER.shutdown()
