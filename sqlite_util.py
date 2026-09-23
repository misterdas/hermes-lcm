"""SQLite lock-contention helpers shared by the TROVE engine.

Isolated from ``engine.py`` (WS5 seam): lock-contention detection, bounded
``busy_timeout`` changes, and transaction-preserving savepoints are pure SQLite
concerns with no engine state. Callers keep their own policy constants (for
example the session-end timeout budget).
"""

from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path
import sqlite3
import stat
import uuid
from contextlib import contextmanager
from typing import Iterator, List


_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


# --- stale shared-memory cleanup (t3) ------------------------------------
#
# The -shm file is pure derived state: a rebuildable index over -wal frames.
# When the -wal goes missing or empty (checkpoint completed, process killed
# mid-recovery, stale sidecars from an earlier incarnation), a leftover -shm
# can confuse the next opener's shm attach and is never load-bearing. Drop it
# before SQLite opens the database. This deliberately never touches the main
# .db file or a non-empty -wal (which may still hold uncheckpointed frames).
#
# Multi-process safety (2026-09-23 split-brain regression): a -wal can be
# legitimately 0 bytes at a checkpoint boundary while sibling connections
# (same OR different process) still use the -shm as their lock/index region.
# Unlinking it under them desyncs their WAL-index view and surfaces later as
# "file is not a database". So cleanup only runs when we can prove the sidecar
# is orphaned: no process holds an open fd on the shm path. Lock probes are
# NOT sufficient — an idle WAL-mode connection holds no shm lock between
# transactions but is still attached — so we check open fds directly.


def _shm_has_any_attachment(shm_path: Path) -> bool:
    """True when ANY process (this one included) holds the shm path open.

    Scans /proc/*/fd readlink targets as path strings. Fail-closed: if the
    scan cannot run (no /proc, permissions), report attached and never delete.
    """
    try:
        self_fds = set(os.listdir("/proc/self/fd"))
    except OSError:
        return True
    target = str(shm_path)
    try:
        pids = [e for e in os.listdir("/proc") if e.isdigit()]
    except OSError:
        return True
    for pid in pids:
        fd_dir = f"/proc/{pid}/fd"
        try:
            entries = os.listdir(fd_dir)
        except OSError:
            continue
        for entry in entries:
            if pid == str(os.getpid()) and entry in self_fds:
                continue  # our own probe fds, checked via self_fds below
            try:
                opened = os.readlink(f"{fd_dir}/{entry}")
            except OSError:
                continue
            if opened == target:
                return True
    # This process's own fds (readlink under /proc/<pid>/fd can be blocked
    # for our own entries on hardened kernels; self_fds scan covers them).
    for entry in self_fds:
        try:
            if os.readlink(f"/proc/self/fd/{entry}") == target:
                return True
        except OSError:
            continue
    return False


def _remove_stale_shm_sidecar(db_path: Path) -> bool:
    """Remove a stale -shm sidecar when no connection is attached to it.

    Returns True when a file was removed. Never raises: cleanup is
    best-effort and SQLite itself tolerates (or rebuilds) a missing -shm.
    """
    try:
        shm = db_path.with_name(db_path.name + "-shm")
        wal = db_path.with_name(db_path.name + "-wal")
        try:
            shm_stat = shm.stat()
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(shm_stat.st_mode):
            return False
        try:
            wal_size: int | None = wal.stat().st_size
        except FileNotFoundError:
            wal_size = None
        if wal_size is not None and wal_size > 0:
            return False
        if _shm_has_any_attachment(shm):
            return False
        shm.unlink()
        return True
    except OSError:
        return False



def _sqlite_artifact_error(path: Path, reason: str) -> OSError:
    return OSError(errno.EPERM, f"refusing SQLite artifact {path.name!r}: {reason}", str(path))


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _validate_sqlite_artifact(path: Path, file_stat: os.stat_result) -> None:
    if not stat.S_ISREG(file_stat.st_mode):
        raise _sqlite_artifact_error(path, "not a regular file")
    if file_stat.st_nlink != 1:
        raise _sqlite_artifact_error(path, "link count is not one")


def _require_sqlite_artifact_absent(path: Path, *, directory_fd: int) -> None:
    """Accept a vanished sidecar only while its directory entry stays absent."""
    try:
        current = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    _validate_sqlite_artifact(path, current)
    raise _sqlite_artifact_error(path, "directory entry changed while opening")


def _open_private_sqlite_directory(path: Path) -> int:
    directory = path.parent
    expected = os.stat(directory, follow_symlinks=False)
    if not stat.S_ISDIR(expected.st_mode):
        raise _sqlite_artifact_error(path, "parent is not a regular directory")
    if expected.st_uid != os.getuid():
        raise _sqlite_artifact_error(path, "parent directory is not owned by the current user")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
    directory_fd = os.open(directory, flags)
    opened = os.fstat(directory_fd)
    if not stat.S_ISDIR(opened.st_mode) or not _same_file_identity(expected, opened):
        os.close(directory_fd)
        raise _sqlite_artifact_error(path, "parent directory changed while opening")
    return directory_fd


def _chmod_sqlite_artifact_at(
    path: Path,
    *,
    directory_fd: int,
    create: bool,
    allow_sidecar_disappearance: bool = False,
) -> bool:
    expected: os.stat_result | None
    try:
        expected = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        if not create:
            return False
        expected = None
    if expected is not None:
        _validate_sqlite_artifact(path, expected)

    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    if expected is None:
        flags |= os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
    except FileExistsError:
        if expected is not None:
            raise
        return _chmod_sqlite_artifact_at(
            path,
            directory_fd=directory_fd,
            create=False,
        )
    except FileNotFoundError:
        if not allow_sidecar_disappearance:
            raise
        _require_sqlite_artifact_absent(path, directory_fd=directory_fd)
        return False
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise _sqlite_artifact_error(path, "not a regular file")
        if expected is not None and not _same_file_identity(expected, opened):
            raise _sqlite_artifact_error(path, "directory entry changed while opening")
        if opened.st_nlink == 0 and allow_sidecar_disappearance:
            _require_sqlite_artifact_absent(path, directory_fd=directory_fd)
            return False
        _validate_sqlite_artifact(path, opened)
        os.fchmod(fd, 0o600)
        restricted = os.fstat(fd)
        if restricted.st_nlink == 0 and allow_sidecar_disappearance:
            _require_sqlite_artifact_absent(path, directory_fd=directory_fd)
            return False
        if restricted.st_nlink != 1:
            raise _sqlite_artifact_error(path, "link count changed while restricting permissions")
    finally:
        os.close(fd)
    return True


def _restrict_existing_sqlite_artifacts(db_path: Path) -> None:
    """Restrict verified, single-link SQLite files without following links."""
    if os.name != "posix":  # pragma: no cover - Windows compatibility fallback
        for artifact in (
            db_path,
            *(db_path.with_name(db_path.name + suffix) for suffix in _SQLITE_SIDECAR_SUFFIXES),
        ):
            try:
                artifact.chmod(0o600)
            except FileNotFoundError:
                continue
        return

    directory_fd = _open_private_sqlite_directory(db_path)
    try:
        _chmod_sqlite_artifact_at(
            db_path,
            directory_fd=directory_fd,
            create=False,
        )
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            _chmod_sqlite_artifact_at(
                db_path.with_name(db_path.name + suffix),
                directory_fd=directory_fd,
                create=False,
                allow_sidecar_disappearance=True,
            )
    finally:
        os.close(directory_fd)


def _prepare_private_sqlite_file(path: Path) -> None:
    """Create or tighten one SQLite file and its existing sidecars safely."""
    if os.name != "posix":  # pragma: no cover - Windows compatibility fallback
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(path, flags, 0o600)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            else:
                path.chmod(0o600)
        finally:
            os.close(fd)
        _restrict_existing_sqlite_artifacts(path)
        return

    directory_fd = _open_private_sqlite_directory(path)
    try:
        _chmod_sqlite_artifact_at(path, directory_fd=directory_fd, create=True)
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            _chmod_sqlite_artifact_at(
                path.with_name(path.name + suffix),
                directory_fd=directory_fd,
                create=False,
                allow_sidecar_disappearance=True,
            )
    finally:
        os.close(directory_fd)


def _is_sqlite_locked_error(exc: BaseException) -> bool:
    """Return True when an exception chain represents SQLite lock contention."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if isinstance(current, sqlite3.Error) and "locked" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _sqlite_busy_timeout_ms(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA busy_timeout").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


@contextmanager
def _sqlite_savepoint(conn: sqlite3.Connection) -> Iterator[None]:
    """Isolate helper writes without taking ownership of a caller transaction."""
    # UUID hex contains only identifier-safe characters and keeps every nested
    # helper's SAVEPOINT name unique with a fixed upper bound on name length.
    name = f"trove_{uuid.uuid4().hex}"
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield
    except BaseException:
        try:
            conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
        finally:
            conn.execute(f"RELEASE SAVEPOINT {name}")
        raise
    else:
        conn.execute(f"RELEASE SAVEPOINT {name}")


@contextmanager
def _temporary_sqlite_busy_timeout(
    connections: List[sqlite3.Connection | None],
    timeout_ms: int,
) -> Iterator[None]:
    """Temporarily bound SQLite lock waits for gateway-critical paths."""
    bounded_timeout = max(0, int(timeout_ms))
    originals: list[tuple[sqlite3.Connection, int]] = []
    for conn in connections:
        if conn is None:
            continue
        original = _sqlite_busy_timeout_ms(conn)
        conn.execute(f"PRAGMA busy_timeout={bounded_timeout}")
        originals.append((conn, original))
    try:
        yield
    finally:
        for conn, original in reversed(originals):
            conn.execute(f"PRAGMA busy_timeout={original}")


# --- startup index self-heal ---------------------------------------------
#
# Observed in production (Sep 2026): under multi-process WAL concurrency the
# ``sqlite_autoindex_metadata_1`` entry for a metadata row can go missing
# while the row itself remains present. ``PRAGMA quick_check`` does not catch
# it (it skips index-vs-table cross-checks); ``PRAGMA integrity_check`` does.
# Rather than quarantining or renaming the database (which would strand other
# processes' open file descriptors on deleted inodes), the store verifies on
# open and rebuilds the damaged index in place. Row data is never touched.

_STARTUP_SELF_HEAL_QUICK_CHECK = "quick_check"

# Cap on index rebuilds per store open: integrity_check is O(db size), and on
# a healthy DB this path costs one check at process boot (rare). A corrupt
# index is an exceptional event, so a single bounded rebuild attempt is right.
_STARTUP_SELF_HEAL_MAX_ATTEMPTS = 1


def _is_index_corruption_detail(detail: object) -> bool:
    """Return True when an integrity_check detail names a missing-wrong index row."""
    message = str(detail or "").lower()
    return bool(message) and ("missing from index" in message or "wrong # of entries in index" in message)


def _integrity_failure_details(conn: sqlite3.Connection) -> list[str]:
    """Run PRAGMA integrity_check and return the raw failure detail rows."""
    try:
        rows = conn.execute("PRAGMA integrity_check").fetchall()
    except sqlite3.Error:
        return []
    details = [str(row[0]) for row in rows if row and str(row[0]).lower() != "ok"]
    return details


def startup_index_self_heal(conn: sqlite3.Connection) -> dict[str, object]:
    """Verify b-tree/index consistency on open, rebuilding damaged indexes.

    Runs the full ``PRAGMA integrity_check`` cross-check on every store open
    (process boot is rare, so the O(db size) cost is acceptable — and it is
    the only pragma that catches index-vs-table drift; ``quick_check``
    provably misses this fault class in production). When the failure details
    name only missing/wrong index rows, it runs the minimal in-place
    ``REINDEX``. Data rows are never modified, the database file is never
    renamed or moved, and sibling processes' open connections are unaffected.
    Genuine page-level or structural damage is never auto-repaired; it is
    returned for the doctor to surface.

    Returns ``{"healed": bool, "details": [...]}``.
    """
    details = _integrity_failure_details(conn)
    if not details:
        return {"healed": False, "details": []}
    if not any(_is_index_corruption_detail(detail) for detail in details):
        # Genuine page-level or structural damage — not an index-only fault.
        # Never attempt to "fix" that automatically; surface it to the doctor.
        return {"healed": False, "details": details}
    healed = False
    for _ in range(_STARTUP_SELF_HEAL_MAX_ATTEMPTS):
        try:
            conn.execute("REINDEX")
        except sqlite3.Error:
            break
        remaining = _integrity_failure_details(conn)
        if not remaining:
            healed = True
            details = []
            break
        details = remaining
    return {"healed": healed, "details": details}
