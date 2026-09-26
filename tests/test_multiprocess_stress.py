"""Multi-process concurrency stress test (subprocess-isolated workers).

Every other test in this suite runs one process with one connection. The
corruption class that damaged a live trove.db on 2026-09-25 cannot be caught
by such a suite, because the suite never creates the condition: several OS
processes opening the same WAL database and writing concurrently.

Workers here are launched as independent ``sys.executable`` subprocesses
(a small child script written to disk) rather than ``multiprocessing``
workers. That isolation is deliberate: a spawned multiprocessing child
inherits the parent's ``__main__`` and dies under pytest. A plain
``python child.py`` is a clean, realistic second process with its own
sqlite3 handle, page cache, and file lock - the same topology as gateway +
desktop + embedding worker.

The child writes a JSON result file (the store_ids it was ACKNOWLEDGED for,
plus any errors) so the parent can assert the two properties that matter:
``PRAGMA integrity_check`` is ``ok``, and every acknowledged write is durably
present (no lost or half-applied rows).
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent

WORKERS = 4
SECONDS = 3.0

CHILD_TEMPLATE = '''\
import faulthandler, importlib.util, json, sys, time
from pathlib import Path
# faulthandler.enable() dumps a Python traceback for fatal signals
# (SIGBUS/SIGSEGV/SIGABRT) to stderr, so a crash is diagnosable rather than
# a silent "no output". SIGBUS itself cannot be register()'d, but enable()
# already covers it. Passing a FILE keeps the dump out of the interleaved
# stderr stream, so a crash's traceback is not lost among other output.
_crash_log = open(str(Path(__file__).with_name("crash_%s.log" % sys.argv[2])), "w")
faulthandler.enable(file=_crash_log, all_threads=True)

# Register the flat-layout plugin as a proper package, mirroring
# tests/conftest.py: the repo root IS the ``hermes_trove`` package dir, its
# ``__init__`` must NOT be exec'd (it tries to register with the host), and
# each top-level submodule is registered under the package name.
_plugin_dir = Path({plugin_root!r})
_pkg = "hermes_trove"
_spec = importlib.util.spec_from_file_location(
    _pkg, str(_plugin_dir / "__init__.py"),
    submodule_search_locations=[str(_plugin_dir)],
)
_mod = importlib.util.module_from_spec(_spec)
_mod.__path__ = [str(_plugin_dir)]
_mod.__package__ = _pkg
sys.modules[_pkg] = _mod
for _py in _plugin_dir.glob("*.py"):
    if _py.name == "__init__.py":
        continue
    _sub = "%s.%s" % (_pkg, _py.stem)
    if _sub in sys.modules:
        continue
    _ss = importlib.util.spec_from_file_location(
        _sub, str(_py), submodule_search_locations=[],
    )
    _sm = importlib.util.module_from_spec(_ss)
    _sm.__package__ = _pkg
    sys.modules[_sub] = _sm
    setattr(_mod, _py.stem, _sm)
    try:
        _ss.loader.exec_module(_sm)
    except Exception:
        pass

from hermes_trove.store import MessageStore

db_path, worker_id, secret, seconds, out_path = sys.argv[1:6]
# Deadline is relative to THIS process's start, so a late-spawned worker
# still gets a full run window.
_stop_at = time.monotonic() + float(seconds)
store = MessageStore(db_path)
ack, errors = [], []
try:
    while time.monotonic() < _stop_at:
        try:
            sid = store.append(
                "stress-%s" % worker_id,
                {{"source": "stress", "role": "user",
                  "content": "%s-%s-%d" % (secret, worker_id, len(ack))}},
            )
            ack.append(int(sid))
        except Exception as exc:
            errors.append("%s: %s" % (type(exc).__name__, exc))
            if len(errors) > 5:
                break
finally:
    try:
        store._conn.close()
    except Exception:
        pass
with open(out_path, "w") as fh:
    json.dump({{"ack": ack, "errors": errors}}, fh)
'''


def test_multi_process_concurrent_appends_keep_store_intact(tmp_path: Path) -> None:
    """Concurrent writers from separate OS processes must not corrupt or lose data.

    Regression guard for multi-process concurrent writes. Exercises the public
    ``MessageStore.append`` path from multiple concurrent OS worker processes.
    Asserts zero lock errors, structural integrity check pass, and that all
    acknowledged writes are present in the database.
    """
    from hermes_trove.store import MessageStore  # seed/migrate the db once

    db = tmp_path / "trove.db"
    seed = MessageStore(db)
    seed._conn.close()

    child_py = tmp_path / "stress_child.py"
    child_py.write_text(CHILD_TEMPLATE.format(plugin_root=str(PLUGIN_ROOT)))

    secret = "trove-stress-marker"
    # Give each child a per-worker deadline computed from ITS OWN start, not
    # the parent's: process spawn under load can take longer than SECONDS, and
    # a parent-computed absolute deadline may already be in the past by the
    # time a late worker begins, making it write nothing.
    procs, out_paths = [], []
    for wid in range(WORKERS):
        out = tmp_path / f"out_{wid}.json"
        out_paths.append(out)
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(child_py),
                    str(db),
                    str(wid),
                    secret,
                    str(SECONDS),
                    str(out),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )

    acknowledged: set[int] = set()
    errors: list[str] = []
    for p, out in zip(procs, out_paths):
        try:
            _, err = p.communicate(timeout=SECONDS + 25)
        except subprocess.TimeoutExpired:
            p.kill()
            errors.append("a worker hung (likely lock deadlock)")
            continue
        if not out.exists():
            tail = (err or b"").decode(errors="replace").strip()
            errors.append(
                "a worker never reported back (rc=%s): " % p.returncode
                + (tail[-400:] if tail else "no stderr")
            )
            continue
        result = json.loads(out.read_text())
        acknowledged.update(int(x) for x in result["ack"])
        errors.extend(result["errors"])

    # 1) No write errors under contention: a "database is locked" that escapes
    #    the busy-retry, or a malformed-image error, fails here.
    assert not errors, f"concurrent writers hit errors: {errors[:5]}"

    # 2) Structural integrity - the assertion that would have caught the
    #    2026-09-25 corruption.
    check = sqlite3.connect(str(db))
    try:
        assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok", (
            "database failed integrity_check after concurrent multi-process writes"
        )
        # 3) Every acknowledged write is durably present (no lost rows).
        total = check.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        present = 0
        ids = list(acknowledged)
        for start in range(0, len(ids), 400):
            chunk = ids[start : start + 400]
            placeholders = ",".join("?" * len(chunk))
            present += check.execute(
                f"SELECT COUNT(*) FROM messages WHERE store_id IN ({placeholders})",
                chunk,
            ).fetchone()[0]
    finally:
        check.close()

    assert len(acknowledged) > 0, "no concurrent writes completed; not exercising the path"
    assert present == len(acknowledged), (
        f"{len(acknowledged) - present} acknowledged writes are missing from the "
        f"store (lost under contention)"
    )
    assert total >= len(acknowledged)
