import threading
import time

import pytest

from hermes_trove.dag import SummaryDAG
from hermes_trove.lifecycle_state import LifecycleStateStore
from hermes_trove.rollup_store import RollupStore
from hermes_trove.store import MessageStore


@pytest.mark.parametrize(
    ("factory", "lock_name"),
    [
        (lambda path: MessageStore(path), "_write_lock"),
        (lambda path: SummaryDAG(path), "_db_lock"),
        (lambda path: LifecycleStateStore(path), "_lock"),
        (lambda path: RollupStore(path), "_write_lock"),
    ],
)
def test_store_close_waits_for_active_write_lock(tmp_path, factory, lock_name):
    store = factory(tmp_path / "close-race.db")
    lock = getattr(store, lock_name)
    thread = None
    try:
        with lock:
            closed = threading.Event()

            def close_store():
                store.close()
                closed.set()

            thread = threading.Thread(target=close_store)
            thread.start()
            time.sleep(0.02)

            assert not closed.is_set()
    finally:
        store.close()
    thread.join(timeout=2)
    assert closed.is_set()
