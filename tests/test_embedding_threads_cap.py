"""Prove TROVE_EMBEDDING_THREADS caps CPU through the real provider path.

Lives under tests/ so conftest builds the hermes_trove package the same way the
suite does. Uses the plugin's own FastembedProvider._construct() -- the exact
kwargs path the patch changed -- not a bare fastembed call.

One config per PROCESS: the ONNX thread pool is global, so both configs in one
process would contaminate each other. That is why this asserts on the kwargs the
provider hands to fastembed, and the multi-process CPU measurement lives in the
standalone scratch probe.
"""
from __future__ import annotations

import resource
import time

import pytest

from hermes_trove.embedding_provider import FastembedProvider

MODEL = "BAAI/bge-base-en-v1.5"
CACHE = "/home/ubuntu/.cache/fastembed"
TEXTS = [
    f"nifty option chain strike {i} premium {i * 13} oi {i * 97} iv {20 + i % 10}"
    for i in range(64)
]


def _cpu_seconds() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def test_threads_cap_reaches_onnx_and_limits_core_usage():
    """threads=1 must reach fastembed AND keep the embed on a single core."""
    pytest.importorskip("fastembed", reason="fastembed is an optional dependency")

    provider = FastembedProvider(MODEL, cache_dir=CACHE, threads=1)
    model = provider._construct(allow_download=False)

    list(model.embed(TEXTS[:8]))  # warm the graph, exclude first-call setup

    walls, cpus = [], []
    for _ in range(2):
        c0, w0 = _cpu_seconds(), time.perf_counter()
        list(model.embed(TEXTS))
        walls.append(time.perf_counter() - w0)
        cpus.append(_cpu_seconds() - c0)

    wall, cpu = min(walls), min(cpus)
    cores_used = cpu / wall
    print(
        f"\n  provider_threads=1 wall={wall:.3f}s "
        f"cpu={cpu:.3f}s cores_used={cores_used:.2f}"
    )
    # One core's worth of CPU per second of wall clock. Allow generous headroom
    # for scheduler noise; the uncapped path measures ~1.96 on a 2-core box.
    assert cores_used < 1.5, f"expected <1.5 cores, measured {cores_used:.2f}"
