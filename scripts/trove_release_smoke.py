#!/usr/bin/env python3
"""Offline release smoke test for the TROVE persistence contract.

Runs a fresh database through ingest, DAG summary, FTS recall, deterministic
semantic recall, shutdown, and reopen. It uses a local fake provider only; no
network or credentials are required.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT.parent):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

if "hermes_trove" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "hermes_trove",
        str(REPO_ROOT / "__init__.py"),
        submodule_search_locations=[str(REPO_ROOT)],
    )
    if spec is not None and spec.loader is not None:
        package = importlib.util.module_from_spec(spec)
        package.__path__ = [str(REPO_ROOT)]
        package.__package__ = "hermes_trove"
        sys.modules["hermes_trove"] = package
        spec.loader.exec_module(package)

from benchmarking.standalone import ensure_agent_context_engine_importable

ensure_agent_context_engine_importable()

from hermes_trove.config import TROVEConfig
from hermes_trove.dag import SummaryNode
from hermes_trove.engine import TROVEEngine
from hermes_trove import tools as trove_tools
from hermes_trove.vector_store import VectorStore


class _FakeProvider:
    provider_id = "smoke"
    model_id = "smoke-model"
    dim = 2

    def embed_query(self, text: str) -> list[float]:
        normalized = text.lower()
        return [1.0, 0.25 if "summary" in normalized else 0.0]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(root: Path) -> dict[str, object]:
    db_path = root / "trove.db"
    config = TROVEConfig(
        database_path=str(db_path),
        embeddings_enabled=True,
        embedding_provider="smoke",
        embedding_model="smoke-model",
    )
    home = root / "home"
    engine = TROVEEngine(config=config, hermes_home=str(home))
    try:
        engine.on_session_start("smoke-session", conversation_id="smoke-conversation")
        store_id = engine._store.append(
            "smoke-session",
            {"role": "user", "content": "release smoke marker"},
            token_estimate=4,
        )
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id="smoke-session",
                depth=0,
                summary="release smoke summary",
                token_count=4,
                source_token_count=4,
                source_ids=[store_id],
                source_type="messages",
                created_at=1.0,
                earliest_at=1.0,
                latest_at=1.0,
                expand_hint="Expand release smoke summary",
            )
        )

        vector_store = VectorStore(db_path, config=config)
        try:
            vector_store.register_profile("smoke-model", "smoke", 2)
            identity = vector_store.capture_identity("smoke-model", provider="smoke")
            vector_store.record_embedding(
                str(node_id), "summary", "smoke-model", [1.0, 0.25], identity=identity
            )
        finally:
            vector_store.close()

        original_resolve = trove_tools.resolve_provider
        trove_tools.resolve_provider = lambda _config: _FakeProvider()
        try:
            fts = json.loads(
                trove_tools.trove_grep(
                    {"query": "release smoke marker", "mode": "full_text"}, engine=engine
                )
            )
            semantic = json.loads(
                trove_tools.trove_grep(
                    {"query": "release smoke summary", "mode": "semantic"}, engine=engine
                )
            )
        finally:
            trove_tools.resolve_provider = original_resolve
        _require(bool(fts.get("results")), "fresh-database FTS recall returned no results")
        _require(bool(semantic.get("results")), "fake-provider semantic recall returned no results")
    finally:
        engine.shutdown()

    reopened = TROVEEngine(config=config, hermes_home=str(home))
    try:
        reopened.on_session_start("smoke-session", conversation_id="smoke-conversation")
        payload = json.loads(
            trove_tools.trove_grep(
                {"query": "release smoke marker", "mode": "full_text"}, engine=reopened
            )
        )
        _require(bool(payload.get("results")), "FTS recall failed after shutdown/reopen")
    finally:
        reopened.shutdown()
    return {"status": "pass", "database": str(db_path), "reopened": True}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="trove-release-smoke-") as directory:
        result = run(Path(directory))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"trove release smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
