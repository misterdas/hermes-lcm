"""TROVE configuration with defaults and env var overrides."""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional fallback for minimal installs
    yaml = None


logger = logging.getLogger(__name__)


def _parse_pattern_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


# Default trove_recall RRF arm weights. Conservative down-weight of the weak FTS
# arm (the LongMemEval harness will tune from here); summary/chunk vector arms
# keep full say. See docs/retrieval-tools.md.
_DEFAULT_RECALL_ARM_WEIGHTS: dict[str, float] = {
    "fts": 0.5,
    "summary": 1.0,
    "chunk": 1.0,
}


def _parse_arm_weights(raw: str, defaults: dict[str, float]) -> dict[str, float]:
    """Leniently parse ``arm=weight`` pairs (e.g. ``fts=0.5,summary=1.0``).

    Unknown arm names, malformed pairs, and non-finite/non-numeric weights are
    skipped; any arm not overridden keeps its default. A wholly unparsable value
    therefore degrades to the defaults rather than erroring the tool.

    A negative weight is invalid -- it would invert RRF rank-monotonicity (a
    rank-1 hit scoring below a rank-2 hit) -- so it is rejected and the arm keeps
    its default with a logged warning. ``0.0`` is legal and cleanly drops the arm.
    """
    weights = dict(defaults)
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, _, value = part.partition("=")
        name = name.strip().lower()
        if name not in defaults:
            continue
        try:
            parsed = float(value.strip())
        except (TypeError, ValueError):
            continue
        if parsed != parsed or parsed in (float("inf"), float("-inf")):
            continue
        if parsed < 0.0:
            logger.warning(
                "TROVE_RECALL_ARM_WEIGHTS: negative weight %r for arm %r is "
                "invalid; using default %r",
                parsed,
                name,
                defaults[name],
            )
            continue
        weights[name] = parsed
    return weights


def _parse_int_env(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _parse_float_env(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _parse_bool_env(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_str_env(key: str, default):
    return os.environ.get(key, default)


def _parse_int_env_with_source(
    key: str,
    default: int,
    *,
    default_source: str = "default",
) -> tuple[int, str, str | None]:
    raw = os.environ.get(key)
    if raw is None:
        return default, default_source, None
    try:
        return int(raw), f"env:{key}", None
    except (TypeError, ValueError):
        return default, default_source, f"invalid env {key}={raw!r} ignored"


def _parse_float_env_with_source(
    key: str,
    default: float,
    *,
    default_source: str = "default",
) -> tuple[float, str, str | None]:
    raw = os.environ.get(key)
    if raw is None:
        return default, default_source, None
    try:
        return float(raw), f"env:{key}", None
    except (TypeError, ValueError):
        return default, default_source, f"invalid env {key}={raw!r} ignored"


def _config_bool_disabled(value) -> bool:
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "off"}:
            return True
        try:
            return float(normalized) == 0
        except ValueError:
            return False
    return False


def _resolve_hermes_home(hermes_home: str | Path | None = None) -> Path:
    """Resolve the active Hermes home without changing process-global state.

    Routed Hermes gateways keep the active profile in a context-local core
    override rather than mutating ``os.environ``.  The explicit argument is
    used by engine lifecycle hooks; the core helper is the standalone fallback
    for callers that do not have a home argument.
    """
    if hermes_home:
        return Path(hermes_home).expanduser()

    for module_name in ("hermes_constants", "hermes_cli.config"):
        try:
            module = __import__(module_name, fromlist=["get_hermes_home"])
            resolver = getattr(module, "get_hermes_home", None)
            if callable(resolver):
                resolved_home = resolver()
                return Path(str(resolved_home)).expanduser()
        except Exception:
            continue

    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()


def _hermes_config_path(hermes_home: str | Path | None = None) -> Path:
    return _resolve_hermes_home(hermes_home) / "config.yaml"


def _load_hermes_config_yaml(
    hermes_home: str | Path | None = None,
) -> dict[str, Any]:
    cfg_path = _hermes_config_path(hermes_home)
    try:
        text = cfg_path.read_text()
    except Exception:
        return {}
    if yaml is not None:
        try:
            loaded = yaml.safe_load(text) or {}
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        key, raw_value = line.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else root
        value = raw_value.strip()
        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
            continue
        value = value.strip("'\"")
        lowered = value.lower()
        if lowered in {"true", "yes", "on"}:
            parsed: Any = True
        elif lowered in {"false", "no", "off"}:
            parsed = False
        else:
            try:
                parsed = float(value) if "." in value else int(value)
            except ValueError:
                parsed = value
        parent[key] = parsed
    return root


_SUPPORTED_TROVE_CONFIG_YAML_KEYS = {"context_threshold"}


def _ignored_trove_config_yaml_keys(
    cfg: dict[str, Any] | None = None,
    *,
    hermes_home: str | Path | None = None,
) -> list[str]:
    cfg = cfg if cfg is not None else _load_hermes_config_yaml(hermes_home)
    trove_section = cfg.get("trove") if isinstance(cfg, dict) else None
    if not isinstance(trove_section, dict):
        return []
    return sorted(
        str(key)
        for key in trove_section
        if str(key) not in _SUPPORTED_TROVE_CONFIG_YAML_KEYS
    )


def _hermes_compression_threshold(
    default: float,
    *,
    hermes_home: str | Path | None = None,
) -> float:
    """Read trove.context_threshold or Hermes compression.threshold from config.yaml.

    Priority when no ``TROVE_CONTEXT_THRESHOLD`` env var is set:
      1. ``trove.context_threshold`` (TROVE-specific override in config.yaml)
      2. ``compression.threshold`` (Hermes global setting, unless compression disabled)

    Hermes gateways may load ``~/.hermes/config.yaml`` without exporting every
    setting into the process environment. The ``trove.context_threshold`` key lets
    operators tune TROVE compaction independently of the Hermes compression setting.
    Disabled Hermes compression should not leak its threshold into TROVE.
    """
    value, _source = _hermes_compression_threshold_with_source(
        default,
        hermes_home=hermes_home,
    )
    return value


def _hermes_compression_threshold_with_source(
    default: float,
    *,
    hermes_home: str | Path | None = None,
) -> tuple[float, str]:
    cfg = _load_hermes_config_yaml(hermes_home)
    try:
        trove_section = cfg.get("trove") or {}
        if isinstance(trove_section, dict):
            trove_val = trove_section.get("context_threshold")
            if trove_val is not None:
                return float(trove_val), "config_yaml:trove.context_threshold"
        compression = cfg.get("compression") or {}
        if not isinstance(compression, dict):
            return default, "default"
        if _config_bool_disabled(compression.get("enabled")):
            return default, "default"
        comp_val = compression.get("threshold")
        if comp_val is not None:
            return float(comp_val), "config_yaml:compression.threshold"
    except Exception:
        return default, "default"
    return default, "default"


def _hermes_auxiliary_compression_timeout_ms(
    default: int,
    *,
    hermes_home: str | Path | None = None,
) -> int:
    """Read Hermes auxiliary.compression.timeout when no TROVE override is present.

    Hermes uses seconds for the auxiliary compression timeout, while TROVE stores
    the summary timeout in milliseconds. Aligning the default keeps TROVE summary
    calls from timing out earlier than the host compression route unless
    ``TROVE_SUMMARY_TIMEOUT_MS`` is explicitly configured.
    """
    value, _source = _hermes_auxiliary_compression_timeout_ms_with_source(
        default,
        hermes_home=hermes_home,
    )
    return value


def _hermes_auxiliary_compression_timeout_ms_with_source(
    default: int,
    *,
    hermes_home: str | Path | None = None,
) -> tuple[int, str]:
    cfg = _load_hermes_config_yaml(hermes_home)
    try:
        auxiliary = cfg.get("auxiliary") or {}
        if not isinstance(auxiliary, dict):
            return default, "default"
        compression = auxiliary.get("compression") or {}
        if not isinstance(compression, dict):
            return default, "default"
        value = compression.get("timeout")
        if value is None:
            return default, "default"
        return int(float(value) * 1000), "config_yaml:auxiliary.compression.timeout"
    except Exception:
        return default, "default"


def _hermes_codex_gpt55_autoraise_with_source(
    default: bool,
    *,
    hermes_home: str | Path | None = None,
) -> tuple[bool, str]:
    cfg = _load_hermes_config_yaml(hermes_home)
    try:
        compression = cfg.get("compression") or {}
        if not isinstance(compression, dict):
            return default, "default"
        value = compression.get("codex_gpt55_autoraise")
        if value is None:
            return default, "default"
        return (not _config_bool_disabled(value)), "config_yaml:compression.codex_gpt55_autoraise"
    except Exception:
        return default, "default"


@dataclass(frozen=True)
class _EnvFieldSpec:
    """One scalar ``TROVE_*`` environment override: which config field it sets,
    its environment variable, and the Python type used to parse it."""

    name: str
    env_key: str
    py_type: type


# Single source of truth for the scalar TROVE_* env overrides. ``from_env`` applies
# the non-source-tracked entries uniformly, and ``presets`` derives its
# preset-field lookups from the same list so the field/env/type mapping is not
# duplicated. Order mirrors the historical ``from_env`` order for readability.
ENV_FIELD_SPECS: tuple[_EnvFieldSpec, ...] = (
    _EnvFieldSpec("fresh_tail_count", "TROVE_FRESH_TAIL_COUNT", int),
    _EnvFieldSpec("fresh_tail_max_tokens", "TROVE_FRESH_TAIL_MAX_TOKENS", int),
    _EnvFieldSpec("leaf_chunk_tokens", "TROVE_LEAF_CHUNK_TOKENS", int),
    _EnvFieldSpec("context_threshold", "TROVE_CONTEXT_THRESHOLD", float),
    _EnvFieldSpec("incremental_max_depth", "TROVE_INCREMENTAL_MAX_DEPTH", int),
    _EnvFieldSpec("condensation_fanin", "TROVE_CONDENSATION_FANIN", int),
    _EnvFieldSpec("dynamic_leaf_chunk_enabled", "TROVE_DYNAMIC_LEAF_CHUNK_ENABLED", bool),
    _EnvFieldSpec("dynamic_leaf_chunk_max", "TROVE_DYNAMIC_LEAF_CHUNK_MAX", int),
    _EnvFieldSpec("cache_friendly_condensation_enabled", "TROVE_CACHE_FRIENDLY_CONDENSATION_ENABLED", bool),
    _EnvFieldSpec("cache_friendly_min_debt_groups", "TROVE_CACHE_FRIENDLY_MIN_DEBT_GROUPS", int),
    _EnvFieldSpec("deferred_maintenance_enabled", "TROVE_DEFERRED_MAINTENANCE_ENABLED", bool),
    _EnvFieldSpec("deferred_maintenance_max_passes", "TROVE_DEFERRED_MAINTENANCE_MAX_PASSES", int),
    _EnvFieldSpec("critical_budget_pressure_ratio", "TROVE_CRITICAL_BUDGET_PRESSURE_RATIO", float),
    _EnvFieldSpec("threshold_full_sweep_enabled", "TROVE_THRESHOLD_FULL_SWEEP_ENABLED", bool),
    _EnvFieldSpec("summary_prefix_target_tokens", "TROVE_SUMMARY_PREFIX_TARGET_TOKENS", int),
    _EnvFieldSpec("l2_budget_ratio", "TROVE_L2_BUDGET_RATIO", float),
    _EnvFieldSpec("l3_truncate_tokens", "TROVE_L3_TRUNCATE_TOKENS", int),
    _EnvFieldSpec("max_assembly_tokens", "TROVE_MAX_ASSEMBLY_TOKENS", int),
    _EnvFieldSpec("reserve_tokens_floor", "TROVE_RESERVE_TOKENS_FLOOR", int),
    _EnvFieldSpec("custom_instructions", "TROVE_CUSTOM_INSTRUCTIONS", str),
    _EnvFieldSpec("extraction_enabled", "TROVE_EXTRACTION_ENABLED", bool),
    _EnvFieldSpec("extraction_model", "TROVE_EXTRACTION_MODEL", str),
    _EnvFieldSpec("extraction_output_path", "TROVE_EXTRACTION_OUTPUT_PATH", str),
    _EnvFieldSpec("assertions_enabled", "TROVE_ASSERTIONS_ENABLED", bool),
    _EnvFieldSpec("query_views_enabled", "TROVE_QUERY_VIEWS_ENABLED", bool),
    _EnvFieldSpec(
        "adaptive_retrieval_enabled", "TROVE_ADAPTIVE_RETRIEVAL_ENABLED", bool
    ),
    _EnvFieldSpec(
        "assertion_extraction_enabled", "TROVE_ASSERTION_EXTRACTION_ENABLED", bool
    ),
    _EnvFieldSpec("assertion_extraction_model", "TROVE_ASSERTION_EXTRACTION_MODEL", str),
    _EnvFieldSpec(
        "assertion_extraction_max_sources_per_pass",
        "TROVE_ASSERTION_EXTRACTION_MAX_SOURCES_PER_PASS",
        int,
    ),
    _EnvFieldSpec(
        "assertion_extraction_timeout_seconds",
        "TROVE_ASSERTION_EXTRACTION_TIMEOUT_SECONDS",
        float,
    ),
    _EnvFieldSpec("sensitive_patterns_enabled", "TROVE_SENSITIVE_PATTERNS_ENABLED", bool),
    _EnvFieldSpec("large_output_externalization_enabled", "TROVE_LARGE_OUTPUT_EXTERNALIZATION_ENABLED", bool),
    _EnvFieldSpec("large_output_externalization_threshold_chars", "TROVE_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS", int),
    _EnvFieldSpec("large_output_externalization_path", "TROVE_LARGE_OUTPUT_EXTERNALIZATION_PATH", str),
    _EnvFieldSpec("large_output_active_replay_stubbing_enabled", "TROVE_LARGE_OUTPUT_ACTIVE_REPLAY_STUBBING_ENABLED", bool),
    _EnvFieldSpec("large_output_active_replay_stub_threshold_tokens", "TROVE_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_THRESHOLD_TOKENS", int),
    _EnvFieldSpec("large_output_transcript_gc_enabled", "TROVE_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED", bool),
    _EnvFieldSpec("summary_model", "TROVE_SUMMARY_MODEL", str),
    _EnvFieldSpec("summary_circuit_breaker_failure_threshold", "TROVE_SUMMARY_CIRCUIT_BREAKER_FAILURE_THRESHOLD", int),
    _EnvFieldSpec("summary_circuit_breaker_cooldown_seconds", "TROVE_SUMMARY_CIRCUIT_BREAKER_COOLDOWN_SECONDS", int),
    _EnvFieldSpec("summary_spend_max_calls", "TROVE_SUMMARY_SPEND_MAX_CALLS", int),
    _EnvFieldSpec("summary_spend_window_seconds", "TROVE_SUMMARY_SPEND_WINDOW_SECONDS", float),
    _EnvFieldSpec("summary_spend_backoff_seconds", "TROVE_SUMMARY_SPEND_BACKOFF_SECONDS", float),
    _EnvFieldSpec("expansion_model", "TROVE_EXPANSION_MODEL", str),
    _EnvFieldSpec("expansion_context_tokens", "TROVE_EXPANSION_CONTEXT_TOKENS", int),
    _EnvFieldSpec("summary_timeout_ms", "TROVE_SUMMARY_TIMEOUT_MS", int),
    _EnvFieldSpec("expansion_timeout_ms", "TROVE_EXPANSION_TIMEOUT_MS", int),
    _EnvFieldSpec("database_path", "TROVE_DATABASE_PATH", str),
    _EnvFieldSpec("embeddings_enabled", "TROVE_EMBEDDINGS_ENABLED", bool),
    _EnvFieldSpec("rerank_enabled", "TROVE_RERANK_ENABLED", bool),
    _EnvFieldSpec("recall_scan_rows", "TROVE_RECALL_SCAN_ROWS", int),
    _EnvFieldSpec("recall_scan_max_rows", "TROVE_RECALL_SCAN_MAX_ROWS", int),
    _EnvFieldSpec("recall_scan_budget_s", "TROVE_RECALL_SCAN_BUDGET_S", float),
    _EnvFieldSpec("recall_reference_strict", "TROVE_RECALL_REFERENCE_STRICT", bool),
    _EnvFieldSpec("proactive_recall_enabled", "TROVE_PROACTIVE_RECALL_ENABLED", bool),
    _EnvFieldSpec("proactive_recall_min_score", "TROVE_PROACTIVE_RECALL_MIN_SCORE", float),
    _EnvFieldSpec("proactive_recall_budget_tokens", "TROVE_PROACTIVE_RECALL_BUDGET_TOKENS", int),
    _EnvFieldSpec("proactive_recall_provider", "TROVE_PROACTIVE_RECALL_PROVIDER", str),
    _EnvFieldSpec("preanswer_evidence_enabled", "TROVE_PREANSWER_EVIDENCE_ENABLED", bool),
    _EnvFieldSpec("preanswer_evidence_mode", "TROVE_PREANSWER_EVIDENCE_MODE", str),
    _EnvFieldSpec("selective_compiler_enabled", "TROVE_SELECTIVE_COMPILER_ENABLED", bool),
    _EnvFieldSpec("selective_compiler_model", "TROVE_SELECTIVE_COMPILER_MODEL", str),
    _EnvFieldSpec("embedding_bounded_scan_rows", "TROVE_EMBEDDING_BOUNDED_SCAN_ROWS", int),
    _EnvFieldSpec("embedding_storage_dtype", "TROVE_EMBEDDING_STORAGE_DTYPE", str),
    _EnvFieldSpec("embedding_store_dim", "TROVE_EMBEDDING_STORE_DIM", int),
    _EnvFieldSpec("embedding_binary_prescreen", "TROVE_EMBEDDING_BINARY_PRESCREEN", bool),
    _EnvFieldSpec("knn_prescreen_multiplier", "TROVE_KNN_PRESCREEN_MULTIPLIER", int),
    _EnvFieldSpec("embedding_provider", "TROVE_EMBEDDING_PROVIDER", str),
    _EnvFieldSpec("embedding_model", "TROVE_EMBEDDING_MODEL", str),
    _EnvFieldSpec("embedding_content_policy", "TROVE_EMBED_CONTENT_POLICY", str),
    _EnvFieldSpec("ollama_base_url", "TROVE_OLLAMA_BASE_URL", str),
    _EnvFieldSpec("fastembed_cache_dir", "TROVE_FASTEMBED_CACHE_DIR", str),
    _EnvFieldSpec("embedding_query_timeout_s", "TROVE_EMBEDDING_QUERY_TIMEOUT_S", float),
    _EnvFieldSpec("recall_query_timeout_s", "TROVE_RECALL_QUERY_TIMEOUT_S", float),
    _EnvFieldSpec("embedding_backfill_timeout_s", "TROVE_EMBEDDING_BACKFILL_TIMEOUT_S", float),
    _EnvFieldSpec("embedding_max_batch_items", "TROVE_EMBEDDING_MAX_BATCH_ITEMS", int),
    _EnvFieldSpec("embedding_query_spend_max_calls", "TROVE_EMBEDDING_QUERY_SPEND_MAX_CALLS", int),
    _EnvFieldSpec("embedding_query_spend_window_seconds", "TROVE_EMBEDDING_QUERY_SPEND_WINDOW_SECONDS", float),
    _EnvFieldSpec("embedding_query_spend_backoff_seconds", "TROVE_EMBEDDING_QUERY_SPEND_BACKOFF_SECONDS", float),
    _EnvFieldSpec("new_session_retain_depth", "TROVE_NEW_SESSION_RETAIN_DEPTH", int),
    _EnvFieldSpec("doctor_clean_apply_enabled", "TROVE_DOCTOR_CLEAN_APPLY_ENABLED", bool),
    _EnvFieldSpec("slash_commands_enabled", "TROVE_ENABLE_SLASH_COMMAND", bool),
    _EnvFieldSpec("empty_lifecycle_gc_enabled", "TROVE_EMPTY_LIFECYCLE_GC_ENABLED", bool),
    _EnvFieldSpec("empty_lifecycle_gc_threshold", "TROVE_EMPTY_LIFECYCLE_GC_THRESHOLD", int),
    _EnvFieldSpec("temporal_rollups_enabled", "TROVE_TEMPORAL_ROLLUPS_ENABLED", bool),
    _EnvFieldSpec("rollup_daily_target_tokens", "TROVE_ROLLUP_DAILY_TARGET_TOKENS", int),
    _EnvFieldSpec("rollup_daily_max_tokens", "TROVE_ROLLUP_DAILY_MAX_TOKENS", int),
    _EnvFieldSpec("rollup_aggregate_max_tokens", "TROVE_ROLLUP_AGGREGATE_MAX_TOKENS", int),
    _EnvFieldSpec("rollup_builds_per_pass", "TROVE_ROLLUP_BUILDS_PER_PASS", int),
    _EnvFieldSpec("rollup_maintenance_budget_ms", "TROVE_ROLLUP_MAINTENANCE_BUDGET_MS", int),
    _EnvFieldSpec("retention_days", "TROVE_RETENTION_DAYS", int),
    _EnvFieldSpec("retention_apply_enabled", "TROVE_RETENTION_APPLY_ENABLED", bool),
)

_PARSER_BY_TYPE = {
    int: _parse_int_env,
    float: _parse_float_env,
    bool: _parse_bool_env,
    str: _parse_str_env,
}

# Fields whose env reading needs provenance tracking or a computed default;
# ``from_env`` handles these explicitly, so the uniform loop skips them.
_SOURCE_TRACKED_ENV_FIELDS = frozenset({
    "fresh_tail_count",
    "fresh_tail_max_tokens",
    "leaf_chunk_tokens",
    "context_threshold",
    "summary_spend_max_calls",
    "summary_spend_window_seconds",
    "summary_spend_backoff_seconds",
    "summary_timeout_ms",
})

# Fields exposed as runtime preset overrides (consumed by presets.py).
_PRESET_ENV_FIELDS = frozenset({
    "context_threshold",
    "fresh_tail_count",
    "leaf_chunk_tokens",
    "condensation_fanin",
    "incremental_max_depth",
})


@dataclass
class TROVEConfig:
    """All tunables for the TROVE engine."""

    # -- Fresh tail: recent messages never compacted ---
    fresh_tail_count: int = 32
    # Optional token cap for the protected suffix (0 = disabled)
    fresh_tail_max_tokens: int = 0

    # -- Compaction thresholds ---
    # Max source tokens in a leaf chunk before summarization triggers
    leaf_chunk_tokens: int = 20_000
    # Fraction of context window that triggers compaction (0.0–1.0)
    context_threshold: float = 0.35
    # Mirror Hermes Agent's Codex gpt-5.5 route-specific threshold auto-raise
    # when TROVE is inheriting the host compression threshold. Explicit TROVE
    # threshold overrides remain authoritative.
    codex_gpt55_autoraise_enabled: bool = True
    # Max condensation depth (-1 = unlimited, 0 = leaf only)
    incremental_max_depth: int = 3
    # How many same-depth summaries trigger condensation
    condensation_fanin: int = 4
    # When enabled, leaf compaction may use a larger working chunk size based on backlog pressure
    dynamic_leaf_chunk_enabled: bool = False
    # Upper bound for the working dynamic leaf chunk threshold
    dynamic_leaf_chunk_max: int = 40_000
    # When enabled, suppress follow-on condensation after a leaf pass unless
    # debt/pressure says the extra churn is worth it
    cache_friendly_condensation_enabled: bool = False
    # Minimum number of same-depth fanin groups before one follow-on
    # condensation pass is allowed in cache-friendly mode
    cache_friendly_min_debt_groups: int = 2
    # When enabled, turns can persist raw-backlog maintenance debt and use
    # later bounded catch-up passes to reduce it.
    deferred_maintenance_enabled: bool = False
    # Maximum extra leaf passes a debt-triggered later turn may spend on
    # catch-up work.
    deferred_maintenance_max_passes: int = 4
    # Disabled at 0.0. When set, only bypass cache-friendly/deferred polite
    # gates once prompt pressure reaches this fraction of the context window.
    critical_budget_pressure_ratio: float = 0.0
    # Opt into one bounded synchronous sweep after threshold pressure is reached.
    threshold_full_sweep_enabled: bool = False
    # Target frontier-summary size after a sweep (0 = derive one leaf budget).
    summary_prefix_target_tokens: int = 0

    # -- Escalation ---
    # L2 bullet budget as fraction of L1
    l2_budget_ratio: float = 0.50
    # L3 deterministic truncate token limit
    l3_truncate_tokens: int = 512

    # -- Assembly guardrails ---
    # Hard cap for the assembled active context (0 = disabled)
    max_assembly_tokens: int = 0
    # Reserve this many tokens from the model context window before assembly
    # (0 = disabled). Effective cap becomes context_length - reserve_tokens_floor.
    reserve_tokens_floor: int = 0

    # -- Session and message filtering ---
    # Sessions to exclude from TROVE storage entirely.
    ignore_session_patterns: list[str] = field(default_factory=list)
    # Sessions that may read carried-over TROVE state but never write new data.
    stateless_session_patterns: list[str] = field(default_factory=list)
    # Per-message regex patterns; matching messages are skipped before TROVE storage.
    ignore_message_patterns: list[str] = field(default_factory=list)
    # Diagnostics: where each pattern list came from.
    ignore_session_patterns_source: str = "default"
    stateless_session_patterns_source: str = "default"
    ignore_message_patterns_source: str = "default"

    # -- Summary instructions ---
    # Custom instructions injected into all summarization prompts
    custom_instructions: str = ""

    # -- Pre-compaction extraction ---
    # Extract decisions/commitments to files before compaction
    extraction_enabled: bool = False
    # Model for extraction (empty = fall back to summary_model)
    extraction_model: str = ""
    # Directory for daily extraction files (empty = auto: ~/.hermes/trove-extractions/)
    extraction_output_path: str = ""

    # -- V4 assertion sidecar --
    # Materializes the rebuildable assertion tables in the same profile trove.db.
    # This does not enable extraction or backfill; it only binds schema/read APIs.
    assertions_enabled: bool = False
    # Materializes demand-shaped query evidence views in the same profile DB.
    # Default off; enabling this store does not invoke a model or retrieval provider.
    query_views_enabled: bool = False
    # Exposes the bounded single-turn retrieval controller. Enabling it also
    # binds the same-database query-view store needed for warm evidence reuse.
    # The controller itself never invokes a model or provider.
    adaptive_retrieval_enabled: bool = False
    # Enables the separate structured exact-row extractor. Default off: merely
    # enabling the assertion store never performs a model/provider call.
    assertion_extraction_enabled: bool = False
    # Empty falls back to extraction_model, then summary_model.
    assertion_extraction_model: str = ""
    # Per pre-compaction batch. Runtime clamps to [1, 8].
    assertion_extraction_max_sources_per_pass: int = 4
    # Per exact source provider call. Runtime clamps to [0.1, 120] seconds.
    assertion_extraction_timeout_seconds: float = 30.0

    # -- Sensitive-pattern handling ---
    # Disabled by default. When enabled, named patterns redact matching secrets
    # before TROVE storage, FTS indexing, summarization, or externalization.
    sensitive_patterns_enabled: bool = False
    # Named pattern catalog entries to apply when sensitive handling is enabled.
    sensitive_patterns: list[str] = field(
        default_factory=lambda: ["api_key", "bearer_token", "password_assignment", "private_key"]
    )
    # Diagnostics: where the sensitive pattern list came from.
    sensitive_patterns_source: str = "default"

    # -- Large tool-output externalization ---
    # When enabled, oversized tool results are written to plugin-managed storage
    # and replaced with compact references in pre-compaction serializer input.
    large_output_externalization_enabled: bool = False
    # Character threshold above which tool results are externalized.
    large_output_externalization_threshold_chars: int = 12_000
    # Explicit storage directory for externalized payloads (empty = auto under hermes home).
    large_output_externalization_path: str = ""
    # Replace eligible textual tool results with durable compact refs in
    # provider-visible replay. Current-turn ingest is intercepted immediately;
    # historical assembly separately respects the protected fresh tail. This
    # remains opt-in and requires large-output externalization.
    large_output_active_replay_stubbing_enabled: bool = False
    # Token-aware active-replay threshold. The character threshold above still
    # controls ordinary ingest externalization; this threshold controls when a
    # provider-visible textual tool result is replaced by its durable ref.
    large_output_active_replay_stub_threshold_tokens: int = 25_000
    # When enabled, already-externalized summarized tool-result transcript rows may
    # be rewritten to compact GC placeholders after successful leaf compaction.
    large_output_transcript_gc_enabled: bool = False

    # -- Models ---
    summary_model: str = ""       # empty = use Hermes auxiliary model
    # Optional fallback summary models tried after summary_model/task default.
    summary_fallback_models: list[str] = field(default_factory=list)
    # Consecutive failed summary calls before a route is skipped temporarily.
    summary_circuit_breaker_failure_threshold: int = 2
    # Seconds to skip an open summary route before allowing a retry.
    summary_circuit_breaker_cooldown_seconds: int = 300
    # Sliding-window cap for paid/auxiliary summarizer calls before falling
    # back to deterministic L3 truncation. 0 disables the spend guard.
    summary_spend_max_calls: int = 24
    # Window, in seconds, over which summary spend calls are counted.
    summary_spend_window_seconds: float = 600.0
    # Backoff, in seconds, after the spend window is exhausted.
    summary_spend_backoff_seconds: float = 1800.0
    expansion_model: str = ""     # empty = fall back to summary_model / Hermes auxiliary model
    # Serialized summary/raw/child-source/externalized context budget fed to trove_expand_query's auxiliary LLM before it returns a bounded answer.
    expansion_context_tokens: int = 32_000

    # -- Timeouts ---
    summary_timeout_ms: int = 60_000
    expansion_timeout_ms: int = 120_000

    # -- Storage ---
    database_path: str = ""       # empty = HERMES_HOME/trove.db; TROVE_DATABASE_PATH may override

    # -- Embeddings (default-off until a provider/model are configured) ---
    embeddings_enabled: bool = False
    # trove_recall cross-encoder rerank stage (voyage rerank-2.5-lite over the top
    # fused candidates). Default-off: recall ships value on RRF order alone, and
    # rerank is one extra billable API call the operator opts into.
    rerank_enabled: bool = False
    embedding_bounded_scan_rows: int = 2_000
    # Vector storage dtype for NEWLY-registered embedding profiles: float32
    # (default; a stock install keeps summary vectors byte-identical) or int8
    # (per-vector symmetric quantization + a binary sign-bit prescreen column,
    # unlocking full-corpus two-stage KNN). dtype is part of the profile identity
    # hash, so an int8 identity never mixes with existing float32 vectors.
    embedding_storage_dtype: str = "float32"
    # Optional Matryoshka store dimension for newly-registered profiles (0 =
    # full profile dim). When >0 and < the provider dim, vectors are truncated
    # to this many leading dims and renormalized before storage/quantization.
    # Also a profile-identity component (the stored dim is hashed), so truncated
    # vectors never mix with full-dim ones.
    embedding_store_dim: int = 0
    # Write the sign-bit prescreen for float32 identities too (int8 always writes
    # it). float32-vec + prescreen = the full-corpus two-stage KNN with EXACT
    # float rescore of survivors (highest recall, ~10x less query RAM than a full
    # float32 scan). Default-off keeps stock float32 identities byte-identical and
    # binary-free (legacy bounded path); enable it on a DISTINCT identity so
    # prescreen rows never mix into a legacy float32 identity.
    embedding_binary_prescreen: bool = False
    # Stage-1 prescreen breadth for the two-stage (binary Hamming -> int8/float
    # rescore) KNN: M = knn_prescreen_multiplier x k survivors are rescored.
    # Larger widens the approximate prescreen toward exact recall at more cost.
    knn_prescreen_multiplier: int = 4
    # trove_recall candidate-scan BATCH SIZE. trove_recall promises "all
    # conversations, all time", so it must NOT inherit the small
    # recency-truncating grep bound above (that structurally hides the oldest
    # memories). It used to be a hard bound, which made the promise false at
    # scale: at 185k vectors it scored only the 25k most-recent and recall for
    # ageing content went to zero (FINDING-F31 §2). The scan now covers the
    # WHOLE corpus and this value bounds only how many vectors are resident per
    # batch (a running top-k spans the batches), so it trades peak memory, not
    # coverage.
    recall_scan_rows: int = 25_000
    # Hard candidate cap for the trove_recall scan; 0 = unlimited (the default:
    # cover everything). Set it only for a pathological corpus -- a capped scan
    # reports coverage='bounded' and discloses the scanned/total ratio, exactly
    # as the old recency window did.
    recall_scan_max_rows: int = 0
    # Optional hard latency budget for the trove_recall scan, in seconds; 0 = no
    # early stop (the default). When set, a scan that overruns it stops between
    # batches and degrades to coverage='bounded' rather than silently paying an
    # unbounded cost. This is the ONLY thing that truncates a default scan.
    recall_scan_budget_s: float = 0.0
    # Per-arm RRF fusion weights for trove_recall's 3-arm hybrid (fts/summary/chunk).
    # Down-weighting the weak FTS arm keeps naive equal-weight fusion from dragging
    # fused recall below its best (vector) arm — measured −21 R@5 on LongMemEval.
    # Override via TROVE_RECALL_ARM_WEIGHTS ("fts=0.5,summary=1.0,chunk=1.0").
    recall_arm_weights: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_RECALL_ARM_WEIGHTS)
    )
    # Reference-strict delivery for detail='answer_ready' (FINDING-F35 §2): a hit
    # that cannot carry a truthful (store_id, char_start, char_end) source span is
    # never delivered as evidence; the next-ranked citable hit takes its slot and
    # the omission count is surfaced in provenance.answer_ready. ON by default --
    # delivering evidence the product cannot cite is a correctness defect, and the
    # consumers that validate references fail CLOSED on an unreferenced card, so a
    # single uncitable hit destroys the whole response. Set False only for a host
    # that renders recall without citations and wants summary hits back; disabled,
    # the answer_ready response is byte-identical to the pre-F35 delivery.
    recall_reference_strict: bool = True
    # -- Proactive memory injection (SPEC F, default-OFF) ---
    # At active-context assembly, embed the newest user message and run the
    # trove_recall pipeline to surface cross-session memories the model would
    # otherwise have to trove_recall by hand. Default-off => byte-identical
    # assembly; when disabled the whole path is skipped before any work.
    proactive_recall_enabled: bool = False
    # Relevance floor on the trove_recall composite score. Two regimes:
    #  - rerank OFF (default): the score is RRF-scale (~0.014-0.05); a single
    #    top-ranked arm hit is ~0.016, so this floor mainly drops ancient or
    #    low-ranked hits. The default keeps fresh top-of-arm hits.
    #  - rerank ON: a cross-encoder relevance in [0,1] dominates the score;
    #    raise this floor (e.g. ~0.3) for a true semantic gate.
    proactive_recall_min_score: float = 0.01
    # Hard token budget for the single injected "relevant memories" block.
    proactive_recall_budget_tokens: int = 500
    # Optional embedding-provider override for the injection query only (e.g.
    # keep a local fastembed provider for the offline injection path even when
    # interactive search uses voyage). Empty => reuse the main provider/model.
    proactive_recall_provider: str = ""
    # Product-owned automatic evidence validation at the official
    # ``pre_llm_call`` seam. Default-off preserves the exact ordinary hook
    # context and performs no retrieval or computation work.
    preanswer_evidence_enabled: bool = False
    # Empty preserves the historical boolean-only behavior: when the master
    # flag is true it resolves to ``legacy_selective``. Explicit values are
    # off | legacy_selective | requirements_v1 | sufficiency_v1. The master
    # flag remains the default-off activation boundary.
    preanswer_evidence_mode: str = ""
    # Optional minimal semantic selector for code-derived closed operations.
    # It is independent and default-off; enabling pre-answer evidence alone
    # still performs only the provider-free session-bundle path.
    selective_compiler_enabled: bool = False
    selective_compiler_model: str = ""
    embedding_provider: str = ""
    embedding_model: str = ""
    # Content-aware chunk policy for the raw-history chunk corpus:
    # conversational (default) | heads | full. Unknown values degrade to the
    # default in the chunker's normalize_content_policy.
    embedding_content_policy: str = "conversational"
    ollama_base_url: str = "http://localhost:11434"
    # fastembed model cache dir (opt-in, default-off). Empty = fall back to
    # the process-level default (~/.cache/fastembed). Set this to a shared
    # cache dir so multiple local-model users share ONE model download
    # instead of each keeping their own copy.
    fastembed_cache_dir: str = ""
    embedding_query_timeout_s: float = 3.0
    # Dedicated deadline for trove_recall. It fans out three sequential arms (FTS +
    # summary KNN + chunk KNN) plus fusion, hydration, and an optional rerank, so
    # it needs more headroom than trove_grep's single-arm query deadline above
    # (which stays 3.0s). sprint-opt-2.
    recall_query_timeout_s: float = 8.0
    # Per-provider-operation deadline for bulk document embedding. This is
    # deliberately separate from the latency-sensitive query deadline; the
    # whole backfill invocation is additionally governed by
    # TROVE_EMBEDDING_BACKFILL_BUDGET_S (0 = unlimited, checked between batches).
    embedding_backfill_timeout_s: float = 120.0
    # Voyage caps a single embeddings request at 1000 input items; document
    # batches split at this many items in addition to the token budget.
    embedding_max_batch_items: int = 1000
    # Sliding-window spend guard for the latency-sensitive QUERY embedding path
    # (resolve_provider(for_backfill=False)). The historical hardcoded default
    # was 60 calls / 60s window / 60s backoff -- a backfill-economy control that
    # silently gutted retrieval when a tight query loop (e.g. a benchmark firing
    # hundreds of query embeds in a minute) crossed 60 calls and every further
    # call was rejected pre-network with ProviderRateLimited (~9k tokens for 451
    # query embeds is negligible spend, so the low ceiling bought nothing). The
    # default is now generous; max_calls=0 disables the guard entirely. Backfill
    # keeps its own bulk contract (max_calls=0) and is unaffected.
    embedding_query_spend_max_calls: int = 600
    embedding_query_spend_window_seconds: float = 60.0
    embedding_query_spend_backoff_seconds: float = 60.0

    # -- Session carry-over ---
    # Depth retained after /new (-1 = all, 0 = nothing, 2 = keep d2+)
    new_session_retain_depth: int = 2
    # Safety gate: destructive `/trove doctor clean apply` workflow is disabled by default.
    doctor_clean_apply_enabled: bool = False
    # Enable the optional `/trove` slash command surface (requires
    # `TROVE_ENABLE_SLASH_COMMAND=1` in the environment).
    slash_commands_enabled: bool = False

    # -- Lifecycle GC ---
    # Enables automatic pruning of lifecycle rows for sessions that never
    # ingested any messages or nodes (gateway restart orphans, ephemeral
    # cron ticks, etc.).  Runs at session-start when the lifecycle table
    # exceeds ``empty_lifecycle_gc_threshold`` rows.
    empty_lifecycle_gc_enabled: bool = True
    # Number of lifecycle rows at which the GC pass fires.  Default 200
    # so fresh installs skip the work until enough churn has occurred.
    empty_lifecycle_gc_threshold: int = 200
    # Age guard for automatic lifecycle GC. Startup GC must not delete
    # recently-bound empty rows because another live engine may not have
    # ingested its first message yet. Set to 0 only in trusted/test
    # environments that intentionally want immediate empty-row pruning.
    empty_lifecycle_gc_max_age_hours: float | None = 24.0

    # -- Session retention ---
    # Age (in days) after which a stored session's RAW messages become
    # eligible for retention cleanup. 0 (default) = retain raw messages
    # forever — TROVE's lossless guarantee is the default and nothing is
    # ever auto-deleted. When set, a session whose LAST activity is older
    # than retention_days is a candidate for `/trove doctor retention apply`,
    # provided it still carries summary nodes (the recallable core). The
    # actively-bound session is always protected regardless of age.
    retention_days: int = 0
    # Destructive `/trove doctor retention apply` workflow. Enabled by default:
    # the RETENTION_DAYS number is the safety gate (0 = never delete). Set this
    # to false only to hard-disable apply on shared/multi-user setups.
    retention_apply_enabled: bool = True

    # -- Temporal rollups ---
    # Disabled by default; the engine's ingest/build hooks are flag-gated.
    temporal_rollups_enabled: bool = False
    rollup_daily_target_tokens: int = 5_000
    rollup_daily_max_tokens: int = 15_000
    rollup_aggregate_max_tokens: int = 20_000
    rollup_builds_per_pass: int = 2
    # Best-effort wall-clock budget checked between builds. A slow summarizer
    # may finish its current build and leave later rollups lagging until a future pass.
    rollup_maintenance_budget_ms: int = 5_000

    # -- Diagnostics ---
    # Field-level provenance for values loaded through from_env(). Manual
    # TROVEConfig(...) instances leave this empty and status treats them as manual/default.
    config_sources: dict[str, str] = field(default_factory=dict)
    config_source_warnings: list[str] = field(default_factory=list)
    ignored_config_yaml_trove_keys: list[str] = field(default_factory=list)
    config_hermes_home: str = ""

    @classmethod
    def from_env(
        cls,
        *,
        hermes_home: str | Path | None = None,
    ) -> "TROVEConfig":
        """Build config from environment variables and one Hermes profile.

        ``hermes_home`` is intentionally an argument instead of a temporary
        ``HERMES_HOME`` environment mutation so concurrent routed profiles
        cannot observe each other's configuration.
        """
        c = cls()
        c.config_hermes_home = str(_resolve_hermes_home(hermes_home))
        config_sources: dict[str, str] = {}
        config_source_warnings: list[str] = []

        def _record(field: str, source: str, warning: str | None = None) -> None:
            config_sources[field] = source
            if warning:
                config_source_warnings.append(warning)

        c.ignored_config_yaml_trove_keys = _ignored_trove_config_yaml_keys(
            hermes_home=hermes_home
        )

        # Source-tracked fields (provenance recording and/or a computed default)
        # stay explicit; the uniform loop below skips them.
        c.fresh_tail_count, source, warning = _parse_int_env_with_source(
            "TROVE_FRESH_TAIL_COUNT", c.fresh_tail_count
        )
        _record("fresh_tail_count", source, warning)
        c.fresh_tail_max_tokens, source, warning = _parse_int_env_with_source(
            "TROVE_FRESH_TAIL_MAX_TOKENS", c.fresh_tail_max_tokens
        )
        c.fresh_tail_max_tokens = max(0, c.fresh_tail_max_tokens)
        _record("fresh_tail_max_tokens", source, warning)
        c.leaf_chunk_tokens, source, warning = _parse_int_env_with_source(
            "TROVE_LEAF_CHUNK_TOKENS", c.leaf_chunk_tokens
        )
        _record("leaf_chunk_tokens", source, warning)
        context_default, context_source = _hermes_compression_threshold_with_source(
            c.context_threshold,
            hermes_home=hermes_home,
        )
        c.context_threshold, source, warning = _parse_float_env_with_source(
            "TROVE_CONTEXT_THRESHOLD",
            context_default,
            default_source=context_source,
        )
        _record("context_threshold", source, warning)
        c.codex_gpt55_autoraise_enabled, source = _hermes_codex_gpt55_autoraise_with_source(
            c.codex_gpt55_autoraise_enabled,
            hermes_home=hermes_home,
        )
        _record("codex_gpt55_autoraise_enabled", source)
        c.summary_spend_max_calls, source, warning = _parse_int_env_with_source(
            "TROVE_SUMMARY_SPEND_MAX_CALLS",
            c.summary_spend_max_calls,
        )
        _record("summary_spend_max_calls", source, warning)
        c.summary_spend_window_seconds, source, warning = _parse_float_env_with_source(
            "TROVE_SUMMARY_SPEND_WINDOW_SECONDS",
            c.summary_spend_window_seconds,
        )
        _record("summary_spend_window_seconds", source, warning)
        c.summary_spend_backoff_seconds, source, warning = _parse_float_env_with_source(
            "TROVE_SUMMARY_SPEND_BACKOFF_SECONDS",
            c.summary_spend_backoff_seconds,
        )
        _record("summary_spend_backoff_seconds", source, warning)
        summary_timeout_default, summary_timeout_source = _hermes_auxiliary_compression_timeout_ms_with_source(
            c.summary_timeout_ms,
            hermes_home=hermes_home,
        )
        c.summary_timeout_ms, source, warning = _parse_int_env_with_source(
            "TROVE_SUMMARY_TIMEOUT_MS",
            summary_timeout_default,
            default_source=summary_timeout_source,
        )
        _record("summary_timeout_ms", source, warning)

        # Every other scalar TROVE_* override is applied uniformly from the spec.
        for spec in ENV_FIELD_SPECS:
            if spec.name in _SOURCE_TRACKED_ENV_FIELDS:
                continue
            parser = _PARSER_BY_TYPE[spec.py_type]
            setattr(c, spec.name, parser(spec.env_key, getattr(c, spec.name)))

        # Pattern-list overrides carry a source sidecar and stay explicit.
        raw_sensitive_patterns = os.environ.get("TROVE_SENSITIVE_PATTERNS")
        if raw_sensitive_patterns is not None:
            c.sensitive_patterns = _parse_pattern_list(raw_sensitive_patterns)
            c.sensitive_patterns_source = "env"
        raw_summary_fallback_models = os.environ.get("TROVE_SUMMARY_FALLBACK_MODELS")
        if raw_summary_fallback_models is not None:
            c.summary_fallback_models = _parse_pattern_list(raw_summary_fallback_models)

        raw_max_age = os.environ.get("TROVE_EMPTY_LIFECYCLE_GC_MAX_AGE_HOURS")
        if raw_max_age is not None:
            try:
                c.empty_lifecycle_gc_max_age_hours = float(raw_max_age)
            except (TypeError, ValueError):
                pass

        raw_arm_weights = os.environ.get("TROVE_RECALL_ARM_WEIGHTS")
        if raw_arm_weights is not None:
            c.recall_arm_weights = _parse_arm_weights(
                raw_arm_weights, _DEFAULT_RECALL_ARM_WEIGHTS
            )

        raw_ignore = os.environ.get("TROVE_IGNORE_SESSION_PATTERNS")
        if raw_ignore is not None:
            c.ignore_session_patterns = _parse_pattern_list(raw_ignore)
            c.ignore_session_patterns_source = "env"

        raw_stateless = os.environ.get("TROVE_STATELESS_SESSION_PATTERNS")
        if raw_stateless is not None:
            c.stateless_session_patterns = _parse_pattern_list(raw_stateless)
            c.stateless_session_patterns_source = "env"

        raw_ignore_messages = os.environ.get("TROVE_IGNORE_MESSAGE_PATTERNS")
        if raw_ignore_messages is not None:
            c.ignore_message_patterns = _parse_pattern_list(raw_ignore_messages)
            c.ignore_message_patterns_source = "env"

        c.config_sources = config_sources
        c.config_source_warnings = config_source_warnings
        return c
