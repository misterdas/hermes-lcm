"""Tests for path containment checks - validates boundary enforcement."""
import tempfile
from pathlib import Path
import pytest

from hermes_trove.externalize import get_large_output_storage_dir


def test_path_containment_within_allowed_base(monkeypatch):
    """Test that hermes_home within allowed base is accepted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set allowed base to tmpdir using monkeypatch
        monkeypatch.setenv("TROVE_HERMES_BASE_DIR", tmpdir)

        from hermes_trove.command import _state_db_path_for_engine

        # Create a mock engine with hermes_home inside allowed base
        hermes_home = str(Path(tmpdir) / "hermes")

        class MockEngine:
            _hermes_home = hermes_home

        engine = MockEngine()
        # Should succeed without raising
        path = _state_db_path_for_engine(engine)
        assert path.is_absolute()
        assert str(path).startswith(tmpdir)


def test_path_containment_outside_allowed_base(monkeypatch):
    """Test that hermes_home outside allowed base raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set allowed base to tmpdir
        monkeypatch.setenv("TROVE_HERMES_BASE_DIR", tmpdir)

        from hermes_trove.command import _state_db_path_for_engine

        # Create a mock engine with hermes_home outside allowed base
        class MockEngine:
            _hermes_home = "/etc"

        engine = MockEngine()
        # Should raise ValueError
        with pytest.raises(ValueError, match="not within allowed base"):
            _state_db_path_for_engine(engine)


def test_engine_state_db_path_outside_allowed_base(monkeypatch):
    """Test TROVEEngine._state_db_path with engine method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("TROVE_HERMES_BASE_DIR", tmpdir)

        from hermes_trove.engine import TROVEEngine

        # Create a mock store with db_path
        class MockStore:
            db_path = str(Path(tmpdir) / "trove.db")

        # Create engine with hermes_home outside allowed base
        engine = TROVEEngine.__new__(TROVEEngine)
        engine._hermes_home = "/etc"
        engine._store = MockStore()

        # Should raise ValueError
        with pytest.raises(ValueError, match="not within allowed base"):
            engine._state_db_path()


def test_state_db_path_fallback_outside_allowed_base_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("TROVE_HERMES_BASE_DIR", str(tmp_path / "allowed"))

    from hermes_trove.command import _state_db_path_for_engine as command_state_db_path
    from hermes_trove.tools import _state_db_path_for_engine as tools_state_db_path

    class MockStore:
        db_path = str(tmp_path / "outside" / "trove.db")

    class MockEngine:
        _hermes_home = ""
        _store = MockStore()

    for state_db_path_for_engine in (command_state_db_path, tools_state_db_path):
        with pytest.raises(ValueError, match="not within allowed base"):
            state_db_path_for_engine(MockEngine())


def test_engine_state_db_path_fallback_outside_allowed_base_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("TROVE_HERMES_BASE_DIR", str(tmp_path / "allowed"))

    from hermes_trove.engine import TROVEEngine

    class MockStore:
        db_path = str(tmp_path / "outside" / "trove.db")

    engine = TROVEEngine.__new__(TROVEEngine)
    engine._hermes_home = ""
    engine._store = MockStore()

    with pytest.raises(ValueError, match="not within allowed base"):
        engine._state_db_path()


def _resolved(p) -> Path:
    return Path(str(p)).expanduser().resolve()


def test_externalization_path_outside_hermes_home_warns_but_does_not_break(monkeypatch, tmp_path, caplog):
    import logging
    from hermes_trove.externalize import get_large_output_storage_dir, _WARNED_EXTERNALIZATION_PATHS

    monkeypatch.delenv("TROVE_HERMES_BASE_DIR", raising=False)
    _WARNED_EXTERNALIZATION_PATHS.clear()
    outside = tmp_path / "other-volume" / "payloads"

    class Config:
        large_output_externalization_path = str(outside)

    with caplog.at_level(logging.WARNING):
        path = get_large_output_storage_dir(
            Config(), hermes_home=str(tmp_path / "hermes"), create=False
        )

    assert path == _resolved(outside)
    assert any("outside the hermes_home base" in r.message for r in caplog.records)


def test_externalization_path_within_hermes_home_does_not_warn(monkeypatch, tmp_path, caplog):
    import logging
    from hermes_trove.externalize import get_large_output_storage_dir, _WARNED_EXTERNALIZATION_PATHS

    monkeypatch.delenv("TROVE_HERMES_BASE_DIR", raising=False)
    _WARNED_EXTERNALIZATION_PATHS.clear()
    hermes_home = tmp_path / "hermes"
    inside = hermes_home / "custom-outputs"

    class Config:
        large_output_externalization_path = str(inside)

    with caplog.at_level(logging.WARNING):
        path = get_large_output_storage_dir(Config(), hermes_home=str(hermes_home), create=False)

    assert path == _resolved(inside)
    assert not any("outside the hermes_home base" in r.message for r in caplog.records)


def test_externalization_path_strict_containment_when_base_set(monkeypatch, tmp_path):
    from hermes_trove.externalize import get_large_output_storage_dir

    monkeypatch.setenv("TROVE_HERMES_BASE_DIR", str(tmp_path / "allowed"))

    class Config:
        large_output_externalization_path = str(tmp_path / "elsewhere" / "payloads")

    with pytest.raises(ValueError):
        get_large_output_storage_dir(
            Config(), hermes_home=str(tmp_path / "allowed" / "hermes"), create=False
        )


# ---------------------------------------------------------------------------
# TROVE_EXTERNALIZATION_STRICT (C4 opt-in hard error)
# ---------------------------------------------------------------------------


def _strict_config(path: str = ""):
    class Config:
        large_output_externalization_path = path
        externalization_strict = True
    return Config()


def test_strict_mode_configured_path_outside_base_raises(monkeypatch, tmp_path):
    """Strict mode: configured externalization path outside hermes_home raises."""
    monkeypatch.delenv("TROVE_HERMES_BASE_DIR", raising=False)
    outside = tmp_path / "other-volume" / "payloads"

    with pytest.raises(ValueError, match="refusing to write in strict mode"):
        get_large_output_storage_dir(
            _strict_config(str(outside)),
            hermes_home=str(tmp_path / "hermes"),
            create=False,
        )


def test_strict_mode_configured_path_inside_base_passes(monkeypatch, tmp_path):
    """Strict mode: configured path inside hermes_home is still allowed."""
    monkeypatch.delenv("TROVE_HERMES_BASE_DIR", raising=False)
    hermes_home = tmp_path / "hermes"
    inside = hermes_home / "custom-outputs"

    path = get_large_output_storage_dir(
        _strict_config(str(inside)),
        hermes_home=str(hermes_home),
        create=False,
    )
    assert path == _resolved(inside)


def test_strict_mode_default_path_inside_base_passes(monkeypatch, tmp_path):
    """Strict mode: default hermes_home/trove-large-outputs is allowed."""
    monkeypatch.delenv("TROVE_HERMES_BASE_DIR", raising=False)
    hermes_home = tmp_path / "hermes"

    path = get_large_output_storage_dir(
        _strict_config(""),
        hermes_home=str(hermes_home),
        create=False,
    )
    assert path == _resolved(hermes_home / "trove-large-outputs")


def test_strict_mode_with_explicit_base_outside_raises(monkeypatch, tmp_path):
    """Strict mode + TROVE_HERMES_BASE_DIR: outside path still hard-raises."""
    monkeypatch.setenv("TROVE_HERMES_BASE_DIR", str(tmp_path / "allowed"))

    with pytest.raises(ValueError, match="not within allowed base"):
        get_large_output_storage_dir(
            _strict_config(str(tmp_path / "elsewhere" / "payloads")),
            hermes_home=str(tmp_path / "allowed" / "hermes"),
            create=False,
        )


def test_strict_mode_with_explicit_base_inside_passes(monkeypatch, tmp_path):
    """Strict mode + TROVE_HERMES_BASE_DIR: inside path is allowed."""
    monkeypatch.setenv("TROVE_HERMES_BASE_DIR", str(tmp_path / "allowed"))
    allowed = tmp_path / "allowed"

    path = get_large_output_storage_dir(
        _strict_config(str(allowed / "payloads")),
        hermes_home=str(allowed / "hermes"),
        create=False,
    )
    assert path == _resolved(allowed / "payloads")


def test_default_mode_still_warns_and_allows(monkeypatch, caplog, tmp_path):
    """Default mode (strict=False) keeps the existing warn-once behavior."""
    import logging
    from hermes_trove.externalize import _WARNED_EXTERNALIZATION_PATHS

    monkeypatch.delenv("TROVE_HERMES_BASE_DIR", raising=False)
    _WARNED_EXTERNALIZATION_PATHS.clear()

    class Config:
        large_output_externalization_path = str(tmp_path / "other" / "payloads")
        externalization_strict = False

    with caplog.at_level(logging.WARNING):
        path = get_large_output_storage_dir(
            Config(),
            hermes_home=str(tmp_path / "hermes"),
            create=False,
        )

    assert path == _resolved(tmp_path / "other" / "payloads")
    assert any("outside the hermes_home base" in r.message for r in caplog.records)


def test_env_var_wires_through_config(monkeypatch):
    """TROVE_EXTERNALIZATION_STRICT=true sets the config field via from_env."""
    monkeypatch.setenv("TROVE_EXTERNALIZATION_STRICT", "true")
    from hermes_trove.config import TROVEConfig
    cfg = TROVEConfig.from_env()
    assert cfg.externalization_strict is True
