"""Tests for config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.loader import AppConfig, load_config


def test_load_default_config(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    config = load_config(missing)
    assert config["server"]["transport"] == "stdio"
    assert config["tools"]["agent_query"]["enabled"] is True


def test_invalid_config_raises(tmp_path: Path) -> None:
    bad_config = tmp_path / "config.yaml"
    bad_config.write_text("server:\n  transport: invalid\n")
    with pytest.raises(ValueError):
        load_config(bad_config)


def test_app_config_defaults() -> None:
    config = AppConfig().model_dump()
    assert config["server"]["name"]
