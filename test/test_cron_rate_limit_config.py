"""Config plumbing tests for the per-agent cron rate-limit section.

Mirrors the CronHistoryConfig wiring: a `cron_rate_limit` section on
KiroCrewConfig carrying `max_concurrent_per_agent` (0 = unlimited). These
tests cover parse-from-disk, the absent-section default, the negative-value
fail-safe clamp, and a serialize/reload round-trip. No scheduler behavior is
exercised here -- enforcement lives in a separate feature that reads this
config.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from kiro_crew.config.loader import KiroCrewConfig
from kiro_crew.config.sections import CronRateLimitConfig


@pytest.fixture()
def cfg_file(tmp_path):
    """Redirect config_path() to a temp file for isolation."""
    p = tmp_path / "config.json"
    p.write_text("{}", encoding="utf-8")
    with patch("kiro_crew.config.loader.config_path", return_value=p):
        yield p


def _load_with(cfg_file, data: dict) -> KiroCrewConfig:
    """Write ``data`` to the redirected config file and load it."""
    cfg_file.write_text(json.dumps(data), encoding="utf-8")
    return KiroCrewConfig.load()


def test_parses_max_concurrent_per_agent(cfg_file):
    """A provided cron_rate_limit section is parsed onto the config."""
    cfg = _load_with(cfg_file, {"cron_rate_limit": {"max_concurrent_per_agent": 3}})
    assert cfg.cron_rate_limit.max_concurrent_per_agent == 3


def test_defaults_to_zero_when_section_absent(cfg_file):
    """Absent section yields the unlimited default (0)."""
    cfg = _load_with(cfg_file, {})
    assert cfg.cron_rate_limit.max_concurrent_per_agent == 0


def test_negative_value_clamps_to_zero(cfg_file):
    """A hand-edited negative value fail-safes to unlimited (0)."""
    cfg = _load_with(cfg_file, {"cron_rate_limit": {"max_concurrent_per_agent": -5}})
    assert cfg.cron_rate_limit.max_concurrent_per_agent == 0


def test_dataclass_post_init_clamps_negative():
    """The dataclass itself clamps a negative value independent of the loader."""
    assert CronRateLimitConfig(max_concurrent_per_agent=-1).max_concurrent_per_agent == 0
    assert CronRateLimitConfig().max_concurrent_per_agent == 0


def test_value_survives_serialize_reload_roundtrip(cfg_file):
    """The value survives a to_dict()/serialize -> reload round-trip."""
    cfg = _load_with(cfg_file, {"cron_rate_limit": {"max_concurrent_per_agent": 7}})
    serialized = cfg.to_dict()
    assert serialized["cron_rate_limit"]["max_concurrent_per_agent"] == 7

    # Reload the serialized form and confirm the value round-trips.
    reloaded = _load_with(cfg_file, serialized)
    assert reloaded.cron_rate_limit.max_concurrent_per_agent == 7
