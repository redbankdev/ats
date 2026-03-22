"""Tests for utils: config, logging, and repro modules."""

import json
import logging
import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from ats_research.utils.config import (
    ATSConfig,
    CostsConfig,
    DataConfig,
    OutputConfig,
    RiskConfig,
    SimulationConfig,
    StrategyConfig,
    _apply_env_overrides,
    load_config,
)
from ats_research.utils.logging import (
    _RunIdFilter,
    get_logger,
    setup_logging,
)
from ats_research.utils.repro import (
    RunContext,
    _parse_pyproject_deps,
    check_dependency_pins,
    generate_run_id,
    set_global_seed,
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestLoadConfigYAML:
    def test_load_yaml_config(self, tmp_path):
        cfg = {
            "data": {"path": "data.csv", "freq": "1D", "tz": "UTC"},
            "strategy": {"name": "sma_crossover", "params": {"fast": 10, "slow": 30}},
            "risk": {"max_pos_pct": 0.10, "per_trade_risk_pct": 0.01,
                     "daily_loss_limit_pct": 0.05, "vol_target_pct": 0.10},
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(cfg))
        result = load_config(p)
        assert isinstance(result, ATSConfig)
        assert result.data.path == "data.csv"
        assert result.strategy.name == "sma_crossover"

    def test_load_yml_extension(self, tmp_path):
        cfg = {
            "data": {"path": "data.csv"},
            "strategy": {"name": "sma_crossover", "params": {}},
            "risk": {"max_pos_pct": 0.10, "daily_loss_limit_pct": 0.05},
        }
        p = tmp_path / "config.yml"
        p.write_text(yaml.dump(cfg))
        result = load_config(p)
        assert result.data.path == "data.csv"

    def test_load_json_config(self, tmp_path):
        cfg = {
            "data": {"path": "data.csv"},
            "strategy": {"name": "sma_crossover", "params": {}},
            "risk": {"max_pos_pct": 0.10, "daily_loss_limit_pct": 0.05},
        }
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg))
        result = load_config(p)
        assert result.data.path == "data.csv"

    def test_load_config_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_load_config_unsupported_extension(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text("something")
        with pytest.raises(ValueError, match="Unsupported config format"):
            load_config(p)


class TestConfigGuardrails:
    def test_max_pos_pct_zero_rejected(self, tmp_path):
        cfg = {
            "data": {"path": "data.csv"},
            "strategy": {"name": "sma", "params": {}},
            "risk": {"max_pos_pct": 0.0, "daily_loss_limit_pct": 0.05},
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(cfg))
        with pytest.raises(Exception, match="max_pos_pct"):
            load_config(p)

    def test_daily_loss_limit_zero_rejected(self, tmp_path):
        cfg = {
            "data": {"path": "data.csv"},
            "strategy": {"name": "sma", "params": {}},
            "risk": {"max_pos_pct": 0.10, "daily_loss_limit_pct": 0.0},
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(cfg))
        with pytest.raises(Exception, match="daily_loss_limit_pct"):
            load_config(p)

    def test_leverage_exceeds_cap(self, tmp_path):
        cfg = {
            "data": {"path": "data.csv"},
            "strategy": {"name": "sma", "params": {"leverage": 2.0}},
            "risk": {"max_pos_pct": 0.10, "daily_loss_limit_pct": 0.05},
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(cfg))
        with pytest.raises(Exception, match="Leverage"):
            load_config(p)


class TestEnvOverrides:
    def test_env_override_data_path(self, tmp_path):
        cfg = {
            "data": {"path": "original.csv"},
            "strategy": {"name": "sma", "params": {}},
            "risk": {"max_pos_pct": 0.10, "daily_loss_limit_pct": 0.05},
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(cfg))

        with patch.dict(os.environ, {"ATS_DATA__PATH": "overridden.csv"}, clear=False):
            result = load_config(p)
        assert result.data.path == "overridden.csv"

    def test_env_override_output_dir(self, tmp_path):
        cfg = {
            "data": {"path": "data.csv"},
            "strategy": {"name": "sma", "params": {}},
            "risk": {"max_pos_pct": 0.10, "daily_loss_limit_pct": 0.05},
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(cfg))

        with patch.dict(os.environ, {"ATS_OUTPUT__DIR": "/tmp/custom_out"}, clear=False):
            result = load_config(p)
        assert result.output.dir == "/tmp/custom_out"

    def test_apply_env_overrides_nested(self):
        raw = {"data": {"path": "original.csv"}}
        with patch.dict(os.environ, {"ATS_DATA__PATH": "new.csv"}, clear=False):
            result = _apply_env_overrides(raw)
        assert result["data"]["path"] == "new.csv"

    def test_apply_env_overrides_creates_missing_keys(self):
        raw = {}
        with patch.dict(os.environ, {"ATS_DATA__PATH": "new.csv"}, clear=False):
            result = _apply_env_overrides(raw)
        assert result["data"]["path"] == "new.csv"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogging:
    def test_setup_logging_creates_handler(self):
        setup_logging(run_id="test_run_001", level="DEBUG")
        logger = logging.getLogger("ats_research")
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) > 0

    def test_setup_logging_with_int_level(self):
        setup_logging(run_id="test_run_002", level=logging.WARNING)
        logger = logging.getLogger("ats_research")
        assert logger.level == logging.WARNING

    def test_setup_logging_idempotent(self):
        """Calling setup_logging twice doesn't duplicate handlers."""
        setup_logging(run_id="run_a")
        n_handlers_1 = len(logging.getLogger("ats_research").handlers)
        setup_logging(run_id="run_b")
        n_handlers_2 = len(logging.getLogger("ats_research").handlers)
        assert n_handlers_2 == n_handlers_1

    def test_get_logger_prefixes(self):
        logger = get_logger("my_module")
        assert logger.name == "ats_research.my_module"

    def test_get_logger_already_prefixed(self):
        logger = get_logger("ats_research.some.module")
        assert logger.name == "ats_research.some.module"

    def test_get_logger_none(self):
        logger = get_logger(None)
        assert logger.name == "ats_research"

    def test_run_id_filter(self):
        f = _RunIdFilter(run_id="test_123")
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        assert f.filter(record) is True
        assert record.run_id == "test_123"

    def test_run_id_filter_existing(self):
        """If record already has run_id, filter doesn't overwrite it."""
        f = _RunIdFilter(run_id="filter_id")
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.run_id = "existing_id"
        f.filter(record)
        assert record.run_id == "existing_id"


# ---------------------------------------------------------------------------
# Repro
# ---------------------------------------------------------------------------


class TestSetGlobalSeed:
    def test_set_global_seed(self):
        import random
        import numpy as np

        set_global_seed(123)
        a = random.random()
        np_a = np.random.random()

        set_global_seed(123)
        b = random.random()
        np_b = np.random.random()

        assert a == b
        assert np_a == np_b

    def test_pythonhashseed_env(self):
        set_global_seed(99)
        assert os.environ["PYTHONHASHSEED"] == "99"


class TestGenerateRunId:
    def test_format(self):
        run_id = generate_run_id()
        # YYYYMMDD_HHMMSS_<hex8>
        assert re.match(r"\d{8}_\d{6}_[a-f0-9]{8}", run_id)

    def test_uniqueness(self):
        ids = {generate_run_id() for _ in range(10)}
        assert len(ids) == 10


class TestRunContext:
    def test_create(self):
        ctx = RunContext.create(seed=42)
        assert ctx.seed == 42
        assert ctx.python_version
        assert ctx.package_version == "0.1.0"
        assert ctx.run_id
        assert ctx.timestamp

    def test_create_with_explicit_run_id(self):
        ctx = RunContext.create(seed=0, run_id="my_run")
        assert ctx.run_id == "my_run"


class TestParsePyprojectDeps:
    def test_parse_deps(self, tmp_path):
        content = '''\
[project]
name = "mypackage"
dependencies = [
    "pandas>=2.1.0,<3.0",
    "numpy>=1.24",
    "click",
]

[build-system]
requires = ["setuptools"]
'''
        p = tmp_path / "pyproject.toml"
        p.write_text(content)
        deps = _parse_pyproject_deps(p)
        assert "pandas" in deps
        assert deps["pandas"] == ">=2.1.0,<3.0"
        assert "numpy" in deps
        assert deps["numpy"] == ">=1.24"
        assert "click" in deps
        assert deps["click"] == ""


class TestCheckDependencyPins:
    def test_no_pyproject(self, tmp_path):
        """Returns empty list when pyproject not found."""
        result = check_dependency_pins(tmp_path / "nonexistent.toml")
        assert result == []

    def test_auto_discover_pyproject(self):
        """Auto-discovery should work from the project root."""
        result = check_dependency_pins()
        # May or may not find mismatches, but should not raise
        assert isinstance(result, list)

    def test_with_explicit_pyproject(self, tmp_path):
        content = '''\
[project]
dependencies = [
    "nonexistent-package-xyz>=1.0",
]
'''
        p = tmp_path / "pyproject.toml"
        p.write_text(content)
        result = check_dependency_pins(p)
        assert len(result) == 1
        assert "not installed" in result[0]

    def test_with_installed_package(self, tmp_path):
        """Check a package we know is installed (pandas)."""
        content = '''\
[project]
dependencies = [
    "pandas>=0.1.0",
]
'''
        p = tmp_path / "pyproject.toml"
        p.write_text(content)
        result = check_dependency_pins(p)
        # pandas IS installed and >=0.1.0, so should be empty or packaging note
        assert isinstance(result, list)
