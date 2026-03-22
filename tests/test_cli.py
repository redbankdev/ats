"""Tests for CLI commands using click.testing.CliRunner."""

import json
import os
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from ats_research.cli.main import cli, _resolve_strategy
from ats_research.data.loaders import generate_demo_data


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def config_and_data(tmp_path):
    """Create a valid config file and matching data file."""
    # Generate and save demo data
    data = generate_demo_data(n_bars=100, freq="1D", seed=42)
    data_path = tmp_path / "data.csv"
    data.to_csv(data_path)

    # Create config YAML
    config_content = f"""\
data:
  path: {data_path}
  freq: "1D"
  tz: "UTC"

strategy:
  name: sma_crossover
  params:
    fast: 10
    slow: 30

risk:
  max_pos_pct: 0.10
  per_trade_risk_pct: 0.01
  daily_loss_limit_pct: 0.05
  vol_target_pct: 0.10

costs:
  slippage_bps: 5.0
  commission_per_share: 0.005

simulation:
  seed: 42
  liquidity_cap_pct_adv: 0.02

output:
  dir: {tmp_path / "runs"}
  save_plots: true
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return str(config_path), str(data_path), tmp_path


class TestCLIBacktest:
    def test_backtest_command_runs(self, runner, config_and_data):
        """The backtest command runs to completion."""
        config_path, _, tmp_path = config_and_data
        result = runner.invoke(cli, ["backtest", "--config", config_path])
        assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
        assert "Outputs saved to" in result.output

    def test_backtest_with_output_override(self, runner, config_and_data):
        """Output directory can be overridden."""
        config_path, _, tmp_path = config_and_data
        custom_out = str(tmp_path / "custom_out")
        result = runner.invoke(
            cli, ["backtest", "--config", config_path, "--output", custom_out]
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
        assert Path(custom_out).exists()

    def test_backtest_produces_output_files(self, runner, config_and_data):
        """Backtest creates metrics.json, equity_curve.csv, and reports."""
        config_path, _, tmp_path = config_and_data
        result = runner.invoke(cli, ["backtest", "--config", config_path])
        assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"

        runs_dir = tmp_path / "runs"
        assert runs_dir.exists()
        # Find the run directory (first subdirectory)
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        assert len(run_dirs) >= 1
        run_dir = run_dirs[0]

        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "equity_curve.csv").exists()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "report.md").exists()
        assert (run_dir / "report.html").exists()


class TestCLIReport:
    def test_report_command_missing_run(self, runner, tmp_path):
        """Report command fails gracefully when run directory is missing."""
        result = runner.invoke(
            cli, ["report", "--run", "nonexistent_run", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code != 0

    def test_report_command_runs(self, runner, config_and_data):
        """Report command succeeds on a completed run."""
        config_path, _, tmp_path = config_and_data

        # First run a backtest
        result = runner.invoke(cli, ["backtest", "--config", config_path])
        assert result.exit_code == 0

        # Find the run_id
        runs_dir = tmp_path / "runs"
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        run_id = run_dirs[0].name

        # Regenerate report
        result = runner.invoke(
            cli, ["report", "--run", run_id, "--output-dir", str(runs_dir)]
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
        assert "Reports regenerated" in result.output

    def test_report_missing_equity_curve(self, runner, tmp_path):
        """Report command fails when equity_curve.csv is missing."""
        run_dir = tmp_path / "fake_run"
        run_dir.mkdir(parents=True)
        result = runner.invoke(
            cli, ["report", "--run", "fake_run", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code != 0


class TestCLIPaper:
    def test_paper_command_runs(self, runner, config_and_data):
        """Paper trading command runs to completion."""
        config_path, data_path, tmp_path = config_and_data
        result = runner.invoke(
            cli, ["paper", "--config", config_path, "--stream", data_path]
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
        assert "Paper Trading Complete" in result.output


class TestResolveStrategy:
    def test_resolve_known_strategy(self):
        """Known strategy name resolves correctly."""
        strategy = _resolve_strategy("sma_crossover", {"fast": 10, "slow": 30})
        assert strategy.name == "SMACrossover"

    def test_resolve_unknown_strategy(self):
        """Unknown strategy raises ClickException."""
        import click
        with pytest.raises(click.ClickException, match="Unknown strategy"):
            _resolve_strategy("nonexistent", {})


class TestCLIVersion:
    def test_version_option(self, runner):
        """--version shows version info."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "atsr" in result.output
