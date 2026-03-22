"""Tests for report generation: plots and markdown/HTML reports."""

import os
from pathlib import Path

import pandas as pd
import pytest

from ats_research.data.loaders import generate_demo_data
from ats_research.report.plots import (
    generate_all_plots,
    plot_drawdown,
    plot_equity_curve,
    plot_monthly_returns_heatmap,
    plot_returns_distribution,
    plot_rolling_sharpe,
    _is_trivial,
    _finalise,
)
from ats_research.report.report import (
    generate_html_report,
    generate_markdown_report,
    _format_metric_value,
    _relative_plot_path,
)


@pytest.fixture
def equity_series():
    """Generate a realistic equity series from demo data."""
    data = generate_demo_data(n_bars=200, seed=42)
    # Simulate a simple equity curve: cumulative returns
    returns = data["close"].pct_change().fillna(0)
    equity = 100_000 * (1 + returns).cumprod()
    equity.name = "equity"
    return equity


@pytest.fixture
def short_equity():
    """Equity series too short for meaningful plots."""
    idx = pd.date_range("2020-01-01", periods=1, freq="D", tz="UTC")
    return pd.Series([100_000.0], index=idx, name="equity")


@pytest.fixture
def sample_metrics():
    return {
        "total_return": 0.1523,
        "cagr": 0.0812,
        "sharpe_ratio": 1.234,
        "sortino_ratio": 1.567,
        "max_drawdown": -0.0845,
        "calmar_ratio": 0.962,
        "annual_volatility": 0.152,
        "hit_rate": 0.55,
        "profit_factor": 1.42,
        "exposure": 0.78,
        "avg_win": 0.0123,
        "avg_loss": -0.0087,
        "daily_turnover": 0.05,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


class TestPlotHelpers:
    def test_is_trivial_none(self):
        assert _is_trivial(None) is True

    def test_is_trivial_short(self):
        s = pd.Series([100.0])
        assert _is_trivial(s) is True

    def test_is_trivial_normal(self):
        s = pd.Series([100.0, 101.0, 102.0])
        assert _is_trivial(s) is False


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------


class TestPlotEquityCurve:
    def test_plot_equity_curve_saves(self, equity_series, tmp_path):
        save_path = str(tmp_path / "equity.png")
        fig = plot_equity_curve(equity_series, save_path=save_path)
        assert Path(save_path).exists()

    def test_plot_equity_curve_trivial(self, short_equity, tmp_path):
        save_path = str(tmp_path / "equity_trivial.png")
        fig = plot_equity_curve(short_equity, save_path=save_path)
        assert Path(save_path).exists()

    def test_plot_equity_curve_no_save(self, equity_series):
        import matplotlib.pyplot as plt
        fig = plot_equity_curve(equity_series, save_path=None)
        assert fig is not None
        plt.close(fig)


class TestPlotDrawdown:
    def test_plot_drawdown_saves(self, equity_series, tmp_path):
        save_path = str(tmp_path / "drawdown.png")
        fig = plot_drawdown(equity_series, save_path=save_path)
        assert Path(save_path).exists()

    def test_plot_drawdown_trivial(self, short_equity, tmp_path):
        save_path = str(tmp_path / "dd_trivial.png")
        fig = plot_drawdown(short_equity, save_path=save_path)
        assert Path(save_path).exists()


class TestPlotRollingSharpe:
    def test_plot_rolling_sharpe_saves(self, equity_series, tmp_path):
        save_path = str(tmp_path / "sharpe.png")
        fig = plot_rolling_sharpe(equity_series, window=20, save_path=save_path)
        assert Path(save_path).exists()

    def test_plot_rolling_sharpe_trivial(self, short_equity, tmp_path):
        save_path = str(tmp_path / "sharpe_trivial.png")
        fig = plot_rolling_sharpe(short_equity, save_path=save_path)
        assert Path(save_path).exists()

    def test_plot_rolling_sharpe_insufficient_window(self, tmp_path):
        """Equity shorter than window shows insufficient data."""
        idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        equity = pd.Series(range(100, 110), index=idx, dtype=float)
        save_path = str(tmp_path / "sharpe_short.png")
        fig = plot_rolling_sharpe(equity, window=63, save_path=save_path)
        assert Path(save_path).exists()


class TestPlotReturnsDistribution:
    def test_plot_returns_distribution_saves(self, equity_series, tmp_path):
        save_path = str(tmp_path / "returns.png")
        fig = plot_returns_distribution(equity_series, save_path=save_path)
        assert Path(save_path).exists()

    def test_plot_returns_distribution_trivial(self, short_equity, tmp_path):
        save_path = str(tmp_path / "returns_trivial.png")
        fig = plot_returns_distribution(short_equity, save_path=save_path)
        assert Path(save_path).exists()


class TestPlotMonthlyReturnsHeatmap:
    def test_plot_monthly_heatmap_saves(self, equity_series, tmp_path):
        save_path = str(tmp_path / "heatmap.png")
        fig = plot_monthly_returns_heatmap(equity_series, save_path=save_path)
        assert Path(save_path).exists()

    def test_plot_monthly_heatmap_trivial(self, short_equity, tmp_path):
        save_path = str(tmp_path / "heatmap_trivial.png")
        fig = plot_monthly_returns_heatmap(short_equity, save_path=save_path)
        assert Path(save_path).exists()


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


class TestGenerateAllPlots:
    def test_generate_all_plots(self, equity_series, tmp_path):
        output_dir = str(tmp_path / "plots")
        saved = generate_all_plots(equity_series, output_dir)
        assert len(saved) == 5
        for path in saved:
            assert os.path.exists(path)

    def test_generate_all_plots_creates_dir(self, equity_series, tmp_path):
        output_dir = str(tmp_path / "nested" / "plots")
        saved = generate_all_plots(equity_series, output_dir)
        assert len(saved) == 5
        assert Path(output_dir).exists()


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


class TestFormatMetricValue:
    def test_none(self):
        assert _format_metric_value("any_key", None) == "N/A"

    def test_nan(self):
        assert _format_metric_value("any_key", float("nan")) == "N/A"

    def test_inf(self):
        assert _format_metric_value("any_key", float("inf")) == "Inf"

    def test_pct_key(self):
        result = _format_metric_value("total_return", 0.1523)
        assert "15.23%" in result

    def test_non_pct_key(self):
        result = _format_metric_value("sharpe_ratio", 1.234)
        assert "1.2340" in result

    def test_string_value(self):
        assert _format_metric_value("name", "test") == "test"

    def test_integer_value(self):
        assert _format_metric_value("count", 42) == "42"

    def test_hit_rate(self):
        result = _format_metric_value("hit_rate", 0.55)
        assert "55.00%" in result

    def test_cagr(self):
        result = _format_metric_value("cagr", 0.12)
        assert "12.00%" in result

    def test_max_drawdown(self):
        result = _format_metric_value("max_drawdown", -0.08)
        assert "-8.00%" in result

    def test_exposure(self):
        result = _format_metric_value("exposure", 0.78)
        assert "78.00%" in result

    def test_avg_win(self):
        result = _format_metric_value("avg_win", 0.012)
        assert "1.20%" in result

    def test_avg_loss(self):
        result = _format_metric_value("avg_loss", -0.008)
        assert "-0.80%" in result

    def test_daily_turnover(self):
        result = _format_metric_value("daily_turnover", 0.05)
        assert "5.00%" in result

    def test_annual_volatility(self):
        result = _format_metric_value("annual_volatility", 0.15)
        assert "15.00%" in result


class TestRelativePlotPath:
    def test_relative_path_same_parent(self, tmp_path):
        plots_dir = str(tmp_path / "plots")
        output_path = str(tmp_path / "report.md")
        result = _relative_plot_path(plots_dir, output_path, "equity_curve.png")
        assert "equity_curve.png" in result

    def test_relative_path_different_parents(self, tmp_path):
        plots_dir = str(tmp_path / "run1" / "plots")
        output_path = str(tmp_path / "run1" / "report.md")
        result = _relative_plot_path(plots_dir, output_path, "equity_curve.png")
        assert "equity_curve.png" in result


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


class TestMarkdownReport:
    def test_generate_markdown_report(self, sample_metrics, tmp_path):
        plots_dir = str(tmp_path / "plots")
        os.makedirs(plots_dir, exist_ok=True)
        output_path = str(tmp_path / "report.md")

        config = {"run_id": "test_run_123", "strategy": "sma_crossover"}
        content = generate_markdown_report(
            metrics=sample_metrics,
            config=config,
            plots_dir=plots_dir,
            output_path=output_path,
        )

        assert Path(output_path).exists()
        assert "# Backtest Report" in content
        assert "DISCLAIMER" in content
        assert "Performance Metrics" in content
        assert "test_run_123" in content
        assert "sma_crossover" in content

    def test_markdown_report_with_empty_config(self, sample_metrics, tmp_path):
        plots_dir = str(tmp_path / "plots")
        os.makedirs(plots_dir, exist_ok=True)
        output_path = str(tmp_path / "report.md")

        content = generate_markdown_report(
            metrics=sample_metrics,
            config={},
            plots_dir=plots_dir,
            output_path=output_path,
        )
        assert "N/A" in content  # run_id defaults to N/A


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


class TestHTMLReport:
    def test_generate_html_report(self, sample_metrics, tmp_path):
        plots_dir = str(tmp_path / "plots")
        os.makedirs(plots_dir, exist_ok=True)
        output_path = str(tmp_path / "report.html")

        config = {"run_id": "test_run_456", "strategy": "sma_crossover"}
        content = generate_html_report(
            metrics=sample_metrics,
            config=config,
            plots_dir=plots_dir,
            output_path=output_path,
        )

        assert Path(output_path).exists()
        assert "<!DOCTYPE html>" in content
        assert "Backtest Report" in content
        assert "test_run_456" in content
        assert "DISCLAIMER" in content

    def test_html_report_with_config_summary(self, sample_metrics, tmp_path):
        plots_dir = str(tmp_path / "plots")
        os.makedirs(plots_dir, exist_ok=True)
        output_path = str(tmp_path / "report.html")

        config = {
            "strategy": "sma_crossover",
            "start_date": "2020-01-01",
            "end_date": "2021-12-31",
        }
        content = generate_html_report(
            metrics=sample_metrics,
            config=config,
            plots_dir=plots_dir,
            output_path=output_path,
        )
        assert "sma_crossover" in content

    def test_html_report_creates_parent_dirs(self, sample_metrics, tmp_path):
        plots_dir = str(tmp_path / "plots")
        os.makedirs(plots_dir, exist_ok=True)
        output_path = str(tmp_path / "nested" / "dir" / "report.html")

        content = generate_html_report(
            metrics=sample_metrics,
            config={},
            plots_dir=plots_dir,
            output_path=output_path,
        )
        assert Path(output_path).exists()
