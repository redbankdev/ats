"""Tests for backtest metrics module."""

import numpy as np
import pandas as pd
import pytest

from ats_research.backtest.metrics import compute_drawdown_series, compute_metrics


def _make_equity(values: list[float], start_date: str = "2020-01-01") -> pd.Series:
    idx = pd.date_range(start_date, periods=len(values), freq="B", tz="UTC")
    return pd.Series(values, index=idx, name="equity")


class TestComputeMetrics:
    def test_compute_metrics_keys(self):
        """All expected keys are present in the result."""
        eq = _make_equity([100, 101, 102, 103, 104])
        m = compute_metrics(eq)
        expected_keys = [
            "total_return",
            "cagr",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "annual_volatility",
            "hit_rate",
            "profit_factor",
            "avg_win",
            "avg_loss",
            "max_consecutive_wins",
            "max_consecutive_losses",
            "exposure",
        ]
        for key in expected_keys:
            assert key in m, f"Missing metric: {key}"

    def test_sharpe_ratio_flat_returns(self):
        """Zero returns (flat equity) gives 0 Sharpe."""
        eq = _make_equity([100] * 50)
        m = compute_metrics(eq)
        assert m["sharpe_ratio"] == 0.0

    def test_max_drawdown_known_values(self):
        """Golden test with known drawdown data."""
        # 100 -> 120 -> 90 -> 100: max DD = (90-120)/120 = -25%
        eq = _make_equity([100, 110, 120, 100, 90, 95, 100])
        m = compute_metrics(eq)
        assert abs(m["max_drawdown"] - (-0.25)) < 1e-10

    def test_cagr_known_values(self):
        """Golden test for CAGR calculation."""
        # ~252 trading days = 1 year, 100 -> 110 = ~10% CAGR
        n = 253
        values = np.linspace(100, 110, n).tolist()
        eq = _make_equity(values)
        m = compute_metrics(eq)
        # Should be close to 10%
        assert abs(m["cagr"] - 0.10) < 0.02

    def test_sortino_ratio(self):
        """Sortino ratio is computed correctly (positive for upward trend)."""
        eq = _make_equity(list(range(100, 200)))
        m = compute_metrics(eq)
        assert m["sortino_ratio"] > 0

    def test_profit_factor(self):
        """Profit factor = gross profits / gross losses."""
        eq = _make_equity([100, 101, 100, 101, 102, 101])
        m = compute_metrics(eq)
        # Has both positive and negative returns, so profit_factor > 0
        assert m["profit_factor"] > 0

    def test_hit_rate(self):
        """Hit rate is the percentage of positive-return days."""
        # 3 up days, 1 down day out of 4 returns -> 75%
        eq = _make_equity([100, 101, 102, 103, 102])
        m = compute_metrics(eq)
        assert 0.5 <= m["hit_rate"] <= 1.0


class TestDrawdownSeries:
    def test_shape(self):
        eq = _make_equity([100, 110, 105, 115])
        dd = compute_drawdown_series(eq)
        assert len(dd) == len(eq)

    def test_starts_at_zero(self):
        eq = _make_equity([100, 110, 105])
        dd = compute_drawdown_series(eq)
        assert dd.iloc[0] == 0.0

    def test_all_negative_or_zero(self):
        eq = _make_equity([100, 110, 105, 115, 90])
        dd = compute_drawdown_series(eq)
        assert (dd <= 0.0).all()

    def test_max_drawdown_matches(self):
        eq = _make_equity([100, 120, 90, 100])
        dd = compute_drawdown_series(eq)
        assert abs(dd.min() - (-0.25)) < 1e-10
