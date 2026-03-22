"""Tests for the backtest engine and portfolio."""

import pandas as pd
import pytest

from ats_research.backtest.engine import BacktestEngine, BacktestResult
from ats_research.backtest.portfolio import Portfolio
from ats_research.data.loaders import generate_demo_data
from ats_research.strategy.sma_crossover import SMACrossover


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    @pytest.fixture
    def engine_and_data(self, sample_config):
        data = generate_demo_data(n_bars=200, freq="1D", seed=42)
        data["symbol"] = "DEMO"
        strategy = SMACrossover(params={"fast": 10, "slow": 30})
        engine = BacktestEngine(
            config=sample_config,
            strategy=strategy,
            data=data,
            initial_cash=100_000.0,
        )
        return engine, data

    def test_engine_runs_without_error(self, engine_and_data):
        """Basic run completes without raising."""
        engine, _ = engine_and_data
        result = engine.run()
        assert isinstance(result, BacktestResult)

    def test_engine_produces_equity_curve(self, engine_and_data):
        """Equity curve has correct length (one per bar)."""
        engine, data = engine_and_data
        result = engine.run()
        assert len(result.equity_curve) == len(data)

    def test_engine_initial_equity(self, engine_and_data):
        """Equity starts at the configured initial cash."""
        engine, _ = engine_and_data
        result = engine.run()
        assert result.equity_curve.iloc[0] == 100_000.0

    def test_engine_result_has_metrics(self, engine_and_data):
        """Result contains expected metrics keys."""
        engine, _ = engine_and_data
        result = engine.run()
        expected_keys = [
            "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "calmar_ratio", "annual_volatility",
            "hit_rate", "profit_factor",
        ]
        for key in expected_keys:
            assert key in result.metrics, f"Missing metric: {key}"


# ---------------------------------------------------------------------------
# Portfolio tests
# ---------------------------------------------------------------------------

class TestPortfolio:
    def test_portfolio_update_fill(self):
        """Fill updates positions and cash correctly."""
        portfolio = Portfolio(initial_cash=100_000.0)
        portfolio.update_fill(
            symbol="TEST",
            quantity=100,
            price=50.0,
            commission=5.0,
            side_is_buy=True,
        )
        # Cash: 100000 - (100*50) - 5 = 94995
        assert abs(portfolio.cash - 94_995.0) < 1e-8
        assert portfolio.positions["TEST"] == 100.0

    def test_portfolio_mark_to_market(self):
        """Mark-to-market updates equity in history."""
        portfolio = Portfolio(initial_cash=100_000.0)
        portfolio.update_fill(
            symbol="TEST",
            quantity=100,
            price=50.0,
            commission=0.0,
            side_is_buy=True,
        )
        ts = pd.Timestamp("2020-01-01", tz="UTC")
        portfolio.mark_to_market(ts, prices={"TEST": 55.0})
        eq_curve = portfolio.get_equity_curve()
        assert len(eq_curve) == 1
        # cash = 100000 - 5000 = 95000; position value = 100*55 = 5500
        # equity = 95000 + 5500 = 100500
        assert abs(eq_curve.iloc[0] - 100_500.0) < 1e-8
