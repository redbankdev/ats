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

    def test_portfolio_sell_realizes_pnl(self):
        """Selling from a long position realizes PnL."""
        portfolio = Portfolio(initial_cash=100_000.0)
        portfolio.update_fill("TEST", 100, 50.0, 0.0, side_is_buy=True)
        portfolio.update_fill("TEST", 100, 60.0, 0.0, side_is_buy=False)
        # Realized PnL: 100 * (60 - 50) = 1000
        assert abs(portfolio.realized_pnl - 1000.0) < 1e-8
        assert "TEST" not in portfolio.positions  # fully closed

    def test_portfolio_short_position(self):
        """Opening and closing a short position."""
        portfolio = Portfolio(initial_cash=100_000.0)
        # Sell to open short
        portfolio.update_fill("TEST", 100, 50.0, 0.0, side_is_buy=False)
        assert portfolio.positions["TEST"] == -100.0
        # Buy to close short
        portfolio.update_fill("TEST", 100, 45.0, 0.0, side_is_buy=True)
        # Profit: 100 * (50 - 45) = 500
        assert abs(portfolio.realized_pnl - 500.0) < 1e-8
        assert "TEST" not in portfolio.positions

    def test_portfolio_cross_from_long_to_short(self):
        """Selling more than long position crosses to short."""
        portfolio = Portfolio(initial_cash=100_000.0)
        portfolio.update_fill("TEST", 50, 100.0, 0.0, side_is_buy=True)
        # Sell 80: close 50 long + open 30 short
        portfolio.update_fill("TEST", 80, 110.0, 0.0, side_is_buy=False)
        assert portfolio.positions["TEST"] == -30.0
        # Realized on the 50 closed: 50 * (110 - 100) = 500
        assert abs(portfolio.realized_pnl - 500.0) < 1e-8
        # Average entry for the short should be 110
        assert abs(portfolio.avg_entry_prices["TEST"] - 110.0) < 1e-8

    def test_portfolio_cross_from_short_to_long(self):
        """Buying more than short position crosses to long."""
        portfolio = Portfolio(initial_cash=100_000.0)
        portfolio.update_fill("TEST", 50, 100.0, 0.0, side_is_buy=False)  # short 50
        # Buy 80: cover 50 short + open 30 long
        portfolio.update_fill("TEST", 80, 90.0, 0.0, side_is_buy=True)
        assert portfolio.positions["TEST"] == 30.0
        # Realized on the 50 covered: 50 * (100 - 90) = 500
        assert abs(portfolio.realized_pnl - 500.0) < 1e-8
        # Average entry for the long should be 90
        assert abs(portfolio.avg_entry_prices["TEST"] - 90.0) < 1e-8

    def test_portfolio_get_state(self):
        """get_state returns correct equity and positions."""
        portfolio = Portfolio(initial_cash=100_000.0)
        portfolio.update_fill("TEST", 100, 50.0, 0.0, side_is_buy=True)
        state = portfolio.get_state({"TEST": 55.0})
        assert abs(state["equity"] - 100_500.0) < 1e-8
        assert state["positions"]["TEST"] == 100.0

    def test_portfolio_new_day_with_history(self):
        """new_day resets daily PnL tracking from last equity."""
        portfolio = Portfolio(initial_cash=100_000.0)
        portfolio.update_fill("TEST", 100, 50.0, 0.0, side_is_buy=True)
        ts = pd.Timestamp("2020-01-01", tz="UTC")
        portfolio.mark_to_market(ts, {"TEST": 55.0})
        portfolio.new_day()
        assert abs(portfolio._daily_start_equity - 100_500.0) < 1e-8

    def test_portfolio_new_day_no_history(self):
        """new_day without history uses cash."""
        portfolio = Portfolio(initial_cash=50_000.0)
        portfolio.new_day()
        assert portfolio._daily_start_equity == 50_000.0

    def test_portfolio_get_history_df_empty(self):
        """Empty portfolio returns empty DataFrame."""
        portfolio = Portfolio()
        df = portfolio.get_history_df()
        assert df.empty

    def test_portfolio_get_equity_curve_empty(self):
        """Empty portfolio returns empty Series."""
        portfolio = Portfolio()
        eq = portfolio.get_equity_curve()
        assert len(eq) == 0

    def test_portfolio_get_history_df(self):
        """History DataFrame has expected columns."""
        portfolio = Portfolio(initial_cash=100_000.0)
        portfolio.update_fill("TEST", 100, 50.0, 0.0, side_is_buy=True)
        ts = pd.Timestamp("2020-01-01", tz="UTC")
        portfolio.mark_to_market(ts, {"TEST": 55.0})
        df = portfolio.get_history_df()
        assert len(df) == 1
        assert "cash" in df.columns
        assert "equity" in df.columns

    def test_portfolio_unrealized_pnl_short(self):
        """Mark-to-market correctly calculates unrealized PnL for shorts."""
        portfolio = Portfolio(initial_cash=100_000.0)
        portfolio.update_fill("TEST", 100, 50.0, 0.0, side_is_buy=False)
        ts = pd.Timestamp("2020-01-01", tz="UTC")
        portfolio.mark_to_market(ts, {"TEST": 45.0})
        # Short 100 @ 50, current 45 -> unrealized = 100 * (50 - 45) = 500
        assert len(portfolio._history) == 1
        assert abs(portfolio._history[0].unrealized_pnl - 500.0) < 1e-8


class TestBacktestResult:
    def test_to_json(self, sample_config, tmp_path):
        """BacktestResult.to_json saves metrics."""
        from ats_research.utils.repro import RunContext
        eq = pd.Series([100_000.0, 100_500.0],
                       index=pd.date_range("2020-01-01", periods=2, freq="D", tz="UTC"),
                       name="equity")
        result = BacktestResult(
            equity_curve=eq,
            metrics={"total_return": 0.005, "sharpe_ratio": float("nan")},
            trades=[],
            portfolio_history=pd.DataFrame(),
            config=sample_config,
            run_context=RunContext.create(seed=42, run_id="test_run"),
        )
        out = tmp_path / "metrics.json"
        result.to_json(out)
        assert out.exists()
        import json
        data = json.loads(out.read_text())
        assert data["metrics"]["total_return"] == 0.005
        assert data["metrics"]["sharpe_ratio"] is None  # NaN -> None

    def test_to_csv(self, sample_config, tmp_path):
        """BacktestResult.to_csv saves equity curve."""
        from ats_research.utils.repro import RunContext
        eq = pd.Series([100_000.0, 100_500.0],
                       index=pd.date_range("2020-01-01", periods=2, freq="D", tz="UTC"),
                       name="equity")
        result = BacktestResult(
            equity_curve=eq,
            metrics={},
            trades=[],
            portfolio_history=pd.DataFrame(),
            config=sample_config,
            run_context=RunContext.create(seed=42, run_id="test_run"),
        )
        out = tmp_path / "eq.csv"
        result.to_csv(out)
        assert out.exists()

    def test_summary(self, sample_config):
        """BacktestResult.summary returns formatted string."""
        from ats_research.utils.repro import RunContext
        eq = pd.Series([100_000.0, 101_000.0],
                       index=pd.date_range("2020-01-01", periods=2, freq="D", tz="UTC"),
                       name="equity")
        result = BacktestResult(
            equity_curve=eq,
            metrics={
                "total_return": 0.01, "cagr": 0.01, "sharpe_ratio": 1.0,
                "sortino_ratio": 1.5, "max_drawdown": -0.05, "calmar_ratio": 0.2,
                "annual_volatility": 0.15, "hit_rate": 0.55, "profit_factor": 1.3,
                "avg_win": 0.01, "avg_loss": -0.008,
                "max_consecutive_wins": 5, "max_consecutive_losses": 3,
                "exposure": 0.8,
            },
            trades=[],
            portfolio_history=pd.DataFrame(),
            config=sample_config,
            run_context=RunContext.create(seed=42, run_id="test_run"),
        )
        text = result.summary()
        assert "Backtest Summary" in text
        assert "Total Return" in text
        assert "Start equity" in text

    def test_summary_with_nan_metrics(self, sample_config):
        """Summary handles NaN metrics gracefully."""
        from ats_research.utils.repro import RunContext
        eq = pd.Series([100_000.0],
                       index=pd.date_range("2020-01-01", periods=1, freq="D", tz="UTC"),
                       name="equity")
        result = BacktestResult(
            equity_curve=eq,
            metrics={"total_return": float("nan"), "cagr": float("nan")},
            trades=[],
            portfolio_history=pd.DataFrame(),
            config=sample_config,
            run_context=RunContext.create(seed=42),
        )
        text = result.summary()
        assert "N/A" in text
