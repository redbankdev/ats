"""Tests for the risk module: rules, position sizing, and risk manager."""

import math
from dataclasses import dataclass
from typing import Any

import pytest

from ats_research.risk.risk_rules import (
    MaxPositionRule,
    PerTradeLossCapRule,
    DailyLossHaltRule,
    RiskManager,
)
from ats_research.risk.position_sizing import (
    FixedFractionSizer,
    VolatilityScaledSizer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeOrder:
    symbol: str = "TEST"
    quantity: float = 100.0
    stop_price: float = None


@dataclass
class _FakeSignal:
    target_position: float = 1.0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def _portfolio_state(
    equity=100_000.0,
    cash=100_000.0,
    daily_pnl=0.0,
    positions=None,
    prices=None,
):
    return {
        "equity": equity,
        "cash": cash,
        "daily_pnl": daily_pnl,
        "positions": positions or {},
        "prices": prices or {"TEST": 100.0},
    }


# ---------------------------------------------------------------------------
# Risk Rules
# ---------------------------------------------------------------------------

class TestMaxPositionRule:
    def test_max_position_rule_pass(self):
        """Within limits: order is allowed."""
        rule = MaxPositionRule(max_pct=0.10)
        order = _FakeOrder(symbol="TEST", quantity=50)
        state = _portfolio_state(equity=100_000.0, prices={"TEST": 100.0})
        allowed, reason = rule.check(order, state)
        # 50 * 100 = 5000, 5% of 100k -> allowed
        assert allowed is True
        assert reason == ""

    def test_max_position_rule_fail(self):
        """Exceeding limit: order is rejected."""
        rule = MaxPositionRule(max_pct=0.10)
        order = _FakeOrder(symbol="TEST", quantity=200)
        state = _portfolio_state(equity=100_000.0, prices={"TEST": 100.0})
        allowed, reason = rule.check(order, state)
        # 200 * 100 = 20000, 20% of 100k -> rejected
        assert allowed is False
        assert "exceeding limit" in reason


class TestPerTradeLossCapRule:
    def test_per_trade_loss_cap_pass(self):
        """Trade with stop within loss cap is allowed."""
        rule = PerTradeLossCapRule(max_loss_pct=0.02)
        order = _FakeOrder(symbol="TEST", quantity=100, stop_price=98.0)
        state = _portfolio_state(equity=100_000.0, prices={"TEST": 100.0})
        allowed, reason = rule.check(order, state)
        # loss = 100 * |100-98| = 200, 0.2% of 100k -> allowed
        assert allowed is True

    def test_per_trade_loss_cap_fail(self):
        """Trade exceeding loss cap is rejected."""
        rule = PerTradeLossCapRule(max_loss_pct=0.01)
        # No stop -> worst case = full position value = 100*100 = 10000 = 10%
        order = _FakeOrder(symbol="TEST", quantity=100, stop_price=None)
        state = _portfolio_state(equity=100_000.0, prices={"TEST": 100.0})
        allowed, reason = rule.check(order, state)
        assert allowed is False
        assert "exceeding limit" in reason


class TestDailyLossHalt:
    def test_daily_loss_halt(self):
        """Triggers halt when daily loss exceeds threshold."""
        rule = DailyLossHaltRule(max_daily_loss_pct=0.05)
        order = _FakeOrder()
        # daily_pnl = -6000 -> loss_pct = 6000/100000 = 6% > 5%
        state = _portfolio_state(equity=100_000.0, daily_pnl=-6_000.0)
        allowed, reason = rule.check(order, state)
        assert allowed is False
        assert "halted" in reason.lower()
        # Subsequent orders also rejected
        state2 = _portfolio_state(equity=100_000.0, daily_pnl=0.0)
        allowed2, _ = rule.check(order, state2)
        assert allowed2 is False

    def test_daily_loss_halt_reset(self):
        """Halt can be reset."""
        rule = DailyLossHaltRule(max_daily_loss_pct=0.05)
        order = _FakeOrder()
        state = _portfolio_state(equity=100_000.0, daily_pnl=-6_000.0)
        rule.check(order, state)
        assert rule._halted is True
        rule.reset()
        assert rule._halted is False
        # Now orders pass again
        state2 = _portfolio_state(equity=100_000.0, daily_pnl=0.0)
        allowed, _ = rule.check(order, state2)
        assert allowed is True


# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------

class TestPositionSizing:
    def test_fixed_fraction_sizer(self):
        """Fixed fraction sizer computes correct number of units."""
        sizer = FixedFractionSizer(fraction=0.02, max_position_pct=0.50)
        signal = _FakeSignal(target_position=1.0, metadata={"stop_price": 95.0})
        # risk = 0.02 * 100000 = 2000; stop_distance = |100 - 95| = 5
        # raw = 2000/5 = 400 units; capped by max_position_pct = 0.5*100000/100 = 500
        # result = int(min(400, 500)) = 400
        result = sizer.size(signal=signal, equity=100_000.0, price=100.0, volatility=0.20)
        assert result == 400.0

    def test_volatility_scaled_sizer(self):
        """Volatility-scaled sizer produces correct sizing."""
        sizer = VolatilityScaledSizer(vol_target=0.10, max_position_pct=1.0)
        signal = _FakeSignal(target_position=1.0)
        # The formula: equity * vol_target / (price * daily_vol * sqrt(252))
        # = 100000 * 0.10 / (100 * 0.20) = 10000 / 20 = 500
        result = sizer.size(signal=signal, equity=100_000.0, price=100.0, volatility=0.20)
        assert result > 0
        assert result == 500.0

    def test_position_sizer_cap(self):
        """Max position cap is enforced."""
        sizer = FixedFractionSizer(fraction=0.10, max_position_pct=0.05)
        signal = _FakeSignal(target_position=1.0, metadata={"stop_price": 99.0})
        # risk = 0.10 * 100000 = 10000; stop_dist = 1 -> raw = 10000 units
        # max_units = 0.05 * 100000 / 100 = 50
        result = sizer.size(signal=signal, equity=100_000.0, price=100.0, volatility=0.20)
        assert result == 50.0


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------

class TestRiskManager:
    def test_risk_manager_all_pass(self):
        """All rules pass -> order is allowed."""
        manager = RiskManager(rules=[
            MaxPositionRule(max_pct=0.50),
            PerTradeLossCapRule(max_loss_pct=0.10),
        ])
        order = _FakeOrder(symbol="TEST", quantity=10)
        state = _portfolio_state(equity=100_000.0, prices={"TEST": 100.0})
        allowed, reason = manager.check_order(order, state)
        assert allowed is True
        assert reason == ""

    def test_risk_manager_one_fails(self):
        """First failure causes rejection."""
        manager = RiskManager(rules=[
            MaxPositionRule(max_pct=0.01),  # very tight
            PerTradeLossCapRule(max_loss_pct=0.10),
        ])
        order = _FakeOrder(symbol="TEST", quantity=100)
        state = _portfolio_state(equity=100_000.0, prices={"TEST": 100.0})
        allowed, reason = manager.check_order(order, state)
        assert allowed is False
        assert "exceeding limit" in reason
