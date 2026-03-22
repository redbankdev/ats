"""Tests for the execution module: slippage, commission, broker_sim."""

import pandas as pd
import pytest
from datetime import datetime

from ats_research.exec.orders import Order, OrderSide, OrderType, OrderStatus, TimeInForce
from ats_research.exec.slippage import FixedBpsSlippage, VolumeDepSlippage, NoSlippage
from ats_research.exec.commission import PerShareCommission, PercentCommission, ZeroCommission
from ats_research.exec.broker_sim import BrokerSim


# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------

class TestSlippage:
    def test_fixed_bps_slippage_buy(self):
        """Buys pay more than the raw price."""
        model = FixedBpsSlippage(bps=10.0)
        order = Order(symbol="TEST", side=OrderSide.BUY, quantity=100)
        price = 100.0
        adjusted = model.apply(order, price, volume=1_000_000)
        assert adjusted > price
        # 10 bps = 0.1% -> 100 * 1.001 = 100.1
        assert abs(adjusted - 100.1) < 1e-8

    def test_fixed_bps_slippage_sell(self):
        """Sells receive less than the raw price."""
        model = FixedBpsSlippage(bps=10.0)
        order = Order(symbol="TEST", side=OrderSide.SELL, quantity=100)
        price = 100.0
        adjusted = model.apply(order, price, volume=1_000_000)
        assert adjusted < price
        assert abs(adjusted - 99.9) < 1e-8

    def test_volume_dep_slippage(self):
        """Larger orders relative to volume have more slippage."""
        model = VolumeDepSlippage(base_bps=5.0, volume_impact=100.0)
        small_order = Order(symbol="TEST", side=OrderSide.BUY, quantity=100)
        large_order = Order(symbol="TEST", side=OrderSide.BUY, quantity=10_000)
        volume = 100_000.0
        price = 100.0

        small_adjusted = model.apply(small_order, price, volume)
        large_adjusted = model.apply(large_order, price, volume)
        # Large order should pay more slippage
        assert large_adjusted > small_adjusted


# ---------------------------------------------------------------------------
# Commission
# ---------------------------------------------------------------------------

class TestCommission:
    def test_per_share_commission(self):
        """Per-share commission math is correct."""
        model = PerShareCommission(rate=0.005, min_commission=1.0)
        # 100 shares * 0.005 = 0.50, but min is 1.0
        assert model.calculate(100, 50.0) == 1.0
        # 1000 shares * 0.005 = 5.0, above min
        assert abs(model.calculate(1000, 50.0) - 5.0) < 1e-8

    def test_percent_commission(self):
        """Percent commission math is correct."""
        model = PercentCommission(rate=0.001)
        # 0.001 * 100 * 50 = 5.0
        result = model.calculate(100, 50.0)
        assert abs(result - 5.0) < 1e-8

    def test_zero_commission(self):
        """Zero commission returns 0."""
        model = ZeroCommission()
        assert model.calculate(100, 50.0) == 0.0


# ---------------------------------------------------------------------------
# BrokerSim
# ---------------------------------------------------------------------------

def _make_bar(open_=100.0, high=105.0, low=95.0, close=102.0, volume=1_000_000):
    return pd.Series({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestBrokerSim:
    def test_market_order_fill(self):
        """Market order fills at bar open price (no slippage model)."""
        broker = BrokerSim(slippage_model=NoSlippage(), commission_model=ZeroCommission())
        order = Order(
            symbol="TEST", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET, time_in_force=TimeInForce.GTC,
        )
        broker.submit_order(order)
        bar = _make_bar(open_=100.0)
        fills = broker.process_bar(bar, current_dt=datetime(2020, 1, 1))
        assert len(fills) == 1
        assert fills[0].price == 100.0
        assert fills[0].quantity == 100

    def test_limit_order_fill(self):
        """Limit buy fills when bar low touches limit price."""
        broker = BrokerSim(slippage_model=NoSlippage(), commission_model=ZeroCommission())
        order = Order(
            symbol="TEST", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.LIMIT, limit_price=96.0,
            time_in_force=TimeInForce.GTC,
        )
        broker.submit_order(order)
        # Bar low=95 touches limit=96
        bar = _make_bar(open_=100.0, high=105.0, low=95.0)
        fills = broker.process_bar(bar, current_dt=datetime(2020, 1, 1))
        assert len(fills) == 1
        assert fills[0].price == 96.0

    def test_limit_order_no_fill(self):
        """Limit buy does not fill when bar low is above limit price."""
        broker = BrokerSim(slippage_model=NoSlippage(), commission_model=ZeroCommission())
        order = Order(
            symbol="TEST", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.LIMIT, limit_price=90.0,
            time_in_force=TimeInForce.GTC,
        )
        broker.submit_order(order)
        # Bar low=95, limit=90 -> no fill
        bar = _make_bar(open_=100.0, high=105.0, low=95.0)
        fills = broker.process_bar(bar, current_dt=datetime(2020, 1, 1))
        assert len(fills) == 0

    def test_partial_fill_liquidity_cap(self):
        """Partial fill when order exceeds volume cap."""
        broker = BrokerSim(
            slippage_model=NoSlippage(),
            commission_model=ZeroCommission(),
            liquidity_cap_pct_adv=0.02,
        )
        # Volume=10000, cap=0.02 -> max fill = 200
        order = Order(
            symbol="TEST", side=OrderSide.BUY, quantity=500,
            order_type=OrderType.MARKET, time_in_force=TimeInForce.GTC,
        )
        broker.submit_order(order)
        bar = _make_bar(volume=10_000)
        fills = broker.process_bar(bar, current_dt=datetime(2020, 1, 1))
        assert len(fills) == 1
        assert fills[0].quantity == 200.0
        assert order.status == OrderStatus.PARTIALLY_FILLED

    def test_stop_order_trigger(self):
        """Stop buy order triggers and fills as market when bar high >= stop price."""
        broker = BrokerSim(slippage_model=NoSlippage(), commission_model=ZeroCommission())
        order = Order(
            symbol="TEST", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.STOP, stop_price=104.0,
            time_in_force=TimeInForce.GTC,
        )
        broker.submit_order(order)
        # Bar high=105 >= stop=104, so stop triggers and fills at open
        bar = _make_bar(open_=100.0, high=105.0, low=95.0)
        fills = broker.process_bar(bar, current_dt=datetime(2020, 1, 1))
        assert len(fills) == 1
        assert fills[0].price == 100.0  # fills at open
        assert order.status == OrderStatus.FILLED

    def test_stop_sell_order_trigger(self):
        """Stop sell triggers when bar low <= stop price."""
        broker = BrokerSim(slippage_model=NoSlippage(), commission_model=ZeroCommission())
        order = Order(
            symbol="TEST", side=OrderSide.SELL, quantity=100,
            order_type=OrderType.STOP, stop_price=96.0,
            time_in_force=TimeInForce.GTC,
        )
        broker.submit_order(order)
        bar = _make_bar(open_=100.0, high=105.0, low=95.0)
        fills = broker.process_bar(bar, current_dt=datetime(2020, 1, 1))
        assert len(fills) == 1
        assert order.status == OrderStatus.FILLED

    def test_stop_order_no_trigger(self):
        """Stop buy does not trigger when bar high < stop price."""
        broker = BrokerSim(slippage_model=NoSlippage(), commission_model=ZeroCommission())
        order = Order(
            symbol="TEST", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.STOP, stop_price=110.0,
            time_in_force=TimeInForce.GTC,
        )
        broker.submit_order(order)
        bar = _make_bar(open_=100.0, high=105.0, low=95.0)
        fills = broker.process_bar(bar, current_dt=datetime(2020, 1, 1))
        assert len(fills) == 0

    def test_day_order_expires(self):
        """DAY orders are cancelled at end of bar cycle."""
        broker = BrokerSim(slippage_model=NoSlippage(), commission_model=ZeroCommission())
        order = Order(
            symbol="TEST", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.LIMIT, limit_price=90.0,
            time_in_force=TimeInForce.DAY,
        )
        broker.submit_order(order)
        # Bar low=95, limit=90 -> no fill, then DAY expires
        bar = _make_bar(open_=100.0, high=105.0, low=95.0)
        fills = broker.process_bar(bar, current_dt=datetime(2020, 1, 1))
        assert len(fills) == 0
        assert order.status == OrderStatus.CANCELLED
        assert len(broker.get_pending_orders()) == 0

    def test_cancel_order(self):
        """Cancel a pending order."""
        broker = BrokerSim(slippage_model=NoSlippage(), commission_model=ZeroCommission())
        order = Order(
            symbol="TEST", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.MARKET, time_in_force=TimeInForce.GTC,
        )
        broker.submit_order(order)
        assert len(broker.get_pending_orders()) == 1
        result = broker.cancel_order(order.order_id)
        assert result is True
        assert order.status == OrderStatus.CANCELLED
        assert len(broker.get_pending_orders()) == 0

    def test_cancel_nonexistent_order(self):
        """Cancelling a non-existent order returns False."""
        broker = BrokerSim()
        assert broker.cancel_order("nonexistent_id") is False

    def test_submit_non_pending_order(self):
        """Submitting non-PENDING order raises ValueError."""
        broker = BrokerSim()
        order = Order(
            symbol="TEST", side=OrderSide.BUY, quantity=100,
            status=OrderStatus.FILLED,
        )
        with pytest.raises(ValueError, match="Cannot submit order"):
            broker.submit_order(order)

    def test_reject_limit_missing_price(self):
        """Limit order missing limit_price is rejected."""
        broker = BrokerSim()
        order = Order(
            symbol="TEST", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.LIMIT, limit_price=None,
        )
        broker.submit_order(order)
        assert order.status == OrderStatus.REJECTED

    def test_reject_stop_missing_price(self):
        """Stop order missing stop_price is rejected."""
        broker = BrokerSim()
        order = Order(
            symbol="TEST", side=OrderSide.BUY, quantity=100,
            order_type=OrderType.STOP, stop_price=None,
        )
        broker.submit_order(order)
        assert order.status == OrderStatus.REJECTED

    def test_reject_zero_quantity(self):
        """Order with zero quantity is rejected."""
        broker = BrokerSim()
        order = Order(
            symbol="TEST", side=OrderSide.BUY, quantity=0,
            order_type=OrderType.MARKET,
        )
        broker.submit_order(order)
        assert order.status == OrderStatus.REJECTED

    def test_invalid_liquidity_cap(self):
        """Negative liquidity_cap_pct_adv raises ValueError."""
        with pytest.raises(ValueError, match="liquidity_cap_pct_adv"):
            BrokerSim(liquidity_cap_pct_adv=-0.01)

    def test_limit_sell_fill(self):
        """Limit sell fills when bar high >= limit price."""
        broker = BrokerSim(slippage_model=NoSlippage(), commission_model=ZeroCommission())
        order = Order(
            symbol="TEST", side=OrderSide.SELL, quantity=50,
            order_type=OrderType.LIMIT, limit_price=104.0,
            time_in_force=TimeInForce.GTC,
        )
        broker.submit_order(order)
        bar = _make_bar(open_=100.0, high=105.0, low=95.0)
        fills = broker.process_bar(bar, current_dt=datetime(2020, 1, 1))
        assert len(fills) == 1
        assert fills[0].price == 104.0

    def test_get_pending_orders(self):
        """get_pending_orders returns all pending orders."""
        broker = BrokerSim()
        o1 = Order(symbol="A", side=OrderSide.BUY, quantity=10, time_in_force=TimeInForce.GTC)
        o2 = Order(symbol="B", side=OrderSide.BUY, quantity=20, time_in_force=TimeInForce.GTC)
        broker.submit_order(o1)
        broker.submit_order(o2)
        assert len(broker.get_pending_orders()) == 2

    def test_partial_fill_then_complete(self):
        """Multi-bar partial fill eventually completes the order."""
        broker = BrokerSim(
            slippage_model=NoSlippage(),
            commission_model=ZeroCommission(),
            liquidity_cap_pct_adv=0.10,
        )
        # Volume=100, cap=0.10 -> max 10 per bar; order quantity=25
        order = Order(
            symbol="TEST", side=OrderSide.BUY, quantity=25,
            order_type=OrderType.MARKET, time_in_force=TimeInForce.GTC,
        )
        broker.submit_order(order)
        bar = _make_bar(volume=100)

        fill1 = broker.process_bar(bar, current_dt=datetime(2020, 1, 1))
        assert len(fill1) == 1
        assert fill1[0].quantity == 10
        assert order.status == OrderStatus.PARTIALLY_FILLED

        fill2 = broker.process_bar(bar, current_dt=datetime(2020, 1, 2))
        assert len(fill2) == 1
        assert fill2[0].quantity == 10

        fill3 = broker.process_bar(bar, current_dt=datetime(2020, 1, 3))
        assert len(fill3) == 1
        assert fill3[0].quantity == 5
        assert order.status == OrderStatus.FILLED


# ---------------------------------------------------------------------------
# Slippage edge cases
# ---------------------------------------------------------------------------

class TestSlippageEdgeCases:
    def test_fixed_bps_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            FixedBpsSlippage(bps=-1)

    def test_volume_dep_zero_volume(self):
        """Volume=0 uses participation=1.0."""
        model = VolumeDepSlippage(base_bps=5.0, volume_impact=100.0)
        order = Order(symbol="TEST", side=OrderSide.BUY, quantity=100)
        adjusted = model.apply(order, 100.0, volume=0.0)
        assert adjusted > 100.0

    def test_volume_dep_sell(self):
        """Sell side gets lower price."""
        model = VolumeDepSlippage(base_bps=5.0, volume_impact=100.0)
        order = Order(symbol="TEST", side=OrderSide.SELL, quantity=100)
        adjusted = model.apply(order, 100.0, volume=100_000.0)
        assert adjusted < 100.0

    def test_volume_dep_negative_base_bps(self):
        with pytest.raises(ValueError, match="non-negative"):
            VolumeDepSlippage(base_bps=-1, volume_impact=10)

    def test_volume_dep_negative_impact(self):
        with pytest.raises(ValueError, match="non-negative"):
            VolumeDepSlippage(base_bps=5, volume_impact=-10)


# ---------------------------------------------------------------------------
# Commission edge cases
# ---------------------------------------------------------------------------

class TestCommissionEdgeCases:
    def test_per_share_negative_rate(self):
        with pytest.raises(ValueError, match="non-negative"):
            PerShareCommission(rate=-0.01)

    def test_per_share_negative_min(self):
        with pytest.raises(ValueError, match="non-negative"):
            PerShareCommission(rate=0.01, min_commission=-1)

    def test_percent_negative_rate(self):
        with pytest.raises(ValueError, match="non-negative"):
            PercentCommission(rate=-0.01)

    def test_composite_commission(self):
        from ats_research.exec.commission import CompositeCommission
        c1 = PerShareCommission(rate=0.01)
        c2 = PercentCommission(rate=0.001)
        composite = CompositeCommission([c1, c2])
        result = composite.calculate(100, 50.0)
        # per_share: 0.01*100=1.0; pct: 0.001*100*50=5.0; total=6.0
        assert abs(result - 6.0) < 1e-8

    def test_composite_commission_empty(self):
        from ats_research.exec.commission import CompositeCommission
        with pytest.raises(ValueError, match="must not be empty"):
            CompositeCommission([])
