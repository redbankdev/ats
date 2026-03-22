"""Simulated brokerage for backtesting execution.

:class:`BrokerSim` accepts orders, matches them against incoming price
bars, and produces :class:`~ats_research.exec.orders.Fill` records that
account for slippage, commissions, and liquidity constraints.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .commission import CommissionModel, ZeroCommission
from .orders import Fill, Order, OrderSide, OrderStatus, OrderType, TimeInForce
from .slippage import NoSlippage, SlippageModel

logger = logging.getLogger(__name__)


class BrokerSim:
    """Event-driven simulated broker.

    Parameters
    ----------
    slippage_model : SlippageModel | None
        Model used to adjust fill prices.  Defaults to :class:`NoSlippage`.
    commission_model : CommissionModel | None
        Model used to compute commission costs.  Defaults to
        :class:`ZeroCommission`.
    liquidity_cap_pct_adv : float
        Maximum fraction of bar volume that a single order may consume in
        one bar (default ``0.02`` = 2 %).
    """

    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        commission_model: Optional[CommissionModel] = None,
        liquidity_cap_pct_adv: float = 0.02,
    ) -> None:
        self.slippage_model: SlippageModel = slippage_model or NoSlippage()
        self.commission_model: CommissionModel = commission_model or ZeroCommission()
        if liquidity_cap_pct_adv <= 0:
            raise ValueError("liquidity_cap_pct_adv must be positive")
        self.liquidity_cap_pct_adv = liquidity_cap_pct_adv

        self._pending: Dict[str, Order] = {}  # order_id -> Order
        self._triggered_stops: Dict[str, Order] = {}  # stops promoted to market

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> None:
        """Add an order to the pending queue.

        Parameters
        ----------
        order : Order
            Must have status ``PENDING``.  The order is rejected if
            required fields are missing (e.g. no ``limit_price`` for a
            limit order).
        """
        if order.status != OrderStatus.PENDING:
            raise ValueError(f"Cannot submit order with status {order.status}")

        # Validate order-type-specific fields.
        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            order.status = OrderStatus.REJECTED
            logger.warning("Rejected order %s: limit order missing limit_price", order.order_id)
            return
        if order.order_type == OrderType.STOP and order.stop_price is None:
            order.status = OrderStatus.REJECTED
            logger.warning("Rejected order %s: stop order missing stop_price", order.order_id)
            return
        if order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            logger.warning("Rejected order %s: quantity must be positive", order.order_id)
            return

        self._pending[order.order_id] = order
        logger.debug("Submitted order %s: %s %s %.4f %s", order.order_id, order.side.value, order.symbol, order.quantity, order.order_type.value)

    def process_bar(self, bar: pd.Series, current_dt: datetime) -> List[Fill]:
        """Match pending orders against a single price bar.

        The bar must contain at least the columns/keys: ``open``,
        ``high``, ``low``, ``close``, ``volume``.

        Processing order:

        1. Check stop orders for trigger conditions and promote to market.
        2. Attempt to fill market orders at the bar's **open** price
           (next-bar-open convention).
        3. Attempt to fill limit orders if the bar's high/low range
           touches or crosses the limit price.
        4. Expire DAY orders at end of cycle.

        Parameters
        ----------
        bar : pd.Series
            OHLCV bar data.
        current_dt : datetime
            Simulation timestamp for the bar.

        Returns
        -------
        list[Fill]
            Fills generated during this bar.
        """
        fills: List[Fill] = []

        bar_open: float = float(bar["open"])
        bar_high: float = float(bar["high"])
        bar_low: float = float(bar["low"])
        bar_volume: float = float(bar["volume"])

        max_fill_qty = self.liquidity_cap_pct_adv * bar_volume

        # Snapshot order ids -- we may mutate _pending during iteration.
        order_ids = list(self._pending.keys())

        # Phase 1: promote triggered stop orders to market orders.
        for oid in order_ids:
            order = self._pending.get(oid)
            if order is None or order.order_type != OrderType.STOP:
                continue
            if self._stop_triggered(order, bar_high, bar_low):
                # Promote: treat as market for the remainder of this bar.
                self._triggered_stops[oid] = order
                logger.debug("Stop triggered for order %s at bar %s", oid, current_dt)

        # Phase 2: fill market orders (including freshly triggered stops).
        for oid in order_ids:
            order = self._pending.get(oid)
            if order is None:
                continue

            is_market = order.order_type == OrderType.MARKET
            is_triggered_stop = oid in self._triggered_stops

            if not (is_market or is_triggered_stop):
                continue

            fill = self._try_fill(order, bar_open, bar_volume, max_fill_qty, current_dt)
            if fill is not None:
                fills.append(fill)

            # Clean up triggered-stop tracking regardless of fill outcome.
            self._triggered_stops.pop(oid, None)

        # Phase 3: fill limit orders.
        for oid in order_ids:
            order = self._pending.get(oid)
            if order is None or order.order_type != OrderType.LIMIT:
                continue

            limit_price = order.limit_price
            assert limit_price is not None  # validated on submit

            if not self._limit_touched(order, limit_price, bar_high, bar_low):
                continue

            fill = self._try_fill(order, limit_price, bar_volume, max_fill_qty, current_dt)
            if fill is not None:
                fills.append(fill)

        # Phase 4: expire DAY orders.
        self._expire_day_orders()

        return fills

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Parameters
        ----------
        order_id : str
            Identifier of the order to cancel.

        Returns
        -------
        bool
            ``True`` if the order was found and cancelled.
        """
        order = self._pending.pop(order_id, None)
        if order is None:
            return False
        order.status = OrderStatus.CANCELLED
        self._triggered_stops.pop(order_id, None)
        logger.debug("Cancelled order %s", order_id)
        return True

    def get_pending_orders(self) -> List[Order]:
        """Return a list of all orders still in the pending queue."""
        return list(self._pending.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stop_triggered(order: Order, bar_high: float, bar_low: float) -> bool:
        """Return ``True`` if the stop price was hit during the bar.

        Buy-stop triggers when the bar high >= stop price.
        Sell-stop triggers when the bar low <= stop price.
        """
        assert order.stop_price is not None
        if order.side == OrderSide.BUY:
            return bar_high >= order.stop_price
        return bar_low <= order.stop_price

    @staticmethod
    def _limit_touched(
        order: Order, limit_price: float, bar_high: float, bar_low: float
    ) -> bool:
        """Return ``True`` if the limit price was reachable within the bar.

        Buy-limit fills when the bar low <= limit price.
        Sell-limit fills when the bar high >= limit price.
        """
        if order.side == OrderSide.BUY:
            return bar_low <= limit_price
        return bar_high >= limit_price

    def _try_fill(
        self,
        order: Order,
        raw_price: float,
        bar_volume: float,
        max_fill_qty: float,
        current_dt: datetime,
    ) -> Optional[Fill]:
        """Attempt to fill (or partially fill) an order.

        Returns a :class:`Fill` if any quantity was executed, or ``None``
        if the liquidity cap was already exhausted (``max_fill_qty <= 0``).
        """
        remaining = order.quantity - order.filled_quantity
        if remaining <= 0:
            return None

        fill_qty = min(remaining, max_fill_qty)
        if fill_qty <= 0:
            return None

        # Apply slippage.
        adjusted_price = self.slippage_model.apply(order, raw_price, bar_volume)
        slippage_per_share = abs(adjusted_price - raw_price)
        slippage_cost = slippage_per_share * fill_qty

        # Apply commission.
        commission = self.commission_model.calculate(fill_qty, adjusted_price)

        # Update order state.
        prev_filled = order.filled_quantity
        order.filled_quantity += fill_qty
        # Maintain volume-weighted average fill price.
        if prev_filled == 0:
            order.filled_price = adjusted_price
        else:
            order.filled_price = (
                (order.filled_price * prev_filled + adjusted_price * fill_qty)
                / order.filled_quantity
            )
        order.commission += commission
        order.slippage += slippage_cost

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            self._pending.pop(order.order_id, None)
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_qty,
            price=adjusted_price,
            commission=commission,
            slippage=slippage_cost,
            timestamp=current_dt,
        )

        logger.debug(
            "Fill: %s %s %.4f @ %.4f (slip=%.4f, comm=%.4f)",
            order.side.value,
            order.symbol,
            fill_qty,
            adjusted_price,
            slippage_cost,
            commission,
        )
        return fill

    def _expire_day_orders(self) -> None:
        """Cancel all pending DAY orders at end of bar cycle."""
        expired_ids = [
            oid
            for oid, order in self._pending.items()
            if order.time_in_force == TimeInForce.DAY
        ]
        for oid in expired_ids:
            order = self._pending.pop(oid)
            order.status = OrderStatus.CANCELLED
            self._triggered_stops.pop(oid, None)
            logger.debug("Expired DAY order %s", oid)
