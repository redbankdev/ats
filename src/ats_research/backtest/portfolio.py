"""Portfolio state tracking: equity, cash, positions, PnL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class PortfolioState:
    """Snapshot of portfolio at a point in time."""

    timestamp: pd.Timestamp
    cash: float
    equity: float
    positions: dict[str, float]  # symbol -> quantity
    position_values: dict[str, float]  # symbol -> market value
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float


class Portfolio:
    """Tracks portfolio state through time.

    Uses weighted-average cost basis for realized PnL calculation.

    Parameters
    ----------
    initial_cash:
        Starting cash balance.  Defaults to ``100_000.0``.
    """

    def __init__(self, initial_cash: float = 100_000.0) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, float] = {}  # symbol -> quantity
        self.avg_entry_prices: dict[str, float] = {}  # symbol -> weighted avg cost
        self.realized_pnl = 0.0
        self._daily_start_equity = initial_cash
        self._daily_pnl = 0.0
        self._history: list[PortfolioState] = []

    # ------------------------------------------------------------------
    # Order fills
    # ------------------------------------------------------------------

    def update_fill(
        self,
        symbol: str,
        quantity: float,
        price: float,
        commission: float,
        side_is_buy: bool,
    ) -> None:
        """Update positions and cash after a fill.

        For buys: cash decreases, position increases.
        For sells: cash increases, position decreases.
        Realized PnL is computed using weighted-average cost basis.

        Parameters
        ----------
        symbol:
            Instrument identifier.
        quantity:
            Number of shares filled (always positive).
        price:
            Fill price per share.
        commission:
            Total commission cost for this fill.
        side_is_buy:
            ``True`` for a buy, ``False`` for a sell.
        """
        current_qty = self.positions.get(symbol, 0.0)
        current_avg = self.avg_entry_prices.get(symbol, 0.0)

        if side_is_buy:
            # Cash outflow
            self.cash -= quantity * price + commission
            new_qty = current_qty + quantity

            # If we had a short and are buying to cover/go long
            if current_qty < 0:
                # Covering short: realize PnL on the covered portion
                cover_qty = min(quantity, abs(current_qty))
                self.realized_pnl += cover_qty * (current_avg - price)

                remaining_buy = quantity - cover_qty
                if remaining_buy > 0 and new_qty != 0.0:
                    # Crossed from short to long
                    self.avg_entry_prices[symbol] = price
                elif new_qty < 0:
                    # Still short, avg cost unchanged
                    pass
                else:
                    # Exactly flat or crossed
                    self.avg_entry_prices[symbol] = 0.0
            else:
                # Adding to long or opening new long
                if new_qty != 0.0:
                    total_cost = current_avg * current_qty + price * quantity
                    self.avg_entry_prices[symbol] = total_cost / new_qty
                else:
                    self.avg_entry_prices[symbol] = 0.0
        else:
            # Sell: cash inflow
            self.cash += quantity * price - commission
            new_qty = current_qty - quantity

            if current_qty > 0:
                # Selling from long position: realize PnL
                sell_qty = min(quantity, current_qty)
                self.realized_pnl += sell_qty * (price - current_avg)

                remaining_sell = quantity - sell_qty
                if remaining_sell > 0 and new_qty != 0.0:
                    # Crossed from long to short
                    self.avg_entry_prices[symbol] = price
                elif new_qty > 0:
                    # Still long, avg cost unchanged
                    pass
                else:
                    # Exactly flat or crossed
                    self.avg_entry_prices[symbol] = 0.0
            else:
                # Adding to short or opening new short
                if new_qty != 0.0:
                    total_cost = current_avg * abs(current_qty) + price * quantity
                    self.avg_entry_prices[symbol] = total_cost / abs(new_qty)
                else:
                    self.avg_entry_prices[symbol] = 0.0

        # Commission always reduces realized PnL
        self.realized_pnl -= commission

        # Update position
        if abs(new_qty) < 1e-10:
            self.positions.pop(symbol, None)
            self.avg_entry_prices.pop(symbol, None)
        else:
            self.positions[symbol] = new_qty

    @property
    def equity(self) -> float:
        """Current equity from last mark-to-market, or initial cash."""
        if self._history:
            return self._history[-1].equity
        return self.cash

    # ------------------------------------------------------------------
    # Mark-to-market
    # ------------------------------------------------------------------

    def mark_to_market(
        self,
        timestamp: pd.Timestamp,
        prices: dict[str, float],
    ) -> None:
        """Revalue positions at current market prices and record state.

        Parameters
        ----------
        timestamp:
            Current bar timestamp.
        prices:
            Mapping of symbol to current price for every held position.
        """
        position_values: dict[str, float] = {}
        unrealized_pnl = 0.0

        for symbol, qty in self.positions.items():
            px = prices.get(symbol, 0.0)
            mkt_val = qty * px
            position_values[symbol] = mkt_val

            avg_cost = self.avg_entry_prices.get(symbol, 0.0)
            if qty > 0:
                unrealized_pnl += qty * (px - avg_cost)
            else:
                unrealized_pnl += abs(qty) * (avg_cost - px)

        equity = self.cash + sum(position_values.values())
        self._daily_pnl = equity - self._daily_start_equity

        state = PortfolioState(
            timestamp=timestamp,
            cash=self.cash,
            equity=equity,
            positions=dict(self.positions),
            position_values=position_values,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.realized_pnl,
            daily_pnl=self._daily_pnl,
        )
        self._history.append(state)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_state(self, prices: dict[str, float]) -> dict[str, Any]:
        """Return current portfolio state as dict for risk checks.

        Parameters
        ----------
        prices:
            Current prices for all held symbols.

        Returns
        -------
        dict
            Keys: ``equity``, ``cash``, ``daily_pnl``, ``positions``,
            ``prices``.
        """
        position_values = {
            sym: qty * prices.get(sym, 0.0)
            for sym, qty in self.positions.items()
        }
        equity = self.cash + sum(position_values.values())

        return {
            "equity": equity,
            "cash": self.cash,
            "daily_pnl": equity - self._daily_start_equity,
            "positions": dict(self.positions),
            "prices": dict(prices),
        }

    def new_day(self) -> None:
        """Reset daily PnL tracking at the start of a new trading day."""
        if self._history:
            self._daily_start_equity = self._history[-1].equity
        else:
            self._daily_start_equity = self.cash
        self._daily_pnl = 0.0

    # ------------------------------------------------------------------
    # History / reporting
    # ------------------------------------------------------------------

    def get_equity_curve(self) -> pd.Series:
        """Return equity over time as a Series indexed by timestamp."""
        if not self._history:
            return pd.Series(dtype=float)
        return pd.Series(
            [s.equity for s in self._history],
            index=pd.DatetimeIndex([s.timestamp for s in self._history]),
            name="equity",
        )

    def get_history_df(self) -> pd.DataFrame:
        """Return full portfolio history as DataFrame."""
        if not self._history:
            return pd.DataFrame()
        records = []
        for s in self._history:
            records.append(
                {
                    "timestamp": s.timestamp,
                    "cash": s.cash,
                    "equity": s.equity,
                    "unrealized_pnl": s.unrealized_pnl,
                    "realized_pnl": s.realized_pnl,
                    "daily_pnl": s.daily_pnl,
                    "n_positions": len(s.positions),
                }
            )
        df = pd.DataFrame(records)
        df = df.set_index("timestamp")
        return df
