"""Risk rule enforcement for order validation and daily loss management.

Each :class:`RiskRule` inspects a proposed order against the current
portfolio state and returns whether the order is allowed.  The
:class:`RiskManager` aggregates multiple rules and provides a single
entry point for order checking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RiskRule(ABC):
    """Abstract base class for risk rules.

    Subclasses implement :meth:`check` which decides whether an order
    should be allowed or blocked.
    """

    @abstractmethod
    def check(self, order: Any, portfolio_state: dict) -> tuple[bool, str]:
        """Evaluate *order* against *portfolio_state*.

        Parameters
        ----------
        order:
            An order object.  At minimum it must expose ``symbol``
            (str) and ``quantity`` (float, signed) attributes.  It may
            also carry ``stop_price`` for loss-cap calculations.
        portfolio_state:
            Dictionary with at least the following keys:

            * ``equity`` -- total portfolio equity (float).
            * ``cash`` -- available cash (float).
            * ``daily_pnl`` -- realised + unrealised PnL today (float).
            * ``positions`` -- ``dict[str, float]`` mapping symbol to
              current quantity.
            * ``prices`` -- ``dict[str, float]`` mapping symbol to
              latest price.

        Returns
        -------
        tuple[bool, str]
            ``(True, "")`` if the order is allowed, or
            ``(False, reason)`` if it is blocked.
        """
        ...


@dataclass
class MaxPositionRule(RiskRule):
    """Reject an order if the resulting position would exceed a
    percentage of equity.

    Parameters
    ----------
    max_pct:
        Maximum position value as a fraction of equity.  Defaults to
        ``0.10`` (10 %).
    """

    max_pct: float = 0.10

    def check(self, order: Any, portfolio_state: dict) -> tuple[bool, str]:
        equity = portfolio_state.get("equity", 0.0)
        if equity <= 0:
            return False, "Equity is zero or negative"

        symbol: str = getattr(order, "symbol", "")
        quantity: float = getattr(order, "quantity", 0.0)

        positions: dict[str, float] = portfolio_state.get("positions", {})
        prices: dict[str, float] = portfolio_state.get("prices", {})

        current_qty = positions.get(symbol, 0.0)
        price = prices.get(symbol, 0.0)

        if price <= 0:
            return False, f"No valid price for {symbol}"

        new_qty = current_qty + quantity
        position_value = abs(new_qty) * price
        position_pct = position_value / equity

        if position_pct > self.max_pct:
            return (
                False,
                f"Position in {symbol} would be {position_pct:.2%} of equity, "
                f"exceeding limit of {self.max_pct:.2%}",
            )

        return True, ""


@dataclass
class PerTradeLossCapRule(RiskRule):
    """Reject an order if the potential loss exceeds a percentage of
    equity.

    When the order carries a ``stop_price``, the loss is computed as the
    distance from the current price to the stop.  Otherwise the worst
    case is assumed to be the full position value.

    Parameters
    ----------
    max_loss_pct:
        Maximum potential loss as a fraction of equity.  Defaults to
        ``0.01`` (1 %).
    """

    max_loss_pct: float = 0.01

    def check(self, order: Any, portfolio_state: dict) -> tuple[bool, str]:
        equity = portfolio_state.get("equity", 0.0)
        if equity <= 0:
            return False, "Equity is zero or negative"

        symbol: str = getattr(order, "symbol", "")
        quantity: float = getattr(order, "quantity", 0.0)
        stop_price: float | None = getattr(order, "stop_price", None)

        prices: dict[str, float] = portfolio_state.get("prices", {})
        price = prices.get(symbol, 0.0)

        if price <= 0:
            return False, f"No valid price for {symbol}"

        abs_qty = abs(quantity)

        if stop_price is not None and stop_price > 0:
            loss_per_unit = abs(price - stop_price)
        else:
            # Worst case: full position value.
            loss_per_unit = price

        potential_loss = abs_qty * loss_per_unit
        loss_pct = potential_loss / equity

        if loss_pct > self.max_loss_pct:
            return (
                False,
                f"Potential loss on {symbol} trade is {loss_pct:.2%} of equity, "
                f"exceeding limit of {self.max_loss_pct:.2%}",
            )

        return True, ""


@dataclass
class DailyLossHaltRule(RiskRule):
    """Halt all trading when the daily PnL loss exceeds a threshold.

    Once triggered the halt persists until :meth:`RiskManager.reset_daily`
    is called (typically at the start of the next trading day).

    Parameters
    ----------
    max_daily_loss_pct:
        Maximum daily loss as a fraction of equity.  Defaults to
        ``0.05`` (5 %).
    """

    max_daily_loss_pct: float = 0.05
    _halted: bool = field(default=False, init=False, repr=False)

    def check(self, order: Any, portfolio_state: dict) -> tuple[bool, str]:
        if self._halted:
            return False, "Trading halted: daily loss limit was breached"

        equity = portfolio_state.get("equity", 0.0)
        daily_pnl = portfolio_state.get("daily_pnl", 0.0)

        if equity <= 0:
            self._halted = True
            return False, "Equity is zero or negative; trading halted"

        loss_pct = -daily_pnl / equity  # daily_pnl is negative when losing

        if loss_pct >= self.max_daily_loss_pct:
            self._halted = True
            return (
                False,
                f"Daily loss is {loss_pct:.2%} of equity, "
                f"exceeding limit of {self.max_daily_loss_pct:.2%}; trading halted",
            )

        return True, ""

    def reset(self) -> None:
        """Reset the halt flag (called at start of a new trading day)."""
        self._halted = False


class RiskManager:
    """Aggregate risk-rule checker.

    Runs a list of :class:`RiskRule` instances against each proposed
    order and rejects the order on the first failure.

    Parameters
    ----------
    rules:
        Ordered list of risk rules to evaluate.

    Examples
    --------
    >>> manager = RiskManager([
    ...     MaxPositionRule(max_pct=0.10),
    ...     PerTradeLossCapRule(max_loss_pct=0.01),
    ...     DailyLossHaltRule(max_daily_loss_pct=0.05),
    ... ])
    >>> allowed, reason = manager.check_order(order, state)
    """

    def __init__(self, rules: list[RiskRule]) -> None:
        self.rules: list[RiskRule] = rules

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_order(self, order: Any, portfolio_state: dict) -> tuple[bool, str]:
        """Run all rules against *order*.

        Parameters
        ----------
        order:
            The proposed order (must expose ``symbol`` and ``quantity``).
        portfolio_state:
            Current portfolio snapshot.  Expected keys: ``equity``,
            ``cash``, ``daily_pnl``, ``positions``, ``prices``.

        Returns
        -------
        tuple[bool, str]
            ``(True, "")`` if every rule passes, or ``(False, reason)``
            on the first failure.
        """
        for rule in self.rules:
            allowed, reason = rule.check(order, portfolio_state)
            if not allowed:
                return False, reason
        return True, ""

    @property
    def is_halted(self) -> bool:
        """Return ``True`` if any :class:`DailyLossHaltRule` has been
        triggered."""
        return any(
            getattr(rule, "_halted", False)
            for rule in self.rules
            if isinstance(rule, DailyLossHaltRule)
        )

    def reset_daily(self) -> None:
        """Reset daily tracking state on all rules.

        Call this at the beginning of each trading day to clear halt
        flags and any other daily accumulators.
        """
        for rule in self.rules:
            if isinstance(rule, DailyLossHaltRule):
                rule.reset()
