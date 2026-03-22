"""Position sizing methods for the risk management framework.

Each sizer takes a signal, current equity, instrument price, and volatility
estimate and returns the number of whole units (shares/contracts) to trade.
All sizers enforce a maximum position size as a percentage of equity.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PositionSizer(ABC):
    """Abstract base class for position sizers.

    Subclasses must implement :meth:`_raw_size` which returns the
    un-capped, un-rounded number of units.  The public :meth:`size`
    method applies the ``max_position_pct`` cap and rounds to whole
    shares.

    Parameters
    ----------
    max_position_pct:
        Maximum position value as a fraction of equity.  Defaults to
        ``0.10`` (10 %).
    """

    max_position_pct: float = 0.10

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def size(
        self,
        signal: Any,
        equity: float,
        price: float,
        volatility: float,
    ) -> float:
        """Return the number of whole units to trade.

        Parameters
        ----------
        signal:
            A :class:`~ats_research.strategy.base.Signal` (or any object
            with ``target_position`` and ``metadata`` attributes).
        equity:
            Current portfolio equity.
        price:
            Current instrument price.
        volatility:
            Annualized volatility estimate for the instrument (as a
            decimal, e.g. ``0.20`` for 20 %).

        Returns
        -------
        float
            Number of whole units (always an integer value stored as
            ``float``).  Returns ``0`` when inputs are invalid.
        """
        if equity <= 0 or price <= 0:
            return 0.0

        raw = self._raw_size(signal, equity, price, volatility)

        # Enforce the maximum-position-pct cap.
        max_units = (equity * self.max_position_pct) / price
        capped = min(abs(raw), max_units)

        # Preserve direction from the raw calculation.
        directed = math.copysign(capped, raw) if raw != 0 else 0.0

        # Round towards zero to whole shares.
        return float(int(directed))

    # ------------------------------------------------------------------
    # Hook for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _raw_size(
        self,
        signal: Any,
        equity: float,
        price: float,
        volatility: float,
    ) -> float:
        """Return the un-capped, un-rounded number of units.

        Implementations may return a negative value to indicate a short
        position.
        """
        ...


@dataclass
class FixedFractionSizer(PositionSizer):
    """Risk a fixed fraction of equity per trade.

    The position size is determined so that if the trade hits its stop
    (or, when no stop is available, moves by one daily-volatility unit)
    the loss equals ``fraction * equity``.

    Parameters
    ----------
    fraction:
        Fraction of equity to risk per trade.  Defaults to ``0.02``
        (2 %).
    """

    fraction: float = 0.02

    def _raw_size(
        self,
        signal: Any,
        equity: float,
        price: float,
        volatility: float,
    ) -> float:
        metadata = getattr(signal, "metadata", {}) or {}
        stop_price = metadata.get("stop_price")

        if stop_price is not None:
            stop_distance = abs(price - stop_price)
        else:
            # Fall back to one daily standard-deviation move.
            daily_vol = volatility / math.sqrt(252) if volatility > 0 else 0.0
            stop_distance = price * daily_vol

        if stop_distance <= 0:
            return 0.0

        dollar_risk = equity * self.fraction
        units = dollar_risk / stop_distance

        # Preserve signal direction.
        target = getattr(signal, "target_position", 0.0)
        if target < 0:
            units = -units

        return units


@dataclass
class VolatilityScaledSizer(PositionSizer):
    """Target a constant annualized portfolio volatility.

    Position size is scaled inversely with the instrument's volatility
    so that every instrument contributes roughly the same risk.

    .. math::

        \\text{units} = \\frac{\\text{equity} \\times \\text{vol\\_target}}
                              {\\text{price} \\times \\text{daily\\_vol}
                               \\times \\sqrt{252}}

    Parameters
    ----------
    vol_target:
        Target annualized portfolio volatility.  Defaults to ``0.10``
        (10 %).
    """

    vol_target: float = 0.10

    def _raw_size(
        self,
        signal: Any,
        equity: float,
        price: float,
        volatility: float,
    ) -> float:
        daily_vol = volatility / math.sqrt(252) if volatility > 0 else 0.0

        if daily_vol <= 0:
            return 0.0

        units = (equity * self.vol_target) / (price * daily_vol * math.sqrt(252))

        # Preserve signal direction.
        target = getattr(signal, "target_position", 0.0)
        if target < 0:
            units = -units

        return units


@dataclass
class KellyFractionSizer(PositionSizer):
    """Kelly criterion position sizing.

    The full Kelly fraction is:

    .. math::

        f^* = \\frac{p}{1} \\;-\\; \\frac{1 - p}{\\text{avg\\_win} / \\text{avg\\_loss}}

    where *p* is the win rate.  By default a ``fraction`` of ``0.5``
    (half-Kelly) is applied to reduce drawdowns.

    Parameters
    ----------
    win_rate:
        Historical win rate in ``[0, 1]``.
    avg_win:
        Average winning trade return (positive value).
    avg_loss:
        Average losing trade return (positive value representing the
        magnitude of the loss).
    fraction:
        Fraction of the full Kelly to use.  Defaults to ``0.5``
        (half-Kelly).
    """

    win_rate: float = 0.50
    avg_win: float = 1.0
    avg_loss: float = 1.0
    fraction: float = 0.5

    def _raw_size(
        self,
        signal: Any,
        equity: float,
        price: float,
        volatility: float,
    ) -> float:
        if self.avg_loss <= 0 or self.avg_win <= 0:
            return 0.0

        win_loss_ratio = self.avg_win / self.avg_loss
        kelly_f = self.win_rate - (1.0 - self.win_rate) / win_loss_ratio

        # A negative Kelly fraction means negative edge -- don't trade.
        if kelly_f <= 0:
            return 0.0

        adjusted = kelly_f * self.fraction
        dollar_amount = equity * adjusted
        units = dollar_amount / price

        # Preserve signal direction.
        target = getattr(signal, "target_position", 0.0)
        if target < 0:
            units = -units

        return units
