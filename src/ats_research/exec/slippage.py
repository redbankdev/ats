"""Slippage models for execution simulation.

Each model takes a raw price and returns an adjusted price that reflects
market-impact or execution-quality assumptions.
"""

from abc import ABC, abstractmethod

from .orders import Order, OrderSide


class SlippageModel(ABC):
    """Abstract base class for slippage models.

    Subclasses implement :meth:`apply` to return a price that has been
    adjusted for estimated slippage.
    """

    @abstractmethod
    def apply(self, order: Order, price: float, volume: float) -> float:
        """Return the slippage-adjusted execution price.

        Parameters
        ----------
        order : Order
            The order being executed.
        price : float
            Raw (unadjusted) execution price.
        volume : float
            Bar volume available for liquidity reference.

        Returns
        -------
        float
            Adjusted execution price.
        """


class NoSlippage(SlippageModel):
    """No slippage -- returns the price unchanged."""

    def apply(self, order: Order, price: float, volume: float) -> float:
        """Return *price* with no adjustment."""
        return price


class FixedBpsSlippage(SlippageModel):
    """Apply a fixed basis-point slippage to every execution.

    Buys pay more; sells receive less.

    Parameters
    ----------
    bps : float
        Slippage in basis points (1 bp = 0.01 %).
    """

    def __init__(self, bps: float) -> None:
        if bps < 0:
            raise ValueError("bps must be non-negative")
        self.bps = bps

    def apply(self, order: Order, price: float, volume: float) -> float:
        """Return *price* shifted by a fixed number of basis points."""
        multiplier = self.bps / 10_000.0
        if order.side == OrderSide.BUY:
            return price * (1.0 + multiplier)
        return price * (1.0 - multiplier)


class VolumeDepSlippage(SlippageModel):
    """Slippage that increases with order size relative to bar volume.

    The effective basis-point cost is::

        effective_bps = base_bps + volume_impact * (order.quantity / volume)

    Parameters
    ----------
    base_bps : float
        Minimum slippage in basis points regardless of size.
    volume_impact : float
        Scaling factor for the participation-rate component (in basis
        points).  For example, ``volume_impact=100`` means an order that
        is 100 % of bar volume adds 100 bps of extra slippage on top of
        *base_bps*.
    """

    def __init__(self, base_bps: float, volume_impact: float) -> None:
        if base_bps < 0:
            raise ValueError("base_bps must be non-negative")
        if volume_impact < 0:
            raise ValueError("volume_impact must be non-negative")
        self.base_bps = base_bps
        self.volume_impact = volume_impact

    def apply(self, order: Order, price: float, volume: float) -> float:
        """Return *price* adjusted for volume-dependent slippage."""
        if volume <= 0:
            participation = 1.0
        else:
            participation = order.quantity / volume

        effective_bps = self.base_bps + self.volume_impact * participation
        multiplier = effective_bps / 10_000.0

        if order.side == OrderSide.BUY:
            return price * (1.0 + multiplier)
        return price * (1.0 - multiplier)
