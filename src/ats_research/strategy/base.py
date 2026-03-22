"""Abstract strategy interface for the backtesting framework.

Every concrete strategy must subclass :class:`Strategy` and implement
:meth:`on_bar`.  The backtesting engine calls :meth:`initialize` once before
the run begins, then :meth:`on_bar` for every bar in the dataset.  Strategies
communicate desired positions back to the engine via :class:`Signal` objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class Signal:
    """Signal produced by a strategy.

    Attributes
    ----------
    symbol:
        Instrument identifier the signal refers to.
    target_position:
        Desired position in units.  Positive values indicate a long
        position, negative values a short position, and zero means flat.
    confidence:
        Confidence level in the range ``[0, 1]``.  Strategies that do not
        model confidence can leave this at the default of ``1.0``.
    metadata:
        Arbitrary extra information attached to the signal (e.g.
        ``stop_price``, ``tp_price``).
    """

    symbol: str
    target_position: float  # positive=long, negative=short, 0=flat
    confidence: float = 1.0  # 0-1 confidence level
    metadata: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """Abstract base class for all strategies.

    Strategies receive bar data and produce :class:`Signal` instances
    indicating desired positions.  The backtesting engine translates signals
    into orders.

    Parameters
    ----------
    params:
        Free-form dictionary of strategy parameters (e.g. look-back
        windows, thresholds).  Accessible via ``self.params``.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = params or {}
        self._is_initialized: bool = False

    # ------------------------------------------------------------------
    # Core hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def on_bar(
        self,
        timestamp: pd.Timestamp,
        bar: pd.Series,
        history: pd.DataFrame,
    ) -> Optional[Signal]:
        """Called for each new bar.

        Parameters
        ----------
        timestamp:
            The timestamp of the current bar.
        bar:
            The current OHLCV bar as a :class:`~pandas.Series`.
        history:
            All bars up to and including the current bar.

        Returns
        -------
        Signal or None
            Return a :class:`Signal` to communicate a desired position
            change, or ``None`` to indicate no action.
        """
        ...

    def on_fill(self, fill: Any) -> None:
        """Called when an order is filled.

        Override this in strategies that need to track fill information
        (e.g. to adjust trailing stops or compute running P&L).

        Parameters
        ----------
        fill:
            Fill information provided by the execution engine.
        """

    def initialize(self, data: pd.DataFrame) -> None:
        """Called once before the backtest starts.

        Use this hook to precompute indicators, validate parameters, or
        perform any other one-time setup.

        Parameters
        ----------
        data:
            The full bar dataset that will be iterated over during the
            backtest.
        """
        self._is_initialized = True

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable strategy name (defaults to the class name)."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}(params={self.params})"
