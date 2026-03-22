"""Simple Moving Average crossover strategy.

A reference implementation of :class:`~ats_research.strategy.base.Strategy`
that goes long when a fast SMA crosses above a slow SMA, and exits (goes
flat) on the opposite cross.  Stop-loss and take-profit levels are derived
from the Average True Range (ATR).

Usage::

    from ats_research.strategy.sma_crossover import SMACrossover

    strategy = SMACrossover(params={"fast": 10, "slow": 30, "atr_stop": 1.5})
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from ats_research.data.features import atr, sma
from ats_research.strategy.base import Signal, Strategy


class SMACrossover(Strategy):
    """SMA crossover strategy with ATR-based stop and take-profit.

    Parameters (via ``params`` dict)
    --------------------------------
    fast : int, default 20
        Look-back window for the fast simple moving average.
    slow : int, default 50
        Look-back window for the slow simple moving average.
    atr_stop : float, default 2.0
        Number of ATR multiples below entry price for the stop-loss.
    tp_mult : float, default 3.0
        Take-profit distance expressed as a multiple of the stop distance
        (i.e. ``tp_mult * atr_stop * ATR``).
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)

        # Resolve parameters with defaults
        self._fast: int = int(self.params.get("fast", 20))
        self._slow: int = int(self.params.get("slow", 50))
        self._atr_stop: float = float(self.params.get("atr_stop", 2.0))
        self._tp_mult: float = float(self.params.get("tp_mult", 3.0))

        # Precomputed indicator series (populated in initialize())
        self._fast_sma: pd.Series | None = None
        self._slow_sma: pd.Series | None = None
        self._atr: pd.Series | None = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def initialize(self, data: pd.DataFrame) -> None:
        """Precompute fast SMA, slow SMA, and ATR over the full dataset.

        Parameters
        ----------
        data:
            DataFrame with at least ``close``, ``high``, and ``low``
            columns.
        """
        self._fast_sma = sma(data["close"], window=self._fast)
        self._slow_sma = sma(data["close"], window=self._slow)
        self._atr = atr(data, window=self._slow)  # use slow window for ATR smoothing
        super().initialize(data)

    def on_bar(
        self,
        timestamp: pd.Timestamp,
        bar: pd.Series,
        history: pd.DataFrame,
    ) -> Optional[Signal]:
        """Generate a signal on each SMA crossover.

        Rules
        -----
        * **Long entry** -- fast SMA crosses *above* slow SMA.
          ``target_position = 1.0`` with stop and take-profit metadata.
        * **Exit** -- fast SMA crosses *below* slow SMA.
          ``target_position = 0.0``.
        * If there is insufficient history (fewer than ``slow`` bars) or
          the indicator values are not yet available, return ``None``.

        Parameters
        ----------
        timestamp:
            Current bar timestamp.
        bar:
            Current OHLCV bar.
        history:
            All bars up to and including the current bar.

        Returns
        -------
        Signal or None
        """
        if self._fast_sma is None or self._slow_sma is None or self._atr is None:
            raise RuntimeError(
                "Strategy has not been initialized. Call initialize() before on_bar()."
            )

        # Need at least `slow` bars of history before generating signals.
        if len(history) <= self._slow:
            return None

        # Current and previous indicator values
        fast_now = self._fast_sma.loc[timestamp]
        slow_now = self._slow_sma.loc[timestamp]

        prev_ts = history.index[-2]
        fast_prev = self._fast_sma.loc[prev_ts]
        slow_prev = self._slow_sma.loc[prev_ts]

        # Guard against NaN values in the warm-up period
        if pd.isna(fast_now) or pd.isna(slow_now) or pd.isna(fast_prev) or pd.isna(slow_prev):
            return None

        atr_now = self._atr.loc[timestamp]
        if pd.isna(atr_now):
            return None

        symbol: str = bar.get("symbol", "UNKNOWN")
        close: float = float(bar["close"])

        # Detect crossovers
        crossed_above = (fast_prev <= slow_prev) and (fast_now > slow_now)
        crossed_below = (fast_prev >= slow_prev) and (fast_now < slow_now)

        if crossed_above:
            stop_distance = self._atr_stop * atr_now
            stop_price = close - stop_distance
            tp_price = close + self._tp_mult * stop_distance

            return Signal(
                symbol=symbol,
                target_position=1.0,
                confidence=1.0,
                metadata={
                    "entry_price": close,
                    "stop_price": round(stop_price, 6),
                    "tp_price": round(tp_price, 6),
                    "atr": round(atr_now, 6),
                },
            )

        if crossed_below:
            return Signal(
                symbol=symbol,
                target_position=0.0,
                confidence=1.0,
                metadata={"exit_price": close},
            )

        # No crossover -- no signal
        return None
