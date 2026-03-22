"""
Common technical indicators implemented as vectorised pandas operations.

All functions return a :class:`~pandas.Series` of the same length as the
input.  Leading values that cannot be computed (due to insufficient
look-back) are left as ``NaN``.

Usage::

    from ats_research.data.features import sma, ema, atr, rsi, rolling_volatility

    df["sma_20"] = sma(df["close"], window=20)
    df["rsi_14"] = rsi(df["close"], window=14)
    df["atr_14"] = atr(df, window=14)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def sma(series: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average.

    Parameters
    ----------
    series:
        Price (or any numeric) series.
    window:
        Look-back window in number of bars.

    Returns
    -------
    pd.Series
        SMA values; the first ``window - 1`` entries are ``NaN``.
    """
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential Moving Average.

    Uses the *span* convention: ``alpha = 2 / (span + 1)``.

    Parameters
    ----------
    series:
        Price (or any numeric) series.
    span:
        EMA span (equivalent to the "period" in most charting packages).

    Returns
    -------
    pd.Series
        EMA values.  The first value is seeded from the available data;
        earlier entries may still be ``NaN`` if the source contains them.
    """
    return series.ewm(span=span, adjust=False).mean()


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range.

    True Range for bar *i* is defined as::

        TR_i = max(high_i - low_i,
                   |high_i - close_{i-1}|,
                   |low_i  - close_{i-1}|)

    ATR is the exponential (Wilder) moving average of TR over *window*
    periods.

    Parameters
    ----------
    df:
        DataFrame containing ``high``, ``low``, and ``close`` columns.
    window:
        Smoothing window (default 14).

    Returns
    -------
    pd.Series
        ATR values.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    true_range.name = "true_range"

    # Wilder smoothing is equivalent to EWM with alpha = 1/window
    return true_range.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()


def rolling_volatility(
    series: pd.Series,
    window: int = 20,
    annualisation_factor: float = 252.0,
) -> pd.Series:
    """
    Annualised rolling volatility of log returns.

    Parameters
    ----------
    series:
        Price series (typically close prices).
    window:
        Rolling window for the standard-deviation calculation.
    annualisation_factor:
        Scaling factor.  Use ``252`` for daily bars, ``52`` for weekly,
        ``12`` for monthly, etc.

    Returns
    -------
    pd.Series
        Annualised volatility.  The first ``window`` entries are ``NaN``
        (``window - 1`` from returns + 1 from rolling).
    """
    log_returns = np.log(series / series.shift(1))
    return log_returns.rolling(window=window, min_periods=window).std() * np.sqrt(
        annualisation_factor
    )


# ---------------------------------------------------------------------------
# Momentum / oscillators
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder's smoothing).

    RSI = 100 - 100 / (1 + RS)

    where RS = avg_gain / avg_loss over *window* periods using exponential
    (Wilder) smoothing.

    Parameters
    ----------
    series:
        Price series (typically close prices).
    window:
        Look-back window (default 14).

    Returns
    -------
    pd.Series
        RSI values in the range ``[0, 100]``.  The first ``window``
        entries are ``NaN``.
    """
    delta = series.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder smoothing: EWM with alpha = 1/window
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss

    # When avg_loss == 0, RS is infinite → RSI = 100
    result = 100.0 - (100.0 / (1.0 + rs))
    result = result.where(avg_loss > 0, 100.0)

    return result
