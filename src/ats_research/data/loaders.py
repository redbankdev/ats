"""
Data loaders for OHLCV market data.

Supports CSV and Parquet files as well as synthetic demo data generation
via geometric Brownian motion.

Usage::

    from ats_research.data.loaders import load_ohlcv, generate_demo_data

    df = load_ohlcv("data/prices.csv", freq="1D", tz="UTC")
    demo = generate_demo_data(n_bars=500, freq="1h", seed=0)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ats_research.data.validators import validate_ohlcv
from ats_research.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ohlcv(
    path: str | Path,
    freq: str = "1D",
    tz: str = "UTC",
) -> pd.DataFrame:
    """
    Load an OHLCV file (CSV or Parquet) and return a validated DataFrame.

    The file must contain a ``timestamp`` column (or use the first column as
    timestamps) along with ``open``, ``high``, ``low``, ``close``, and
    ``volume`` columns.

    Parameters
    ----------
    path:
        Path to a ``.csv`` or ``.parquet`` file.
    freq:
        Frequency hint (e.g. ``"1D"``, ``"1h"``).  Currently stored as
        metadata but not enforced.
    tz:
        IANA timezone string used to localise timezone-naive timestamps.

    Returns
    -------
    pd.DataFrame
        Validated OHLCV DataFrame with a timezone-aware
        :class:`~pandas.DatetimeIndex`.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file extension is unsupported or the data fails validation.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = path.suffix.lower()
    log.info("Loading OHLCV data from %s (freq=%s, tz=%s)", path, freq, tz)

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Expected .csv or .parquet."
        )

    # --- Normalise column names -------------------------------------------
    df.columns = df.columns.str.lower().str.strip()

    # --- Parse timestamp --------------------------------------------------
    ts_col = _find_timestamp_column(df)
    df[ts_col] = pd.to_datetime(df[ts_col], utc=False)

    # Localise to requested timezone
    if df[ts_col].dt.tz is None:
        df[ts_col] = df[ts_col].dt.tz_localize(tz)
    else:
        df[ts_col] = df[ts_col].dt.tz_convert(tz)

    df = df.set_index(ts_col)
    df.index.name = "timestamp"

    # Attach frequency as metadata
    df.attrs["freq"] = freq

    # --- Validate ---------------------------------------------------------
    df = validate_ohlcv(df)

    log.info("Loaded %d bars from %s.", len(df), path.name)
    return df


def generate_demo_data(
    n_bars: int = 1000,
    freq: str = "1D",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data using geometric Brownian motion.

    Parameters
    ----------
    n_bars:
        Number of bars (rows) to generate.
    freq:
        Pandas frequency string for the timestamp index.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``open``, ``high``, ``low``, ``close``,
        ``volume`` and a timezone-aware UTC :class:`~pandas.DatetimeIndex`.
    """
    rng = np.random.default_rng(seed)

    # --- GBM parameters ---------------------------------------------------
    mu = 0.0002  # daily drift
    sigma = 0.015  # daily volatility
    s0 = 100.0  # starting price

    # Generate log returns and cumulative price path
    log_returns = (mu - 0.5 * sigma**2) + sigma * rng.standard_normal(n_bars)
    cum_returns = np.cumsum(log_returns)
    close_prices = s0 * np.exp(cum_returns)

    # --- Derive OHLC from close -------------------------------------------
    # open ≈ previous close (first open = s0)
    open_prices = np.empty(n_bars)
    open_prices[0] = s0
    open_prices[1:] = close_prices[:-1]

    # Intrabar noise for high/low
    intrabar_range = sigma * close_prices * rng.uniform(0.2, 1.5, size=n_bars)
    bar_max = np.maximum(open_prices, close_prices)
    bar_min = np.minimum(open_prices, close_prices)

    high_prices = bar_max + rng.uniform(0.0, 1.0, size=n_bars) * intrabar_range
    low_prices = bar_min - rng.uniform(0.0, 1.0, size=n_bars) * intrabar_range

    # Ensure low > 0
    low_prices = np.maximum(low_prices, bar_min * 0.995)

    # --- Volume (log-normal) ----------------------------------------------
    base_volume = 1_000_000
    volume = rng.lognormal(mean=np.log(base_volume), sigma=0.4, size=n_bars)
    volume = np.round(volume).astype(np.int64)

    # --- Build DataFrame --------------------------------------------------
    timestamps = pd.date_range(
        start="2020-01-01",
        periods=n_bars,
        freq=freq,
        tz="UTC",
    )

    df = pd.DataFrame(
        {
            "open": np.round(open_prices, 4),
            "high": np.round(high_prices, 4),
            "low": np.round(low_prices, 4),
            "close": np.round(close_prices, 4),
            "volume": volume,
        },
        index=timestamps,
    )
    df.index.name = "timestamp"
    df.attrs["freq"] = freq

    return df


def save_demo_data(
    path: str | Path,
    n_bars: int = 1000,
) -> None:
    """
    Generate demo OHLCV data and save it to a CSV file.

    Parameters
    ----------
    path:
        Destination file path (will be created / overwritten).
    n_bars:
        Number of bars to generate.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_demo_data(n_bars=n_bars)
    df.to_csv(path)

    log.info("Saved %d demo bars to %s.", n_bars, path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_timestamp_column(df: pd.DataFrame) -> str:
    """Return the name of the timestamp column, or fall back to the first column."""
    candidates = ["timestamp", "date", "datetime", "time", "ts"]
    for col in candidates:
        if col in df.columns:
            return col
    # Fall back to the first column
    first = df.columns[0]
    log.info("No standard timestamp column found; using '%s'.", first)
    return first
