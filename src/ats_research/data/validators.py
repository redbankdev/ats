"""
Data validation utilities for OHLCV DataFrames and config paths.

Usage::

    from ats_research.data.validators import validate_ohlcv, validate_config_data_path

    df = validate_ohlcv(raw_df)
    path = validate_config_data_path("data/prices.csv")
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from ats_research.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]

_SUPPORTED_EXTENSIONS: set[str] = {".csv", ".parquet"}


# ---------------------------------------------------------------------------
# OHLCV validation
# ---------------------------------------------------------------------------

def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean an OHLCV DataFrame.

    Checks performed (in order):

    1. All required columns present (case-insensitive, normalised to lowercase).
    2. Index is a timezone-aware :class:`~pandas.DatetimeIndex`.
    3. No duplicate timestamps (duplicates dropped with warning).
    4. Monotonically increasing timestamps (sorted if needed with warning).
    5. No NaN in OHLC columns (raises :class:`ValueError`); NaN in volume
       filled with ``0``.
    6. ``high >= low``, ``high >= open``, ``high >= close``,
       ``low <= open``, ``low <= close`` (warns on violations).
    7. ``volume >= 0`` (warns on violations).

    Parameters
    ----------
    df:
        Raw OHLCV DataFrame.  The index must be a
        :class:`~pandas.DatetimeIndex`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with lowercase column names, sorted unique
        timestamps, and volume NaNs filled.

    Raises
    ------
    ValueError
        If required columns are missing, the index is not a
        :class:`~pandas.DatetimeIndex` or is timezone-naive, or OHLC
        columns contain NaN values.
    """
    df = df.copy()

    # --- 1. Column presence (case-insensitive) ----------------------------
    df.columns = df.columns.str.lower().str.strip()

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"DataFrame has: {sorted(df.columns.tolist())}"
        )

    # --- 2. DatetimeIndex & timezone awareness ----------------------------
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"Index must be a DatetimeIndex, got {type(df.index).__name__}."
        )

    if df.index.tz is None:
        raise ValueError(
            "DatetimeIndex must be timezone-aware. "
            "Localise with df.index = df.index.tz_localize('UTC')."
        )

    # --- 3. Duplicate timestamps ------------------------------------------
    dup_mask = df.index.duplicated(keep="first")
    if dup_mask.any():
        n_dups = int(dup_mask.sum())
        warnings.warn(
            f"Dropping {n_dups} duplicate timestamp(s).",
            stacklevel=2,
        )
        log.warning("Dropping %d duplicate timestamp(s).", n_dups)
        df = df[~dup_mask]

    # --- 4. Monotonically increasing timestamps ---------------------------
    if not df.index.is_monotonic_increasing:
        warnings.warn(
            "Timestamps are not monotonically increasing — sorting.",
            stacklevel=2,
        )
        log.warning("Sorting non-monotonic timestamps.")
        df = df.sort_index()

    # --- 5. NaN handling --------------------------------------------------
    ohlc_cols = ["open", "high", "low", "close"]
    ohlc_nans = df[ohlc_cols].isna().sum()
    ohlc_nans = ohlc_nans[ohlc_nans > 0]
    if not ohlc_nans.empty:
        raise ValueError(
            f"NaN values found in OHLC columns: "
            f"{ohlc_nans.to_dict()}"
        )

    if df["volume"].isna().any():
        n_filled = int(df["volume"].isna().sum())
        log.info("Filling %d NaN volume value(s) with 0.", n_filled)
        df["volume"] = df["volume"].fillna(0)

    # --- 6. Price relationship sanity checks ------------------------------
    violations: list[str] = []

    mask_hl = df["high"] < df["low"]
    if mask_hl.any():
        violations.append(f"high < low on {int(mask_hl.sum())} bar(s)")

    mask_ho = df["high"] < df["open"]
    if mask_ho.any():
        violations.append(f"high < open on {int(mask_ho.sum())} bar(s)")

    mask_hc = df["high"] < df["close"]
    if mask_hc.any():
        violations.append(f"high < close on {int(mask_hc.sum())} bar(s)")

    mask_lo = df["low"] > df["open"]
    if mask_lo.any():
        violations.append(f"low > open on {int(mask_lo.sum())} bar(s)")

    mask_lc = df["low"] > df["close"]
    if mask_lc.any():
        violations.append(f"low > close on {int(mask_lc.sum())} bar(s)")

    if violations:
        msg = "OHLC relationship violations: " + "; ".join(violations)
        warnings.warn(msg, stacklevel=2)
        log.warning(msg)

    # --- 7. Non-negative volume -------------------------------------------
    neg_vol = df["volume"] < 0
    if neg_vol.any():
        msg = f"Negative volume on {int(neg_vol.sum())} bar(s)."
        warnings.warn(msg, stacklevel=2)
        log.warning(msg)

    return df


# ---------------------------------------------------------------------------
# Config path validation
# ---------------------------------------------------------------------------

def validate_config_data_path(path: str) -> Path:
    """
    Validate that *path* points to an existing data file with a supported
    extension (``.csv`` or ``.parquet``).

    Parameters
    ----------
    path:
        File-system path to validate.

    Returns
    -------
    pathlib.Path
        Resolved :class:`~pathlib.Path` object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not ``.csv`` or ``.parquet``.
    """
    p = Path(path).resolve()

    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    if p.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{p.suffix}'. "
            f"Expected one of: {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    return p
