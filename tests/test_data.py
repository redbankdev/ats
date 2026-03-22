"""Tests for the data module: validators, loaders, and features."""

import numpy as np
import pandas as pd
import pytest

from ats_research.data.validators import validate_ohlcv
from ats_research.data.loaders import load_ohlcv, generate_demo_data
from ats_research.data.features import sma, ema, atr, rsi


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

class TestValidateOHLCV:
    def test_validate_ohlcv_valid(self, sample_ohlcv_df):
        """Valid data passes validation unchanged (aside from cleaning)."""
        result = validate_ohlcv(sample_ohlcv_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_validate_ohlcv_missing_columns(self):
        """Raises ValueError when required columns are missing."""
        idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
        df = pd.DataFrame({"open": [1, 2, 3, 4, 5]}, index=idx)
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_ohlcv(df)

    def test_validate_ohlcv_nan_in_ohlc(self, sample_ohlcv_df):
        """Raises ValueError when OHLC columns contain NaN."""
        df = sample_ohlcv_df.copy()
        df.loc[df.index[5], "close"] = np.nan
        with pytest.raises(ValueError, match="NaN values found in OHLC"):
            validate_ohlcv(df)

    def test_validate_ohlcv_duplicate_timestamps(self, sample_ohlcv_df):
        """Duplicate timestamps are dropped with a warning."""
        dup = pd.concat([sample_ohlcv_df, sample_ohlcv_df.iloc[[0]]])
        with pytest.warns(UserWarning, match="duplicate timestamp"):
            result = validate_ohlcv(dup)
        assert not result.index.duplicated().any()

    def test_validate_ohlcv_unordered(self, sample_ohlcv_df):
        """Unordered timestamps are sorted."""
        df = sample_ohlcv_df.iloc[::-1]  # reverse
        assert not df.index.is_monotonic_increasing
        with pytest.warns(UserWarning, match="not monotonically increasing"):
            result = validate_ohlcv(df)
        assert result.index.is_monotonic_increasing

    def test_validate_ohlcv_nan_volume_filled(self, sample_ohlcv_df):
        """NaN volume values are filled with 0."""
        df = sample_ohlcv_df.copy()
        df.loc[df.index[3], "volume"] = np.nan
        result = validate_ohlcv(df)
        assert result["volume"].isna().sum() == 0
        assert result.loc[result.index[3], "volume"] == 0


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

class TestLoadCSV:
    def test_load_csv(self, sample_ohlcv_df, tmp_path):
        """Load from CSV round-trips correctly."""
        csv_path = tmp_path / "test_data.csv"
        sample_ohlcv_df.to_csv(csv_path)

        loaded = load_ohlcv(str(csv_path), freq="1D", tz="UTC")
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == len(sample_ohlcv_df)
        assert isinstance(loaded.index, pd.DatetimeIndex)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in loaded.columns


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_features_sma(self, sample_ohlcv_df):
        """SMA calculation produces correct values."""
        window = 10
        result = sma(sample_ohlcv_df["close"], window=window)
        assert len(result) == len(sample_ohlcv_df)
        # First window-1 values should be NaN
        assert result.iloc[: window - 1].isna().all()
        # The value at index window-1 should equal the mean of first `window` closes
        expected = sample_ohlcv_df["close"].iloc[:window].mean()
        assert abs(result.iloc[window - 1] - expected) < 1e-8

    def test_features_ema(self, sample_ohlcv_df):
        """EMA produces a series of the same length with no NaN after first value."""
        result = ema(sample_ohlcv_df["close"], span=10)
        assert len(result) == len(sample_ohlcv_df)
        # First value should equal first close (adjust=False seeds from first value)
        assert abs(result.iloc[0] - sample_ohlcv_df["close"].iloc[0]) < 1e-8

    def test_features_atr(self, sample_ohlcv_df):
        """ATR produces positive values after warmup."""
        window = 14
        result = atr(sample_ohlcv_df, window=window)
        assert len(result) == len(sample_ohlcv_df)
        # After warmup, values should be positive
        valid = result.dropna()
        assert (valid > 0).all()

    def test_features_rsi(self, sample_ohlcv_df):
        """RSI values are in [0, 100] range."""
        result = rsi(sample_ohlcv_df["close"], window=14)
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()
        assert (valid <= 100).all()
