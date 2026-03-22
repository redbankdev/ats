"""Tests for the data module: validators, loaders, and features."""

import numpy as np
import pandas as pd
import pytest

from ats_research.data.validators import validate_ohlcv, validate_config_data_path
from ats_research.data.loaders import load_ohlcv, generate_demo_data, save_demo_data
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

    def test_validate_ohlcv_not_datetimeindex(self):
        """Raises ValueError when index is not DatetimeIndex."""
        df = pd.DataFrame({
            "open": [1, 2], "high": [2, 3], "low": [0.5, 1.5],
            "close": [1.5, 2.5], "volume": [100, 200],
        }, index=[0, 1])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            validate_ohlcv(df)

    def test_validate_ohlcv_tz_naive(self):
        """Raises ValueError when DatetimeIndex is tz-naive."""
        idx = pd.date_range("2020-01-01", periods=3, freq="D")  # no tz
        df = pd.DataFrame({
            "open": [1, 2, 3], "high": [2, 3, 4], "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5], "volume": [100, 200, 300],
        }, index=idx)
        with pytest.raises(ValueError, match="timezone-aware"):
            validate_ohlcv(df)

    def test_validate_ohlcv_high_low_violation(self):
        """Warns when high < low."""
        idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({
            "open": [10, 10, 10],
            "high": [8, 12, 12],  # first bar: high < low
            "low": [9, 9, 9],
            "close": [10, 10, 10],
            "volume": [100, 100, 100],
        }, index=idx)
        with pytest.warns(UserWarning, match="OHLC relationship violations"):
            result = validate_ohlcv(df)
        assert len(result) == 3

    def test_validate_ohlcv_high_lt_open(self):
        """Warns when high < open."""
        idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({
            "open": [12, 10, 10],
            "high": [11, 12, 12],  # first: high < open
            "low": [9, 9, 9],
            "close": [10, 10, 10],
            "volume": [100, 100, 100],
        }, index=idx)
        with pytest.warns(UserWarning, match="OHLC relationship violations"):
            validate_ohlcv(df)

    def test_validate_ohlcv_high_lt_close(self):
        """Warns when high < close."""
        idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({
            "open": [10, 10, 10],
            "high": [11, 12, 12],
            "low": [9, 9, 9],
            "close": [13, 10, 10],  # first: close > high
            "volume": [100, 100, 100],
        }, index=idx)
        with pytest.warns(UserWarning, match="OHLC relationship violations"):
            validate_ohlcv(df)

    def test_validate_ohlcv_low_gt_open(self):
        """Warns when low > open."""
        idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({
            "open": [8, 10, 10],  # first: open < low
            "high": [12, 12, 12],
            "low": [9, 9, 9],
            "close": [10, 10, 10],
            "volume": [100, 100, 100],
        }, index=idx)
        with pytest.warns(UserWarning, match="OHLC relationship violations"):
            validate_ohlcv(df)

    def test_validate_ohlcv_low_gt_close(self):
        """Warns when low > close."""
        idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({
            "open": [10, 10, 10],
            "high": [12, 12, 12],
            "low": [9, 9, 9],
            "close": [8, 10, 10],  # first: close < low
            "volume": [100, 100, 100],
        }, index=idx)
        with pytest.warns(UserWarning, match="OHLC relationship violations"):
            validate_ohlcv(df)

    def test_validate_ohlcv_negative_volume(self):
        """Warns when volume is negative."""
        idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({
            "open": [10, 10, 10],
            "high": [12, 12, 12],
            "low": [9, 9, 9],
            "close": [11, 11, 11],
            "volume": [-100, 100, 100],
        }, index=idx)
        with pytest.warns(UserWarning, match="Negative volume"):
            validate_ohlcv(df)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

class TestValidateConfigDataPath:
    def test_valid_csv(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,2\n")
        result = validate_config_data_path(str(p))
        assert result.suffix == ".csv"

    def test_valid_parquet(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.write_text("fake")  # just needs to exist
        result = validate_config_data_path(str(p))
        assert result.suffix == ".parquet"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            validate_config_data_path(str(tmp_path / "missing.csv"))

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "data.xlsx"
        p.write_text("fake")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            validate_config_data_path(str(p))


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

    def test_load_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            load_ohlcv(str(tmp_path / "missing.csv"))

    def test_load_unsupported_extension(self, tmp_path):
        p = tmp_path / "data.xlsx"
        p.write_text("fake")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_ohlcv(str(p))

    def test_load_parquet(self, sample_ohlcv_df, tmp_path):
        """Load from parquet format."""
        pq_path = tmp_path / "test_data.parquet"
        # Reset index so timestamp is a column, as load_ohlcv expects
        df = sample_ohlcv_df.reset_index()
        df.to_parquet(pq_path, index=False)
        loaded = load_ohlcv(str(pq_path), freq="1D", tz="UTC")
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == len(sample_ohlcv_df)

    def test_load_csv_tz_convert(self, sample_ohlcv_df, tmp_path):
        """Loading with already tz-aware data converts correctly."""
        csv_path = tmp_path / "test_data.csv"
        sample_ohlcv_df.to_csv(csv_path)
        loaded = load_ohlcv(str(csv_path), freq="1D", tz="UTC")
        assert str(loaded.index.tz) == "UTC"


class TestSaveDemoData:
    def test_save_demo_data(self, tmp_path):
        out = tmp_path / "demo.csv"
        save_demo_data(str(out), n_bars=50)
        assert out.exists()
        df = pd.read_csv(out, index_col=0)
        assert len(df) == 50

    def test_save_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "demo.csv"
        save_demo_data(str(out), n_bars=10)
        assert out.exists()


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
