"""Tests for strategy module."""

import pandas as pd
import pytest

from ats_research.data.loaders import generate_demo_data
from ats_research.strategy.base import Signal, Strategy
from ats_research.strategy.sma_crossover import SMACrossover


@pytest.fixture
def demo_data():
    data = generate_demo_data(n_bars=200, seed=42)
    data["symbol"] = "DEMO"
    return data


class TestSMACrossover:
    def test_sma_crossover_initialization(self, demo_data):
        """Strategy initializes without error."""
        s = SMACrossover(params={"fast": 10, "slow": 30})
        s.initialize(demo_data)
        assert s._is_initialized
        assert s._fast_sma is not None
        assert s._slow_sma is not None
        assert s._atr is not None

    def test_sma_crossover_warmup(self, demo_data):
        """Returns None during warmup period (first `slow` bars)."""
        s = SMACrossover(params={"fast": 10, "slow": 30})
        s.initialize(demo_data)
        # First bar during warmup should return None
        ts = demo_data.index[0]
        bar = demo_data.iloc[0]
        history = demo_data.iloc[:1]
        result = s.on_bar(ts, bar, history)
        assert result is None

    def test_sma_crossover_generates_signal(self, demo_data):
        """Produces at least one signal after warmup."""
        s = SMACrossover(params={"fast": 10, "slow": 30})
        s.initialize(demo_data)
        signals = []
        for i in range(len(demo_data)):
            ts = demo_data.index[i]
            bar = demo_data.iloc[i]
            history = demo_data.iloc[: i + 1]
            sig = s.on_bar(ts, bar, history)
            if sig is not None:
                signals.append(sig)
        # Should generate at least one signal over 200 bars
        assert len(signals) > 0
        assert all(isinstance(s, Signal) for s in signals)

    def test_signal_has_metadata(self, demo_data):
        """Long-entry signals have stop_price and tp_price in metadata."""
        s = SMACrossover(params={"fast": 10, "slow": 30, "atr_stop": 2.0, "tp_mult": 3.0})
        s.initialize(demo_data)
        for i in range(len(demo_data)):
            ts = demo_data.index[i]
            bar = demo_data.iloc[i]
            history = demo_data.iloc[: i + 1]
            sig = s.on_bar(ts, bar, history)
            if sig is not None and sig.target_position > 0:
                assert "stop_price" in sig.metadata
                assert "tp_price" in sig.metadata
                assert sig.metadata["stop_price"] < bar["close"]
                assert sig.metadata["tp_price"] > bar["close"]
                break
        else:
            pytest.skip("No long signal generated in test data")
