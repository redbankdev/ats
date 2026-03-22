"""Tests for walk-forward analysis module."""

import pandas as pd
import pytest

from ats_research.backtest.walkforward import WalkForwardSplitter
from ats_research.data.loaders import generate_demo_data


@pytest.fixture
def demo_data():
    return generate_demo_data(n_bars=500, seed=42)


class TestWalkForwardSplitter:
    def test_splitter_rolling(self, demo_data):
        """Rolling method produces the correct number of splits."""
        splitter = WalkForwardSplitter(n_splits=5, train_pct=0.6, method="rolling")
        splits = list(splitter.split(demo_data))
        assert len(splits) == 5
        for train, test in splits:
            assert len(train) > 0
            assert len(test) > 0

    def test_splitter_expanding(self, demo_data):
        """Expanding method: training set grows with each split."""
        splitter = WalkForwardSplitter(n_splits=3, train_pct=0.6, method="expanding")
        splits = list(splitter.split(demo_data))
        assert len(splits) == 3
        train_sizes = [len(train) for train, _ in splits]
        # Expanding: each train set should be >= the previous
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]

    def test_splitter_purge(self, demo_data):
        """Purge removes days from the end of the training set."""
        splitter_no_purge = WalkForwardSplitter(
            n_splits=3, train_pct=0.6, method="rolling", purge_days=0
        )
        splitter_with_purge = WalkForwardSplitter(
            n_splits=3, train_pct=0.6, method="rolling", purge_days=5
        )
        splits_no = list(splitter_no_purge.split(demo_data))
        splits_with = list(splitter_with_purge.split(demo_data))
        # With purge, train should be shorter
        for (train_no, _), (train_with, _) in zip(splits_no, splits_with):
            assert len(train_with) <= len(train_no)

    def test_splitter_embargo(self, demo_data):
        """Embargo skips days at the start of the test set."""
        splitter_no_embargo = WalkForwardSplitter(
            n_splits=3, train_pct=0.6, method="rolling", embargo_days=0
        )
        splitter_with_embargo = WalkForwardSplitter(
            n_splits=3, train_pct=0.6, method="rolling", embargo_days=5
        )
        splits_no = list(splitter_no_embargo.split(demo_data))
        splits_with = list(splitter_with_embargo.split(demo_data))
        # With embargo, test should start later (fewer or equal bars)
        for (_, test_no), (_, test_with) in zip(splits_no, splits_with):
            assert len(test_with) <= len(test_no)

    def test_invalid_method(self):
        """Raises ValueError for an invalid method."""
        with pytest.raises(ValueError, match="method"):
            WalkForwardSplitter(n_splits=3, train_pct=0.6, method="invalid")
