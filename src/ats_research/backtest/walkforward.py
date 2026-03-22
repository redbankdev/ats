"""Walk-forward analysis for overfitting detection and robustness testing.

Provides :class:`WalkForwardSplitter` for time-series cross-validation with
purge and embargo gaps, and :func:`run_walkforward` for grid-search evaluation
across multiple train/test splits.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any, Iterator, Type

import pandas as pd

from ats_research.backtest.engine import BacktestEngine, BacktestResult
from ats_research.backtest.metrics import compute_metrics
from ats_research.strategy.base import Strategy
from ats_research.utils.config import ATSConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WalkForwardSplitter
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardSplitter:
    """Time-series cross-validation splitter with purge and embargo.

    Splits a time-indexed DataFrame into sequential train/test folds
    suitable for walk-forward analysis of trading strategies.

    Parameters
    ----------
    n_splits:
        Number of test folds to generate.
    train_pct:
        Fraction of the data window used for training in each fold
        (only applies to ``"rolling"`` method; for ``"expanding"`` the
        training set grows with each fold).
    method:
        ``"rolling"`` for a fixed-size training window that slides
        forward, or ``"expanding"`` for a training window that grows
        over time.
    purge_days:
        Number of calendar days to remove from the end of each training
        set to prevent look-ahead leakage from overlapping labels.
    embargo_days:
        Number of calendar days to skip at the beginning of each test
        set to further guard against information leakage.
    """

    n_splits: int = 5
    train_pct: float = 0.60
    method: str = "rolling"  # "rolling" or "expanding"
    purge_days: int = 0
    embargo_days: int = 0

    def __post_init__(self) -> None:
        if self.n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if not 0.0 < self.train_pct < 1.0:
            raise ValueError("train_pct must be between 0 and 1 (exclusive)")
        if self.method not in ("rolling", "expanding"):
            raise ValueError(f"method must be 'rolling' or 'expanding', got {self.method!r}")
        if self.purge_days < 0:
            raise ValueError("purge_days must be >= 0")
        if self.embargo_days < 0:
            raise ValueError("embargo_days must be >= 0")

    def split(self, data: pd.DataFrame) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate train/test splits from a time-indexed DataFrame.

        Parameters
        ----------
        data:
            OHLCV DataFrame with a :class:`~pandas.DatetimeIndex`.

        Yields
        ------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(train_df, test_df)`` pairs for each fold.

        Raises
        ------
        ValueError
            If the data has fewer rows than required for the requested
            number of splits.
        """
        n_rows = len(data)
        if n_rows < self.n_splits + 1:
            raise ValueError(
                f"Not enough data ({n_rows} rows) for {self.n_splits} splits"
            )

        if self.method == "rolling":
            yield from self._split_rolling(data, n_rows)
        else:
            yield from self._split_expanding(data, n_rows)

    def _split_rolling(
        self, data: pd.DataFrame, n_rows: int
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Rolling window: fixed train size, slides forward."""
        # Total usable rows divided into n_splits test segments
        # The first fold starts after the initial training window
        train_size = int(n_rows * self.train_pct)
        remaining = n_rows - train_size
        test_size = remaining // self.n_splits

        if train_size < 2 or test_size < 1:
            raise ValueError(
                f"Data too small for rolling split: train_size={train_size}, "
                f"test_size={test_size}"
            )

        for i in range(self.n_splits):
            test_start_idx = train_size + i * test_size
            test_end_idx = test_start_idx + test_size
            if i == self.n_splits - 1:
                # Last fold takes remaining data
                test_end_idx = n_rows

            # Train window slides: ends at test_start_idx
            train_start_idx = test_start_idx - train_size
            train_end_idx = test_start_idx

            # Apply purge: remove purge_days from end of train
            train_df = data.iloc[train_start_idx:train_end_idx]
            if self.purge_days > 0 and len(train_df) > 0:
                purge_cutoff = train_df.index[-1] - pd.Timedelta(days=self.purge_days)
                train_df = train_df[train_df.index <= purge_cutoff]

            # Apply embargo: skip embargo_days at start of test
            test_df = data.iloc[test_start_idx:test_end_idx]
            if self.embargo_days > 0 and len(test_df) > 0:
                embargo_cutoff = test_df.index[0] + pd.Timedelta(days=self.embargo_days)
                test_df = test_df[test_df.index >= embargo_cutoff]

            if len(train_df) < 2 or len(test_df) < 1:
                logger.warning(
                    "Skipping split %d: insufficient data after purge/embargo "
                    "(train=%d, test=%d)",
                    i,
                    len(train_df),
                    len(test_df),
                )
                continue

            yield train_df, test_df

    def _split_expanding(
        self, data: pd.DataFrame, n_rows: int
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Expanding window: train grows, test segments are fixed size."""
        # Minimum initial train size
        min_train_size = int(n_rows * self.train_pct)
        remaining = n_rows - min_train_size
        test_size = remaining // self.n_splits

        if min_train_size < 2 or test_size < 1:
            raise ValueError(
                f"Data too small for expanding split: min_train={min_train_size}, "
                f"test_size={test_size}"
            )

        for i in range(self.n_splits):
            test_start_idx = min_train_size + i * test_size
            test_end_idx = test_start_idx + test_size
            if i == self.n_splits - 1:
                test_end_idx = n_rows

            # Expanding: train always starts from the beginning
            train_end_idx = test_start_idx

            train_df = data.iloc[0:train_end_idx]
            if self.purge_days > 0 and len(train_df) > 0:
                purge_cutoff = train_df.index[-1] - pd.Timedelta(days=self.purge_days)
                train_df = train_df[train_df.index <= purge_cutoff]

            test_df = data.iloc[test_start_idx:test_end_idx]
            if self.embargo_days > 0 and len(test_df) > 0:
                embargo_cutoff = test_df.index[0] + pd.Timedelta(days=self.embargo_days)
                test_df = test_df[test_df.index >= embargo_cutoff]

            if len(train_df) < 2 or len(test_df) < 1:
                logger.warning(
                    "Skipping split %d: insufficient data after purge/embargo "
                    "(train=%d, test=%d)",
                    i,
                    len(train_df),
                    len(test_df),
                )
                continue

            yield train_df, test_df


# ---------------------------------------------------------------------------
# Walk-forward runner
# ---------------------------------------------------------------------------

def run_walkforward(
    config: ATSConfig,
    strategy_cls: Type[Strategy],
    strategy_params_grid: dict[str, list[Any]],
    data: pd.DataFrame,
    splitter: WalkForwardSplitter,
    initial_cash: float = 100_000.0,
) -> pd.DataFrame:
    """Run walk-forward analysis over a parameter grid and splitter.

    For each combination of strategy parameters and each train/test split,
    runs a backtest on the training data and the test data independently,
    then collects key metrics for comparison.  This is useful for detecting
    overfitting: strategies that perform well in-sample but poorly
    out-of-sample are suspect.

    Parameters
    ----------
    config:
        Base configuration.  The ``strategy.params`` field is overridden
        for each parameter combination in the grid.
    strategy_cls:
        Strategy class to instantiate for each run.
    strategy_params_grid:
        Dictionary mapping parameter names to lists of values to try.
        All combinations are evaluated (Cartesian product).
        Example: ``{"fast": [10, 20], "slow": [50, 100]}``.
    data:
        Full OHLCV dataset.
    splitter:
        :class:`WalkForwardSplitter` defining the train/test splits.
    initial_cash:
        Starting cash for each backtest run.

    Returns
    -------
    pd.DataFrame
        Results with columns: ``params``, ``split_idx``,
        ``train_sharpe``, ``test_sharpe``, ``train_cagr``, ``test_cagr``,
        ``train_max_drawdown``, ``test_max_drawdown``,
        ``train_total_return``, ``test_total_return``.
    """
    # Build all parameter combinations
    param_names = list(strategy_params_grid.keys())
    param_values = list(strategy_params_grid.values())
    param_combos = list(itertools.product(*param_values))

    results: list[dict[str, Any]] = []

    total_combos = len(param_combos)
    logger.info(
        "Walk-forward: %d param combos x %d splits",
        total_combos,
        splitter.n_splits,
    )

    for combo_idx, combo in enumerate(param_combos):
        params = dict(zip(param_names, combo))
        params_str = str(params)

        logger.info(
            "Evaluating combo %d/%d: %s", combo_idx + 1, total_combos, params_str
        )

        # Generate splits from the full dataset
        for split_idx, (train_df, test_df) in enumerate(splitter.split(data)):
            # Build config override for this param combo
            run_config = config.model_copy(deep=True)
            run_config.strategy.params.update(params)

            # --- Train backtest ---
            train_metrics = _run_single(
                run_config, strategy_cls, params, train_df, initial_cash
            )

            # --- Test backtest ---
            test_metrics = _run_single(
                run_config, strategy_cls, params, test_df, initial_cash
            )

            results.append(
                {
                    "params": params_str,
                    "split_idx": split_idx,
                    "train_sharpe": train_metrics.get("sharpe_ratio", float("nan")),
                    "test_sharpe": test_metrics.get("sharpe_ratio", float("nan")),
                    "train_cagr": train_metrics.get("cagr", float("nan")),
                    "test_cagr": test_metrics.get("cagr", float("nan")),
                    "train_max_drawdown": train_metrics.get("max_drawdown", float("nan")),
                    "test_max_drawdown": test_metrics.get("max_drawdown", float("nan")),
                    "train_total_return": train_metrics.get("total_return", float("nan")),
                    "test_total_return": test_metrics.get("total_return", float("nan")),
                }
            )

    if not results:
        return pd.DataFrame(
            columns=[
                "params",
                "split_idx",
                "train_sharpe",
                "test_sharpe",
                "train_cagr",
                "test_cagr",
                "train_max_drawdown",
                "test_max_drawdown",
                "train_total_return",
                "test_total_return",
            ]
        )

    return pd.DataFrame(results)


def _run_single(
    config: ATSConfig,
    strategy_cls: Type[Strategy],
    params: dict[str, Any],
    data: pd.DataFrame,
    initial_cash: float,
) -> dict[str, float]:
    """Run a single backtest and return its metrics dict.

    Parameters
    ----------
    config:
        Run configuration (strategy params should already be set).
    strategy_cls:
        Strategy class to instantiate.
    params:
        Strategy parameters dict.
    data:
        OHLCV data for this run segment.
    initial_cash:
        Starting cash.

    Returns
    -------
    dict[str, float]
        Computed performance metrics, or empty-ish defaults on failure.
    """
    try:
        strategy = strategy_cls(params=params)
        engine = BacktestEngine(
            config=config,
            strategy=strategy,
            data=data,
            initial_cash=initial_cash,
        )
        result = engine.run()
        return result.metrics
    except Exception:
        logger.exception("Backtest failed for params=%s", params)
        return {
            "sharpe_ratio": float("nan"),
            "cagr": float("nan"),
            "max_drawdown": float("nan"),
            "total_return": float("nan"),
        }
