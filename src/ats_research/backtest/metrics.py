"""Performance metrics for backtest evaluation.

Provides functions to compute standard risk-adjusted performance statistics
from an equity curve (a :class:`~pandas.Series` indexed by time).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute a comprehensive set of performance metrics.

    Parameters
    ----------
    equity_curve:
        Portfolio equity indexed by timestamp.  Must contain at least two
        data points to produce meaningful results.
    risk_free_rate:
        Annualized risk-free rate as a decimal (e.g. ``0.04`` for 4 %).
        Used in Sharpe and Sortino calculations.

    Returns
    -------
    dict[str, float]
        Dictionary of metric name to value.  Includes:
        ``total_return``, ``cagr``, ``sharpe_ratio``, ``sortino_ratio``,
        ``max_drawdown``, ``calmar_ratio``, ``annual_volatility``,
        ``hit_rate``, ``profit_factor``, ``avg_win``, ``avg_loss``,
        ``max_consecutive_wins``, ``max_consecutive_losses``,
        ``exposure``, ``daily_turnover``.
    """
    metrics: dict[str, float] = {}

    if equity_curve.empty or len(equity_curve) < 2:
        return _empty_metrics()

    # Daily returns
    daily_returns = equity_curve.pct_change().dropna()

    if daily_returns.empty:
        return _empty_metrics()

    # --- Total return ---
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0
    metrics["total_return"] = total_return

    # --- CAGR ---
    n_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if n_days <= 0:
        n_days = len(equity_curve)  # fallback for intraday
    years = n_days / 365.25
    if years > 0 and equity_curve.iloc[0] > 0:
        end_over_start = equity_curve.iloc[-1] / equity_curve.iloc[0]
        if end_over_start > 0:
            metrics["cagr"] = end_over_start ** (1.0 / years) - 1.0
        else:
            metrics["cagr"] = -1.0
    else:
        metrics["cagr"] = 0.0

    # --- Annualized volatility ---
    daily_vol = daily_returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    metrics["annual_volatility"] = annual_vol

    # --- Sharpe ratio (annualized) ---
    daily_rf = risk_free_rate / 252.0
    excess_returns = daily_returns - daily_rf
    if daily_vol > 0:
        metrics["sharpe_ratio"] = (excess_returns.mean() / daily_vol) * np.sqrt(252)
    else:
        metrics["sharpe_ratio"] = 0.0

    # --- Sortino ratio (annualized, downside deviation only) ---
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) > 0:
        downside_std = np.sqrt((downside_returns**2).mean())
        if downside_std > 0:
            metrics["sortino_ratio"] = (excess_returns.mean() / downside_std) * np.sqrt(252)
        else:
            metrics["sortino_ratio"] = float("inf") if excess_returns.mean() > 0 else 0.0
    else:
        metrics["sortino_ratio"] = float("inf") if excess_returns.mean() > 0 else 0.0

    # --- Max drawdown ---
    dd_series = compute_drawdown_series(equity_curve)
    max_dd = dd_series.min()  # most negative value
    metrics["max_drawdown"] = max_dd

    # --- Calmar ratio ---
    if max_dd != 0.0:
        metrics["calmar_ratio"] = metrics["cagr"] / abs(max_dd)
    else:
        metrics["calmar_ratio"] = float("inf") if metrics["cagr"] > 0 else 0.0

    # --- Hit rate ---
    positive_days = (daily_returns > 0).sum()
    total_days = len(daily_returns)
    metrics["hit_rate"] = positive_days / total_days if total_days > 0 else 0.0

    # --- Profit factor ---
    gross_profit = daily_returns[daily_returns > 0].sum()
    gross_loss = abs(daily_returns[daily_returns < 0].sum())
    if gross_loss > 0:
        metrics["profit_factor"] = gross_profit / gross_loss
    else:
        metrics["profit_factor"] = float("inf") if gross_profit > 0 else 0.0

    # --- Avg win / avg loss ---
    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    metrics["avg_win"] = wins.mean() if len(wins) > 0 else 0.0
    metrics["avg_loss"] = losses.mean() if len(losses) > 0 else 0.0

    # --- Max consecutive wins / losses ---
    metrics["max_consecutive_wins"] = _max_consecutive(daily_returns > 0)
    metrics["max_consecutive_losses"] = _max_consecutive(daily_returns < 0)

    # --- Exposure (approximation: pct of days with non-zero return) ---
    non_zero_days = (daily_returns.abs() > 1e-10).sum()
    metrics["exposure"] = non_zero_days / total_days if total_days > 0 else 0.0

    # --- Daily turnover (requires trade info, set to NaN) ---
    metrics["daily_turnover"] = float("nan")

    return metrics


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Compute the drawdown at each point in time.

    Drawdown is expressed as a negative fraction (e.g. ``-0.10`` means the
    equity is 10 % below the running peak).

    Parameters
    ----------
    equity_curve:
        Portfolio equity indexed by timestamp.

    Returns
    -------
    pd.Series
        Drawdown fraction at each time step.
    """
    if equity_curve.empty:
        return pd.Series(dtype=float)

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    drawdown = drawdown.fillna(0.0)
    drawdown.name = "drawdown"
    return drawdown


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_metrics() -> dict[str, float]:
    """Return a metrics dict with safe default values for edge cases."""
    return {
        "total_return": 0.0,
        "cagr": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "calmar_ratio": 0.0,
        "annual_volatility": 0.0,
        "hit_rate": 0.0,
        "profit_factor": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
        "exposure": 0.0,
        "daily_turnover": float("nan"),
    }


def _max_consecutive(mask: pd.Series) -> int:
    """Return the longest run of ``True`` values in a boolean Series."""
    if mask.empty or not mask.any():
        return 0
    # Group consecutive identical values and find the longest True run
    groups = mask.ne(mask.shift()).cumsum()
    true_groups = groups[mask]
    if true_groups.empty:
        return 0
    return int(true_groups.value_counts().max())
