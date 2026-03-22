"""Plotting functions for backtest performance reports.

All plots use matplotlib with a clean, professional style and the ``Agg``
backend so they can be generated in headless environments.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy import stats as sp_stats  # noqa: E402

from ats_research.backtest.metrics import compute_drawdown_series  # noqa: E402

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_STYLE_PREFERENCES = ["seaborn-v0_8-whitegrid", "seaborn-whitegrid"]

def _apply_style() -> None:
    """Apply a clean plot style, falling back gracefully."""
    for style in _STYLE_PREFERENCES:
        if style in plt.style.available:
            plt.style.use(style)
            return
    # Fallback: use reasonable rcParams for a clean look
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


_apply_style()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finalise(fig: Figure, save_path: str | None) -> Figure:
    """Tight-layout, optionally save, and return the figure."""
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def _is_trivial(series: pd.Series) -> bool:
    """Return ``True`` when the series is too short to plot meaningfully."""
    return series is None or len(series) < 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_equity_curve(
    equity: pd.Series,
    title: str = "Equity Curve",
    save_path: str | None = None,
) -> Figure:
    """Plot portfolio equity over time.

    A horizontal dashed line marks the initial equity value.

    Parameters
    ----------
    equity:
        Equity series indexed by datetime.
    title:
        Plot title.
    save_path:
        If provided, save the figure to this path and close it.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    if _is_trivial(equity):
        ax.set_title(title)
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="grey")
        return _finalise(fig, save_path)

    ax.plot(equity.index, equity.values, linewidth=1.2, color="#1f77b4")
    ax.axhline(equity.iloc[0], linestyle="--", linewidth=0.8, color="grey",
               label=f"Initial: {equity.iloc[0]:,.2f}")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend(loc="best")

    return _finalise(fig, save_path)


def plot_drawdown(
    equity: pd.Series,
    title: str = "Drawdown",
    save_path: str | None = None,
) -> Figure:
    """Plot the drawdown (underwater chart) over time.

    The area below zero is filled in red to highlight drawdown periods.

    Parameters
    ----------
    equity:
        Equity series indexed by datetime.
    title:
        Plot title.
    save_path:
        If provided, save the figure to this path and close it.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    if _is_trivial(equity):
        ax.set_title(title)
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="grey")
        return _finalise(fig, save_path)

    dd = compute_drawdown_series(equity) * 100  # percent
    ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.35)
    ax.plot(dd.index, dd.values, linewidth=0.8, color="red")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")

    return _finalise(fig, save_path)


def plot_rolling_sharpe(
    equity: pd.Series,
    window: int = 63,
    title: str = "Rolling Sharpe (63d)",
    save_path: str | None = None,
) -> Figure:
    """Plot the rolling annualized Sharpe ratio.

    Parameters
    ----------
    equity:
        Equity series indexed by datetime.
    window:
        Rolling window in trading days.
    title:
        Plot title.
    save_path:
        If provided, save the figure to this path and close it.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    if _is_trivial(equity) or len(equity) < window + 1:
        ax.set_title(title)
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="grey")
        return _finalise(fig, save_path)

    daily_returns = equity.pct_change().dropna()
    rolling_mean = daily_returns.rolling(window).mean()
    rolling_std = daily_returns.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.0,
            color="#2ca02c")
    ax.axhline(0, linestyle="--", linewidth=0.7, color="grey")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")

    return _finalise(fig, save_path)


def plot_returns_distribution(
    equity: pd.Series,
    title: str = "Daily Returns Distribution",
    save_path: str | None = None,
) -> Figure:
    """Plot a histogram of daily returns with a normal distribution overlay.

    A vertical line marks the mean return.

    Parameters
    ----------
    equity:
        Equity series indexed by datetime.
    title:
        Plot title.
    save_path:
        If provided, save the figure to this path and close it.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    if _is_trivial(equity):
        ax.set_title(title)
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="grey")
        return _finalise(fig, save_path)

    daily_returns = equity.pct_change().dropna()

    if daily_returns.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No returns to display", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="grey")
        return _finalise(fig, save_path)

    mu = daily_returns.mean()
    sigma = daily_returns.std()

    ax.hist(daily_returns.values, bins=50, density=True, alpha=0.6,
            color="#1f77b4", edgecolor="white", linewidth=0.5)

    # Normal distribution overlay
    if sigma > 0:
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        ax.plot(x, sp_stats.norm.pdf(x, mu, sigma), linewidth=1.5,
                color="#d62728", label="Normal fit")

    ax.axvline(mu, linestyle="--", linewidth=1.0, color="black",
               label=f"Mean: {mu:.4%}")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Density")
    ax.legend(loc="best")

    return _finalise(fig, save_path)


def plot_monthly_returns_heatmap(
    equity: pd.Series,
    title: str = "Monthly Returns (%)",
    save_path: str | None = None,
) -> Figure:
    """Plot monthly returns as a year-by-month heatmap table.

    Parameters
    ----------
    equity:
        Equity series indexed by datetime.
    title:
        Plot title.
    save_path:
        If provided, save the figure to this path and close it.

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, max(4, 0.6 * 8)))

    if _is_trivial(equity):
        ax.set_title(title)
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="grey")
        ax.axis("off")
        return _finalise(fig, save_path)

    daily_returns = equity.pct_change().dropna()

    if daily_returns.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No returns to display", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="grey")
        ax.axis("off")
        return _finalise(fig, save_path)

    # Compute monthly returns
    monthly = daily_returns.groupby(
        [daily_returns.index.year, daily_returns.index.month]
    ).apply(lambda x: (1 + x).prod() - 1)
    monthly.index.names = ["Year", "Month"]
    monthly_table = monthly.unstack(level="Month") * 100  # percent

    # Ensure all 12 months present
    for m in range(1, 13):
        if m not in monthly_table.columns:
            monthly_table[m] = np.nan
    monthly_table = monthly_table[sorted(monthly_table.columns)]

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Resize figure based on actual data
    n_years = len(monthly_table)
    fig.set_size_inches(12, max(3, 0.6 * n_years + 1.5))

    ax.clear()

    data = monthly_table.values
    vmax = np.nanmax(np.abs(data)) if not np.all(np.isnan(data)) else 1.0
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels)
    ax.set_yticks(range(n_years))
    ax.set_yticklabels([str(y) for y in monthly_table.index])

    # Annotate cells with values
    for i in range(n_years):
        for j in range(12):
            val = data[i, j]
            if not np.isnan(val):
                colour = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color=colour)

    ax.set_title(title, fontsize=14, pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Return (%)")

    return _finalise(fig, save_path)


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_all_plots(equity: pd.Series, output_dir: str) -> list[str]:
    """Generate all standard plots and save them to *output_dir*.

    Parameters
    ----------
    equity:
        Equity series indexed by datetime.
    output_dir:
        Directory to write PNG files.

    Returns
    -------
    list[str]
        Absolute paths of saved files.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    saved: list[str] = []

    specs: list[tuple[str, ...]] = [
        ("equity_curve.png",),
        ("drawdown.png",),
        ("rolling_sharpe.png",),
        ("returns_distribution.png",),
        ("monthly_returns_heatmap.png",),
    ]

    plot_funcs = [
        plot_equity_curve,
        plot_drawdown,
        plot_rolling_sharpe,
        plot_returns_distribution,
        plot_monthly_returns_heatmap,
    ]

    for func, (filename,) in zip(plot_funcs, specs):
        path = os.path.join(output_dir, filename)
        func(equity, save_path=path)
        saved.append(os.path.abspath(path))

    return saved
