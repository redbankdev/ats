"""Event-driven backtest engine.

Orchestrates the interaction between :class:`~ats_research.strategy.base.Strategy`,
:class:`~ats_research.exec.broker_sim.BrokerSim`,
:class:`~ats_research.risk.risk_rules.RiskManager`,
:class:`~ats_research.risk.position_sizing.PositionSizer`, and
:class:`~ats_research.backtest.portfolio.Portfolio` to produce a complete
backtest result.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ats_research.backtest.metrics import compute_metrics
from ats_research.backtest.portfolio import Portfolio
from ats_research.data.features import rolling_volatility
from ats_research.exec.broker_sim import BrokerSim
from ats_research.exec.commission import PerShareCommission, PercentCommission
from ats_research.exec.orders import Fill, Order, OrderSide, OrderType, TimeInForce
from ats_research.exec.slippage import FixedBpsSlippage
from ats_research.risk.position_sizing import (
    FixedFractionSizer,
    PositionSizer,
    VolatilityScaledSizer,
)
from ats_research.risk.risk_rules import (
    DailyLossHaltRule,
    MaxPositionRule,
    PerTradeLossCapRule,
    RiskManager,
)
from ats_research.strategy.base import Signal, Strategy
from ats_research.utils.config import ATSConfig
from ats_research.utils.repro import RunContext, generate_run_id, set_global_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Container for all outputs of a single backtest run.

    Attributes
    ----------
    equity_curve:
        Time-indexed equity series.
    metrics:
        Dictionary of computed performance metrics.
    trades:
        List of all :class:`~ats_research.exec.orders.Fill` records.
    portfolio_history:
        Full portfolio state history as a DataFrame.
    config:
        The :class:`~ats_research.utils.config.ATSConfig` used for the run.
    run_context:
        Reproducibility context capturing seed, versions, and run ID.
    """

    equity_curve: pd.Series
    metrics: dict[str, float]
    trades: list[Fill]
    portfolio_history: pd.DataFrame
    config: ATSConfig
    run_context: RunContext

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_json(self, path: str | Path) -> None:
        """Save metrics and run context to a JSON file.

        Parameters
        ----------
        path:
            Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "run_id": self.run_context.run_id,
            "seed": self.run_context.seed,
            "timestamp": self.run_context.timestamp,
            "python_version": self.run_context.python_version,
            "package_version": self.run_context.package_version,
            "strategy": self.config.strategy.name,
            "strategy_params": self.config.strategy.params,
            "metrics": {},
        }
        # Ensure all metric values are JSON-serialisable
        for k, v in self.metrics.items():
            if pd.isna(v):
                payload["metrics"][k] = None
            else:
                payload["metrics"][k] = v

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved metrics to %s", path)

    def to_csv(self, path: str | Path) -> None:
        """Save the equity curve to a CSV file.

        Parameters
        ----------
        path:
            Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.equity_curve.to_csv(path, header=True)
        logger.info("Saved equity curve to %s", path)

    def summary(self) -> str:
        """Return a human-readable summary of the backtest results.

        Returns
        -------
        str
            Formatted multi-line string with key metrics.
        """
        lines = [
            "=" * 60,
            f"  Backtest Summary — {self.run_context.run_id}",
            "=" * 60,
            f"  Strategy      : {self.config.strategy.name}",
            f"  Params        : {self.config.strategy.params}",
            f"  Seed          : {self.run_context.seed}",
            "-" * 60,
        ]

        fmt_items = [
            ("Total Return", "total_return", ".2%"),
            ("CAGR", "cagr", ".2%"),
            ("Sharpe Ratio", "sharpe_ratio", ".3f"),
            ("Sortino Ratio", "sortino_ratio", ".3f"),
            ("Max Drawdown", "max_drawdown", ".2%"),
            ("Calmar Ratio", "calmar_ratio", ".3f"),
            ("Annual Vol", "annual_volatility", ".2%"),
            ("Hit Rate", "hit_rate", ".2%"),
            ("Profit Factor", "profit_factor", ".3f"),
            ("Avg Win", "avg_win", ".4%"),
            ("Avg Loss", "avg_loss", ".4%"),
            ("Max Consec Wins", "max_consecutive_wins", "d"),
            ("Max Consec Losses", "max_consecutive_losses", "d"),
            ("Exposure", "exposure", ".2%"),
        ]

        for label, key, fmt in fmt_items:
            value = self.metrics.get(key, float("nan"))
            if pd.isna(value):
                lines.append(f"  {label:<20s}: N/A")
            elif fmt == "d":
                lines.append(f"  {label:<20s}: {int(value)}")
            else:
                lines.append(f"  {label:<20s}: {value:{fmt}}")

        lines.append("-" * 60)
        lines.append(f"  Total trades  : {len(self.trades)}")
        if self.equity_curve is not None and not self.equity_curve.empty:
            lines.append(f"  Start equity  : {self.equity_curve.iloc[0]:,.2f}")
            lines.append(f"  End equity    : {self.equity_curve.iloc[-1]:,.2f}")
        lines.append("=" * 60)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Event-driven backtesting engine.

    Iterates over bars chronologically, feeding them through the strategy,
    position sizer, risk manager, and simulated broker to produce a complete
    :class:`BacktestResult`.

    Parameters
    ----------
    config:
        Full run configuration.
    strategy:
        Strategy instance to evaluate.
    data:
        OHLCV DataFrame indexed by timestamp.  Must contain at least
        ``open``, ``high``, ``low``, ``close``, ``volume`` columns and
        a ``symbol`` column (or a single-instrument dataset).
    initial_cash:
        Starting cash.  Defaults to ``100_000.0``.
    """

    def __init__(
        self,
        config: ATSConfig,
        strategy: Strategy,
        data: pd.DataFrame,
        initial_cash: float = 100_000.0,
    ) -> None:
        self.config = config
        self.strategy = strategy
        self.data = data.sort_index()
        self.initial_cash = initial_cash

        # ----- Set up execution components from config -----
        slippage_model = FixedBpsSlippage(bps=config.costs.slippage_bps)

        # Choose commission model: prefer per-share, fall back to percent
        if config.costs.commission_per_share > 0:
            commission_model = PerShareCommission(rate=config.costs.commission_per_share)
        elif config.costs.commission_pct > 0:
            commission_model = PercentCommission(rate=config.costs.commission_pct)
        else:
            commission_model = PerShareCommission(rate=0.0)

        self.broker = BrokerSim(
            slippage_model=slippage_model,
            commission_model=commission_model,
            liquidity_cap_pct_adv=config.simulation.liquidity_cap_pct_adv,
        )

        # ----- Risk manager -----
        self.risk_manager = RiskManager(
            rules=[
                MaxPositionRule(max_pct=config.risk.max_pos_pct),
                PerTradeLossCapRule(max_loss_pct=config.risk.per_trade_risk_pct),
                DailyLossHaltRule(max_daily_loss_pct=config.risk.daily_loss_limit_pct),
            ]
        )

        # ----- Portfolio -----
        self.portfolio = Portfolio(initial_cash=initial_cash)

        # ----- Position sizer -----
        self.position_sizer: PositionSizer = VolatilityScaledSizer(
            vol_target=config.risk.vol_target_pct,
            max_position_pct=config.risk.max_pos_pct,
        )

        # ----- Pre-compute rolling volatility for sizing -----
        self._volatility: pd.Series = rolling_volatility(
            self.data["close"], window=20
        )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the backtest and return results.

        The event loop processes bars chronologically:

        1. Process pending orders through broker (fill at bar open for
           market orders).
        2. Notify strategy of fills via ``on_fill()``.
        3. Mark portfolio to market at close prices.
        4. Check daily loss halt.
        5. Call ``strategy.on_bar()`` to get signal.
        6. If signal, use position sizer to determine units.
        7. Calculate orders needed to reach target position.
        8. Run orders through risk manager.
        9. Submit approved orders to broker.
        10. Record state.

        Returns
        -------
        BacktestResult
            Complete backtest output including equity curve, metrics,
            trades, and portfolio history.
        """
        # Seed for reproducibility
        set_global_seed(self.config.simulation.seed)

        # Build run context
        run_context = RunContext.create(
            seed=self.config.simulation.seed,
        )

        # Initialize strategy
        self.strategy.initialize(self.data)

        all_fills: list[Fill] = []
        prev_date: Optional[pd.Timestamp] = None

        # Determine symbol: use column if present, else default
        has_symbol_col = "symbol" in self.data.columns

        timestamps = self.data.index
        n_bars = len(timestamps)

        for bar_idx in range(n_bars):
            timestamp = timestamps[bar_idx]
            bar = self.data.iloc[bar_idx]

            # Detect new trading day for daily reset
            current_date = pd.Timestamp(timestamp).normalize()
            if prev_date is not None and current_date != prev_date:
                self.portfolio.new_day()
                self.risk_manager.reset_daily()
            prev_date = current_date

            # Step 1: Process pending orders through broker at this bar
            fills = self.broker.process_bar(bar, current_dt=timestamp)

            # Step 2: Update portfolio and notify strategy for each fill
            for fill in fills:
                self.portfolio.update_fill(
                    symbol=fill.symbol,
                    quantity=fill.quantity,
                    price=fill.price,
                    commission=fill.commission,
                    side_is_buy=(fill.side == OrderSide.BUY),
                )
                self.strategy.on_fill(fill)
                all_fills.append(fill)

            # Step 3: Mark portfolio to market at close prices
            symbol = str(bar["symbol"]) if has_symbol_col else "UNKNOWN"
            close_price = float(bar["close"])
            prices = {symbol: close_price}
            # Include any other held positions at their last known price
            for sym in self.portfolio.positions:
                if sym not in prices:
                    prices[sym] = close_price  # best approximation for single-asset
            self.portfolio.mark_to_market(pd.Timestamp(timestamp), prices)

            # Step 4: Check daily loss halt — skip signal generation if halted
            if self.risk_manager.is_halted:
                continue

            # Step 5: Call strategy.on_bar() to get signal
            history = self.data.iloc[: bar_idx + 1]
            signal = self.strategy.on_bar(
                timestamp=pd.Timestamp(timestamp),
                bar=bar,
                history=history,
            )

            if signal is None:
                continue

            # Step 6: Use position sizer to determine target units
            vol = self._volatility.iloc[bar_idx] if bar_idx < len(self._volatility) else 0.0
            if pd.isna(vol):
                vol = 0.0

            portfolio_state = self.portfolio.get_state(prices)
            equity = portfolio_state["equity"]

            sized_units = self.position_sizer.size(
                signal=signal,
                equity=equity,
                price=close_price,
                volatility=vol,
            )

            if sized_units == 0:
                # If signal says go flat, we still need to close
                if signal.target_position == 0.0:
                    sized_units = 0.0
                else:
                    continue

            # Step 7: Calculate orders needed to reach target position
            current_qty = self.portfolio.positions.get(signal.symbol, 0.0)

            if signal.target_position == 0.0:
                # Flatten position
                target_qty = 0.0
            elif signal.target_position > 0:
                target_qty = abs(sized_units)
            else:
                target_qty = -abs(sized_units)

            delta = target_qty - current_qty

            if abs(delta) < 1e-10:
                continue

            if delta > 0:
                side = OrderSide.BUY
                quantity = delta
            else:
                side = OrderSide.SELL
                quantity = abs(delta)

            order = Order(
                symbol=signal.symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
                created_at=timestamp,
            )

            # Attach stop price from signal metadata if available
            stop_price = signal.metadata.get("stop_price")
            if stop_price is not None:
                order.stop_price = stop_price

            # Step 8: Run order through risk manager
            allowed, reason = self.risk_manager.check_order(order, portfolio_state)

            if not allowed:
                logger.debug(
                    "Order rejected by risk manager: %s (symbol=%s, qty=%.2f)",
                    reason,
                    signal.symbol,
                    quantity,
                )
                continue

            # Step 9: Submit approved order to broker
            self.broker.submit_order(order)

        # ----- Post-loop: compute metrics -----
        equity_curve = self.portfolio.get_equity_curve()
        metrics = compute_metrics(equity_curve)
        portfolio_history = self.portfolio.get_history_df()

        result = BacktestResult(
            equity_curve=equity_curve,
            metrics=metrics,
            trades=all_fills,
            portfolio_history=portfolio_history,
            config=self.config,
            run_context=run_context,
        )

        logger.info(
            "Backtest complete: %d bars, %d trades, total return %.2f%%",
            n_bars,
            len(all_fills),
            metrics.get("total_return", 0.0) * 100,
        )

        return result
