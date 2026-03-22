"""CLI entry point for ATS Research.

Commands:
    atsr backtest --config config.yaml
    atsr paper --config config.yaml --stream data/demo_stream.csv
    atsr report --run <run_id>

DISCLAIMER: NOT FINANCIAL ADVICE — EDUCATIONAL AND RESEARCH USE ONLY.
"""

import json
import sys
from pathlib import Path

import click

import ats_research

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║              ATS Research — Trading Research System          ║
║                                                              ║
║  ⚠  NOT FINANCIAL ADVICE — EDUCATIONAL USE ONLY  ⚠          ║
║  No live trading. No real-money brokerage connections.        ║
║  Simulation and paper trading ONLY.                          ║
╚══════════════════════════════════════════════════════════════╝
"""


@click.group()
@click.version_option(version=ats_research.__version__, prog_name="atsr")
def cli():
    """ATS Research — Automated Trading Research System.

    DISCLAIMER: NOT FINANCIAL ADVICE. EDUCATIONAL AND RESEARCH USE ONLY.
    No live trading or real-money brokerage connections.
    """
    click.echo(BANNER)


@cli.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML/JSON config file.",
)
@click.option("--output", "output_dir", default=None, help="Override output directory.")
def backtest(config_path: str, output_dir: str | None):
    """Run a backtest with the given configuration."""
    from ats_research.backtest.engine import BacktestEngine
    from ats_research.data.loaders import load_ohlcv
    from ats_research.report.plots import generate_all_plots
    from ats_research.report.report import generate_html_report, generate_markdown_report
    from ats_research.utils.config import load_config
    from ats_research.utils.logging import get_logger, setup_logging
    from ats_research.utils.repro import generate_run_id, set_global_seed

    config = load_config(config_path)
    run_id = generate_run_id()

    if output_dir:
        config.output.dir = output_dir

    out = Path(config.output.dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "plots"
    plots_dir.mkdir(exist_ok=True)

    setup_logging(run_id)
    logger = get_logger(__name__)

    logger.info("Starting backtest run %s", run_id)
    logger.info("Config: %s", config_path)

    # Seed for reproducibility
    set_global_seed(config.simulation.seed)

    # Load data
    click.echo(f"Loading data from {config.data.path} ...")
    data = load_ohlcv(config.data.path, freq=config.data.freq, tz=config.data.tz)
    click.echo(f"Loaded {len(data)} bars from {data.index[0]} to {data.index[-1]}")

    # Resolve strategy
    strategy = _resolve_strategy(config.strategy.name, config.strategy.params)
    click.echo(f"Strategy: {strategy}")

    # Run engine
    click.echo("Running backtest ...")
    engine = BacktestEngine(config=config, strategy=strategy, data=data)
    result = engine.run()

    click.echo(f"\n{'='*60}")
    click.echo(result.summary())

    # Save outputs
    result.to_json(out / "metrics.json")
    result.to_csv(out / "equity_curve.csv")

    # Save config
    with open(out / "config.json", "w") as f:
        json.dump(config.model_dump(), f, indent=2, default=str)

    # Generate plots
    if config.output.save_plots and len(result.equity_curve) > 1:
        click.echo("Generating plots ...")
        generate_all_plots(result.equity_curve, str(plots_dir))

    # Generate reports
    click.echo("Generating reports ...")
    generate_markdown_report(
        metrics=result.metrics,
        config=config.model_dump(),
        plots_dir=str(plots_dir),
        output_path=str(out / "report.md"),
    )
    generate_html_report(
        metrics=result.metrics,
        config=config.model_dump(),
        plots_dir=str(plots_dir),
        output_path=str(out / "report.html"),
    )

    click.echo(f"\nOutputs saved to: {out}")
    click.echo(f"  - metrics.json")
    click.echo(f"  - equity_curve.csv")
    click.echo(f"  - report.md / report.html")
    click.echo(f"  - plots/ (equity curve, drawdown, etc.)")
    logger.info("Backtest run %s complete", run_id)


@cli.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML/JSON config file.",
)
@click.option(
    "--stream",
    "stream_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to streaming data CSV (simulates real-time bar arrival).",
)
def paper(config_path: str, stream_path: str | None):
    """Run paper trading simulation (no real money, no live APIs).

    Reuses the backtest engine in streaming mode, processing one bar
    at a time to simulate real-time paper trading.
    """
    import time

    from ats_research.backtest.engine import BacktestEngine
    from ats_research.data.loaders import load_ohlcv
    from ats_research.utils.config import load_config
    from ats_research.utils.logging import get_logger, setup_logging
    from ats_research.utils.repro import generate_run_id, set_global_seed

    config = load_config(config_path)
    run_id = generate_run_id()

    out = Path(config.output.dir) / run_id / "paper"
    out.mkdir(parents=True, exist_ok=True)

    setup_logging(run_id)
    logger = get_logger(__name__)

    set_global_seed(config.simulation.seed)

    # Load stream data
    data_path = stream_path or config.data.path
    click.echo(f"Loading stream data from {data_path} ...")
    data = load_ohlcv(data_path, freq=config.data.freq, tz=config.data.tz)
    click.echo(f"Paper trading {len(data)} bars (simulated streaming)")

    strategy = _resolve_strategy(config.strategy.name, config.strategy.params)

    # Run as a streaming backtest (bar-by-bar with output)
    engine = BacktestEngine(config=config, strategy=strategy, data=data)

    click.echo("\n--- Paper Trading Session ---")
    click.echo("(Simulated — no real money or live connections)\n")

    strategy.initialize(data)

    for i, (ts, bar) in enumerate(data.iterrows()):
        # Process one bar at a time
        fills = engine.broker.process_bar(bar, ts)
        for fill in fills:
            strategy.on_fill(fill)
            engine.portfolio.update_fill(
                fill.symbol, fill.quantity, fill.price, fill.commission,
                fill.side.value == "buy"
            )

        prices = {config.strategy.params.get("symbol", "SIM"): bar["close"]}
        engine.portfolio.mark_to_market(ts, prices)

        # Get signal
        history = data.iloc[: i + 1]
        signal = strategy.on_bar(ts, bar, history)

        if signal:
            click.echo(
                f"[{ts}] Signal: target={signal.target_position:.2f} "
                f"| Equity={engine.portfolio.equity:.2f}"
            )

        # Every 50 bars, print status
        if (i + 1) % 50 == 0:
            click.echo(
                f"[{ts}] Bar {i+1}/{len(data)} | "
                f"Equity: {engine.portfolio.equity:,.2f} | "
                f"Cash: {engine.portfolio.cash:,.2f}"
            )

    click.echo(f"\n--- Paper Trading Complete ---")
    click.echo(f"Final equity: {engine.portfolio.equity:,.2f}")
    click.echo(f"Outputs saved to: {out}")

    # Save final state
    history_df = engine.portfolio.get_history_df()
    if not history_df.empty:
        history_df.to_csv(out / "portfolio_history.csv")

    logger.info("Paper trading run %s complete", run_id)


@cli.command()
@click.option("--run", "run_id", required=True, help="Run ID to generate report for.")
@click.option("--output-dir", default="runs", help="Base output directory.")
def report(run_id: str, output_dir: str):
    """Generate or regenerate reports for a completed run."""
    import pandas as pd

    from ats_research.backtest.metrics import compute_metrics
    from ats_research.report.plots import generate_all_plots
    from ats_research.report.report import generate_html_report, generate_markdown_report

    run_dir = Path(output_dir) / run_id
    if not run_dir.exists():
        click.echo(f"Error: Run directory not found: {run_dir}", err=True)
        sys.exit(1)

    # Load equity curve
    eq_path = run_dir / "equity_curve.csv"
    if not eq_path.exists():
        click.echo(f"Error: equity_curve.csv not found in {run_dir}", err=True)
        sys.exit(1)

    eq = pd.read_csv(eq_path, index_col=0, parse_dates=True).squeeze()
    metrics = compute_metrics(eq)

    # Load config if available
    config = {}
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    generate_all_plots(eq, str(plots_dir))
    generate_markdown_report(metrics, config, str(plots_dir), str(run_dir / "report.md"))
    generate_html_report(metrics, config, str(plots_dir), str(run_dir / "report.html"))

    click.echo(f"Reports regenerated in {run_dir}")


def _resolve_strategy(name: str, params: dict):
    """Resolve strategy class by name and instantiate with params."""
    from ats_research.strategy.sma_crossover import SMACrossover

    strategies = {
        "sma_crossover": SMACrossover,
        "SMACrossover": SMACrossover,
    }

    cls = strategies.get(name)
    if cls is None:
        raise click.ClickException(
            f"Unknown strategy: {name}. Available: {list(strategies.keys())}"
        )

    return cls(params=params)


if __name__ == "__main__":
    cli()
