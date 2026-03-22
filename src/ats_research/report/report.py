"""Report generation in Markdown and HTML formats.

Produces self-contained backtest reports that include performance metrics,
configuration details, and embedded plot images.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from jinja2 import Template
except ImportError:  # pragma: no cover
    Template = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_metric_value(key: str, value: Any) -> str:
    """Format a single metric value for display."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if value != value:  # NaN check
            return "N/A"
        if value == float("inf"):
            return "Inf"
        # Percentage-style metrics
        pct_keys = {
            "total_return", "cagr", "max_drawdown", "annual_volatility",
            "hit_rate", "exposure", "avg_win", "avg_loss", "daily_turnover",
        }
        if key in pct_keys:
            return f"{value:.2%}"
        return f"{value:.4f}"
    return str(value)


def _relative_plot_path(plots_dir: str, output_path: str, filename: str) -> str:
    """Compute a relative path from the report file to the plot image."""
    report_parent = Path(output_path).parent
    plot_abs = Path(plots_dir) / filename
    try:
        return str(plot_abs.relative_to(report_parent))
    except ValueError:
        return os.path.relpath(str(plot_abs), str(report_parent))


_DISCLAIMER = (
    "DISCLAIMER: This report is generated for research and informational "
    "purposes only. It does not constitute investment advice. Past performance "
    "is not indicative of future results. Trading involves risk of loss."
)

_PLOT_FILES = [
    ("Equity Curve", "equity_curve.png"),
    ("Drawdown", "drawdown.png"),
    ("Rolling Sharpe", "rolling_sharpe.png"),
    ("Daily Returns Distribution", "returns_distribution.png"),
    ("Monthly Returns Heatmap", "monthly_returns_heatmap.png"),
]


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_markdown_report(
    metrics: dict[str, Any],
    config: dict[str, Any],
    plots_dir: str,
    output_path: str,
) -> str:
    """Generate a Markdown backtest report.

    Parameters
    ----------
    metrics:
        Dictionary of metric name to value (as returned by
        :func:`~ats_research.backtest.metrics.compute_metrics`).
    config:
        Run configuration dictionary.
    plots_dir:
        Directory containing plot PNG files.
    output_path:
        Path to write the Markdown file.

    Returns
    -------
    str
        The full Markdown content.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    run_id = config.get("run_id", "N/A")

    lines: list[str] = []

    # Header & disclaimer
    lines.append("# Backtest Report")
    lines.append("")
    lines.append(f"> **{_DISCLAIMER}**")
    lines.append("")

    # Run info
    lines.append("## Run Information")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| Run ID | {run_id} |")
    lines.append(f"| Generated | {timestamp} |")

    # Summarise key config fields
    for key in ("strategy", "universe", "start_date", "end_date", "frequency"):
        if key in config:
            lines.append(f"| {key.replace('_', ' ').title()} | {config[key]} |")
    lines.append("")

    # Metrics table
    lines.append("## Performance Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for key, value in metrics.items():
        display_name = key.replace("_", " ").title()
        lines.append(f"| {display_name} | {_format_metric_value(key, value)} |")
    lines.append("")

    # Plots
    lines.append("## Charts")
    lines.append("")
    for label, filename in _PLOT_FILES:
        rel_path = _relative_plot_path(plots_dir, output_path, filename)
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"![{label}]({rel_path})")
        lines.append("")

    # Parameters
    lines.append("## Parameters")
    lines.append("")
    lines.append("```yaml")
    for key, value in config.items():
        lines.append(f"{key}: {value}")
    lines.append("```")
    lines.append("")

    content = "\n".join(lines)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(content, encoding="utf-8")

    return content


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backtest Report{{ " — " + run_id if run_id != "N/A" else "" }}</title>
<style>
  :root {
    --bg: #ffffff;
    --fg: #1a1a2e;
    --accent: #0f3460;
    --border: #dee2e6;
    --light-bg: #f8f9fa;
    --red: #c0392b;
    --green: #27ae60;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    color: var(--fg);
    background: var(--bg);
    line-height: 1.6;
    padding: 2rem;
    max-width: 1100px;
    margin: 0 auto;
  }
  h1 { font-size: 1.8rem; color: var(--accent); margin-bottom: 0.5rem; }
  h2 { font-size: 1.3rem; color: var(--accent); margin: 2rem 0 0.75rem; border-bottom: 2px solid var(--border); padding-bottom: 0.3rem; }
  h3 { font-size: 1.1rem; margin: 1.5rem 0 0.5rem; }
  .disclaimer {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 1rem 1.25rem;
    margin: 1rem 0 1.5rem;
    font-size: 0.9rem;
    color: #856404;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.75rem 0;
    font-size: 0.92rem;
  }
  th, td {
    text-align: left;
    padding: 0.55rem 0.75rem;
    border: 1px solid var(--border);
  }
  th { background: var(--light-bg); font-weight: 600; }
  tr:nth-child(even) td { background: var(--light-bg); }
  .plot-section { margin: 1.5rem 0; }
  .plot-section img { max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 4px; }
  .params { background: var(--light-bg); padding: 1rem; border-radius: 4px; font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.85rem; white-space: pre-wrap; overflow-x: auto; }
  .footer { margin-top: 3rem; font-size: 0.8rem; color: #6c757d; text-align: center; }
</style>
</head>
<body>

<h1>Backtest Report</h1>

<div class="disclaimer">{{ disclaimer }}</div>

<h2>Run Information</h2>
<table>
  <tr><th>Field</th><th>Value</th></tr>
  <tr><td>Run ID</td><td>{{ run_id }}</td></tr>
  <tr><td>Generated</td><td>{{ timestamp }}</td></tr>
  {% for key, val in config_summary %}
  <tr><td>{{ key }}</td><td>{{ val }}</td></tr>
  {% endfor %}
</table>

<h2>Performance Metrics</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  {% for name, value in metrics_rows %}
  <tr><td>{{ name }}</td><td>{{ value }}</td></tr>
  {% endfor %}
</table>

<h2>Charts</h2>
{% for label, rel_path in plots %}
<div class="plot-section">
  <h3>{{ label }}</h3>
  <img src="{{ rel_path }}" alt="{{ label }}">
</div>
{% endfor %}

<h2>Parameters</h2>
<div class="params">{% for key, val in config_items %}{{ key }}: {{ val }}
{% endfor %}</div>

<div class="footer">
  Generated by ats_research &middot; {{ timestamp }}
</div>

</body>
</html>
"""


def generate_html_report(
    metrics: dict[str, Any],
    config: dict[str, Any],
    plots_dir: str,
    output_path: str,
) -> str:
    """Generate an HTML backtest report.

    Uses a Jinja2 template (inlined) to produce a professional, self-contained
    HTML page with CSS styling.

    Parameters
    ----------
    metrics:
        Dictionary of metric name to value.
    config:
        Run configuration dictionary.
    plots_dir:
        Directory containing plot PNG files.
    output_path:
        Path to write the HTML file.

    Returns
    -------
    str
        The full HTML content.

    Raises
    ------
    ImportError
        If Jinja2 is not installed.
    """
    if Template is None:
        raise ImportError(
            "Jinja2 is required for HTML report generation. "
            "Install it with: pip install jinja2"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    run_id = config.get("run_id", "N/A")

    # Config summary rows
    config_summary: list[tuple[str, str]] = []
    for key in ("strategy", "universe", "start_date", "end_date", "frequency"):
        if key in config:
            config_summary.append((key.replace("_", " ").title(), str(config[key])))

    # Metrics rows
    metrics_rows: list[tuple[str, str]] = []
    for key, value in metrics.items():
        display_name = key.replace("_", " ").title()
        metrics_rows.append((display_name, _format_metric_value(key, value)))

    # Plot relative paths
    plots: list[tuple[str, str]] = []
    for label, filename in _PLOT_FILES:
        rel_path = _relative_plot_path(plots_dir, output_path, filename)
        plots.append((label, rel_path))

    # Config items for params section
    config_items = list(config.items())

    template = Template(_HTML_TEMPLATE)
    content = template.render(
        disclaimer=_DISCLAIMER,
        run_id=run_id,
        timestamp=timestamp,
        config_summary=config_summary,
        metrics_rows=metrics_rows,
        plots=plots,
        config_items=config_items,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(content, encoding="utf-8")

    return content
