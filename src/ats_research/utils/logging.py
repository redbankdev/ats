"""
Structured logging with run-ID propagation.

Usage::

    from ats_research.utils.logging import setup_logging, get_logger

    setup_logging(run_id="20260322_abc123", level="DEBUG")
    log = get_logger(__name__)
    log.info("backtest started", extra={"symbols": 42})
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Custom formatter
# ---------------------------------------------------------------------------

_LOG_FMT = "%(asctime)s | %(run_id)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%dT%H:%M:%S%z"

# Sentinel used when no run_id has been injected yet.
_DEFAULT_RUN_ID = "no_run_id"


class _RunIdFilter(logging.Filter):
    """Inject *run_id* into every log record."""

    def __init__(self, run_id: str = _DEFAULT_RUN_ID) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if not hasattr(record, "run_id"):
            record.run_id = self.run_id  # type: ignore[attr-defined]
        return True


# Module-level state so that `get_logger` can attach the current filter.
_current_filter: _RunIdFilter = _RunIdFilter()
_initialized: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logging(
    run_id: str = _DEFAULT_RUN_ID,
    level: str | int = "INFO",
) -> None:
    """
    Configure the root ``ats_research`` logger with structured formatting.

    Parameters
    ----------
    run_id:
        Unique identifier for the current run.  Appears in every log line.
    level:
        Logging level — name (``"DEBUG"``, ``"INFO"``, …) or ``int``.
    """
    global _current_filter, _initialized  # noqa: PLW0603

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger("ats_research")
    root.setLevel(level)

    # Remove previous handlers set up by us (idempotent re-init).
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for filt in list(root.filters):
        root.removeFilter(filt)

    _current_filter = _RunIdFilter(run_id)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    formatter = logging.Formatter(fmt=_LOG_FMT, datefmt=_DATE_FMT)
    handler.setFormatter(formatter)
    handler.addFilter(_current_filter)

    root.addHandler(handler)
    root.addFilter(_current_filter)

    # Prevent duplicate messages from bubbling up to the root logger.
    root.propagate = False
    _initialized = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a child logger under ``ats_research``.

    If *name* already starts with ``ats_research`` it is used as-is;
    otherwise it is prefixed automatically.

    Parameters
    ----------
    name:
        Logger name.  Typically ``__name__`` from the calling module.

    Returns
    -------
    logging.Logger
    """
    if name is None:
        name = "ats_research"
    elif not name.startswith("ats_research"):
        name = f"ats_research.{name}"

    logger = logging.getLogger(name)

    # Ensure the run-id filter is present even if setup_logging hasn't been
    # called yet (graceful degradation).
    if not any(isinstance(f, _RunIdFilter) for f in logger.filters):
        logger.addFilter(_current_filter)

    return logger
