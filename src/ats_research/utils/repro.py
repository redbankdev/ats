"""
Reproducibility utilities — seeding, dependency auditing, and run context.

Usage::

    from ats_research.utils.repro import set_global_seed, generate_run_id, RunContext

    set_global_seed(42)
    ctx = RunContext.create(seed=42)
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import platform
import random
import secrets
from dataclasses import dataclass, field
from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """
    Set deterministic seeds for ``random``, ``numpy``, and Python hash
    randomisation.

    Parameters
    ----------
    seed:
        Non-negative integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 — intentional global seed
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Dependency pin checking
# ---------------------------------------------------------------------------

def _parse_pyproject_deps(pyproject_path: Path) -> dict[str, str]:
    """
    Extract ``dependencies`` from *pyproject.toml* and return a mapping of
    package-name -> version-spec string (e.g. ``">=2.1.0,<3.0"``).
    """
    deps: dict[str, str] = {}
    in_deps = False
    for line in pyproject_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "dependencies = [":
            in_deps = True
            continue
        if in_deps:
            if stripped == "]":
                break
            # Lines look like:  "pandas>=2.1.0,<3.0",
            raw = stripped.strip('",').strip()
            if not raw:
                continue
            # Split on the first version specifier character.
            for i, ch in enumerate(raw):
                if ch in ">=<!=~":
                    name = raw[:i].strip()
                    spec = raw[i:].strip()
                    deps[name.lower()] = spec
                    break
            else:
                # No version pin at all.
                deps[raw.lower()] = ""
    return deps


def check_dependency_pins(
    pyproject_path: Optional[str | Path] = None,
) -> list[str]:
    """
    Compare installed package versions against the pins declared in
    ``pyproject.toml``.

    Parameters
    ----------
    pyproject_path:
        Explicit path to ``pyproject.toml``.  When *None* the function
        walks upward from this file to locate the project root.

    Returns
    -------
    list[str]
        Human-readable warnings for each mismatch.  Empty when everything
        lines up (or the file cannot be found).
    """
    if pyproject_path is None:
        candidate = Path(__file__).resolve().parent
        while candidate != candidate.parent:
            if (candidate / "pyproject.toml").exists():
                pyproject_path = candidate / "pyproject.toml"
                break
            candidate = candidate.parent
        if pyproject_path is None:
            logger.debug("pyproject.toml not found — skipping dependency check")
            return []

    pyproject_path = Path(pyproject_path)
    if not pyproject_path.exists():
        logger.debug("pyproject.toml not found at %s", pyproject_path)
        return []

    pinned = _parse_pyproject_deps(pyproject_path)
    warnings: list[str] = []

    for pkg, spec in pinned.items():
        try:
            inst_ver = installed_version(pkg)
        except Exception:
            warnings.append(f"{pkg}: not installed (expected {spec})")
            continue

        if not spec:
            continue

        # Light-weight check: use packaging if available, otherwise warn.
        try:
            from packaging.specifiers import SpecifierSet
            from packaging.version import Version

            if not SpecifierSet(spec).contains(Version(inst_ver)):
                warnings.append(f"{pkg}: installed {inst_ver} does not match {spec}")
        except ImportError:
            # packaging not available — just log the versions for manual review.
            warnings.append(
                f"{pkg}: installed {inst_ver}, pinned {spec} (install 'packaging' for auto-check)"
            )

    for w in warnings:
        logger.warning("Dependency mismatch — %s", w)

    return warnings


# ---------------------------------------------------------------------------
# Run ID generation
# ---------------------------------------------------------------------------

def generate_run_id() -> str:
    """
    Create a unique run identifier: ``YYYYMMDD_HHMMSS_<hex8>``.

    The timestamp gives rough chronological ordering; the random suffix
    avoids collisions when runs launch within the same second.
    """
    ts = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(4)  # 8 hex chars
    return f"{ts}_{suffix}"


# ---------------------------------------------------------------------------
# Run context dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunContext:
    """Immutable snapshot of everything needed to reproduce a run."""

    run_id: str
    seed: int
    python_version: str
    package_version: str
    timestamp: str

    @classmethod
    def create(
        cls,
        seed: int = 42,
        run_id: Optional[str] = None,
    ) -> "RunContext":
        """
        Build a :class:`RunContext` capturing the current environment.

        Parameters
        ----------
        seed:
            RNG seed used for this run.
        run_id:
            Explicit run ID.  Generated automatically when *None*.
        """
        from ats_research import __version__

        return cls(
            run_id=run_id or generate_run_id(),
            seed=seed,
            python_version=platform.python_version(),
            package_version=__version__,
            timestamp=dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        )
