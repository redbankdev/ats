"""
Pydantic-based configuration with YAML/JSON loading and env-var overrides.

Usage::

    from ats_research.utils.config import load_config
    cfg = load_config("config.yaml")
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Freq(str, Enum):
    """Supported bar frequencies."""
    ONE_MIN = "1min"
    ONE_DAY = "1day"
    # Aliases for common pandas frequency strings
    ONE_MIN_T = "1T"
    ONE_DAY_D = "1D"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class DataConfig(BaseModel):
    """Data source configuration."""
    path: str
    freq: Freq = Freq.ONE_DAY
    tz: str = "UTC"


class StrategyConfig(BaseModel):
    """Strategy identification and parameters."""
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class RiskConfig(BaseModel):
    """Risk guardrails — all expressed as fractions (0–1)."""
    max_pos_pct: float = Field(default=0.10, ge=0.0, le=1.0)
    per_trade_risk_pct: float = Field(default=0.01, ge=0.0, le=1.0)
    daily_loss_limit_pct: float = Field(default=0.05, ge=0.0, le=1.0)
    vol_target_pct: float = Field(default=0.10, ge=0.0, le=1.0)


class CostsConfig(BaseModel):
    """Transaction-cost modelling."""
    slippage_bps: float = Field(default=5.0, ge=0.0)
    commission_per_share: float = Field(default=0.005, ge=0.0)
    commission_pct: float = Field(default=0.0, ge=0.0)


class SimulationConfig(BaseModel):
    """Simulation / backtest parameters."""
    seed: int = Field(default=42, ge=0)
    liquidity_cap_pct_adv: float = Field(default=0.02, ge=0.0, le=1.0)


class OutputConfig(BaseModel):
    """Where and how results are persisted."""
    dir: str = "runs"
    save_plots: bool = True


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

_MAX_LEVERAGE: float = 1.0  # hard cap — no margin


class ATSConfig(BaseModel):
    """
    Master configuration combining every sub-config.

    Validation ensures risk guardrails are present and leverage stays <= 1.
    """
    data: DataConfig
    strategy: StrategyConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _enforce_guardrails(self) -> "ATSConfig":
        # Refuse to run without risk guardrails
        risk = self.risk
        if risk.max_pos_pct <= 0:
            raise ValueError("risk.max_pos_pct must be > 0 — risk guardrails are required")
        if risk.daily_loss_limit_pct <= 0:
            raise ValueError("risk.daily_loss_limit_pct must be > 0 — risk guardrails are required")

        # Enforce hard leverage cap
        implied_leverage = 1.0 / risk.max_pos_pct if risk.max_pos_pct > 0 else float("inf")
        # max_pos_pct=0.10 implies up to 10 independent positions — that's
        # fine. What we block is *explicit* leverage > 1× equity in a single
        # position, which would require max_pos_pct > 1.0.  The field
        # constraint (le=1.0) already blocks that.  As an additional guard we
        # check the strategy params for an explicit leverage key.
        strategy_leverage = float(self.strategy.params.get("leverage", 1.0))
        if strategy_leverage > _MAX_LEVERAGE:
            raise ValueError(
                f"Leverage {strategy_leverage} exceeds hard cap of {_MAX_LEVERAGE}. "
                "Leveraged strategies are not permitted."
            )
        return self


# ---------------------------------------------------------------------------
# Env-var overrides
# ---------------------------------------------------------------------------

def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Apply environment-variable overrides using ``ATS_`` prefix.

    Mapping (case-insensitive env vars):

    * ``ATS_DATA__PATH``           -> data.path
    * ``ATS_RISK__MAX_POS_PCT``    -> risk.max_pos_pct
    * ``ATS_OUTPUT__DIR``           -> output.dir

    Double-underscore separates nesting levels.
    """
    prefix = "ATS_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("__")
        target = raw
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return raw


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> ATSConfig:
    """
    Read a YAML or JSON configuration file and return a validated
    :class:`ATSConfig`.

    Environment variables prefixed with ``ATS_`` override file values (see
    :func:`_apply_env_overrides`).

    Parameters
    ----------
    path:
        Filesystem path to a ``.yaml``, ``.yml``, or ``.json`` config file.

    Returns
    -------
    ATSConfig
        Fully validated configuration object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file extension is unsupported or validation fails.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    text = filepath.read_text(encoding="utf-8")

    if filepath.suffix in (".yaml", ".yml"):
        raw: dict[str, Any] = yaml.safe_load(text) or {}
    elif filepath.suffix == ".json":
        raw = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {filepath.suffix!r} (use .yaml or .json)")

    raw = _apply_env_overrides(raw)
    return ATSConfig.model_validate(raw)
