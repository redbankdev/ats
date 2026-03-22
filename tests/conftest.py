"""Shared fixtures for ats_research tests."""

import pytest
import pandas as pd

from ats_research.data.loaders import generate_demo_data
from ats_research.utils.config import (
    ATSConfig,
    DataConfig,
    StrategyConfig,
    RiskConfig,
    CostsConfig,
    SimulationConfig,
    OutputConfig,
)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Generate a valid OHLCV DataFrame with 100 bars."""
    return generate_demo_data(n_bars=100, freq="1D", seed=42)


@pytest.fixture
def sample_config(tmp_path) -> ATSConfig:
    """Return a valid ATSConfig for testing."""
    return ATSConfig(
        data=DataConfig(path=str(tmp_path / "data.csv")),
        strategy=StrategyConfig(name="sma_crossover", params={"fast": 10, "slow": 30}),
        risk=RiskConfig(
            max_pos_pct=0.10,
            per_trade_risk_pct=0.01,
            daily_loss_limit_pct=0.05,
            vol_target_pct=0.10,
        ),
        costs=CostsConfig(slippage_bps=5.0, commission_per_share=0.005),
        simulation=SimulationConfig(seed=42, liquidity_cap_pct_adv=0.02),
        output=OutputConfig(dir=str(tmp_path / "runs")),
    )
