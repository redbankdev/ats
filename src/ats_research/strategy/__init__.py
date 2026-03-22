"""Strategy interface and reference implementations."""

from ats_research.strategy.base import Signal, Strategy
from ats_research.strategy.sma_crossover import SMACrossover

__all__ = ["Signal", "Strategy", "SMACrossover"]
