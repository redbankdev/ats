"""Commission models for execution simulation.

Each model computes the dollar cost of commissions for a given fill
quantity and price.
"""

from abc import ABC, abstractmethod
from typing import List


class CommissionModel(ABC):
    """Abstract base class for commission models."""

    @abstractmethod
    def calculate(self, quantity: float, price: float) -> float:
        """Return the commission cost for a fill.

        Parameters
        ----------
        quantity : float
            Number of shares/units filled.
        price : float
            Execution price per share/unit.

        Returns
        -------
        float
            Dollar commission cost (non-negative).
        """


class ZeroCommission(CommissionModel):
    """No commission charged."""

    def calculate(self, quantity: float, price: float) -> float:
        """Return zero."""
        return 0.0


class PerShareCommission(CommissionModel):
    """Flat rate per share with an optional minimum.

    Parameters
    ----------
    rate : float
        Dollar cost per share (default ``0.005``).
    min_commission : float
        Minimum commission per fill (default ``0.0``).
    """

    def __init__(self, rate: float = 0.005, min_commission: float = 0.0) -> None:
        if rate < 0:
            raise ValueError("rate must be non-negative")
        if min_commission < 0:
            raise ValueError("min_commission must be non-negative")
        self.rate = rate
        self.min_commission = min_commission

    def calculate(self, quantity: float, price: float) -> float:
        """Return ``max(rate * quantity, min_commission)``."""
        return max(self.rate * abs(quantity), self.min_commission)


class PercentCommission(CommissionModel):
    """Commission as a percentage of notional trade value.

    Parameters
    ----------
    rate : float
        Fractional rate (default ``0.001`` = 0.1 %).
    """

    def __init__(self, rate: float = 0.001) -> None:
        if rate < 0:
            raise ValueError("rate must be non-negative")
        self.rate = rate

    def calculate(self, quantity: float, price: float) -> float:
        """Return ``rate * quantity * price``."""
        return self.rate * abs(quantity) * price


class CompositeCommission(CommissionModel):
    """Sum of multiple commission models.

    Parameters
    ----------
    models : list[CommissionModel]
        Child models whose results are summed.
    """

    def __init__(self, models: List[CommissionModel]) -> None:
        if not models:
            raise ValueError("models list must not be empty")
        self.models = list(models)

    def calculate(self, quantity: float, price: float) -> float:
        """Return the sum of commissions from all child models."""
        return sum(m.calculate(quantity, price) for m in self.models)
