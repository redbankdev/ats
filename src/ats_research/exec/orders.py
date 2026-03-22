"""Order and fill data models for execution simulation."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class OrderType(Enum):
    """Supported order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(Enum):
    """Direction of an order."""

    BUY = "buy"
    SELL = "sell"


class TimeInForce(Enum):
    """Duration policy for an order."""

    GTC = "gtc"  # good till cancelled
    DAY = "day"  # expires end of day


class OrderStatus(Enum):
    """Lifecycle status of an order."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a single order submitted to the simulated broker.

    Parameters
    ----------
    symbol : str
        Ticker or instrument identifier.
    side : OrderSide
        Buy or sell.
    quantity : float
        Number of shares/units (always positive).
    order_type : OrderType
        Market, limit, or stop.
    limit_price : float | None
        Required for limit orders.
    stop_price : float | None
        Required for stop orders.
    time_in_force : TimeInForce
        GTC or DAY.
    order_id : str
        Unique identifier (auto-generated if omitted).
    created_at : datetime | None
        Timestamp when the order was created.
    status : OrderStatus
        Current lifecycle status.
    filled_quantity : float
        Cumulative quantity filled so far.
    filled_price : float
        Volume-weighted average fill price.
    commission : float
        Cumulative commission paid.
    slippage : float
        Cumulative slippage cost.
    """

    symbol: str
    side: OrderSide
    quantity: float  # always positive
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Fill:
    """Record of a single execution event.

    Parameters
    ----------
    order_id : str
        The order that generated this fill.
    symbol : str
        Instrument identifier.
    side : OrderSide
        Buy or sell.
    quantity : float
        Number of shares/units filled in this event.
    price : float
        Execution price (after slippage adjustment).
    commission : float
        Commission charged for this fill.
    slippage : float
        Slippage cost incurred in this fill.
    timestamp : datetime
        Time the fill occurred.
    """

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float
    timestamp: datetime
