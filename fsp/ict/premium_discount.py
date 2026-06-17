"""ICT premium / discount / OTE within a dealing range.

A *dealing range* is bounded by the most recent confirmed swing high and swing
low. Its 50% midpoint is *equilibrium*:

  - above equilibrium = **premium** (expensive — favour selling)
  - below equilibrium = **discount** (cheap — favour buying)

The *OTE* (Optimal Trade Entry) is the 0.62–0.79 retracement of the impulse
leg (the Fibonacci 'golden zone', sweet spot 0.705). For a bullish leg the OTE
sits in discount; for a bearish leg, in premium. The engine enters only when a
setup retraces into discount/OTE (for longs) or premium/OTE (for shorts).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from fsp.structure.swings import find_swings

OTE_LOW = 0.62
OTE_HIGH = 0.79
OTE_SWEET = 0.705


@dataclass
class DealingRange:
    high: float
    high_ts: datetime
    low: float
    low_ts: datetime

    @property
    def size(self) -> float:
        return self.high - self.low

    @property
    def equilibrium(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def direction(self) -> str:
        """Impulse-leg direction: 'bull' if the low formed before the high
        (last leg up), else 'bear'."""
        return "bull" if self.low_ts < self.high_ts else "bear"

    def position(self, price: float) -> float:
        """Where `price` sits in the range: 0.0 = low, 1.0 = high (may exceed)."""
        if self.size <= 0:
            return 0.5
        return (price - self.low) / self.size

    def zone(self, price: float, eq_band: float = 0.0) -> str:
        p = self.position(price)
        if p > 0.5 + eq_band:
            return "premium"
        if p < 0.5 - eq_band:
            return "discount"
        return "equilibrium"

    def in_discount(self, price: float, eq_band: float = 0.0) -> bool:
        return self.zone(price, eq_band) == "discount"

    def in_premium(self, price: float, eq_band: float = 0.0) -> bool:
        return self.zone(price, eq_band) == "premium"

    def ote(self, direction: str | None = None) -> tuple[float, float]:
        """(lower, upper) price bounds of the 0.62–0.79 OTE band."""
        d = direction or self.direction
        if d == "bull":  # retrace down into discount
            return (self.high - OTE_HIGH * self.size, self.high - OTE_LOW * self.size)
        return (self.low + OTE_LOW * self.size, self.low + OTE_HIGH * self.size)

    def ote_sweet_spot(self, direction: str | None = None) -> float:
        d = direction or self.direction
        if d == "bull":
            return self.high - OTE_SWEET * self.size
        return self.low + OTE_SWEET * self.size

    def in_ote(self, price: float, direction: str | None = None) -> bool:
        lo, hi = self.ote(direction)
        return lo <= price <= hi


def dealing_range(df: pd.DataFrame, swing_length: int = 5) -> DealingRange | None:
    """Build the dealing range from the most recent confirmed swing high & low."""
    swings = find_swings(df, length=swing_length)
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if not highs or not lows:
        return None
    h, l = highs[-1], lows[-1]
    if h.price <= l.price:  # degenerate / inverted range
        return None
    return DealingRange(high=h.price, high_ts=h.ts, low=l.price, low_ts=l.ts)
