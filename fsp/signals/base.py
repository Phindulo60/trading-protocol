"""Shared Signal dataclass for intraday strategies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

StrategyName = Literal["ECM", "ARB", "TREND_RSI", "LEVEL_OB", "ASIA_HL", "ICT_SHADOW"]


@dataclass
class Signal:
    strategy: StrategyName
    pair: str
    direction: str          # "long" | "short"
    entry: float
    sl: float
    tp1: float
    tp2: float | None
    inv_pips: float
    rr_tp1: float
    rr_tp2: float | None
    risk_r: float           # suggested R (1.0 or 0.5)
    note: str               # brief human-readable reason
    ts: str                 # ISO UTC timestamp of the signal bar
    context: dict = field(default_factory=dict)

    @property
    def dedup_key(self) -> str:
        return f"{self.pair}|{self.strategy}|{self.direction}|{self.entry:.5f}"
