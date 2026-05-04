"""Common dataclasses shared across the engine."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal

import pandas as pd

TF = Literal["M1", "M5", "M15", "M30", "H1", "H4", "D", "W", "MN"]
Pair = Literal["EURUSD", "GBPUSD", "DXY", "USDJPY"]
Direction = Literal["long", "short"]


class Session(str, Enum):
    ASIA = "ASIA"
    LONDON = "LO"
    NY_AM = "NY-AM"
    LUNCH = "LUNCH"
    NY_PM = "NY-PM"
    OFF = "OFF"


class Cycle(str, Enum):
    EXPANSION = "EXPANSION"
    CONSOLIDATION = "CONSOLIDATION"
    NEUTRAL = "NEUTRAL"


class OFBias(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"


class Grade(str, Enum):
    A_PLUS = "A+"
    A = "A"
    B = "B"
    SKIP = "SKIP"


@dataclass(frozen=True)
class Candle:
    ts: datetime
    o: float
    h: float
    l: float
    c: float
    v: float = 0.0


@dataclass
class Swing:
    ts: datetime
    price: float
    kind: Literal["high", "low"]
    strong: bool  # took opposing liquidity before forming
    broken: bool = False


@dataclass
class FVG:
    ts: datetime       # bar that completed the 3-bar pattern
    top: float
    bottom: float
    direction: Literal["bull", "bear"]
    tf: TF
    mitigated: bool = False
    inverted: bool = False  # IFVG after a close through


@dataclass
class OrderBlock:
    ts: datetime
    top: float
    bottom: float
    direction: Literal["bull", "bear"]
    tf: TF
    mitigated: bool = False


@dataclass
class Level:
    """A tracked liquidity level with provenance."""
    price: float
    label: str        # e.g. "PDH", "LOH", "WeakHigh", "EQH"
    kind: Literal["high", "low"]
    ts: datetime
    swept: bool = False


def candles_to_df(candles: list[Candle]) -> pd.DataFrame:
    return pd.DataFrame(
        [(c.ts, c.o, c.h, c.l, c.c, c.v) for c in candles],
        columns=["ts", "open", "high", "low", "close", "volume"],
    ).set_index("ts")
