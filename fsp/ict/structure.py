"""ICT market-structure state machine: BOS / CHoCH / MSS detection.

Walks confirmed swing pivots forward in time and classifies every structural
break relative to the prevailing trend:

  BOS   (Break of Structure)    — close beyond the reference swing IN the
                                  direction of the trend. Continuation signal.
  CHoCH (Change of Character)    — close beyond the reference swing AGAINST the
                                  trend. First sign of a reversal.
  MSS   (Market-Structure Shift) — a CHoCH delivered by a displacement candle
                                  (body > atr_mult * ATR). High-conviction
                                  reversal; the entries we actually want.

Causality: a pivot found by ``find_swings(length=L)`` is only *confirmed* L bars
after it prints (it needs L bars on either side). A swing therefore cannot act
as a reference level until bar ``pivot_index + L`` has closed. This keeps the
machine strictly causal — no look-ahead — which matters for honest backtests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from fsp.data.types import Swing
from fsp.structure.swings import find_swings
from fsp.structure.displacement import atr


@dataclass
class StructureEvent:
    ts: datetime
    event_type: Literal["BOS", "CHoCH"]
    direction: Literal["bull", "bear"]
    broken_level: float
    broken_swing_ts: datetime
    close: float
    displacement: bool = False

    @property
    def is_mss(self) -> bool:
        """An MSS is a displacement-backed change of character."""
        return self.event_type == "CHoCH" and self.displacement


@dataclass
class StructureState:
    trend: int = 0  # +1 bull, -1 bear, 0 undefined
    ref_high: float | None = None
    ref_high_ts: datetime | None = None
    ref_low: float | None = None
    ref_low_ts: datetime | None = None
    events: list[StructureEvent] = field(default_factory=list)
    last_event: StructureEvent | None = None

    @property
    def bias(self) -> str:
        return {1: "bull", -1: "bear", 0: "neutral"}[self.trend]

    @property
    def last_mss(self) -> StructureEvent | None:
        for ev in reversed(self.events):
            if ev.is_mss:
                return ev
        return None


def analyze_structure(
    df: pd.DataFrame,
    swing_length: int = 5,
    atr_mult: float = 1.5,
    atr_len: int = 20,
) -> StructureState:
    """Reduce an OHLC frame to a market-structure state.

    ``df`` must have columns open/high/low/close and a (preferably tz-aware UTC)
    DatetimeIndex. Returns a :class:`StructureState` whose ``events`` list is in
    chronological order.
    """
    state = StructureState()
    if len(df) < 2 * swing_length + 1:
        return state

    swings = find_swings(df, length=swing_length)
    if not swings:
        return state

    # Map each swing -> (confirmation_index, swing). A pivot at positional index
    # p is confirmed `swing_length` bars later, at p + swing_length.
    pending: list[tuple[int, Swing]] = []
    for s in swings:
        try:
            pos = df.index.get_loc(pd.Timestamp(s.ts))
        except KeyError:
            continue
        if isinstance(pos, slice):  # duplicate timestamps -> take first
            pos = pos.start
        pending.append((int(pos) + swing_length, s))
    pending.sort(key=lambda t: t[0])

    a = atr(df, atr_len).to_numpy()
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    idx = df.index
    n = len(df)

    ref_high_broken = False
    ref_low_broken = False
    pi = 0  # pointer into `pending`

    for i in range(n):
        # 1) activate every swing confirmed by the close of bar i. The newest
        #    confirmed pivot of each kind becomes the live reference level.
        while pi < len(pending) and pending[pi][0] <= i:
            _, s = pending[pi]
            if s.kind == "high":
                state.ref_high = s.price
                state.ref_high_ts = s.ts
                ref_high_broken = False
            else:
                state.ref_low = s.price
                state.ref_low_ts = s.ts
                ref_low_broken = False
            pi += 1

        # 2) is candle i a displacement? (body dwarfs recent ATR)
        disp = False
        if not np.isnan(a[i]) and a[i] > 0:
            body = abs(c[i] - o[i])
            disp = bool(body > atr_mult * a[i])

        ts_i = idx[i].to_pydatetime()

        # 3) bullish break — close above the reference high
        if state.ref_high is not None and not ref_high_broken and c[i] > state.ref_high:
            etype = "CHoCH" if state.trend == -1 else "BOS"
            ev = StructureEvent(
                ts=ts_i, event_type=etype, direction="bull",
                broken_level=state.ref_high, broken_swing_ts=state.ref_high_ts,
                close=float(c[i]), displacement=disp,
            )
            state.events.append(ev)
            state.last_event = ev
            state.trend = 1
            ref_high_broken = True

        # 4) bearish break — close below the reference low
        elif state.ref_low is not None and not ref_low_broken and c[i] < state.ref_low:
            etype = "CHoCH" if state.trend == 1 else "BOS"
            ev = StructureEvent(
                ts=ts_i, event_type=etype, direction="bear",
                broken_level=state.ref_low, broken_swing_ts=state.ref_low_ts,
                close=float(c[i]), displacement=disp,
            )
            state.events.append(ev)
            state.last_event = ev
            state.trend = -1
            ref_low_broken = True

    return state
