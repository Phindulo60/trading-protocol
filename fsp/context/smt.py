"""SMT (Smart Money Technique) divergence detector.

Two correlation regimes we use:
  POSITIVE pair:  EURUSD ↔ GBPUSD    (both move together vs USD)
  NEGATIVE pair:  EURUSD ↔ DXY       (USD strength index inverse to EUR)

SMT divergence at a SWING HIGH (positive pair):
  Asset A prints a new HH while asset B fails to (makes LH) → bearish SMT
  (smart money not supporting the new high).

SMT divergence at a SWING LOW (positive pair):
  Asset A prints a new LL while asset B holds (makes HL) → bullish SMT.

For a NEGATIVE pair (EUR vs DXY):
  EUR new LL + DXY fails to make new HH = bullish SMT for EUR.
  EUR new HH + DXY fails to make new LL = bearish SMT for EUR.

We return the most recent SMT signals aligned in time, matched on nearest swing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import pandas as pd

from fsp.structure.swings import find_swings


@dataclass
class SMTEvent:
    ts: datetime
    pair_a: str          # reference asset (the one we're trading)
    pair_b: str          # correlated asset
    kind: Literal["bull", "bear"]
    at: Literal["high", "low"]
    note: str


def _closest_swing(swings, ts, tol_minutes: int = 180):
    best = None
    best_dt = None
    for s in swings:
        dt = abs((s.ts - ts).total_seconds())
        if dt <= tol_minutes * 60 and (best_dt is None or dt < best_dt):
            best = s
            best_dt = dt
    return best


def detect_smt_positive(df_a: pd.DataFrame, df_b: pd.DataFrame,
                        pair_a: str, pair_b: str,
                        length: int = 3, tol_minutes: int = 180) -> list[SMTEvent]:
    sa = find_swings(df_a, length=length)
    sb = find_swings(df_b, length=length)
    if len(sa) < 2 or len(sb) < 2:
        return []

    events: list[SMTEvent] = []

    # Walk A's swings; for each kind, find the prior same-kind on A and matching B swings
    highs_a = [s for s in sa if s.kind == "high"]
    lows_a = [s for s in sa if s.kind == "low"]

    for i in range(1, len(highs_a)):
        cur, prev = highs_a[i], highs_a[i - 1]
        if cur.price <= prev.price:
            continue  # not a new HH on A
        b_cur = _closest_swing([s for s in sb if s.kind == "high"], cur.ts, tol_minutes)
        b_prev = _closest_swing([s for s in sb if s.kind == "high"], prev.ts, tol_minutes)
        if b_cur is None or b_prev is None:
            continue
        if b_cur.price < b_prev.price:
            events.append(SMTEvent(cur.ts, pair_a, pair_b, "bear", "high",
                                   f"{pair_a} new HH @{cur.price:.5f}, {pair_b} LH"))

    for i in range(1, len(lows_a)):
        cur, prev = lows_a[i], lows_a[i - 1]
        if cur.price >= prev.price:
            continue
        b_cur = _closest_swing([s for s in sb if s.kind == "low"], cur.ts, tol_minutes)
        b_prev = _closest_swing([s for s in sb if s.kind == "low"], prev.ts, tol_minutes)
        if b_cur is None or b_prev is None:
            continue
        if b_cur.price > b_prev.price:
            events.append(SMTEvent(cur.ts, pair_a, pair_b, "bull", "low",
                                   f"{pair_a} new LL @{cur.price:.5f}, {pair_b} HL"))

    return sorted(events, key=lambda e: e.ts)


def detect_smt_negative(df_a: pd.DataFrame, df_b: pd.DataFrame,
                        pair_a: str, pair_b: str,
                        length: int = 3, tol_minutes: int = 180) -> list[SMTEvent]:
    """A ↕ B negatively correlated (e.g. EUR vs DXY).
    Bullish SMT for A at low: A new LL, B fails to make new HH.
    Bearish SMT for A at high: A new HH, B fails to make new LL.
    """
    sa = find_swings(df_a, length=length)
    sb = find_swings(df_b, length=length)
    if len(sa) < 2 or len(sb) < 2:
        return []

    events: list[SMTEvent] = []
    highs_a = [s for s in sa if s.kind == "high"]
    lows_a = [s for s in sa if s.kind == "low"]

    for i in range(1, len(highs_a)):
        cur, prev = highs_a[i], highs_a[i - 1]
        if cur.price <= prev.price:
            continue
        b_cur = _closest_swing([s for s in sb if s.kind == "low"], cur.ts, tol_minutes)
        b_prev = _closest_swing([s for s in sb if s.kind == "low"], prev.ts, tol_minutes)
        if b_cur is None or b_prev is None:
            continue
        # If B failed to make a new low (b_cur > b_prev), SMT bearish for A
        if b_cur.price > b_prev.price:
            events.append(SMTEvent(cur.ts, pair_a, pair_b, "bear", "high",
                                   f"{pair_a} new HH, {pair_b} failed new LL"))

    for i in range(1, len(lows_a)):
        cur, prev = lows_a[i], lows_a[i - 1]
        if cur.price >= prev.price:
            continue
        b_cur = _closest_swing([s for s in sb if s.kind == "high"], cur.ts, tol_minutes)
        b_prev = _closest_swing([s for s in sb if s.kind == "high"], prev.ts, tol_minutes)
        if b_cur is None or b_prev is None:
            continue
        if b_cur.price < b_prev.price:
            events.append(SMTEvent(cur.ts, pair_a, pair_b, "bull", "low",
                                   f"{pair_a} new LL, {pair_b} failed new HH"))

    return sorted(events, key=lambda e: e.ts)
