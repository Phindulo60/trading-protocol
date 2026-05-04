"""Fair Value Gap detection (3-candle imbalance) + mitigation tracking."""
from __future__ import annotations

import pandas as pd

from fsp.data.types import FVG, TF


def find_fvgs(df: pd.DataFrame, tf: TF) -> list[FVG]:
    fvgs: list[FVG] = []
    if len(df) < 3:
        return fvgs
    highs = df["high"].values
    lows = df["low"].values
    idx = df.index

    for i in range(2, len(df)):
        # Bullish FVG: low[i] > high[i-2]
        if lows[i] > highs[i - 2]:
            fvgs.append(FVG(ts=idx[i].to_pydatetime(), top=float(lows[i]),
                            bottom=float(highs[i - 2]), direction="bull", tf=tf))
        # Bearish FVG: high[i] < low[i-2]
        if highs[i] < lows[i - 2]:
            fvgs.append(FVG(ts=idx[i].to_pydatetime(), top=float(lows[i - 2]),
                            bottom=float(highs[i]), direction="bear", tf=tf))
    return fvgs


def mark_mitigation(fvgs: list[FVG], df: pd.DataFrame) -> list[FVG]:
    """An FVG is mitigated once a later candle wicks into its range.
    Inverted (IFVG) = a candle closes fully through the FVG in the opposite direction.
    """
    for f in fvgs:
        future = df[df.index > pd.Timestamp(f.ts)]
        if future.empty:
            continue
        # Mitigated: any wick enters [bottom, top]
        hit = ((future["low"] <= f.top) & (future["high"] >= f.bottom))
        f.mitigated = bool(hit.any())
        # Inverted: close fully on the opposite side after mitigation
        if f.mitigated:
            if f.direction == "bull":
                f.inverted = bool((future["close"] < f.bottom).any())
            else:
                f.inverted = bool((future["close"] > f.top).any())
    return fvgs
