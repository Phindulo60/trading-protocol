"""Order Block detection.

Bullish OB: last down-close candle immediately before an up-displacement.
Bearish OB: last up-close candle immediately before a down-displacement.
OB range = [low, high] of that prior candle. Mitigation = any wick re-enters.
"""
from __future__ import annotations

import pandas as pd

from fsp.data.types import OrderBlock, TF
from fsp.structure.displacement import atr


def find_order_blocks(df: pd.DataFrame, tf: TF, mult: float = 1.5, length: int = 20) -> list[OrderBlock]:
    if len(df) < length + 2:
        return []
    a = atr(df, length).values
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    idx = df.index
    obs: list[OrderBlock] = []
    for i in range(length + 1, len(df)):
        if a[i] == 0 or pd.isna(a[i]):
            continue
        body = abs(c[i] - o[i])
        if body < mult * a[i]:
            continue
        # bullish displacement + prior candle is bearish
        if c[i] > o[i] and c[i] > h[i - 1] and c[i - 1] < o[i - 1]:
            obs.append(OrderBlock(ts=idx[i - 1].to_pydatetime(),
                                  top=float(h[i - 1]), bottom=float(l[i - 1]),
                                  direction="bull", tf=tf))
        # bearish displacement + prior candle is bullish
        elif c[i] < o[i] and c[i] < l[i - 1] and c[i - 1] > o[i - 1]:
            obs.append(OrderBlock(ts=idx[i - 1].to_pydatetime(),
                                  top=float(h[i - 1]), bottom=float(l[i - 1]),
                                  direction="bear", tf=tf))
    return obs


def mark_ob_mitigation(obs: list[OrderBlock], df: pd.DataFrame) -> list[OrderBlock]:
    for ob in obs:
        future = df[df.index > pd.Timestamp(ob.ts)]
        if future.empty:
            continue
        hit = (future["low"] <= ob.top) & (future["high"] >= ob.bottom)
        ob.mitigated = bool(hit.any())
    return obs
