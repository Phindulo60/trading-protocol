"""Displacement candle detection.

A displacement candle is a strong, directional move that leaves an FVG behind.
PDF definition: body > 1.5 * ATR(20) and closes beyond prior structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class Displacement:
    ts: datetime
    direction: Literal["up", "down"]
    body: float
    atr_mult: float
    close: float


def atr(df: pd.DataFrame, length: int = 20) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def find_displacements(df: pd.DataFrame, mult: float = 1.5, length: int = 20) -> list[Displacement]:
    if len(df) < length + 2:
        return []
    a = atr(df, length).values
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    idx = df.index
    out: list[Displacement] = []
    for i in range(length + 1, len(df)):
        if np.isnan(a[i]) or a[i] == 0:
            continue
        body = abs(c[i] - o[i])
        if body < mult * a[i]:
            continue
        if c[i] > o[i] and c[i] > h[i - 1]:
            out.append(Displacement(idx[i].to_pydatetime(), "up", body, body / a[i], float(c[i])))
        elif c[i] < o[i] and c[i] < l[i - 1]:
            out.append(Displacement(idx[i].to_pydatetime(), "down", body, body / a[i], float(c[i])))
    return out
