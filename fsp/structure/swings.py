"""Pivot-based swing detection with ICT-flavoured strong/weak classification.

A pivot-high at bar i with pivot length L means high[i] is the strictly highest
high in [i-L, i+L]. A pivot-high is 'strong' if, since the prior pivot-low, the
market has traded through (taken the liquidity of) that prior pivot-low.
Mirror logic for pivot-lows.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from fsp.data.types import Swing


def find_swings(df: pd.DataFrame, length: int = 5) -> list[Swing]:
    if len(df) < 2 * length + 1:
        return []

    highs = df["high"].values
    lows = df["low"].values
    idx = df.index

    swings: list[Swing] = []
    prev_low: float | None = None   # last confirmed pivot-low price
    prev_low_i: int | None = None
    prev_high: float | None = None
    prev_high_i: int | None = None

    for i in range(length, len(df) - length):
        win_h = highs[i - length : i + length + 1]
        win_l = lows[i - length : i + length + 1]
        is_ph = highs[i] == win_h.max() and (highs[i] > np.delete(win_h, length)).all()
        is_pl = lows[i] == win_l.min() and (lows[i] < np.delete(win_l, length)).all()

        if is_ph:
            # strong high = since prior pivot-low, price traded BELOW prior pivot-low
            strong = False
            if prev_low is not None and prev_low_i is not None:
                segment_low = lows[prev_low_i : i + 1].min()
                strong = segment_low < prev_low
            swings.append(Swing(ts=idx[i].to_pydatetime(), price=float(highs[i]),
                                kind="high", strong=strong))
            prev_high, prev_high_i = float(highs[i]), i

        if is_pl:
            strong = False
            if prev_high is not None and prev_high_i is not None:
                segment_high = highs[prev_high_i : i + 1].max()
                strong = segment_high > prev_high
            swings.append(Swing(ts=idx[i].to_pydatetime(), price=float(lows[i]),
                                kind="low", strong=strong))
            prev_low, prev_low_i = float(lows[i]), i

    return swings


def mark_broken(swings: list[Swing], df: pd.DataFrame) -> list[Swing]:
    """Flag swings whose price has since been traded through."""
    for s in swings:
        future = df[df.index > pd.Timestamp(s.ts)]
        if future.empty:
            continue
        if s.kind == "high":
            s.broken = bool((future["high"] > s.price).any())
        else:
            s.broken = bool((future["low"] < s.price).any())
    return swings
