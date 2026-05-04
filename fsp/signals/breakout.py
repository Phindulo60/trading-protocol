"""Asian Range Breakout (ARB) — intraday signal strategy.

Asian range: 00:00–07:00 UTC (pre-London consolidation window).
Breakout window: 07:00–09:30 UTC (first 2.5 hours of London open).

Entry rules (LONG):
  1. Asian range is 15–60 pips
  2. First M5 bar that closes above asian_high during breakout window
  3. Day filter: not Friday after 14:00 UTC (close risk), not Monday first hour (range fake)
  4. ADR% < 90% (needs room for a full range extension move)

SL = asian_high - 5 pips  (treat broken resistance as new support)
  → risk ≈ (entry - asian_high) + 5 pips, typically 5–9 pips
TP1 = entry + range_size  (1× range extension)
TP2 = entry + range_size * 1.5

SHORT is symmetric from asian_low.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fsp.context.cycle import classify_cycle
from fsp.signals.base import Signal
from fsp.structure.displacement import atr


def _pip(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def _asian_range(m5_df: pd.DataFrame) -> tuple[float, float] | None:
    """Return (asian_high, asian_low) from 00:00–07:00 UTC of the MOST RECENT day.
    Returns None if fewer than 5 bars in the window (thin/holiday session).
    """
    now = m5_df.index[-1]
    # Asian window: today 00:00–07:00 UTC
    asia_start = now.normalize()                         # 00:00 UTC today
    asia_end = asia_start + pd.Timedelta(hours=7)        # 07:00 UTC

    # If we're before 07:00 UTC, look at the previous day's Asian session
    if now < asia_end:
        asia_start -= pd.Timedelta(days=1)
        asia_end -= pd.Timedelta(days=1)

    window = m5_df[(m5_df.index >= asia_start) & (m5_df.index < asia_end)]
    if len(window) < 5:
        return None

    return float(window["high"].max()), float(window["low"].min())


def scan_breakout(pair: str,
                  m5_df: pd.DataFrame,
                  h1_df: pd.DataFrame,
                  daily_df: pd.DataFrame) -> Signal | None:
    """
    Check the last closed M5 bar for an Asian range breakout.

    Parameters
    ----------
    m5_df    : M5 bars (UTC-indexed), enough history to cover the Asian session
    h1_df    : H1 bars for ADR/cycle context
    daily_df : Daily bars for ADR
    """
    if len(m5_df) < 20:
        return None

    pip = _pip(pair)
    ts = m5_df.index[-1]

    # --- Breakout window: 07:00–09:30 UTC ---
    ts_utc = ts
    hour_utc = ts_utc.hour + ts_utc.minute / 60
    if not (7.0 <= hour_utc < 9.5):
        return None

    # --- Day filters ---
    dow = ts_utc.dayofweek  # 0=Mon … 4=Fri
    if dow == 4 and hour_utc >= 14.0:   # Friday afternoon — skip
        return None
    if dow == 0 and hour_utc < 8.0:     # Monday pre-08:00 — skip (unreliable range)
        return None

    # --- Asian range ---
    result = _asian_range(m5_df)
    if result is None:
        return None
    asian_high, asian_low = result
    range_size = asian_high - asian_low
    range_pips = range_size / pip

    if not (15 <= range_pips <= 60):
        return None

    # --- ADR filter ---
    try:
        cyc = classify_cycle(h1_df, daily_df)
        if cyc.adr_pct >= 90:
            return None
        adr_pct = round(cyc.adr_pct, 1)
    except Exception:
        adr_pct = 0.0

    # --- Check for breakout on last closed bar ---
    last_close = float(m5_df["close"].iloc[-1])
    last_open = float(m5_df["open"].iloc[-1])

    bull_break = last_close > asian_high and last_open <= asian_high
    bear_break = last_close < asian_low and last_open >= asian_low

    # Also allow entries a few bars after the break if we just crossed
    if not bull_break and not bear_break:
        # Check if the PREVIOUS bar broke and last bar is still in the breakout zone
        # (entry window within 2 bars of actual break)
        prev2 = m5_df.iloc[-3:-1]
        bull_recent = (prev2["close"] > asian_high).any()
        bear_recent = (prev2["close"] < asian_low).any()

        if not bull_recent and not bear_recent:
            return None

        # Directional continuation: price still above asian_high or below asian_low
        bull_break = bull_recent and last_close > asian_high
        bear_break = bear_recent and last_close < asian_low
        if not bull_break and not bear_break:
            return None

    direction = "long" if bull_break else "short"

    # --- SL / TP ---
    buffer = 5 * pip  # 5 pips beyond broken level as SL

    if direction == "long":
        sl = asian_high - buffer
        risk = last_close - sl
    else:
        sl = asian_low + buffer
        risk = sl - last_close

    if risk <= 0:
        return None

    inv_pips = risk / pip
    if inv_pips > 20:   # if entry ran far from the level, risk too wide
        return None

    tp1 = last_close + range_size if direction == "long" else last_close - range_size
    tp2 = last_close + range_size * 1.5 if direction == "long" else last_close - range_size * 1.5
    rr1 = range_size / risk
    rr2 = (range_size * 1.5) / risk

    if rr1 < 1.5:
        return None

    atr_val = float(atr(m5_df, 14).iloc[-1]) if len(m5_df) >= 15 else pip * 20

    return Signal(
        strategy="ARB",
        pair=pair,
        direction=direction,
        entry=last_close,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        inv_pips=round(inv_pips, 1),
        rr_tp1=round(rr1, 2),
        rr_tp2=round(rr2, 2),
        risk_r=1.0,
        note=(f"Asian range {range_pips:.0f}p [{asian_low:.5f}–{asian_high:.5f}] "
              f"{'bull' if bull_break else 'bear'} break"),
        ts=ts.isoformat(),
        context={
            "asian_high": round(asian_high, 5),
            "asian_low": round(asian_low, 5),
            "range_pips": round(range_pips, 1),
            "adr_pct": adr_pct,
            "hour_utc": round(hour_utc, 2),
        },
    )
