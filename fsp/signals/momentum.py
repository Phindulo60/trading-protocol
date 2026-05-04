"""EMA Cross Momentum (ECM) — intraday signal strategy.

Entry rules (LONG):
  1. EMA8 crossed above EMA21 on the last closed M15 bar (new cross only)
  2. M15 close is above H1 EMA50 (trend-aligned)
  3. RSI(14) on M15 is between 45 and 68 (momentum zone, not overbought)
  4. Session = LONDON or NY-AM (07:00–17:00 UTC)
  5. ADR% < 120% (room to run)
  6. SL = lowest low of last 4 bars minus ATR(14)*0.25 buffer
  7. Max SL 25 pips; skip if wider
  8. RR to TP1 >= 1.5

SHORT is the mirror image. RSI range: 32–55.

TP1 = entry + 1.5 × risk (fixed)
TP2 = entry + 2.5 × risk (optional, extended target)
Suggested risk: 1.0R
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fsp.context.cycle import classify_cycle
from fsp.context.sessions import session_of
from fsp.data.types import Session
from fsp.signals.base import Signal
from fsp.structure.displacement import atr


# pip size
def _pip(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    # When avg_loss = 0, all moves are gains → RSI = 100 (no NaN)
    rsi = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))
    return rsi.where(avg_loss != 0, other=100.0)


def scan_momentum(pair: str,
                  m15_df: pd.DataFrame,
                  h1_df: pd.DataFrame,
                  daily_df: pd.DataFrame) -> Signal | None:
    """
    Inspect the last closed M15 bar and return a Signal if ECM criteria are met,
    otherwise None.

    Parameters
    ----------
    m15_df   : M15 OHLCV bars (UTC-indexed), at least 60 bars
    h1_df    : H1 bars, at least 60 bars
    daily_df : Daily bars, at least 10 bars
    """
    if len(m15_df) < 55 or len(h1_df) < 55:
        return None

    pip = _pip(pair)

    # --- Indicators on M15 ---
    close = m15_df["close"]
    low = m15_df["low"]
    high = m15_df["high"]

    ema8 = _ema(close, 8)
    ema21 = _ema(close, 21)
    rsi14 = _rsi(close, 14)
    atr14 = atr(m15_df, 14)

    # Check last two bars for fresh cross (bar[-2] crossed, bar[-1] still valid)
    prev_diff = (ema8.iloc[-2] - ema21.iloc[-2])   # before last bar
    curr_diff = (ema8.iloc[-1] - ema21.iloc[-1])   # last closed bar

    # Require a FRESH cross: signs differ
    bull_cross = (prev_diff < 0) and (curr_diff > 0)
    bear_cross = (prev_diff > 0) and (curr_diff < 0)

    if not bull_cross and not bear_cross:
        return None

    direction = "long" if bull_cross else "short"

    # --- H1 EMA50 trend filter ---
    h1_ema50 = _ema(h1_df["close"], 50)
    price_vs_h1 = m15_df["close"].iloc[-1]
    h1_trend_ok = (direction == "long" and price_vs_h1 > h1_ema50.iloc[-1]) or \
                  (direction == "short" and price_vs_h1 < h1_ema50.iloc[-1])
    if not h1_trend_ok:
        return None

    # --- RSI filter ---
    rsi_val = float(rsi14.iloc[-1])
    if direction == "long" and not (45 <= rsi_val <= 68):
        return None
    if direction == "short" and not (32 <= rsi_val <= 55):
        return None

    # --- Session filter (London 07:00–12:00 UTC, NY-AM 12:00–17:00 UTC) ---
    ts = m15_df.index[-1]
    sess = session_of(ts)
    if sess not in (Session.LONDON, Session.NY_AM):
        return None

    # --- ADR filter ---
    try:
        from fsp.context.cycle import classify_cycle
        cyc = classify_cycle(h1_df, daily_df)
        if cyc.adr_pct >= 120:
            return None
        adr_pct = round(cyc.adr_pct, 1)
    except Exception:
        adr_pct = 0.0

    # --- Entry / SL / TP ---
    price = float(close.iloc[-1])
    atr_val = float(atr14.iloc[-1])

    if direction == "long":
        sl = float(low.iloc[-4:].min()) - atr_val * 0.25
    else:
        sl = float(high.iloc[-4:].max()) + atr_val * 0.25

    inv_pips = abs(price - sl) / pip
    if inv_pips > 25:
        return None
    if inv_pips < 3:
        return None

    risk = abs(price - sl)
    tp1 = price + risk * 1.5 if direction == "long" else price - risk * 1.5
    tp2 = price + risk * 2.5 if direction == "long" else price - risk * 2.5
    rr1 = 1.5
    rr2 = 2.5

    return Signal(
        strategy="ECM",
        pair=pair,
        direction=direction,
        entry=price,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        inv_pips=round(inv_pips, 1),
        rr_tp1=rr1,
        rr_tp2=rr2,
        risk_r=1.0,
        note=f"EMA8×EMA21 {'bull' if bull_cross else 'bear'} cross | RSI={rsi_val:.0f} | {sess.value}",
        ts=ts.isoformat(),
        context={
            "session": sess.value,
            "rsi": round(rsi_val, 1),
            "ema8": round(float(ema8.iloc[-1]), 5),
            "ema21": round(float(ema21.iloc[-1]), 5),
            "h1_ema50": round(float(h1_ema50.iloc[-1]), 5),
            "adr_pct": adr_pct,
        },
    )
