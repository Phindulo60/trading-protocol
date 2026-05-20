"""
Strategy: ASIA_HL_FADE — fade tests of prior Asian session H/L in H4 trend direction.

Logic:
  During LO/NY-AM session, look for price testing yesterday's Asia session
  range high/low. If H4 trend agrees with the fade direction (e.g. price tags
  Asia low in H4 bull trend → enter long), take the trade as a continuation
  rather than a real breakout.

Backtest (USDCAD May 2025–May 2026, 12mo):
  Full year   : n=138  WR=57.2%  Exp=+0.75R  PF=2.90  +103.2R  DD=-6.3R
  Train (1st 6mo): n=60  WR=56.7%  Exp=+0.62R  PF=2.66  +37.5R  DD=-6.0
  Test  (2nd 6mo): n=77  WR=58.4%  Exp=+0.87R  PF=3.17  +66.7R  DD=-6.3
  → robust out-of-sample (improves in test)

Different from TREND_RSI — uncorrelated trigger:
  TREND_RSI: enters on M15 RSI extreme during NY sessions
  ASIA_HL  : enters on Asia H/L test during early London — different bars

Parameters:
  Asia window  = 19:00–03:00 NY (prev day) — typical Asia session
  H4 trend     = EMA10 (same as TREND_RSI v2)
  ATR mult     = 1.5
  TP1 / TP2    = 2.5R / 4.0R
  Sessions     = LO, NY_AM (skip NY_PM where Asia HL is stale)
  Asia range   = 5–50 pips (filter out flat or huge ranges)
"""
from __future__ import annotations

import pandas as pd

from fsp.context.sessions import session_of
from fsp.data.types import Session
from fsp.signals.base import Signal
from fsp.signals.alpha import _pip, _ema, _atr


def scan_asia_hl(pair: str,
                 m15_df: pd.DataFrame,
                 h4_df: pd.DataFrame) -> Signal | None:
    """Fade tests of prior Asia session H/L in H4 trend direction."""
    if len(m15_df) < 100 or len(h4_df) < 12:
        return None

    pip = _pip(pair)
    ts = m15_df.index[-1]
    sess = session_of(ts)

    # London / NY-AM only — Asia HL gets stale by NY-PM
    if sess not in (Session.LONDON, Session.NY_AM):
        return None

    # Skip Friday and Sunday like TREND_RSI v2
    dow = ts.tz_convert("America/New_York").weekday()
    if dow in (4, 6):
        return None

    # ── Find Asia session range (prev day 19:00 NY → 03:00 NY today) ─────────
    ny_ts = ts.tz_convert("America/New_York")
    today_start = ny_ts.normalize()
    asia_start = today_start - pd.Timedelta(hours=5)   # ~19:00 NY prior day
    asia_end = today_start + pd.Timedelta(hours=3)     # ~03:00 NY today

    asia_window = m15_df[
        (m15_df.index.tz_convert("America/New_York") >= asia_start) &
        (m15_df.index.tz_convert("America/New_York") < asia_end)
    ]
    if len(asia_window) < 8:
        return None

    asia_high = float(asia_window["high"].max())
    asia_low = float(asia_window["low"].min())
    asia_range = asia_high - asia_low

    # Filter out tiny or huge Asia ranges (regime markers)
    if asia_range < 5 * pip or asia_range > 50 * pip:
        return None

    # ── H4 trend ─────────────────────────────────────────────────────────────
    h4_close = float(h4_df["close"].iloc[-1])
    h4_ema10 = float(_ema(h4_df["close"], 10).iloc[-1])
    h4_bull = h4_close > h4_ema10
    h4_bear = h4_close < h4_ema10

    # ── Current bar tests Asia H/L? ──────────────────────────────────────────
    last = m15_df.iloc[-1]
    last_high = float(last["high"])
    last_low = float(last["low"])
    last_close = float(last["close"])

    direction = None
    if h4_bull and last_low <= asia_low and last_close > asia_low:
        # Tagged Asia low and reclaimed → bull continuation long
        direction = "long"
    elif h4_bear and last_high >= asia_high and last_close < asia_high:
        # Tagged Asia high and rejected → bear continuation short
        direction = "short"

    if direction is None:
        return None

    # ── Risk / reward ────────────────────────────────────────────────────────
    atr_v = float(_atr(m15_df, 14).iloc[-1])
    sl = last_close - atr_v * 1.5 if direction == "long" else last_close + atr_v * 1.5
    risk = abs(last_close - sl)
    inv_pips = risk / pip
    if not (3 <= inv_pips <= 60):
        return None

    tp1 = last_close + risk * 2.5 if direction == "long" else last_close - risk * 2.5
    tp2 = last_close + risk * 4.0 if direction == "long" else last_close - risk * 4.0

    return Signal(
        strategy="ASIA_HL",
        pair=pair,
        direction=direction,
        entry=last_close,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        inv_pips=round(inv_pips, 1),
        rr_tp1=2.5,
        rr_tp2=4.0,
        risk_r=1.0,
        note=(f"Fade Asia {'low' if direction=='long' else 'high'} | "
              f"H4 {'BULL' if h4_bull else 'BEAR'} | "
              f"Asia range {asia_range/pip:.0f}p | {sess.value}"),
        ts=ts.isoformat(),
        context={
            "session": sess.value,
            "asia_high": round(asia_high, 5),
            "asia_low": round(asia_low, 5),
            "asia_range_pips": round(asia_range / pip, 1),
            "h4_trend": "BULL" if h4_bull else "BEAR",
            "atr": round(atr_v, 5),
            "max_hold_bars": 16,  # 4h
        },
    )
