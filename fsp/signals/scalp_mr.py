"""SCALP_MR — M5 Bollinger Band mean-reversion scalper.

Entry logic:
  1. Compute 20-period Bollinger Bands (2σ) on M5 close
  2. Bar[-2] closed OUTSIDE the band (price spike away from mean)
  3. Bar[-1] closed back INSIDE the band (reversion started)
  4. Enter in the reversion direction (fade the spike)

Targets: TP = middle band (SMA20), capped 5–10 pips. SL = spike extreme + buffer.
Max SL: 12 pips (skip if wider). Min TP: 5 pips (skip if too thin).
Session: London + NY (07:00–16:00 UTC).
Max hold: 3 bars (15 min).
"""
from __future__ import annotations

import pandas as pd

from fsp.context.sessions import session_of
from fsp.data.types import Session
from fsp.signals.base import Signal


def _pip(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    """Return (sma, upper, lower) Bollinger Bands."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = sma + num_std * std
    lower = sma - num_std * std
    return sma, upper, lower


def scan_scalp_mr(pair: str, m5_df: pd.DataFrame) -> Signal | None:
    """Scan the last closed M5 bar for a BB mean-reversion scalp entry.

    Parameters
    ----------
    pair   : currency pair (e.g. "EURUSD")
    m5_df  : M5 OHLCV bars (UTC-indexed), at least 25 bars
    """
    if len(m5_df) < 25:
        return None

    pip = _pip(pair)
    close = m5_df["close"]
    high = m5_df["high"]
    low = m5_df["low"]

    sma, upper, lower = _bollinger(close, 20, 2.0)

    # Need at least 2 bars with valid BB values
    if pd.isna(sma.iloc[-1]) or pd.isna(sma.iloc[-2]):
        return None

    # --- Session filter: London + NY only (07:00–16:00 UTC) ---
    ts = m5_df.index[-1]
    sess = session_of(ts)
    if sess not in (Session.LONDON, Session.NY_AM):
        return None

    # --- Spike + reversion detection ---
    # Bar[-2] closed outside band, bar[-1] closed back inside
    prev_close = float(close.iloc[-2])
    curr_close = float(close.iloc[-1])
    prev_upper = float(upper.iloc[-2])
    prev_lower = float(lower.iloc[-2])
    curr_upper = float(upper.iloc[-1])
    curr_lower = float(lower.iloc[-1])

    spike_above = prev_close > prev_upper and curr_close <= curr_upper
    spike_below = prev_close < prev_lower and curr_close >= curr_lower

    if not spike_above and not spike_below:
        return None

    direction = "short" if spike_above else "long"
    mid = float(sma.iloc[-1])

    # --- Entry / SL / TP ---
    entry = curr_close

    if direction == "long":
        # Spike was below lower band; SL below the spike low + buffer
        sl = float(low.iloc[-2]) - 2 * pip
        # TP toward middle band
        tp_raw = mid
    else:
        # Spike was above upper band; SL above the spike high + buffer
        sl = float(high.iloc[-2]) + 2 * pip
        # TP toward middle band
        tp_raw = mid

    inv_pips = abs(entry - sl) / pip
    tp_pips = abs(tp_raw - entry) / pip

    # --- Filters ---
    # SL too wide for a scalp (> 12 pips)
    if inv_pips > 12:
        return None
    # SL too tight (< 2 pips — likely noise)
    if inv_pips < 2:
        return None
    # TP too thin (< 5 pips — not worth after spread)
    if tp_pips < 5:
        return None
    # Cap TP at 10 pips
    if tp_pips > 10:
        tp_pips = 10.0
        if direction == "long":
            tp_raw = entry + 10 * pip
        else:
            tp_raw = entry - 10 * pip

    rr = tp_pips / inv_pips if inv_pips > 0 else 0.0

    # Skip if R:R is terrible (< 0.8 — need at least near 1:1)
    if rr < 0.8:
        return None

    # --- Band width filter: skip dead/trending markets ---
    band_width_pct = (float(upper.iloc[-1]) - float(lower.iloc[-1])) / entry * 100
    if band_width_pct < 0.04:
        return None  # dead market, no reversion opportunity
    if band_width_pct > 0.35:
        return None  # wide bands = trending, mean-rev fails

    ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

    return Signal(
        strategy="SCALP_MR",
        pair=pair,
        direction=direction,
        entry=entry,
        sl=sl,
        tp1=tp_raw,
        tp2=None,
        inv_pips=round(inv_pips, 1),
        rr_tp1=round(rr, 2),
        rr_tp2=None,
        risk_r=1.0,
        note=f"BB mean-rev: spike {'above' if spike_above else 'below'} band, "
             f"reverted. TP={tp_pips:.0f}p SL={inv_pips:.0f}p",
        ts=ts_str,
        context={
            "band_width_pct": round(band_width_pct, 3),
            "mid_band": round(mid, 5),
            "max_hold_bars": 3,
        },
    )
