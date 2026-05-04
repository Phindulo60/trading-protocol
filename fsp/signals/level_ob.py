"""LEVEL_OB strategy — Previous Session/Day High/Low + Order Block + H4 Trend.

Statistical edge (8 pairs | Apr 2024–May 2025 | 33,715 level-touch events):

  Three signal tiers based on which OB timeframes confirm the level:

  ┌─────────┬──────────────────────────────┬──────┬──────────┬───────────┬──────────────┐
  │  Tier   │  Condition                   │   n  │  Rev %   │  Avg Move │  Empty wks   │
  ├─────────┼──────────────────────────────┼──────┼──────────┼───────────┼──────────────┤
  │  CONF   │  M15 OB + H1 OB + H4 aligned│   12 │  100 %   │   76 pips │     90 %     │
  │  H1     │  H1 OB + H4 aligned          │  194 │   87 %   │   58 pips │     39 %     │
  │  M15    │  M15 OB + H4 aligned         │  109 │   81 %   │   51 pips │     53 %     │
  └─────────┴──────────────────────────────┴──────┴──────────┴───────────┴──────────────┘

  "Either" (M15 or H1, any) → 291 events, 84% rev, ~2.6 setups/week.

Level types scanned:  PSH / PSL (previous session high/low)
                      PDH / PDL (previous day high/low)
Sessions:  London 86%  |  NY 90%  |  Asia 88%  — all viable with H4+H1 OB
Pairs:     GBPUSD 100% | NZDUSD 96% | EURUSD 91% | USDCHF 88% | USDCAD 86%
           GBPJPY 79% (avg 139p) | EURJPY 75% (avg 115p) | AUDUSD 65%

SL:  just below/above the OB range + 0.3×ATR buffer (OB invalidation logic)
TP1: 2.5R   TP2: 4.0R   Max hold: 8 bars (~2h)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fsp.signals.base import Signal
from fsp.structure.displacement import atr as _atr_series

log = logging.getLogger(__name__)

# ── Strategy parameters ────────────────────────────────────────────────────────
TOUCH_PIPS      = 5     # pips from level → qualifies as "touch"
OB_PROX_M15     = 10    # pips: M15 OB range must overlap level within this
OB_PROX_H1      = 15    # pips: H1 OB (larger candles) proximity
OB_LOOKBACK_M15 = 100   # M15 bars scanned for OBs  (~25 h)
OB_LOOKBACK_H1  = 60    # H1 bars scanned for OBs   (~60 h)
OB_ATR_MULT     = 1.5   # displacement body must be ≥ this × ATR
OB_ATR_LEN      = 14    # ATR period used in OB detection
TP1_R           = 2.5
TP2_R           = 4.0
SL_OB_BUFFER    = 0.3   # ATR multiples added below OB bottom / above OB top
SL_ATR_FALLBACK = 1.5   # fallback SL if OB-based SL is out of range
MIN_INV_PIPS    = 3
MAX_INV_PIPS    = 50

# Historical stats communicated in the signal note / Telegram message
_TIER_STATS = {
    "CONF": {"rev_pct": 100, "avg_pips": 76,  "sample": 12},
    "H1":   {"rev_pct": 87,  "avg_pips": 58,  "sample": 194},
    "M15":  {"rev_pct": 81,  "avg_pips": 51,  "sample": 109},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pip(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _sess(ts: pd.Timestamp) -> str:
    h = ts.hour
    if  7 <= h < 13: return "LON"
    if 13 <= h < 21: return "NY"
    return "ASIA"


def _prev_session_hl(m15_df: pd.DataFrame) -> tuple[float | None, float | None]:
    """High / low of the last COMPLETED session in m15_df."""
    labels = [_sess(ts) for ts in m15_df.index]
    cur = labels[-1]
    n   = len(labels)

    # Find where current session block starts
    cur_start = n - 1
    for i in range(n - 2, -1, -1):
        if labels[i] != cur:
            cur_start = i + 1
            break

    if cur_start == 0:
        return None, None

    # Walk back through the previous session block
    prev     = labels[cur_start - 1]
    prev_end = cur_start - 1
    prev_start = 0
    for i in range(prev_end - 1, -1, -1):
        if labels[i] != prev:
            prev_start = i + 1
            break

    prev_bars = m15_df.iloc[prev_start: prev_end + 1]
    if len(prev_bars) < 2:
        return None, None

    return float(prev_bars["high"].max()), float(prev_bars["low"].min())


def _prev_day_hl(m15_df: pd.DataFrame) -> tuple[float | None, float | None]:
    """High / low of yesterday built from m15_df."""
    dh = m15_df["high"].resample("D").max()
    dl = m15_df["low"].resample("D").min()
    if len(dh) < 2:
        return None, None
    return float(dh.iloc[-2]), float(dl.iloc[-2])


def _find_active_obs(bars: pd.DataFrame, direction: str,
                     level: float, prox: float) -> list[dict]:
    """
    Find unmitigated OBs of `direction` whose range overlaps `level` within `prox`.

    An OB is:
      bull — last bearish candle immediately before a bullish displacement
      bear — last bullish candle immediately before a bearish displacement
    Mitigated = any later candle's high/low wicks back into the OB [bottom, top].

    Returns list of dicts: {top, bottom, ts} — most recent last.
    """
    n = len(bars)
    if n < OB_ATR_LEN + 2:
        return []

    atr_v = _atr_series(bars, OB_ATR_LEN).values
    o = bars["open"].values
    h = bars["high"].values
    l = bars["low"].values
    c = bars["close"].values

    results: list[dict] = []

    for i in range(OB_ATR_LEN + 1, n):
        if atr_v[i] == 0 or np.isnan(atr_v[i]):
            continue
        if abs(c[i] - o[i]) < OB_ATR_MULT * atr_v[i]:
            continue  # not a displacement candle

        is_bull = c[i] > o[i] and c[i] > h[i-1] and c[i-1] < o[i-1]
        is_bear = c[i] < o[i] and c[i] < l[i-1] and c[i-1] > o[i-1]

        if direction == "bull" and not is_bull:
            continue
        if direction == "bear" and not is_bear:
            continue

        ob_top = float(h[i-1])
        ob_bot = float(l[i-1])

        # Proximity check
        if ob_bot > level + prox or ob_top < level - prox:
            continue

        # Mitigation check — any bar after formation wicks into range
        mitigated = False
        for j in range(i, n):
            if l[j] <= ob_top and h[j] >= ob_bot:
                mitigated = True
                break
        if mitigated:
            continue

        results.append({"top": ob_top, "bot": ob_bot, "ts": bars.index[i-1]})

    return results


# ── Main strategy function ────────────────────────────────────────────────────

def scan_level_ob(pair: str,
                  m15_df: pd.DataFrame,
                  h1_df: pd.DataFrame,
                  daily_df: pd.DataFrame) -> Signal | None:
    """
    Scan for LEVEL_OB setups on the latest M15 bar.

    Returns the single highest-priority qualifying Signal, or None.
    Priority: CONF > H1 > M15, then PSH/PSL > PDH/PDL.

    Signal context includes 'tier' key so the Telegram formatter can
    display which filter combination triggered the alert.
    """
    if len(m15_df) < 60 or len(h1_df) < 30:
        return None

    p     = _pip(pair)
    ts    = m15_df.index[-1]
    bar   = m15_df.iloc[-1]
    price = float(bar["close"])

    # Skip Friday (liquidity thins, levels often run without follow-through)
    dow = ts.tz_convert("America/New_York").weekday()
    if dow == 4:
        return None

    sess = _sess(ts)

    # ── H4 trend direction ────────────────────────────────────────────────────
    h4 = m15_df.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    if len(h4) < 22:
        return None

    h4_ema20 = float(_ema(h4["close"], 20).iloc[-1])
    h4_bull  = float(h4["close"].iloc[-1]) > h4_ema20

    # ── Key levels ────────────────────────────────────────────────────────────
    psh, psl = _prev_session_hl(m15_df)
    pdh, pdl = _prev_day_hl(m15_df)

    # ── OB lookback windows ───────────────────────────────────────────────────
    m15_win = m15_df.iloc[-OB_LOOKBACK_M15:]
    h1_win  = h1_df.iloc[-OB_LOOKBACK_H1:] if len(h1_df) >= OB_LOOKBACK_H1 else h1_df

    # ATR for SL sizing
    atr_v = float(_atr_series(m15_df, 14).iloc[-1])

    # ── Scan each level ───────────────────────────────────────────────────────
    touch_p = TOUCH_PIPS * p
    candidates: list[tuple[tuple, Signal]] = []

    for level, lname, is_hi in [
        (psh, "PSH", True),  (psl, "PSL", False),
        (pdh, "PDH", True),  (pdl, "PDL", False),
    ]:
        if level is None:
            continue

        # Touch check: current bar approaches level from the correct side
        if is_hi:
            if float(bar["high"]) < level - touch_p:
                continue
        else:
            if float(bar["low"]) > level + touch_p:
                continue

        # H4 must support the reversal direction
        # At a HIGH level  → need H4 bearish (reversal down follows trend)
        # At a LOW level   → need H4 bullish (reversal up follows trend)
        h4_ok = (not h4_bull) if is_hi else h4_bull
        if not h4_ok:
            continue

        ob_dir  = "bear" if is_hi else "bull"
        prox_m15 = OB_PROX_M15 * p
        prox_h1  = OB_PROX_H1  * p

        obs_m15 = _find_active_obs(m15_win, ob_dir, level, prox_m15)
        obs_h1  = _find_active_obs(h1_win,  ob_dir, level, prox_h1)

        has_m15 = bool(obs_m15)
        has_h1  = bool(obs_h1)

        if not has_m15 and not has_h1:
            continue

        # ── Tier + best OB for SL ─────────────────────────────────────────────
        if has_m15 and has_h1:
            tier    = "CONF"
            best_ob = obs_h1[-1]   # H1 OB sets the SL (wider, more authoritative)
            t_prio  = 0
        elif has_h1:
            tier    = "H1"
            best_ob = obs_h1[-1]
            t_prio  = 1
        else:
            tier    = "M15"
            best_ob = obs_m15[-1]
            t_prio  = 2

        # PSH/PSL preferred over PDH/PDL within the same tier
        l_prio = 0 if lname in ("PSH", "PSL") else 1

        # ── SL: just below the OB (long) or above it (short) ─────────────────
        buf = atr_v * SL_OB_BUFFER
        if not is_hi:   # long at low level
            sl_ob  = best_ob["bot"] - buf
            sl_atr = price - atr_v * SL_ATR_FALLBACK
            sl     = max(sl_ob, sl_atr)   # tightest (highest) of both
        else:           # short at high level
            sl_ob  = best_ob["top"] + buf
            sl_atr = price + atr_v * SL_ATR_FALLBACK
            sl     = min(sl_ob, sl_atr)   # tightest (lowest) of both

        risk = abs(price - sl)
        inv  = risk / p
        if not (MIN_INV_PIPS <= inv <= MAX_INV_PIPS):
            continue

        direction = "short" if is_hi else "long"
        tp1 = price + risk * TP1_R if not is_hi else price - risk * TP1_R
        tp2 = price + risk * TP2_R if not is_hi else price - risk * TP2_R

        stats = _TIER_STATS[tier]
        ob_label = "M15+H1" if tier == "CONF" else tier

        note = (
            f"{lname} | {ob_label} OB | Tier:{tier} "
            f"({stats['rev_pct']}% hist rev, avg {stats['avg_pips']}p, "
            f"n={stats['sample']}) | "
            f"H4:{'BEAR' if is_hi else 'BULL'} | {sess}"
        )

        sig = Signal(
            strategy  = "LEVEL_OB",
            pair      = pair,
            direction = direction,
            entry     = price,
            sl        = sl,
            tp1       = tp1,
            tp2       = tp2,
            inv_pips  = round(inv, 1),
            rr_tp1    = TP1_R,
            rr_tp2    = TP2_R,
            risk_r    = 1.0,
            note      = note,
            ts        = ts.isoformat(),
            context   = {
                "level_type":    lname,
                "level_price":   round(level, 5),
                "tier":          tier,
                "ob_m15":        has_m15,
                "ob_h1":         has_h1,
                "ob_range":      [round(best_ob["bot"], 5), round(best_ob["top"], 5)],
                "h4_trend":      "BEAR" if is_hi else "BULL",
                "session":       sess,
                "atr":           round(atr_v, 5),
                "max_hold_bars": 8,
                "hist_rev_pct":  stats["rev_pct"],
                "hist_avg_pips": stats["avg_pips"],
            },
        )

        candidates.append(((t_prio, l_prio), sig))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]
