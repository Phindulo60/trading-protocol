"""Signal scanner — runs all strategies and deduplicates.

Active strategies (in priority order):
  1. TREND_RSI  — H4 trend + M15 RSI oversold/overbought      backtested: 990 trades, WR=61%, +504R
  2. LEVEL_OB   — Key level + Order Block + H4 trend alignment  studied:  ~2/week avg, 84-100% rev rate
  3. ECM        — EMA Cross Momentum (legacy)
  4. ARB        — Asian Range Breakout (legacy)

LEVEL_OB tiers (shown in signal note — Phindulo decides whether to execute):
  CONF : M15 OB + H1 OB + H4  →  100% hist reversal, avg 76 pips  (rare ~1/month)
  H1   : H1 OB + H4            →   87% hist reversal, avg 58 pips  (~1.8/week)
  M15  : M15 OB + H4           →   81% hist reversal, avg 51 pips  (~1.0/week)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from fsp.data.feed import DataFeed, default_feed
from fsp.signals.base import Signal
from fsp.signals.alpha import scan_trend_rsi
from fsp.signals.level_ob import scan_level_ob
from fsp.signals.momentum import scan_momentum
from fsp.signals.breakout import scan_breakout

log = logging.getLogger(__name__)


def scan_all(pair: str,
             m5_df: pd.DataFrame,
             m15_df: pd.DataFrame,
             h1_df: pd.DataFrame,
             daily_df: pd.DataFrame) -> list[Signal]:
    """Run all strategies and return fired signals (may be empty or multiple)."""
    signals: list[Signal] = []

    # Build H4 from M15 for TREND_RSI (or use H1 if M15 unavailable)
    h4_df = pd.DataFrame()
    try:
        if len(m15_df) >= 55:
            h4_df = m15_df.resample("4h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"
            }).dropna()
    except Exception:
        pass

    # 1. TREND_RSI — H4 trend + M15 RSI oversold/overbought
    try:
        if len(h4_df) >= 22:
            sig = scan_trend_rsi(pair, m15_df, h4_df)
            if sig is not None:
                signals.append(sig)
    except Exception:
        log.exception("TREND_RSI scan failed for %s", pair)

    # 2. LEVEL_OB — Key level + Order Block + H4 trend
    try:
        sig = scan_level_ob(pair, m15_df, h1_df, daily_df)
        if sig is not None:
            signals.append(sig)
    except Exception:
        log.exception("LEVEL_OB scan failed for %s", pair)

    # 4. ECM (legacy)
    try:
        sig = scan_momentum(pair, m15_df, h1_df, daily_df)
        if sig is not None:
            signals.append(sig)
    except Exception:
        log.exception("ECM scan failed for %s", pair)

    # 5. ARB (legacy)
    try:
        sig = scan_breakout(pair, m5_df, h1_df, daily_df)
        if sig is not None:
            signals.append(sig)
    except Exception:
        log.exception("ARB scan failed for %s", pair)

    return signals


async def scan_pair_live(pair: str, feed_kind: str) -> list[Signal]:
    """Fetch fresh data and run all strategies. Used by the live loop."""
    f = default_feed(feed_kind)
    end = datetime.now(timezone.utc)

    try:
        m5_df    = f.history(pair, "M5",  end - timedelta(days=3), end)
        m15_df   = f.history(pair, "M15", end - timedelta(days=5), end)
        h1_df    = f.history(pair, "H1",  end - timedelta(days=30), end)
        daily_df = f.history(pair, "D",   end - timedelta(days=30), end)
    except Exception as e:
        log.error("Data fetch failed for %s: %s", pair, e)
        return []

    return scan_all(pair, m5_df, m15_df, h1_df, daily_df)
