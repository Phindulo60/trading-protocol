"""Signal scanner — runs all strategies and deduplicates.

Active strategies (in priority order):
  1. TREND_RSI  — H4 trend + M15 RSI deep oversold/overbought (backtested edge)
  2. ECM        — EMA Cross Momentum (legacy, lower priority)
  3. ARB        — Asian Range Breakout (legacy, lower priority)

Validated pairs (TREND_RSI, 60-day mini-backtest Mar–May 2025):
  ✅  EURUSD  — 11-month full backtest: WR=66.7%, PF=3.34, TotalR=+62.6R
  ✅  USDJPY  — 60d: WR=71%, PF=2.90, TotalR=+15.4R
  ✅  USDCAD  — 60d: WR=71%, PF=5.58, TotalR=+10.1R
  ✅  EURJPY  — 60d: WR=71%, PF=2.86, DD=-1.0R
  ✅  AUDUSD  — 60d: WR=67%, PF=2.40, TotalR=+10.8R
  ✅  GBPJPY  — 60d: WR=61%, PF=2.49, TotalR=+7.1R
  ✅  GBPUSD  — 60d: WR=52%, marginal (use with caution)
  ⚠️   NZDUSD  — 60d: WR=53%, high DD — needs full backtest
  ⚠️   USDCHF  — 60d: WR=50%, weak — needs full backtest
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from fsp.data.feed import DataFeed, default_feed
from fsp.signals.base import Signal
from fsp.signals.alpha import scan_trend_rsi
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

    # 1. TREND_RSI (primary — backtested edge)
    try:
        if len(h4_df) >= 22:
            sig = scan_trend_rsi(pair, m15_df, h4_df)
            if sig is not None:
                signals.append(sig)
    except Exception:
        log.exception("TREND_RSI scan failed for %s", pair)

    # 2. ECM (legacy)
    try:
        sig = scan_momentum(pair, m15_df, h1_df, daily_df)
        if sig is not None:
            signals.append(sig)
    except Exception:
        log.exception("ECM scan failed for %s", pair)

    # 3. ARB (legacy)
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
