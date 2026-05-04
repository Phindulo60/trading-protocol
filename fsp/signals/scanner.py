"""Signal scanner — runs all strategies and deduplicates.

═══════════════════════════════════════════════════════════════════════════════
ACTIVE STRATEGIES  (priority order — all fire simultaneously if conditions met)
═══════════════════════════════════════════════════════════════════════════════

  1. TREND_RSI  — H4 EMA20 trend + M15 RSI deep oversold/overbought
                  990 trades | 8 pairs | Jun 2024–Apr 2025
                  WR=60.7%  Exp=+0.51R  PF=2.8  TotalR=+504R  MaxDD=-5.3R
                  TP1=3.5R  SL=1.5×ATR  MaxHold=8 bars (~2h)

  2. LEVEL_OB   — PSH/PSL/PDH/PDL + Order Block + H4 trend alignment
                  Three tiers (shown in every Telegram signal — you decide):
                    CONF : M15+H1 OB + H4  → 100% hist rev, avg 76p  (~1/month)
                    H1   : H1 OB + H4       →  87% hist rev, avg 58p  (~1.8/wk)
                    M15  : M15 OB + H4      →  81% hist rev, avg 51p  (~1.0/wk)
                  TP1=2.5R  TP2=4.0R  SL=below/above OB range

  3. ECM        — EMA Cross Momentum (legacy, lower priority)
  4. ARB        — Asian Range Breakout (legacy, lower priority)

═══════════════════════════════════════════════════════════════════════════════
RESEARCH FINDINGS  (8 pairs | Apr 2024–May 2025 | 50,918 level-touch events)
═══════════════════════════════════════════════════════════════════════════════

── Level Type Hierarchy ────────────────────────────────────────────────────────
  Level       Baseline Rev%   With OB+H4   Notes
  PSH/PSL         27%           78-87%     Best reversal levels — session H/L
  PDH/PDL         18-20%        75-91%     Strong with OB+H4
  PWH/PWL          4-5%          n/a       Break levels (93% break-through rate)
                                           OB filter finds zero qualifying events
                                           — by the time price reaches weekly
                                           levels, all OBs near it are mitigated

── OB Timeframe Comparison ─────────────────────────────────────────────────────
  Filter            n      Rev%    Avg Move    Empty wks
  Baseline       33,715    24%       36p          —
  H4 only         7,437    44%       40p          —
  H4 + M15 OB       109    81%       51p         53%
  H4 + H1 OB        194    87%       58p         39%    ← primary filter
  H4 + Either        291    84%       55p         29%
  H4 + Both (conf)    12   100%       76p         90%    ← premium / rare

  → Use H1 OB as the primary OB filter.
  → "Either" (M15 or H1) gives most setups at 84% — good for active scanning.

── RSI Divergence (counterintuitive, critical finding) ──────────────────────────
  At a LOW level touch:
    RSI  0–30  (very oversold)  → 82% BREAK  (momentum aligned = blows through)
    RSI 30–40  (oversold)       → 66% break
    RSI 40–50  (neutral)        → 54% break / 27% reversal
    RSI 60–70  (divergence)     → 47% break / 46% reversal  ← rising
    RSI 70+    (strong diverg.) → 17% break / 72% reversal  ← best

  At a HIGH level touch:
    RSI 70+    (very overbought) → 84% BREAK  (momentum = continuation)
    RSI 60–70  (overbought)      → 67% break  / 21% reversal
    RSI 50–60  (neutral)         → 54% break  / 28% reversal
    RSI 30–40  (divergence)      → 51% break  / 40% reversal  ← rising
    RSI  0–30  (strong diverg.)  → 35% break  / 60% reversal  ← best

  KEY INSIGHT: Do NOT use RSI as confirmation at key levels.
  Use it as a DIVERGENCE signal — extreme RSI in the break direction = break,
  extreme RSI against the test direction = reversal.

── Break + Pullback Study ───────────────────────────────────────────────────────
  23,043 break events | 48% had pullback within 50 bars

  Filter              Continuation%   Notes
  No filter                37%        Baseline — worse than reversal
  H4 aligned               42%        Still below 50% — no clear edge
  OB filter                n/a        OBs at level get MITIGATED during pullback
                                      — filter returns no qualifying events

  Why OBs don't work for break+pullback:
  The pullback itself wicks through the OB range, mitigating it.
  To trade break+pullback with OBs you'd need OBs in the NEW territory
  (above old resistance for bullish breaks) — a different setup entirely.

  VERDICT: Break+pullback does NOT have a statistical edge with these filters.
  → Do not build as a strategy. Use the reversal (LEVEL_OB) instead.

── PWH/PWL Momentum Breakout Study ─────────────────────────────────────────────
  2,659 break events across 8 pairs

  Filter                      WR     Exp/trade   Notes
  No filter                   20%    +0.02R       Near random
  H4 aligned                  22%    +0.10R       Marginal
  H4 + strong close           26%    +0.09R       Marginal
  H4 + strong close + RSI     25%    +0.16R       Best — still thin

  Exceptions worth noting:
    GBPJPY  n=45   WR=44%  Exp=+0.41R  — only pair with real breakout edge
    GBPUSD  n=123  WR=33%  Exp=+0.33R  — strong but small sample

  Best conditions: Thursday, London/Asia session, RSI 0–40 bullish or RSI 45–60 bearish

  VERDICT: Do not build as standalone strategy (+0.16R is too thin after costs).
  Use as CONTEXT to filter TREND_RSI signals:
    → If TREND_RSI fires SHORT at PSH but H4 is breaking above PWH with a
      strong conviction candle on Thursday London → skip the short.
    → Conversely, a PWH/PWL break aligned with TREND_RSI = higher conviction.

── Per-Pair Quality Summary ─────────────────────────────────────────────────────
  LEVEL_OB reversal (H4 + H1 OB):
    GBPUSD  100%   NZDUSD  96%   EURUSD  91%   USDCHF  88%   USDCAD  86%
    GBPJPY   79% (avg 139p)      EURJPY   75% (avg 115p)      AUDUSD   65%

  PWH/PWL breakout (H4 + strong close):
    GBPJPY  Exp=+0.41R   GBPUSD  Exp=+0.33R   (all others marginal)

  TREND_RSI (Jun 2024–Apr 2025):
    EURUSD  +69.7R   AUDUSD  +75.3R   USDCAD  +79.1R   NZDUSD  +69.0R
    USDCHF  +60.2R   GBPUSD  +56.5R   EURJPY  +47.1R   GBPJPY  +47.6R

═══════════════════════════════════════════════════════════════════════════════
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
