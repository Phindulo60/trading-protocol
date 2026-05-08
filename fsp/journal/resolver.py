"""Outcome resolver — walks unresolved journal signals against subsequent M15 bars.

For each signal with outcome=NULL:
  1. Fetch M15 data for pair from signal_ts onward (yfinance, up to today)
  2. Walk forward max_hold_bars=8 bars
  3. Check intra-bar: low <= sl → 'loss' | high >= tp1 → 'win' | else → 'timeout'
  4. Write outcome + r_multiple back to DB

Run periodically (e.g. once a day) or after each trading session.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from fsp.data.feed import default_feed
from fsp.journal.db import unresolved_signals, update_outcome

log = logging.getLogger("fsp.resolver")

# Default max hold per strategy (M15 bars)
_DEFAULT_HOLD = {"TREND_RSI": 8, "LEVEL_OB": 8, "ECM": 32, "ARB": 12}
FETCH_DAYS = 3      # days of M15 data to fetch per signal (covers the hold window)


def _fetch_m15(pair: str, from_ts: datetime) -> pd.DataFrame:
    """Return M15 bars from from_ts onwards using yfinance."""
    f = default_feed("yf")
    end = datetime.now(timezone.utc)
    # Need enough history to cover from_ts + MAX_HOLD bars
    start = from_ts - timedelta(hours=1)   # slight buffer
    return f.history(pair, "M15", start, end)


def _resolve_one(sig: dict, m15: pd.DataFrame) -> tuple[str, float, str] | None:
    """
    Walk M15 bars after the signal timestamp.
    Returns (outcome, r_multiple, exit_ts) or None if bars not available.
    """
    sig_ts = pd.Timestamp(sig["ts"]).tz_localize("UTC") if not sig["ts"].endswith("+00:00") \
             else pd.Timestamp(sig["ts"])
    # Find bars strictly after the signal bar
    max_hold = sig.get("context", {}).get("max_hold_bars",
                  _DEFAULT_HOLD.get(sig.get("strategy", ""), 16))
    future = m15[m15.index > sig_ts].head(max_hold)
    if future.empty:
        return None  # data not yet available (signal too recent)

    entry    = sig["entry"]
    sl       = sig["sl"]
    tp1      = sig["tp1"]
    rr_tp1   = sig["rr_tp1"]
    direction = sig["direction"]
    risk     = abs(entry - sl)

    for ts_bar, bar in future.iterrows():
        hi = bar["high"]
        lo = bar["low"]
        if direction == "long":
            if lo <= sl:
                r = -1.0
                return "loss", r, ts_bar.isoformat()
            if hi >= tp1:
                r = rr_tp1
                return "win", r, ts_bar.isoformat()
        else:  # short
            if hi >= sl:
                r = -1.0
                return "loss", r, ts_bar.isoformat()
            if lo <= tp1:
                r = rr_tp1
                return "win", r, ts_bar.isoformat()

    # No SL/TP hit within max_hold — closed at last bar close
    last_close = float(future["close"].iloc[-1])
    if direction == "long":
        r = (last_close - entry) / risk if risk > 0 else 0.0
    else:
        r = (entry - last_close) / risk if risk > 0 else 0.0
    exit_ts = future.index[-1].isoformat()
    return "timeout", round(r, 3), exit_ts


def resolve_all(verbose: bool = False) -> dict[str, int]:
    """
    Resolve all unresolved signals (TREND_RSI, LEVEL_OB, ECM, ARB) in the journal.
    Returns counts: {'resolved': N, 'skipped': M, 'errors': K}
    """
    # Resolve ALL strategies (not just TREND_RSI)
    signals = unresolved_signals(None)
    if not signals:
        return {"resolved": 0, "skipped": 0, "errors": 0}

    # Pre-filter: skip signals fired within the last 3 hours — price hasn't settled
    cutoff = datetime.now(timezone.utc) - timedelta(hours=3)
    ready, too_recent = [], []
    for s in signals:
        ts = pd.Timestamp(s["ts"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if ts.to_pydatetime() < cutoff:
            ready.append(s)
        else:
            too_recent.append(s)
    skipped = len(too_recent)
    signals = ready

    # Group by pair to minimise fetches
    by_pair: dict[str, list[dict]] = {}
    for s in signals:
        by_pair.setdefault(s["pair"], []).append(s)

    resolved = errors = 0
    # skipped already initialized above with too_recent count
    for pair, sigs in by_pair.items():
        # Fetch once per pair (covers all signals for that pair)
        earliest_ts = min(pd.Timestamp(s["ts"]) for s in sigs)
        if earliest_ts.tzinfo is None:
            earliest_ts = earliest_ts.tz_localize("UTC")
        try:
            m15 = _fetch_m15(pair, earliest_ts.to_pydatetime())
        except Exception as e:
            log.error("M15 fetch failed for %s: %s", pair, e)
            errors += len(sigs)
            continue

        for sig in sigs:
            try:
                result = _resolve_one(sig, m15)
                if result is None:
                    skipped += 1
                    if verbose:
                        print(f"  SKIP {pair} {sig['ts'][:16]} — bars not available yet")
                else:
                    outcome, r, exit_ts = result
                    update_outcome(sig["id"], outcome, r, exit_ts)
                    resolved += 1
                    if verbose:
                        icon = "✅" if outcome == "win" else ("❌" if outcome == "loss" else "⏱")
                        print(f"  {icon} {pair} {sig['ts'][:16]} {sig['direction'].upper()} "
                              f"→ {outcome} {r:+.2f}R")
            except Exception as e:
                log.error("Resolve failed for signal %d: %s", sig["id"], e)
                errors += 1

    return {"resolved": resolved, "skipped": skipped, "errors": errors}
