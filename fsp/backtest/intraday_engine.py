"""Intraday strategies backtester — ECM (EMA Cross Momentum) and ARB (Asian Range Breakout).

Replays history bar-by-bar with no look-ahead. At each decision bar:
  1. Slice all reference frames up to the current timestamp.
  2. Run the strategy scanner on the slice.
  3. If a signal fires and no trade is open for that strategy, enter at bar-close.
  4. Walk forward bar-by-bar, check SL/TP hits on every subsequent bar.
  5. Log outcome as R-multiple.

Fill model (both strategies):
  - Market entry at close of signal bar + half-spread
  - SL / TP checked intra-bar via high/low of each subsequent bar
  - Partial exit at TP1 (50%), runner to TP2 or timeout
  - Max hold: ECM = 32 M15 bars (~8h), ARB = 96 M5 bars (~8h on M5)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

import pandas as pd

from fsp.backtest.engine import (
    BacktestResult, ExecConfig, Trade,
    _realised_r, _update_open,
)
from fsp.data.feed import default_feed
from fsp.signals.momentum import scan_momentum
from fsp.signals.breakout import scan_breakout

log = logging.getLogger("fsp.intraday_bt")


@dataclass
class IntradayExecConfig:
    spread_pips: float = 0.3
    sl_slippage_pips: float = 0.3
    partial_pct: float = 0.5
    ecm_max_hold_bars: int = 32   # M15 bars → max 8 h
    arb_max_hold_bars: int = 96   # M5 bars  → max 8 h
    cooldown_bars: int = 2        # bars between signals on same strategy
    dedup_minutes: int = 60       # skip if same entry re-fires within N min


def _slice_upto(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    return df[df.index <= ts]


def _enter_trade(sig, ts: pd.Timestamp, cfg: IntradayExecConfig,
                 pip: float) -> Trade:
    """Convert a Signal into an open Trade at market (close of signal bar).
    outcome is set to 'open' immediately — intraday signals fill at close,
    not via limit order (no pending state needed).
    """
    spread = cfg.spread_pips * pip
    fill = sig.entry + spread / 2 if sig.direction == "long" else sig.entry - spread / 2
    t = Trade(
        open_ts=ts.to_pydatetime(),
        close_ts=None,
        pair=sig.pair,
        direction=sig.direction,
        grade=sig.strategy,     # reuse grade field for strategy label
        entry=sig.entry,
        fill=fill,
        sl=sig.sl,
        tp1=sig.tp1,
        tp2=sig.tp2,
        risk_r=sig.risk_r,
        rr_tp1=sig.rr_tp1,
        rr_tp2=sig.rr_tp2,
        checklist_passed=0,
        checklist_total=0,
        session=sig.context.get("session", "-"),
        dow=ts.tz_convert("America/New_York").weekday(),
    )
    # Market fill — trade is open immediately
    t.outcome = "open"
    t.filled_ts = ts.to_pydatetime()
    return t


def _update_intraday(t: Trade, bar, bar_ts: pd.Timestamp,
                     cfg: IntradayExecConfig, pip: float,
                     max_hold: int) -> None:
    """Reuse the core SL/TP logic from engine.py with intraday hold limits."""
    # Temporarily set cfg-compatible values
    _cfg = ExecConfig(
        spread_pips=cfg.spread_pips,
        sl_slippage_pips=cfg.sl_slippage_pips,
        partial_pct=cfg.partial_pct,
        max_hold_bars=max_hold,
    )
    # Pass a minimal df (we only need the index for timestamp lookup)
    _update_open(t, bar, pd.DataFrame(), 0, _cfg, pip)
    if t.outcome in ("loss", "win1", "win2", "be", "timeout"):
        t.close_ts = bar_ts.to_pydatetime()


def run_intraday_backtest(
    pair: str,
    start: datetime,
    end: datetime,
    strategies: list[str] = ("ECM", "ARB"),
    feed_kind: str = "duka",
    stride_m15: int = 1,
    stride_m5: int = 1,
    cfg: IntradayExecConfig | None = None,
    verbose: bool = False,
) -> dict[str, BacktestResult]:
    """
    Run ECM and/or ARB backtests for one pair over [start, end].

    Returns a dict keyed by strategy name, each a BacktestResult.
    """
    cfg = cfg or IntradayExecConfig()
    pip = 0.01 if "JPY" in pair else 0.0001

    f = default_feed(feed_kind)
    warmup = timedelta(days=35)
    w_start = start - warmup

    if verbose:
        print(f"Loading data for {pair} …", flush=True)

    # yfinance caps M5/M15 at ~60 days; Dukascopy has no such limit.
    # For sub-hourly TFs we only need enough bars for indicators (55+ bars).
    now_utc = datetime.now(timezone.utc)
    ltf_warmup_start = max(w_start, now_utc - timedelta(days=58)) if feed_kind == "yf" else w_start

    # Load all data upfront
    m5_all  = f.history(pair, "M5",  ltf_warmup_start, end) if "ARB" in strategies else pd.DataFrame()
    m15_all = f.history(pair, "M15", ltf_warmup_start, end) if "ECM" in strategies else pd.DataFrame()
    h1_all  = f.history(pair, "H1",  w_start, end)
    d_all   = f.history(pair, "D",   w_start - timedelta(days=10), end)

    if verbose:
        print(f"  M5={len(m5_all)} M15={len(m15_all)} H1={len(h1_all)} D={len(d_all)}", flush=True)

    results: dict[str, BacktestResult] = {}

    for strat in strategies:
        if verbose:
            print(f"\n--- {strat} ---", flush=True)

        if strat == "ECM":
            sig_df = m15_all
            max_hold = cfg.ecm_max_hold_bars
            stride = stride_m15
            scan_fn = lambda ts, sdf, hdf, ddf: scan_momentum(pair, sdf, hdf, ddf)
            # We need enough M15 bars for indicators to warm up
            min_bars = 55
        elif strat == "ARB":
            sig_df = m5_all
            max_hold = cfg.arb_max_hold_bars
            stride = stride_m5
            scan_fn = lambda ts, sdf, hdf, ddf: scan_breakout(pair, sdf, hdf, ddf)
            min_bars = 100  # need full Asian session history
        else:
            continue

        if sig_df.empty:
            results[strat] = BacktestResult(start=start, end=end)
            continue

        _s = pd.Timestamp(start)
        _e = pd.Timestamp(end)
        bars_in_range = sig_df[(sig_df.index >= _s) & (sig_df.index <= _e)]
        decision_idx = list(range(0, len(bars_in_range), max(1, stride)))

        result = BacktestResult(start=start, end=end)
        open_trade: Trade | None = None
        last_signal_bar = -999
        last_entry_key: str | None = None
        n_scanned = 0
        n_fired = 0

        for i, li in enumerate(decision_idx):
            ts = bars_in_range.index[li]
            bar = bars_in_range.iloc[li]

            # ── Manage open trade ──────────────────────────────────
            if open_trade is not None and open_trade.outcome == "open":
                _update_intraday(open_trade, bar, ts, cfg, pip, max_hold)
                if open_trade.outcome not in ("open", "pending"):
                    result.trades.append(open_trade)
                    open_trade = None

            # If still in a trade, skip signal scan
            if open_trade is not None:
                continue
            if li - last_signal_bar < cfg.cooldown_bars:
                continue

            # ── Slice data up to current bar ───────────────────────
            sig_slice = sig_df[sig_df.index <= ts]
            if len(sig_slice) < min_bars:
                continue
            h1_slice = h1_all[h1_all.index <= ts]
            d_slice  = d_all[d_all.index <= ts]
            if len(h1_slice) < 55 or len(d_slice) < 5:
                continue

            # ── Run strategy ───────────────────────────────────────
            try:
                sig = scan_fn(ts, sig_slice, h1_slice, d_slice)
            except Exception as e:
                log.debug("%s scan err at %s: %s", strat, ts, e)
                continue
            n_scanned += 1

            if sig is None:
                continue

            # Dedup: skip if same entry fired recently
            entry_key = f"{sig.direction}|{sig.entry:.5f}"
            if entry_key == last_entry_key:
                continue
            last_entry_key = entry_key
            last_signal_bar = li
            n_fired += 1

            if verbose and n_fired <= 5:
                print(f"  Signal #{n_fired} at {ts}: {sig.direction} "
                      f"entry={sig.entry:.5f} sl={sig.sl:.5f} tp1={sig.tp1:.5f}", flush=True)

            open_trade = _enter_trade(sig, ts, cfg, pip)

        # Close any open trade at end of period
        if open_trade is not None and open_trade.outcome == "open":
            last_bar = bars_in_range.iloc[-1]
            r = _realised_r(open_trade.fill, float(last_bar["close"]),
                             open_trade.sl, open_trade.direction, pip,
                             ExecConfig(spread_pips=cfg.spread_pips))
            open_trade.outcome = "eop"
            open_trade.r_multiple = r
            open_trade.weighted_r = r * open_trade.risk_r
            open_trade.exit_price = float(last_bar["close"])
            open_trade.close_ts = bars_in_range.index[-1].to_pydatetime()
            result.trades.append(open_trade)

        if verbose:
            st = result.stats()
            print(f"  Scanned={n_scanned} fired={n_fired} trades={st.get('total',0)}", flush=True)
            if st.get("total", 0) > 0:
                print(f"  WR={st['win_rate']*100:.1f}%  Exp={st['expectancy']:+.3f}R  "
                      f"PF={st['profit_factor']:.2f}  TotalR={st['total_r']:+.1f}R  "
                      f"MaxDD={st['max_dd']:.1f}R", flush=True)

        results[strat] = result

    return results
