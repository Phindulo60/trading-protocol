"""
USDCAD Strategy Experiment Runner
==================================

Builds a parameterized TREND_RSI scanner, runs many variants over 12 months
of historical data, and produces a leaderboard.

Trade execution:
  - Fill at signal-bar close + spread/2
  - Walk forward bar-by-bar; check intra-bar high/low for SL or TP1/TP2 hits
  - SL first → -1R; TP1 first → partial 50% close, runner SL → BE, runs to TP2
  - Max-hold timeout → exit at close
  - One trade at a time (no stacking)
"""
from __future__ import annotations

import sys
import json
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional
import numpy as np
import pandas as pd

# Load cached data once
ROOT = "/Users/pmakhado/.aki/aki_workspace/trading-protocol/research/usdcad"
M15 = pd.read_parquet(f"{ROOT}/m15.parquet")
H4 = pd.read_parquet(f"{ROOT}/h4.parquet")
H1 = pd.read_parquet(f"{ROOT}/h1.parquet")
D = pd.read_parquet(f"{ROOT}/d.parquet")

PIP = 0.0001  # USDCAD


# ── Indicators ────────────────────────────────────────────────────────────────

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0)
    lo = (-d).clip(lower=0)
    ag = g.ewm(com=n - 1, adjust=False).mean()
    al = lo.ewm(com=n - 1, adjust=False).mean()
    r = ag / al.replace(0, np.nan)
    v = 100 - 100 / (1 + r)
    return v.where(al != 0, 100.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=n - 1, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    dm_p = (hi - hi.shift()).clip(lower=0).where((hi-hi.shift()) > (lo.shift()-lo), 0.0)
    dm_m = (lo.shift() - lo).clip(lower=0).where((lo.shift()-lo) > (hi-hi.shift()), 0.0)
    atr_n = tr.ewm(com=n-1, adjust=False).mean()
    di_p = 100 * dm_p.ewm(com=n-1, adjust=False).mean() / atr_n.replace(0, np.nan)
    di_m = 100 * dm_m.ewm(com=n-1, adjust=False).mean() / atr_n.replace(0, np.nan)
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    return dx.ewm(com=n-1, adjust=False).mean().fillna(0)


# ── Session detection (NY time) ──────────────────────────────────────────────

def session_name(ts: pd.Timestamp) -> str:
    """Return session label based on NY time."""
    ny = ts.tz_convert("America/New_York")
    hour = ny.hour
    if 3 <= hour < 8:
        return "LONDON"
    elif 8 <= hour < 12:
        return "NY_AM"
    elif 12 <= hour < 17:
        return "NY_PM"
    elif 19 <= hour or hour < 3:
        return "ASIA"
    return "OFF"


# ── Strategy parameters ───────────────────────────────────────────────────────

@dataclass
class TrendRsiParams:
    """All knobs of TREND_RSI strategy. Defaults match production v1."""
    # Trend filter
    htf: str = "H4"  # H4 or H1
    htf_ema: int = 20
    # RSI
    rsi_period: int = 14
    rsi_long: float = 38.0  # enter long when RSI < this
    rsi_short: float = 62.0  # enter short when RSI > this
    # Stop / TP
    atr_period: int = 14
    atr_mult: float = 1.5
    tp1_r: float = 3.5
    tp2_r: float = 4.0
    # Exit
    max_hold_bars: int = 8  # M15 bars
    # Filters
    sessions: tuple = ("NY_AM", "NY_PM")
    skip_friday: bool = True
    skip_sunday: bool = True
    min_inv_pips: float = 3.0
    max_inv_pips: float = 60.0
    # Optional ADX filter
    require_adx_below: Optional[float] = None  # e.g. 25 = mean-reversion regime
    require_adx_above: Optional[float] = None  # e.g. 18 = some structure


# ── Trade ────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    ts: pd.Timestamp
    direction: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    risk_pips: float
    outcome: str = "open"  # win1, win2, loss, be, timeout
    exit_ts: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    r_realized: float = 0.0
    bars_held: int = 0


# ── Scanner factory ──────────────────────────────────────────────────────────

def build_scanner(p: TrendRsiParams):
    """Return a function that takes (m15_df, htf_df) and returns a Signal dict or None."""
    def scan(m15_df: pd.DataFrame, htf_df: pd.DataFrame, full_m15: pd.DataFrame):
        if len(m15_df) < 55 or len(htf_df) < p.htf_ema + 2:
            return None

        ts = m15_df.index[-1]
        sess = session_name(ts)
        if sess not in p.sessions:
            return None

        ny = ts.tz_convert("America/New_York")
        dow = ny.weekday()
        if p.skip_friday and dow == 4:
            return None
        if p.skip_sunday and dow == 6:
            return None

        # HTF trend
        htf_close = float(htf_df["close"].iloc[-1])
        htf_ema_val = float(ema(htf_df["close"], p.htf_ema).iloc[-1])
        htf_bull = htf_close > htf_ema_val
        htf_bear = htf_close < htf_ema_val

        # M15 RSI + ATR
        rsi_v = float(rsi(m15_df["close"], p.rsi_period).iloc[-1])
        atr_v = float(atr(m15_df, p.atr_period).iloc[-1])
        price = float(m15_df["close"].iloc[-1])

        # Optional ADX filter
        if p.require_adx_below is not None or p.require_adx_above is not None:
            adx_v = float(adx(m15_df, 14).iloc[-1])
            if p.require_adx_below is not None and adx_v >= p.require_adx_below:
                return None
            if p.require_adx_above is not None and adx_v <= p.require_adx_above:
                return None

        direction = None
        if htf_bull and rsi_v < p.rsi_long:
            direction = "long"
        elif htf_bear and rsi_v > p.rsi_short:
            direction = "short"
        if direction is None:
            return None

        sl = price - atr_v * p.atr_mult if direction == "long" else price + atr_v * p.atr_mult
        risk = abs(price - sl)
        inv_pips = risk / PIP
        if not (p.min_inv_pips <= inv_pips <= p.max_inv_pips):
            return None

        tp1 = price + risk * p.tp1_r if direction == "long" else price - risk * p.tp1_r
        tp2 = price + risk * p.tp2_r if direction == "long" else price - risk * p.tp2_r

        return {
            "ts": ts,
            "direction": direction,
            "entry": price,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "risk_pips": inv_pips,
        }

    return scan


# ── Trade simulator ──────────────────────────────────────────────────────────

def simulate_trade(sig: dict, future_bars: pd.DataFrame, max_hold: int,
                   spread_pips: float = 0.3, sl_slippage_pips: float = 0.3) -> Trade:
    """Walk forward through future_bars, compute exit."""
    direction = sig["direction"]
    fill = sig["entry"] + (spread_pips * PIP / 2 if direction == "long" else -spread_pips * PIP / 2)
    sl = sig["sl"]
    tp1 = sig["tp1"]
    tp2 = sig["tp2"]
    risk = abs(fill - sl)

    t = Trade(
        ts=sig["ts"], direction=direction, entry=sig["entry"],
        sl=sl, tp1=tp1, tp2=tp2, risk_pips=sig["risk_pips"],
    )

    hit_tp1 = False
    runner_sl = sl  # moved to BE after TP1

    for i in range(min(max_hold, len(future_bars))):
        bar = future_bars.iloc[i]
        hi, lo = float(bar["high"]), float(bar["low"])
        bar_ts = future_bars.index[i]

        if direction == "long":
            if not hit_tp1:
                # Check if both SL and TP1 in same bar (tie → SL wins, conservative)
                if lo <= sl:
                    t.outcome = "loss"
                    t.exit_ts = bar_ts
                    t.exit_price = sl - sl_slippage_pips * PIP
                    t.r_realized = -1.0
                    t.bars_held = i + 1
                    return t
                if hi >= tp1:
                    hit_tp1 = True
                    runner_sl = fill  # BE
                    # Continue same bar to check TP2
                    if hi >= tp2:
                        t.outcome = "win2"
                        t.exit_ts = bar_ts
                        t.exit_price = tp2
                        t.r_realized = 0.5 * (tp1 - fill) / risk + 0.5 * (tp2 - fill) / risk
                        t.bars_held = i + 1
                        return t
            else:
                # In runner mode
                if lo <= runner_sl:
                    t.outcome = "be" if runner_sl == fill else "win1"
                    t.exit_ts = bar_ts
                    t.exit_price = runner_sl
                    # Realised: 50% at TP1, 50% at runner SL (BE)
                    r1 = 0.5 * (tp1 - fill) / risk
                    r2 = 0.5 * (runner_sl - fill) / risk
                    t.r_realized = r1 + r2
                    t.bars_held = i + 1
                    return t
                if hi >= tp2:
                    t.outcome = "win2"
                    t.exit_ts = bar_ts
                    t.exit_price = tp2
                    t.r_realized = 0.5 * (tp1 - fill) / risk + 0.5 * (tp2 - fill) / risk
                    t.bars_held = i + 1
                    return t
        else:  # short
            if not hit_tp1:
                if hi >= sl:
                    t.outcome = "loss"
                    t.exit_ts = bar_ts
                    t.exit_price = sl + sl_slippage_pips * PIP
                    t.r_realized = -1.0
                    t.bars_held = i + 1
                    return t
                if lo <= tp1:
                    hit_tp1 = True
                    runner_sl = fill
                    if lo <= tp2:
                        t.outcome = "win2"
                        t.exit_ts = bar_ts
                        t.exit_price = tp2
                        t.r_realized = 0.5 * (fill - tp1) / risk + 0.5 * (fill - tp2) / risk
                        t.bars_held = i + 1
                        return t
            else:
                if hi >= runner_sl:
                    t.outcome = "be" if runner_sl == fill else "win1"
                    t.exit_ts = bar_ts
                    t.exit_price = runner_sl
                    r1 = 0.5 * (fill - tp1) / risk
                    r2 = 0.5 * (fill - runner_sl) / risk
                    t.r_realized = r1 + r2
                    t.bars_held = i + 1
                    return t
                if lo <= tp2:
                    t.outcome = "win2"
                    t.exit_ts = bar_ts
                    t.exit_price = tp2
                    t.r_realized = 0.5 * (fill - tp1) / risk + 0.5 * (fill - tp2) / risk
                    t.bars_held = i + 1
                    return t

    # Timeout
    if len(future_bars) > 0:
        last_idx = min(max_hold - 1, len(future_bars) - 1)
        last_bar = future_bars.iloc[last_idx]
        last_close = float(last_bar["close"])
        t.bars_held = last_idx + 1
        t.exit_ts = future_bars.index[last_idx]
        t.exit_price = last_close
        if not hit_tp1:
            r = (last_close - fill) / risk if direction == "long" else (fill - last_close) / risk
            t.outcome = "timeout"
            t.r_realized = r
        else:
            r1 = 0.5 * (tp1 - fill) / risk if direction == "long" else 0.5 * (fill - tp1) / risk
            r2 = 0.5 * (last_close - fill) / risk if direction == "long" else 0.5 * (fill - last_close) / risk
            t.outcome = "timeout"
            t.r_realized = r1 + r2
    return t


# ── Backtest loop ────────────────────────────────────────────────────────────

def run_backtest(p: TrendRsiParams, m15_full: pd.DataFrame, htf_full: pd.DataFrame,
                 cooldown_bars: int = 4) -> list[Trade]:
    """Replay history bar-by-bar."""
    scan = build_scanner(p)
    trades = []
    last_signal_idx = -999
    open_trade_until = -1  # bar idx until which we're in a trade

    # Pre-compute all M15 indicators once for speed (no look-ahead in selection)
    # We still slice for the strategy call, but precompute speeds up the scan.
    n_total = len(m15_full)

    for i in range(60, n_total):
        if i < open_trade_until:
            continue  # in trade
        if i - last_signal_idx < cooldown_bars:
            continue  # cooldown

        ts = m15_full.index[i]
        # Slice M15 up to current bar
        m15_slice = m15_full.iloc[:i + 1]
        # Slice HTF up to current bar
        htf_slice = htf_full[htf_full.index <= ts]
        if len(htf_slice) < p.htf_ema + 2:
            continue

        sig = scan(m15_slice, htf_slice, m15_full)
        if sig is None:
            continue

        # Simulate trade using future bars
        future = m15_full.iloc[i + 1:]
        t = simulate_trade(sig, future, p.max_hold_bars)
        trades.append(t)

        last_signal_idx = i
        open_trade_until = i + t.bars_held + 1

    return trades


# ── Stats ────────────────────────────────────────────────────────────────────

def compute_stats(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0}
    n = len(trades)
    wins = [t for t in trades if t.r_realized > 0]
    losses = [t for t in trades if t.r_realized < 0]
    bes = [t for t in trades if t.r_realized == 0]
    total_r = sum(t.r_realized for t in trades)
    win_r = sum(t.r_realized for t in wins)
    loss_r = -sum(t.r_realized for t in losses)
    pf = win_r / loss_r if loss_r > 0 else float("inf")
    wr = len(wins) / n if n > 0 else 0

    # Max drawdown
    eq = [0.0]
    for t in trades:
        eq.append(eq[-1] + t.r_realized)
    peak = 0.0
    max_dd = 0.0
    for v in eq:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd

    return {
        "n": n,
        "wr": round(wr, 3),
        "exp": round(total_r / n, 3),
        "pf": round(pf, 2),
        "total_r": round(total_r, 1),
        "max_dd": round(-max_dd, 1),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "n_bes": len(bes),
        "n_long": sum(1 for t in trades if t.direction == "long"),
        "n_short": sum(1 for t in trades if t.direction == "short"),
        "avg_win_r": round(win_r / len(wins), 2) if wins else 0,
        "avg_loss_r": round(-loss_r / len(losses), 2) if losses else 0,
        "avg_bars_held": round(sum(t.bars_held for t in trades) / n, 1),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run_variant(name: str, params: TrendRsiParams) -> dict:
    """Run a single experiment and return stats + label."""
    htf_df = H4 if params.htf == "H4" else H1
    trades = run_backtest(params, M15, htf_df)
    stats = compute_stats(trades)
    return {"name": name, **stats, "params": asdict(params)}


if __name__ == "__main__":
    import time
    t0 = time.time()

    print(f"Data range: {M15.index[0]} to {M15.index[-1]}")
    print(f"M15 bars: {len(M15)}, H4 bars: {len(H4)}\n")

    # ── Baseline ──
    baseline = TrendRsiParams()
    print("Running BASELINE...")
    res = run_variant("baseline", baseline)
    print(f"  n={res['n']} WR={res['wr']*100:.1f}% Exp={res['exp']:+.2f}R "
          f"PF={res['pf']} TotalR={res['total_r']:+.1f} MaxDD={res['max_dd']}")
    print(f"  Long={res['n_long']} Short={res['n_short']} Wins={res['n_wins']} "
          f"Losses={res['n_losses']} BE={res['n_bes']}")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")
