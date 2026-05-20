"""
Phase 4 — explore COMPLETELY DIFFERENT strategy types beyond TREND_RSI.

Each strategy is implemented from scratch on USDCAD data, not just parameter
tweaks of mean reversion.

Candidates:
  1. DONCHIAN_BO — close above N-bar high in H4 trend direction (breakout)
  2. BB_TOUCH   — Bollinger Band touch with H4 trend filter (mean reversion)
  3. EMA_BOUNCE — H4 trend + price tags M15 EMA21 + bounces (continuation)
  4. ATR_BREAKOUT — sudden ATR expansion + directional close (volatility breakout)
  5. SESSION_HL — fade prior session high/low test in trend direction
"""
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

import runner
from runner import (
    M15, H4, H1, D, PIP,
    ema, rsi, atr, adx, session_name,
    Trade, simulate_trade, compute_stats
)


def _ny_dow(ts):
    return ts.tz_convert("America/New_York").weekday()


def _ny_hour(ts):
    ny = ts.tz_convert("America/New_York")
    return ny.hour + ny.minute / 60.0


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: DONCHIAN_BO — Breakout above 20-bar high in H4 trend
# ─────────────────────────────────────────────────────────────────────────────

def scan_donchian(m15_df, h4_df, lookback=20, ema_period=10,
                  atr_mult=1.5, tp1_r=3.0, tp2_r=4.5,
                  sessions=("LONDON", "NY_AM", "NY_PM")):
    if len(m15_df) < max(lookback, 55) + 5 or len(h4_df) < ema_period + 2:
        return None

    ts = m15_df.index[-1]
    sess = session_name(ts)
    if sess not in sessions:
        return None
    dow = _ny_dow(ts)
    if dow in (4, 6):  # no Friday/Sunday
        return None

    # H4 trend
    h4_close = float(h4_df["close"].iloc[-1])
    h4_ema = float(ema(h4_df["close"], ema_period).iloc[-1])
    h4_bull = h4_close > h4_ema
    h4_bear = h4_close < h4_ema
    if not h4_bull and not h4_bear:
        return None

    # Donchian channels: high of last `lookback` bars (excluding current)
    prior_high = float(m15_df["high"].iloc[-lookback - 1:-1].max())
    prior_low = float(m15_df["low"].iloc[-lookback - 1:-1].min())
    last_close = float(m15_df["close"].iloc[-1])
    last_open = float(m15_df["open"].iloc[-1])

    # Bull breakout: closed above prior high, in H4 bull trend
    direction = None
    if h4_bull and last_close > prior_high and last_open <= prior_high:
        direction = "long"
    elif h4_bear and last_close < prior_low and last_open >= prior_low:
        direction = "short"
    if direction is None:
        return None

    atr_v = float(atr(m15_df, 14).iloc[-1])
    sl = last_close - atr_v * atr_mult if direction == "long" else last_close + atr_v * atr_mult
    risk = abs(last_close - sl)
    inv_pips = risk / PIP
    if not (3 <= inv_pips <= 60):
        return None

    tp1 = last_close + risk * tp1_r if direction == "long" else last_close - risk * tp1_r
    tp2 = last_close + risk * tp2_r if direction == "long" else last_close - risk * tp2_r

    return {
        "ts": ts, "direction": direction, "entry": last_close,
        "sl": sl, "tp1": tp1, "tp2": tp2, "risk_pips": inv_pips,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: BB_TOUCH — Bollinger Band tag in H4 trend (mean reversion alt)
# ─────────────────────────────────────────────────────────────────────────────

def scan_bb(m15_df, h4_df, bb_period=20, bb_std=2.0, ema_period=10,
            atr_mult=1.5, tp1_r=2.5, tp2_r=3.5,
            sessions=("LONDON", "NY_AM", "NY_PM")):
    if len(m15_df) < max(bb_period, 55) + 5 or len(h4_df) < ema_period + 2:
        return None

    ts = m15_df.index[-1]
    sess = session_name(ts)
    if sess not in sessions:
        return None
    dow = _ny_dow(ts)
    if dow in (4, 6):
        return None

    h4_close = float(h4_df["close"].iloc[-1])
    h4_ema = float(ema(h4_df["close"], ema_period).iloc[-1])
    h4_bull = h4_close > h4_ema
    h4_bear = h4_close < h4_ema
    if not h4_bull and not h4_bear:
        return None

    close = m15_df["close"]
    sma = close.rolling(bb_period).mean()
    std = close.rolling(bb_period).std()
    upper = float((sma + bb_std * std).iloc[-1])
    lower = float((sma - bb_std * std).iloc[-1])
    last_low = float(m15_df["low"].iloc[-1])
    last_high = float(m15_df["high"].iloc[-1])
    last_close = float(close.iloc[-1])

    # Bull setup: tagged lower band in bull trend → reversion long
    direction = None
    if h4_bull and last_low <= lower and last_close > lower:
        direction = "long"
    elif h4_bear and last_high >= upper and last_close < upper:
        direction = "short"
    if direction is None:
        return None

    atr_v = float(atr(m15_df, 14).iloc[-1])
    sl = last_close - atr_v * atr_mult if direction == "long" else last_close + atr_v * atr_mult
    risk = abs(last_close - sl)
    inv_pips = risk / PIP
    if not (3 <= inv_pips <= 60):
        return None

    tp1 = last_close + risk * tp1_r if direction == "long" else last_close - risk * tp1_r
    tp2 = last_close + risk * tp2_r if direction == "long" else last_close - risk * tp2_r

    return {
        "ts": ts, "direction": direction, "entry": last_close,
        "sl": sl, "tp1": tp1, "tp2": tp2, "risk_pips": inv_pips,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: EMA_BOUNCE — H4 trend + M15 EMA21 pullback bounce (continuation)
# ─────────────────────────────────────────────────────────────────────────────

def scan_ema_bounce(m15_df, h4_df, m15_ema=21, h4_ema_period=10,
                    atr_mult=1.5, tp1_r=2.5, tp2_r=4.0,
                    sessions=("LONDON", "NY_AM", "NY_PM")):
    if len(m15_df) < 55 or len(h4_df) < h4_ema_period + 2:
        return None

    ts = m15_df.index[-1]
    sess = session_name(ts)
    if sess not in sessions:
        return None
    dow = _ny_dow(ts)
    if dow in (4, 6):
        return None

    h4_close = float(h4_df["close"].iloc[-1])
    h4_ema_v = float(ema(h4_df["close"], h4_ema_period).iloc[-1])
    h4_bull = h4_close > h4_ema_v
    h4_bear = h4_close < h4_ema_v
    if not h4_bull and not h4_bear:
        return None

    close = m15_df["close"]
    ema21 = ema(close, m15_ema)
    e_curr = float(ema21.iloc[-1])
    e_prev = float(ema21.iloc[-2])

    last = m15_df.iloc[-1]
    prev = m15_df.iloc[-2]
    last_low = float(last["low"])
    last_high = float(last["high"])
    last_close = float(last["close"])
    prev_close = float(prev["close"])

    # Bull bounce: prev bar low touched/below EMA21, current closes back above
    direction = None
    if h4_bull and prev["low"] <= e_prev and last_close > e_curr and prev_close <= e_prev * 1.0005:
        direction = "long"
    elif h4_bear and prev["high"] >= e_prev and last_close < e_curr and prev_close >= e_prev * 0.9995:
        direction = "short"
    if direction is None:
        return None

    atr_v = float(atr(m15_df, 14).iloc[-1])
    sl = last_close - atr_v * atr_mult if direction == "long" else last_close + atr_v * atr_mult
    risk = abs(last_close - sl)
    inv_pips = risk / PIP
    if not (3 <= inv_pips <= 60):
        return None

    tp1 = last_close + risk * tp1_r if direction == "long" else last_close - risk * tp1_r
    tp2 = last_close + risk * tp2_r if direction == "long" else last_close - risk * tp2_r

    return {
        "ts": ts, "direction": direction, "entry": last_close,
        "sl": sl, "tp1": tp1, "tp2": tp2, "risk_pips": inv_pips,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 4: ATR_BREAKOUT — sudden ATR expansion + directional close
# ─────────────────────────────────────────────────────────────────────────────

def scan_atr_breakout(m15_df, h4_df, atr_period=14, expansion_mult=1.6,
                      ema_period=10, atr_mult=1.5, tp1_r=2.5, tp2_r=4.0,
                      sessions=("LONDON", "NY_AM", "NY_PM")):
    """Triggers when current ATR > expansion_mult × ATR-50 average,
    and the bar closes in the H4 trend direction."""
    if len(m15_df) < 60 or len(h4_df) < ema_period + 2:
        return None

    ts = m15_df.index[-1]
    sess = session_name(ts)
    if sess not in sessions:
        return None
    dow = _ny_dow(ts)
    if dow in (4, 6):
        return None

    h4_close = float(h4_df["close"].iloc[-1])
    h4_ema_v = float(ema(h4_df["close"], ema_period).iloc[-1])
    h4_bull = h4_close > h4_ema_v
    h4_bear = h4_close < h4_ema_v
    if not h4_bull and not h4_bear:
        return None

    atr_series = atr(m15_df, atr_period)
    atr_curr = float(atr_series.iloc[-1])
    atr_avg = float(atr_series.iloc[-50:-1].mean())
    if atr_curr < atr_avg * expansion_mult:
        return None

    last = m15_df.iloc[-1]
    last_close = float(last["close"])
    last_open = float(last["open"])

    # Need bar to close in trend direction with strong body
    body = abs(last_close - last_open)
    rng = float(last["high"] - last["low"])
    if rng <= 0 or body / rng < 0.5:
        return None

    direction = None
    if h4_bull and last_close > last_open:
        direction = "long"
    elif h4_bear and last_close < last_open:
        direction = "short"
    if direction is None:
        return None

    sl = last_close - atr_curr * atr_mult if direction == "long" else last_close + atr_curr * atr_mult
    risk = abs(last_close - sl)
    inv_pips = risk / PIP
    if not (3 <= inv_pips <= 60):
        return None

    tp1 = last_close + risk * tp1_r if direction == "long" else last_close - risk * tp1_r
    tp2 = last_close + risk * tp2_r if direction == "long" else last_close - risk * tp2_r

    return {
        "ts": ts, "direction": direction, "entry": last_close,
        "sl": sl, "tp1": tp1, "tp2": tp2, "risk_pips": inv_pips,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 5: SESSION_HL_FADE — fade ASIA session high/low in H4 trend
# ─────────────────────────────────────────────────────────────────────────────

def scan_session_fade(m15_df, h4_df, ema_period=10, atr_mult=1.5,
                      tp1_r=2.5, tp2_r=4.0,
                      sessions=("LONDON", "NY_AM")):
    """During London session, fade tests of Asian session H/L
    (looking for trend continuation against fakeout)."""
    if len(m15_df) < 100 or len(h4_df) < ema_period + 2:
        return None

    ts = m15_df.index[-1]
    sess = session_name(ts)
    if sess not in sessions:
        return None
    dow = _ny_dow(ts)
    if dow in (4, 6):
        return None

    # Find Asian session range from yesterday's Asia
    ny_ts = ts.tz_convert("America/New_York")
    today_start = ny_ts.normalize()
    asia_start = today_start - pd.Timedelta(hours=5)  # ~19:00 NY prior day
    asia_end = today_start + pd.Timedelta(hours=3)  # ~03:00 NY today

    asia_window = m15_df[
        (m15_df.index.tz_convert("America/New_York") >= asia_start) &
        (m15_df.index.tz_convert("America/New_York") < asia_end)
    ]
    if len(asia_window) < 8:
        return None

    asia_high = float(asia_window["high"].max())
    asia_low = float(asia_window["low"].min())
    asia_range = asia_high - asia_low
    if asia_range < 5 * PIP or asia_range > 50 * PIP:
        return None

    h4_close = float(h4_df["close"].iloc[-1])
    h4_ema_v = float(ema(h4_df["close"], ema_period).iloc[-1])
    h4_bull = h4_close > h4_ema_v
    h4_bear = h4_close < h4_ema_v

    last = m15_df.iloc[-1]
    last_high = float(last["high"])
    last_low = float(last["low"])
    last_close = float(last["close"])

    # Bull setup: H4 bull, price tested asia_low, closed back above
    direction = None
    if h4_bull and last_low <= asia_low and last_close > asia_low:
        direction = "long"
    elif h4_bear and last_high >= asia_high and last_close < asia_high:
        direction = "short"
    if direction is None:
        return None

    atr_v = float(atr(m15_df, 14).iloc[-1])
    sl = last_close - atr_v * atr_mult if direction == "long" else last_close + atr_v * atr_mult
    risk = abs(last_close - sl)
    inv_pips = risk / PIP
    if not (3 <= inv_pips <= 60):
        return None

    tp1 = last_close + risk * tp1_r if direction == "long" else last_close - risk * tp1_r
    tp2 = last_close + risk * tp2_r if direction == "long" else last_close - risk * tp2_r

    return {
        "ts": ts, "direction": direction, "entry": last_close,
        "sl": sl, "tp1": tp1, "tp2": tp2, "risk_pips": inv_pips,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Generic backtest runner
# ─────────────────────────────────────────────────────────────────────────────

def run_strategy(name: str, scan_fn, scan_kwargs: dict, max_hold: int = 16,
                 cooldown_bars: int = 4) -> dict:
    """Backtest a strategy on M15+H4 data."""
    trades = []
    last_signal_idx = -999
    open_until = -1

    for i in range(60, len(M15)):
        if i < open_until:
            continue
        if i - last_signal_idx < cooldown_bars:
            continue

        ts = M15.index[i]
        m15_slice = M15.iloc[:i + 1]
        h4_slice = H4[H4.index <= ts]
        if len(h4_slice) < 12:
            continue

        sig = scan_fn(m15_slice, h4_slice, **scan_kwargs)
        if sig is None:
            continue

        future = M15.iloc[i + 1:]
        t = simulate_trade(sig, future, max_hold)
        trades.append(t)
        last_signal_idx = i
        open_until = i + t.bars_held + 1

    stats = compute_stats(trades)
    return {"name": name, **stats}


def fmt(res: dict) -> str:
    if res.get("n", 0) == 0:
        return f"{res['name']:<35} no trades"
    return (f"{res['name']:<35} n={res['n']:>3}  "
            f"WR={res['wr']*100:5.1f}%  Exp={res['exp']:+.2f}R  "
            f"PF={res['pf']:5.2f}  TotalR={res['total_r']:+7.1f}  "
            f"DD={res['max_dd']:+5.1f}  "
            f"L={res['n_long']:>2}/S={res['n_short']:>2}")


def main():
    t0 = time.time()
    results = []

    print(f"Phase 4 — exploring NEW strategy types on USDCAD 12mo")
    print(f"{'─' * 100}\n")

    # 1. Donchian
    for lb in [10, 15, 20, 30]:
        r = run_strategy(f"Donchian_{lb}b", scan_donchian, {"lookback": lb})
        results.append(r); print(fmt(r))

    # 2. Bollinger Bands
    for bb_p, bb_s in [(20, 2.0), (20, 2.5), (15, 2.0)]:
        r = run_strategy(f"BB_{bb_p}/{bb_s}", scan_bb,
                         {"bb_period": bb_p, "bb_std": bb_s})
        results.append(r); print(fmt(r))

    # 3. EMA bounce
    for em in [13, 21, 34]:
        r = run_strategy(f"EMA{em}_bounce", scan_ema_bounce, {"m15_ema": em})
        results.append(r); print(fmt(r))

    # 4. ATR expansion breakout
    for em_mult in [1.4, 1.6, 1.8, 2.0]:
        r = run_strategy(f"ATR_BO_{em_mult}x", scan_atr_breakout,
                         {"expansion_mult": em_mult})
        results.append(r); print(fmt(r))

    # 5. Asia HL fade
    r = run_strategy("Session_AsiaHL_fade", scan_session_fade, {})
    results.append(r); print(fmt(r))

    # ─────────────────────────────────────────────────────────────────────────
    # Leaderboard
    valid = [r for r in results if r.get("n", 0) >= 15]
    valid.sort(key=lambda r: r["total_r"], reverse=True)

    elapsed = time.time() - t0
    print(f"\n{'═' * 100}")
    print(f"PHASE 4 — TOP STRATEGIES by Total R  (elapsed {elapsed:.0f}s)")
    print(f"{'═' * 100}")
    for r in valid:
        print(fmt(r))

    print(f"\n{'═' * 100}")
    print(f"PHASE 4 — Compared to MEGA_bal (TREND_RSI v2): n=150 WR=64.7% Exp=+1.02R "
          f"PF=4.65 TotalR=+152.5R")
    print(f"{'═' * 100}")
    if valid:
        winner = valid[0]
        if winner["total_r"] > 100:
            print(f"⭐ Best new strategy: {winner['name']} — "
                  f"could COMPLEMENT MEGA_bal (different setups)")
        else:
            print("None of the new strategies match MEGA_bal in raw R.")
            print("But high-WR ones might add diversification value.")


if __name__ == "__main__":
    main()
