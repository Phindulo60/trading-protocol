"""
Phase 3 — walk-forward validation of phase 2 winners + new angles.

Splits the 12-month data into:
  - Train (first 6 months): May 2025 - Nov 2025
  - Test (last 6 months):   Nov 2025 - May 2026

A robust strategy should be profitable on BOTH halves.
"""
import time
from dataclasses import asdict, replace, dataclass
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

import runner
from runner import TrendRsiParams, run_backtest, compute_stats


def fmt(label: str, stats: dict) -> str:
    if stats.get("n", 0) == 0:
        return f"{label:<40} no trades"
    return (f"{label:<40} n={stats['n']:>3}  "
            f"WR={stats['wr']*100:5.1f}%  Exp={stats['exp']:+.2f}R  "
            f"PF={stats['pf']:5.2f}  TotalR={stats['total_r']:+7.1f}  "
            f"DD={stats['max_dd']:+5.1f}")


def split_data(m15, htf, split_ratio: float = 0.5):
    """Split data into train/test halves at given ratio."""
    n_m15 = len(m15)
    split_idx = int(n_m15 * split_ratio)
    split_ts = m15.index[split_idx]

    train_m15 = m15.iloc[:split_idx]
    test_m15 = m15.iloc[split_idx:]
    train_htf = htf[htf.index < split_ts]
    test_htf = htf[htf.index >= split_ts]

    return train_m15, test_m15, train_htf, test_htf


def run_walk_forward(name: str, p: TrendRsiParams):
    """Run on train then test, separately."""
    htf = runner.H4 if p.htf == "H4" else runner.H1
    train_m15, test_m15, train_htf, test_htf = split_data(runner.M15, htf, 0.5)

    train_trades = run_backtest(p, train_m15, train_htf)
    test_trades = run_backtest(p, test_m15, test_htf)

    return {
        "name": name,
        "train": compute_stats(train_trades),
        "test": compute_stats(test_trades),
    }


def run_full(name: str, p: TrendRsiParams):
    """Run on full data."""
    htf = runner.H4 if p.htf == "H4" else runner.H1
    trades = run_backtest(p, runner.M15, htf)
    return {"name": name, "full": compute_stats(trades)}


# Top phase 2 winners — must hold up out-of-sample
WINNERS = {
    "production_baseline": TrendRsiParams(),  # current live
    "MEGA_max": TrendRsiParams(htf_ema=10, rsi_long=44, rsi_short=56,
                                sessions=("LONDON","NY_AM","NY_PM")),
    "MEGA_bal": TrendRsiParams(htf_ema=10, atr_mult=1.75, max_hold_bars=16,
                                sessions=("LONDON","NY_AM","NY_PM")),
    "MEGA_qual": TrendRsiParams(htf_ema=10, atr_mult=2.0, rsi_long=32, rsi_short=68),
    "ema10+rsi44": TrendRsiParams(htf_ema=10, rsi_long=44, rsi_short=56),
    "ema10+all_sess": TrendRsiParams(htf_ema=10,
                                       sessions=("ASIA","LONDON","NY_AM","NY_PM")),
}


def main():
    t0 = time.time()
    base = TrendRsiParams()

    # ── Step 1: Walk-forward validation of top winners ──
    print("=" * 110)
    print("STEP 1 — WALK-FORWARD VALIDATION (Train: 1st 6mo | Test: last 6mo)")
    print("=" * 110)
    print(f"{'Variant':<40}     n   WR     Exp     PF    TotalR    DD")
    print("-" * 110)

    for name, p in WINNERS.items():
        result = run_walk_forward(name, p)
        train_s, test_s = result["train"], result["test"]
        train_str = (f"n={train_s['n']:>3} WR={train_s['wr']*100:4.0f}% "
                     f"Exp={train_s['exp']:+.2f}R PF={train_s['pf']:5.2f} "
                     f"R={train_s['total_r']:+6.1f} DD={train_s['max_dd']:+5.1f}")
        test_str = (f"n={test_s['n']:>3} WR={test_s['wr']*100:4.0f}% "
                    f"Exp={test_s['exp']:+.2f}R PF={test_s['pf']:5.2f} "
                    f"R={test_s['total_r']:+6.1f} DD={test_s['max_dd']:+5.1f}")
        # Robustness flag
        train_pos = train_s.get('total_r', 0) > 0
        test_pos = test_s.get('total_r', 0) > 0
        flag = "✅" if (train_pos and test_pos) else "⚠️" if (train_pos or test_pos) else "❌"
        print(f"{name:<35} {flag} TRAIN: {train_str}")
        print(f"{'':<35}    TEST:  {test_str}")
        print()

    # ── Step 2: Per-direction analysis (long-only vs short-only) ──
    print("\n" + "=" * 110)
    print("STEP 2 — DIRECTION SPLIT (does long or short alone beat balanced?)")
    print("=" * 110)

    # We need to extract direction-specific stats from MEGA_bal trades
    p_megabal = WINNERS["MEGA_bal"]
    htf = runner.H4
    trades = run_backtest(p_megabal, runner.M15, htf)
    longs = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]

    long_stats = compute_stats(longs)
    short_stats = compute_stats(shorts)
    print(fmt("MEGA_bal LONGS only", long_stats))
    print(fmt("MEGA_bal SHORTS only", short_stats))

    # ── Step 3: Day-of-week filter ──
    print("\n" + "=" * 110)
    print("STEP 3 — DAY-OF-WEEK PERFORMANCE (which days actually win?)")
    print("=" * 110)
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for d in range(7):
        d_trades = [t for t in trades if t.ts.tz_convert("America/New_York").weekday() == d]
        if d_trades:
            stats = compute_stats(d_trades)
            print(fmt(f"DOW {dow_names[d]}", stats))

    # ── Step 4: Hour-of-day analysis (NY time) ──
    print("\n" + "=" * 110)
    print("STEP 4 — HOUR-OF-DAY (NY time) PERFORMANCE")
    print("=" * 110)
    for h in range(24):
        h_trades = [t for t in trades if t.ts.tz_convert("America/New_York").hour == h]
        if len(h_trades) >= 5:
            stats = compute_stats(h_trades)
            print(fmt(f"NY hour {h:02d}:00", stats))

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
