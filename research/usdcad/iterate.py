"""
USDCAD Strategy Iteration — explore parameter space and rank.

Runs many variants of TREND_RSI on USDCAD and ranks by total R / Expectancy.
Prints a leaderboard.
"""
from __future__ import annotations

import time
from dataclasses import replace
from runner import TrendRsiParams, run_variant


def fmt(res: dict) -> str:
    """Pretty-format a result line."""
    return (f"{res['name']:<35} n={res['n']:>3}  "
            f"WR={res['wr']*100:5.1f}%  Exp={res['exp']:+.2f}R  "
            f"PF={res['pf']:5.2f}  TotalR={res['total_r']:+7.1f}  "
            f"DD={res['max_dd']:+5.1f}  "
            f"L={res['n_long']:>2}/S={res['n_short']:>2}")


def grid_search() -> list[dict]:
    """Enumerate variants to test."""
    base = TrendRsiParams()
    variants = [("baseline", base)]

    # 1. RSI threshold variations (symmetric)
    for thr in [30, 32, 34, 36, 38, 40, 42, 44]:
        p = replace(base, rsi_long=thr, rsi_short=100 - thr)
        variants.append((f"rsi_{thr}/{100 - thr}", p))

    # 2. ATR multiplier
    for m in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
        p = replace(base, atr_mult=m)
        variants.append((f"atr_mult_{m}", p))

    # 3. TP1 R-multiple (TP2 = TP1 + 0.5)
    for tp in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        p = replace(base, tp1_r=tp, tp2_r=tp + 0.5)
        variants.append((f"tp1_{tp}R", p))

    # 4. Max hold bars (M15)
    for h in [4, 6, 8, 12, 16, 24, 32]:
        p = replace(base, max_hold_bars=h)
        variants.append((f"hold_{h}b", p))

    # 5. HTF EMA period
    for n in [10, 15, 20, 30, 50]:
        p = replace(base, htf_ema=n)
        variants.append((f"h4_ema_{n}", p))

    # 6. Different HTF (H1 instead of H4)
    p = replace(base, htf="H1", htf_ema=100)  # equivalent ~25h trend
    variants.append(("h1_ema_100", p))
    p = replace(base, htf="H1", htf_ema=50)
    variants.append(("h1_ema_50", p))

    # 7. Session filters
    for sess_set, label in [
        (("LONDON",), "lo_only"),
        (("NY_AM",), "nyam_only"),
        (("NY_PM",), "nypm_only"),
        (("LONDON", "NY_AM"), "lo+nyam"),
        (("LONDON", "NY_AM", "NY_PM"), "lo+ny"),
        (("ASIA", "LONDON", "NY_AM", "NY_PM"), "all_sess"),
    ]:
        p = replace(base, sessions=sess_set)
        variants.append((f"sess_{label}", p))

    # 8. ADX filter — keep only when not strongly trending (mean-reversion)
    for adx_max in [20, 25, 30]:
        p = replace(base, require_adx_below=adx_max)
        variants.append((f"adx<{adx_max}", p))

    # Combo: best ideas often stack
    # We'll add these after we see initial results

    return variants


def main():
    t0 = time.time()
    variants = grid_search()
    print(f"Running {len(variants)} variants on USDCAD (12 months)...\n")

    results = []
    for i, (name, params) in enumerate(variants, 1):
        try:
            res = run_variant(name, params)
            results.append(res)
            print(f"[{i:>2}/{len(variants)}] {fmt(res)}")
        except Exception as e:
            print(f"[{i:>2}/{len(variants)}] {name}: FAILED ({e})")

    elapsed = time.time() - t0

    # ── Leaderboard ──
    valid = [r for r in results if r.get("n", 0) >= 20]  # need stat power
    valid.sort(key=lambda r: r["total_r"], reverse=True)

    print(f"\n{'═' * 95}")
    print(f"TOP 15 by Total R (n>=20 trades)  —  Elapsed: {elapsed:.1f}s")
    print(f"{'═' * 95}")
    for r in valid[:15]:
        print(fmt(r))

    print(f"\n{'═' * 95}")
    print("TOP 10 by Expectancy (n>=20 trades)")
    print(f"{'═' * 95}")
    by_exp = sorted(valid, key=lambda r: r["exp"], reverse=True)[:10]
    for r in by_exp:
        print(fmt(r))

    print(f"\n{'═' * 95}")
    print("WORST 5 — what NOT to do")
    print(f"{'═' * 95}")
    for r in sorted(valid, key=lambda r: r["total_r"])[:5]:
        print(fmt(r))


if __name__ == "__main__":
    main()
