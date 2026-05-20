"""
Phase 2 — combine the best ideas from phase 1.

Phase 1 winners:
  - h4_ema_10: 68% WR, +0.79R (PF=4.21)  ← HUGE
  - rsi_30/70: 75% WR, +1.14R (only 36 trades)
  - sess_all/lo+ny: more trades, similar quality
  - atr_mult_1.75-2.0: higher WR but smaller R
  - hold_12-24b: lets winners run more

Phase 2 hypotheses:
  H1: h4_ema_10 + sessions widening = more high-quality trades
  H2: h4_ema_10 + atr_mult_2.0 = stack quality boosters
  H3: h4_ema_10 + hold_16b + tp1_4R = ride trends longer
  H4: per-session adaptive RSI (different threshold per session)
  H5: DOW filter (avoid Mondays where structure forms)
"""
import time
from dataclasses import replace
from runner import TrendRsiParams, run_variant


def fmt(res: dict) -> str:
    return (f"{res['name']:<40} n={res['n']:>3}  "
            f"WR={res['wr']*100:5.1f}%  Exp={res['exp']:+.2f}R  "
            f"PF={res['pf']:5.2f}  TotalR={res['total_r']:+7.1f}  "
            f"DD={res['max_dd']:+5.1f}  "
            f"L={res['n_long']:>2}/S={res['n_short']:>2}")


def main():
    t0 = time.time()
    base = TrendRsiParams()

    variants = []

    # ── Combo 1: h4_ema_10 + various session sets ──
    for sess, label in [
        (("NY_AM", "NY_PM"), "ny_only"),
        (("LONDON", "NY_AM"), "lo+nyam"),
        (("LONDON", "NY_AM", "NY_PM"), "lo+ny"),
        (("ASIA", "LONDON", "NY_AM", "NY_PM"), "all_sess"),
    ]:
        p = replace(base, htf_ema=10, sessions=sess)
        variants.append((f"ema10+{label}", p))

    # ── Combo 2: h4_ema_10 + ATR mult ──
    for m in [1.5, 1.75, 2.0, 2.5]:
        p = replace(base, htf_ema=10, atr_mult=m)
        variants.append((f"ema10+atr{m}", p))

    # ── Combo 3: h4_ema_10 + max hold variations ──
    for h in [8, 12, 16, 24, 32]:
        p = replace(base, htf_ema=10, max_hold_bars=h)
        variants.append((f"ema10+hold{h}b", p))

    # ── Combo 4: h4_ema_10 + RSI variations ──
    for thr in [30, 32, 34, 36, 38, 40, 44]:
        p = replace(base, htf_ema=10, rsi_long=thr, rsi_short=100 - thr)
        variants.append((f"ema10+rsi{thr}/{100-thr}", p))

    # ── Combo 5: h4_ema_10 + TP variations ──
    for tp in [2.5, 3.0, 3.5, 4.0, 5.0]:
        p = replace(base, htf_ema=10, tp1_r=tp, tp2_r=tp + 0.5)
        variants.append((f"ema10+tp{tp}R", p))

    # ── Mega combos: stack 3+ winners ──
    # Best frequency:  ema10 + lo+ny + rsi40/60
    p = replace(base, htf_ema=10, sessions=("LONDON","NY_AM","NY_PM"),
                rsi_long=40, rsi_short=60)
    variants.append(("MEGA_freq: ema10+lo+ny+rsi40", p))

    # Best quality: ema10 + atr2.0 + rsi32
    p = replace(base, htf_ema=10, atr_mult=2.0, rsi_long=32, rsi_short=68)
    variants.append(("MEGA_qual: ema10+atr2+rsi32", p))

    # Balanced: ema10 + lo+ny + atr1.75 + hold16
    p = replace(base, htf_ema=10, sessions=("LONDON","NY_AM","NY_PM"),
                atr_mult=1.75, max_hold_bars=16)
    variants.append(("MEGA_bal: ema10+lo+ny+atr1.75+hold16", p))

    # Best from phase 1: rsi_44/56
    p = replace(base, htf_ema=10, rsi_long=44, rsi_short=56,
                sessions=("LONDON","NY_AM","NY_PM"))
    variants.append(("MEGA_max: ema10+rsi44+lo+ny", p))

    # ── Run all ──
    print(f"Running {len(variants)} phase 2 variants...\n")
    results = []
    for i, (name, params) in enumerate(variants, 1):
        try:
            res = run_variant(name, params)
            results.append(res)
            print(f"[{i:>2}/{len(variants)}] {fmt(res)}")
        except Exception as e:
            print(f"[{i:>2}/{len(variants)}] {name}: FAILED ({e})")

    elapsed = time.time() - t0

    # Sort and rank
    valid = [r for r in results if r.get("n", 0) >= 20]
    valid.sort(key=lambda r: r["total_r"], reverse=True)

    print(f"\n{'═' * 100}")
    print(f"PHASE 2 — TOP 15 by TotalR  (elapsed {elapsed:.0f}s)")
    print(f"{'═' * 100}")
    for r in valid[:15]:
        print(fmt(r))

    print(f"\n{'═' * 100}")
    print(f"PHASE 2 — TOP 10 by Expectancy")
    print(f"{'═' * 100}")
    for r in sorted(valid, key=lambda r: r["exp"], reverse=True)[:10]:
        print(fmt(r))


if __name__ == "__main__":
    main()
