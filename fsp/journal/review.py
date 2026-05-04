"""Learning report — aggregates resolved signals to surface pattern insights.

Shows: overall stats, breakdown by RSI depth, session, day-of-week, pair.
Goal: understand which conditions produce the best outcomes so you can
      prioritise those setups in live trading.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from fsp.journal.db import resolved_signals


def _bucket_rsi(rsi: float, direction: str) -> str:
    """Bucket RSI depth relative to the 38/62 threshold."""
    depth = (38 - rsi) if direction == "long" else (rsi - 62)
    if depth >= 8:   return "deep  (>8 from threshold)"
    if depth >= 4:   return "mid   (4–8)"
    if depth >= 0:   return "edge  (0–4)"
    return "outside"  # shouldn't happen — signal shouldn't have fired


def _stats(trades: list[dict]) -> dict[str, Any]:
    if not trades:
        return {}
    n = len(trades)
    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    timeouts = [t for t in trades if t["outcome"] == "timeout"]
    rs = [t["r_multiple"] for t in trades if t["r_multiple"] is not None]
    total_r = sum(rs)
    win_r = sum(r for r in rs if r > 0)
    loss_r = abs(sum(r for r in rs if r < 0))
    pf = win_r / loss_r if loss_r > 0 else float("inf")
    return {
        "n": n,
        "wr": len(wins) / n,
        "exp": total_r / n if n else 0.0,
        "total_r": total_r,
        "pf": pf,
        "wins": len(wins),
        "losses": len(losses),
        "timeouts": len(timeouts),
    }


def _fmt_row(label: str, s: dict, width: int = 28) -> str:
    if not s:
        return f"  {label:<{width}}  (no data)"
    bar = "█" * int(s["wr"] * 10)
    return (
        f"  {label:<{width}}"
        f"  n={s['n']:>3}"
        f"  WR={s['wr']*100:4.0f}%  {bar:<10}"
        f"  Exp={s['exp']:+.2f}R"
        f"  PF={s['pf']:.1f}"
        f"  Total={s['total_r']:+.1f}R"
        f"  [W:{s['wins']} L:{s['losses']} T:{s['timeouts']}]"
    )


def build_report(strategy: str = "TREND_RSI") -> str:
    trades = resolved_signals(strategy)
    if not trades:
        return "No resolved signals yet. Run `fsp resolve-outcomes` first."

    lines = [
        f"\n{'='*80}",
        f"SIGNAL REVIEW  ({strategy})  —  {len(trades)} resolved trades",
        f"{'='*80}",
    ]

    # ── Overall ──────────────────────────────────────────────────────────────
    overall = _stats(trades)
    lines += [
        "",
        "OVERALL",
        _fmt_row("All signals", overall),
    ]

    # ── By pair ──────────────────────────────────────────────────────────────
    by_pair: dict[str, list] = defaultdict(list)
    for t in trades:
        by_pair[t["pair"]].append(t)
    lines += ["", "BY PAIR"]
    for pair in sorted(by_pair, key=lambda p: -_stats(by_pair[p]).get("total_r", 0)):
        lines.append(_fmt_row(pair, _stats(by_pair[pair])))

    # ── By RSI depth ─────────────────────────────────────────────────────────
    by_rsi: dict[str, list] = defaultdict(list)
    for t in trades:
        rsi = t["context"].get("rsi")
        if rsi is not None:
            bucket = _bucket_rsi(float(rsi), t["direction"])
            by_rsi[bucket].append(t)
    if by_rsi:
        lines += ["", "BY RSI DEPTH AT ENTRY  (how far into oversold/overbought)"]
        order = ["deep  (>8 from threshold)", "mid   (4–8)", "edge  (0–4)"]
        for b in order:
            if b in by_rsi:
                lines.append(_fmt_row(b, _stats(by_rsi[b])))

    # ── By session ───────────────────────────────────────────────────────────
    by_sess: dict[str, list] = defaultdict(list)
    for t in trades:
        sess = t["context"].get("session", "?")
        by_sess[sess].append(t)
    if by_sess:
        lines += ["", "BY SESSION"]
        for sess in sorted(by_sess):
            lines.append(_fmt_row(sess, _stats(by_sess[sess])))

    # ── By day of week ───────────────────────────────────────────────────────
    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    by_dow: dict[str, list] = defaultdict(list)
    for t in trades:
        try:
            import pandas as pd
            ts = pd.Timestamp(t["ts"]).tz_localize("UTC") if not t["ts"].endswith("+00:00") \
                 else pd.Timestamp(t["ts"])
            ny_dow = ts.tz_convert("America/New_York").weekday()
            by_dow[dow_names[ny_dow]].append(t)
        except Exception:
            pass
    if by_dow:
        lines += ["", "BY DAY OF WEEK"]
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
            if day in by_dow:
                lines.append(_fmt_row(day, _stats(by_dow[day])))

    # ── By H4 trend strength (if ADX stored) ─────────────────────────────────
    by_adx: dict[str, list] = defaultdict(list)
    for t in trades:
        adx = t["context"].get("adx")
        if adx is not None:
            b = "strong (ADX>25)" if float(adx) > 25 else "weak   (ADX≤25)"
            by_adx[b].append(t)
    if by_adx:
        lines += ["", "BY H4 ADX STRENGTH"]
        for b in sorted(by_adx):
            lines.append(_fmt_row(b, _stats(by_adx[b])))

    # ── Recent signals (last 20) ──────────────────────────────────────────────
    lines += ["", f"RECENT SIGNALS (last {min(20, len(trades))})"]
    lines.append(
        f"  {'ts':<17} {'pair':<8} {'dir':<6} {'RSI':>5} {'sess':<8} "
        f"{'outcome':<9} {'R':>6}  {'exit_ts':<17}"
    )
    lines.append("  " + "-" * 80)
    for t in reversed(trades[-20:]):
        rsi = t["context"].get("rsi", "?")
        sess = t["context"].get("session", "?")
        outcome = t.get("outcome") or "-"
        r = f"{t['r_multiple']:+.2f}R" if t.get("r_multiple") is not None else "-"
        exit_ts = (t.get("exit_ts") or "")[:16]
        icon = {"win": "✅", "loss": "❌", "timeout": "⏱"}.get(outcome, " ")
        lines.append(
            f"  {t['ts'][:16]:<17} {t['pair']:<8} {t['direction']:<6} "
            f"{str(rsi):>5} {sess:<8} {icon}{outcome:<8} {r:>6}  {exit_ts}"
        )

    lines.append(f"\n{'='*80}")
    return "\n".join(lines)
