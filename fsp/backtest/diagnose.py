"""Backtest diagnostics — explain WHY trades win or lose.

For each closed trade we re-inspect its checklist (stored on the Trade via
grade_setup output), correlate pass/fail of each item with trade outcome,
and slice the trade population by session, direction, day-of-week, outcome type.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from fsp.backtest.engine import BacktestResult, ExecConfig, run_backtest
from fsp.data.feed import default_feed
from fsp.data.types import Grade
from fsp.grader.setup import grade_setup


@dataclass
class TradeDiag:
    trade_idx: int
    ts: datetime
    pair: str
    direction: str
    grade: str
    session: str
    dow: str
    weighted_r: float
    won: bool
    checklist: list[tuple[str, bool]]
    adr_pct: float
    cycle: str
    bias_event: str


def rebuild_diagnostics(
    pair: str,
    start: datetime,
    end: datetime,
    ltf: str = "M15",
    min_grade: Grade = Grade.A,
    stride: int = 4,
) -> tuple[BacktestResult, list[TradeDiag]]:
    """Rerun backtest and for each trade, recompute the checklist snapshot
    at the time of signal so we can correlate pass/fail with win/loss."""
    res = run_backtest(pair, start, end, ltf=ltf, min_grade=min_grade, stride=stride)

    f = default_feed("duka")
    warmup = start - timedelta(days=35)
    ltf_all = f.history(pair, ltf, warmup, end)
    h1_all = f.history(pair, "H1", warmup, end)
    daily_all = f.history(pair, "D", warmup - timedelta(days=30), end)
    other = "GBPUSD" if pair == "EURUSD" else "EURUSD"
    other_all = None
    try:
        other_all = f.history(other, "H1", warmup, end)
    except Exception:
        pass

    diags: list[TradeDiag] = []
    dows = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    for idx, t in enumerate(res.trades):
        if t.outcome in ("open", "pending"):
            continue
        ts = pd.Timestamp(t.open_ts)
        ltf_slice = ltf_all[ltf_all.index <= ts]
        h1_slice = h1_all[h1_all.index <= ts]
        daily_slice = daily_all[daily_all.index <= ts]
        other_slice = None
        if other_all is not None:
            other_slice = other_all[other_all.index <= ts]
        try:
            s = grade_setup(pair, ltf_slice, h1_slice, daily_slice,
                            other_df=other_slice, other_pair=other)
        except Exception:
            continue
        diags.append(TradeDiag(
            trade_idx=idx, ts=t.open_ts, pair=pair, direction=t.direction,
            grade=t.grade, session=t.session,
            dow=dows[t.dow] if t.dow < 7 else "?",
            weighted_r=t.weighted_r, won=t.weighted_r > 0,
            checklist=[(c.name, bool(c.passed)) for c in s.checklist],
            adr_pct=float(s.context.get("adr_pct", 0)),
            cycle=s.context.get("cycle", "-"),
            bias_event=s.context.get("bias_event", "-"),
        ))
    return res, diags


def checklist_correlation(diags: list[TradeDiag]) -> list[dict]:
    """For each checklist item, compute win rate when passed vs when failed."""
    if not diags:
        return []
    item_names = [n for n, _ in diags[0].checklist]
    out = []
    for name in item_names:
        passed_wins = passed_total = failed_wins = failed_total = 0
        for d in diags:
            for cn, cp in d.checklist:
                if cn != name:
                    continue
                if cp:
                    passed_total += 1
                    if d.won: passed_wins += 1
                else:
                    failed_total += 1
                    if d.won: failed_wins += 1
        p_wr = passed_wins / passed_total if passed_total else 0
        f_wr = failed_wins / failed_total if failed_total else 0
        lift = (p_wr - f_wr) * 100
        out.append({
            "item": name,
            "pass_n": passed_total, "pass_wr": p_wr * 100,
            "fail_n": failed_total, "fail_wr": f_wr * 100,
            "lift_pp": lift,
        })
    return sorted(out, key=lambda x: -x["lift_pp"])


def slice_stats(diags: list[TradeDiag], key_fn, min_n: int = 3) -> list[dict]:
    buckets: dict = defaultdict(list)
    for d in diags:
        buckets[key_fn(d)].append(d)
    rows = []
    for k, group in buckets.items():
        if len(group) < min_n:
            continue
        wins = sum(1 for g in group if g.won)
        total_r = sum(g.weighted_r for g in group)
        rows.append({
            "key": str(k), "n": len(group),
            "wr": wins / len(group) * 100,
            "exp": total_r / len(group),
        })
    return sorted(rows, key=lambda r: -r["exp"])
