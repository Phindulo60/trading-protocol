"""Setup grader — combines all signals into a tradeable A+/A/B/SKIP decision.

Direction comes from H1 OF bias. We look for a nearby unmitigated PDA (OB or FVG)
on the LTF in direction of bias, and verify the checklist around it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

import pandas as pd

from fsp.context.bias import compute_of_bias
from fsp.context.cycle import classify_cycle
from fsp.context.levels import htf_levels, mark_swept, monday_range
from fsp.context.sessions import session_of
from fsp.context.smt import detect_smt_positive, detect_smt_negative
from fsp.data.types import Direction, FVG, Grade, Level, OFBias, OrderBlock, Session
from fsp.structure.displacement import find_displacements, atr
from fsp.structure.fvg import find_fvgs, mark_mitigation
from fsp.structure.order_blocks import find_order_blocks, mark_ob_mitigation
from fsp.structure.swings import find_swings, mark_broken


@dataclass
class ChecklistItem:
    name: str
    passed: bool
    weight: float = 1.0
    note: str = ""

    @property
    def score(self) -> float:
        return self.weight if self.passed else 0.0


@dataclass
class SetupCandidate:
    pair: str
    direction: Direction | None
    grade: Grade
    entry: float | None
    sl: float | None
    tp1: float | None
    tp2: float | None
    invalidation_pips: float | None
    rr_tp1: float | None
    rr_tp2: float | None
    risk_r: float                   # 1.5 / 1.0 / 0.5 / 0.0
    key_level_ref: str              # "FVG@1.17580" etc.
    checklist: list[ChecklistItem] = field(default_factory=list)
    context: dict = field(default_factory=dict)

    def passed(self) -> int:
        return sum(1 for c in self.checklist if c.passed)

    def total(self) -> int:
        return len(self.checklist)


def _pip(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def _nearest_in_direction(price: float, fvgs: list[FVG], obs: list[OrderBlock],
                          direction: Direction) -> tuple[str, float, float, float] | None:
    """Find the closest unmitigated bull (long) or bear (short) PDA to current price.
    Returns (label, top, bottom, midpoint) or None.
    For LONG: need a bullish FVG/OB below price we can buy at.
    For SHORT: need a bearish FVG/OB above price we can sell at.
    """
    candidates = []
    for f in fvgs:
        if f.mitigated:
            continue
        if direction == "long" and f.direction == "bull" and f.top < price:
            candidates.append((f"FVG@{f.bottom:.5f}", f.top, f.bottom))
        if direction == "short" and f.direction == "bear" and f.bottom > price:
            candidates.append((f"FVG@{f.top:.5f}", f.top, f.bottom))
    for ob in obs:
        if ob.mitigated:
            continue
        if direction == "long" and ob.direction == "bull" and ob.top < price:
            candidates.append((f"OB@{ob.bottom:.5f}", ob.top, ob.bottom))
        if direction == "short" and ob.direction == "bear" and ob.bottom > price:
            candidates.append((f"OB@{ob.top:.5f}", ob.top, ob.bottom))
    if not candidates:
        return None
    # Closest by distance to midpoint
    best = min(candidates, key=lambda c: abs(price - (c[1] + c[2]) / 2))
    mid = (best[1] + best[2]) / 2
    return (best[0], best[1], best[2], mid)


def _nearest_opposing_level(price: float, direction: Direction,
                            levels: dict[str, Level]) -> Level | None:
    """For a LONG, find nearest unswept high above price.
    For a SHORT, find nearest unswept low below price.
    """
    best: Level | None = None
    for lvl in levels.values():
        if direction == "long" and lvl.kind == "high" and lvl.price > price and not lvl.swept:
            if best is None or lvl.price < best.price:
                best = lvl
        if direction == "short" and lvl.kind == "low" and lvl.price < price and not lvl.swept:
            if best is None or lvl.price > best.price:
                best = lvl
    return best


def grade_setup(pair: str, ltf_df: pd.DataFrame, h1_df: pd.DataFrame,
                daily_df: pd.DataFrame, other_df: pd.DataFrame | None = None,
                other_pair: str | None = None,
                dxy_df: pd.DataFrame | None = None,
                account_equity: float = 10_000.0,
                base_risk_pct: float = 0.005) -> SetupCandidate:
    """
    Produce a SetupCandidate with grade + checklist.

    ltf_df : M5/M15 bars (execution TF)
    h1_df  : H1 bars (bias TF)
    daily_df : daily bars (cycle + ADR)
    other_df : correlated pair for positive SMT (GBPUSD if pair=EURUSD, else EURUSD)
    dxy_df : DXY bars for negative SMT (optional)
    """
    pip = _pip(pair)
    now = ltf_df.index[-1]
    price = float(ltf_df["close"].iloc[-1])
    session = session_of(now)

    # --- Context signals ---
    cyc = classify_cycle(h1_df, daily_df)
    bias = compute_of_bias(h1_df, length=5)
    levels = mark_swept(htf_levels(h1_df), h1_df)
    mr = monday_range(h1_df) or {}
    all_levels = {**levels, **mr}

    # LTF swings / FVGs / OBs / displacements
    swings = mark_broken(find_swings(ltf_df, length=3), ltf_df)
    fvgs = mark_mitigation(find_fvgs(ltf_df, tf="M15"), ltf_df)
    obs = mark_ob_mitigation(find_order_blocks(ltf_df, tf="M15"), ltf_df)
    disps = find_displacements(ltf_df, mult=1.5, length=20)

    # SMT
    smt_events = []
    if other_df is not None and other_pair is not None:
        smt_events += detect_smt_positive(h1_df, other_df, pair, other_pair)
    if dxy_df is not None and not dxy_df.empty:
        smt_events += detect_smt_negative(h1_df, dxy_df, pair, "DXY")

    # Direction from H1 OF bias
    direction: Direction | None = None
    if bias.bias == OFBias.BULL:
        direction = "long"
    elif bias.bias == OFBias.BEAR:
        direction = "short"

    # Find a nearby LTF PDA aligned with direction
    pda = None
    if direction:
        pda = _nearest_in_direction(price, fvgs, obs, direction)

    # --- Checklist ---
    cl: list[ChecklistItem] = []

    cl.append(ChecklistItem(
        "Active execution session",
        passed=session in (Session.LONDON, Session.NY_AM, Session.NY_PM),
        note=f"session={session.value}"))

    cl.append(ChecklistItem(
        "Not lunch / off-hours",
        passed=session not in (Session.LUNCH, Session.OFF),
        note=f"session={session.value}"))

    cl.append(ChecklistItem(
        "Clear H1 OF bias (not neutral)",
        passed=bias.bias != OFBias.NEUTRAL,
        note=f"bias={bias.bias.value} ({bias.last_event})"))

    cl.append(ChecklistItem(
        "Cycle = EXPANSION",
        passed=cyc.cycle.value == "EXPANSION",
        note=f"cycle={cyc.cycle.value} ratio={cyc.atr_ratio:.2f}"))

    cl.append(ChecklistItem(
        "ADR% < 100%  (room to expand)",
        passed=cyc.adr_pct < 100,
        note=f"ADR%={cyc.adr_pct:.0f}"))

    # Recent sweep in direction of bias (within last 20 LTF bars)
    recent = ltf_df.tail(20)
    recent_sweep = False
    if direction == "long":
        for lvl in all_levels.values():
            if lvl.kind == "low":
                wicked = (recent["low"] < lvl.price) & (recent["close"] > lvl.price)
                if wicked.any():
                    recent_sweep = True
                    break
    elif direction == "short":
        for lvl in all_levels.values():
            if lvl.kind == "high":
                wicked = (recent["high"] > lvl.price) & (recent["close"] < lvl.price)
                if wicked.any():
                    recent_sweep = True
                    break
    cl.append(ChecklistItem("Recent liquidity sweep (≤20 LTF bars)",
                            passed=recent_sweep))


    cl.append(ChecklistItem("Unmitigated PDA available in direction",
                            passed=pda is not None,
                            note=pda[0] if pda else "none"))

    # SMT in same direction in last 24h
    smt_ok = False
    cutoff = now - pd.Timedelta(days=1)
    for ev in smt_events:
        if pd.Timestamp(ev.ts) < cutoff:
            continue
        if direction == "long" and ev.kind == "bull":
            smt_ok = True
        if direction == "short" and ev.kind == "bear":
            smt_ok = True
    cl.append(ChecklistItem("SMT aligned (≤24h)", passed=smt_ok))

    # Entry, SL, TP calc
    entry = sl = tp1 = tp2 = None
    inv_pips = rr1 = rr2 = None
    opposing = None
    if pda and direction:
        top, bot = pda[1], pda[2]
        entry = (top + bot) / 2
        # Buffer beyond the PDA edge for SL
        buffer = atr(ltf_df, 14).iloc[-1] * 0.3
        if direction == "long":
            sl = bot - buffer
        else:
            sl = top + buffer
        inv_pips = abs(entry - sl) / pip

        opposing = _nearest_opposing_level(entry, direction, all_levels)
        if opposing:
            risk = abs(entry - sl)
            reward1 = abs(opposing.price - entry)
            tp1 = opposing.price
            rr1 = reward1 / risk if risk else None
            # TP2 = one HTF level beyond
            others = [lvl for lvl in all_levels.values()
                      if lvl.kind == ("high" if direction == "long" else "low")
                      and not lvl.swept
                      and ((direction == "long" and lvl.price > opposing.price) or
                           (direction == "short" and lvl.price < opposing.price))]
            if others:
                target2 = min(others, key=lambda lv: abs(lv.price - opposing.price))
                tp2 = target2.price
                rr2 = abs(tp2 - entry) / risk if risk else None

    cl.append(ChecklistItem("Invalidation ≤ 30 pips",
                            passed=inv_pips is not None and inv_pips <= 30,
                            note=f"{inv_pips:.1f} pips" if inv_pips else "n/a"))
    cl.append(ChecklistItem("RR to TP1 ≥ 2.0",
                            passed=rr1 is not None and rr1 >= 2.0,
                            note=f"{rr1:.2f}R" if rr1 else "n/a"))

    # ---- Grade ----
    passed_count = sum(1 for c in cl if c.passed)
    total = len(cl)
    # Build a map of checklist items for convenient lookup
    cmap = {c.name: c for c in cl}
    # Append a new hard-filter item: only CHoCH entries (not BOS continuations)
    is_choch = "CHoCH" in bias.last_event
    cl.append(ChecklistItem("Entry is CHoCH (reversal, not BOS)",
                            passed=is_choch, note=bias.last_event))
    # Append ADR hard cap
    adr_ok = cyc.adr_pct < 100
    hard_blockers = [
        not cl[0].passed,                                  # session inactive
        cl[1].passed is False,                             # lunch/off
        direction is None,                                 # no bias
        pda is None,                                       # no PDA
        inv_pips is None or inv_pips > 40,                 # can't size
        not cmap.get("Cycle = EXPANSION",
                     ChecklistItem("x", True)).passed,     # require expansion
        not cmap.get("Recent liquidity sweep (≤20 LTF bars)",
                     ChecklistItem("x", True)).passed,     # require sweep
        not is_choch,                                      # require CHoCH (key edge finding)
        not adr_ok,                                        # require room to run
    ]
    if any(hard_blockers):
        grade = Grade.SKIP
        risk_r = 0.0
    elif passed_count >= total - 1:
        grade = Grade.A_PLUS
        risk_r = 1.5
    elif passed_count >= total - 3:
        grade = Grade.A
        risk_r = 1.0
    elif passed_count >= total - 5:
        grade = Grade.B
        risk_r = 0.5
    else:
        grade = Grade.SKIP
        risk_r = 0.0

    return SetupCandidate(
        pair=pair,
        direction=direction,
        grade=grade,
        entry=entry, sl=sl, tp1=tp1, tp2=tp2,
        invalidation_pips=inv_pips, rr_tp1=rr1, rr_tp2=rr2,
        risk_r=risk_r,
        key_level_ref=pda[0] if pda else "—",
        checklist=cl,
        context={
            "price": price,
            "session": session.value,
            "cycle": cyc.cycle.value,
            "adr_pct": round(cyc.adr_pct, 1),
            "bias": bias.bias.value,
            "bias_event": bias.last_event,
            "opposing_target": opposing.label if opposing else None,
            "opposing_price": opposing.price if opposing else None,
            "now": now.isoformat(),
        },
    )
