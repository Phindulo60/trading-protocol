"""ICT confluence engine — turns the structure/liquidity/PD primitives into a
single, concise trade decision.

The model (long; short is the mirror):

  1. HTF bias       — bullish (from fsp.ict.bias)
  2. Liquidity      — a recent *sell-side* sweep ran the lows and reclaimed
                      (the trigger; defines the trade direction)
  3. Structure      — a bullish MSS / structure shift after the sweep
  4. Premium/Disc.  — price sits in discount, ideally the OTE golden zone
  5. PD array       — an unmitigated bullish FVG / order block to enter from
  6. Killzone       — London or NY-AM timing

Each confluence scores; the total maps to a grade (A+/A/B/skip). When a tradable
setup exists we attach a concrete plan: entry at the PD array (or OTE), stop
below the swept liquidity, target at the opposing unswept liquidity pool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from fsp.data.types import TF
from fsp.context.sessions import session_of, Session, DEFAULT_TZ
from fsp.ict.bias import htf_bias
from fsp.ict.structure import analyze_structure, StructureEvent
from fsp.ict.liquidity import find_liquidity_pools, find_sweeps, nearest_unswept, LiquiditySweep
from fsp.ict.premium_discount import dealing_range, DealingRange
from fsp.ict.smt import smt_divergence
from fsp.ict.draw import significant_levels, best_target, sweep_significance
from fsp.structure.fvg import find_fvgs, mark_mitigation
from fsp.structure.order_blocks import find_order_blocks, mark_ob_mitigation
from fsp.structure.displacement import atr

# Prime entry windows (NY clock). LONDON 02:00-05:00, NY_AM 07:00-12:00.
KILLZONES = {Session.LONDON, Session.NY_AM}

# score -> grade
GRADE_THRESHOLDS: list[tuple[str, int]] = [("A+", 9), ("A", 7), ("B", 5)]


@dataclass
class TradeDecision:
    ts: datetime
    direction: Literal["long", "short", "none"]
    grade: Literal["A+", "A", "B", "skip"]
    score: int
    htf_bias: str
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    rr: float | None = None
    pair: str | None = None
    confluences: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: str = ""
    smt: str | None = None          # 'confirmed' / 'none' / None (not evaluated)
    target_kind: str | None = None  # draw-on-liquidity target: PDH/PWH/EQH/swing/DR
    sweep_major: str | None = None  # major standing level the sweep ran (Turtle Soup tell)

    @property
    def is_tradable(self) -> bool:
        return self.direction != "none" and self.grade != "skip" \
            and self.entry is not None and self.stop is not None and self.target is not None

    def summary(self) -> str:
        if self.direction == "none":
            return f"[{self.grade}] no setup — {self.notes or 'no liquidity sweep'}"
        rr = f"{self.rr:.2f}R" if self.rr is not None else "n/a"
        plan = (f" entry={self.entry:.5f} stop={self.stop:.5f} "
                f"target={self.target:.5f} ({rr})") if self.entry is not None else ""
        return (f"[{self.grade}] {self.direction.upper()} score={self.score} "
                f"bias={self.htf_bias}{plan} | " + ", ".join(self.confluences))


def in_killzone(ts, tz: str = DEFAULT_TZ) -> bool:
    return session_of(pd.Timestamp(ts), tz) in KILLZONES


def _structure_after(events: list[StructureEvent], after_ts, want: str) -> StructureEvent | None:
    cands = [e for e in events if e.ts >= after_ts and e.direction == want]
    if not cands:
        return None
    mss = [e for e in cands if e.is_mss]
    return mss[-1] if mss else cands[-1]


def _entry_array(fvgs, obs, dr: DealingRange | None, direction: str):
    """Pick the proximal unmitigated PD array in the correct half.
    Returns (label, top, bottom) or None."""
    want = "bull" if direction == "long" else "bear"
    cands: list[tuple[str, float, float]] = []
    for f in fvgs:
        if f.direction == want and not f.mitigated:
            cands.append(("FVG", f.top, f.bottom))
    for ob in obs:
        if ob.direction == want and not ob.mitigated:
            cands.append(("OB", ob.top, ob.bottom))
    if not cands:
        return None
    if dr is not None:
        eq = dr.equilibrium
        if direction == "long":
            zoned = [c for c in cands if c[1] <= eq]            # array bottom in discount
        else:
            zoned = [c for c in cands if c[2] >= eq]            # array top in premium
        cands = zoned or cands
    # proximal: for long the highest array (entered from above), for short the lowest
    return max(cands, key=lambda c: c[1]) if direction == "long" \
        else min(cands, key=lambda c: c[2])


def _atr_ref(df: pd.DataFrame, atr_len: int) -> float:
    a = atr(df, atr_len).to_numpy()
    amed = np.nanmedian(a) if np.isfinite(a).any() else np.nan
    if not np.isfinite(amed) or amed == 0:
        amed = float((df["high"] - df["low"]).mean())
    return float(amed)


def decide(
    ltf_df: pd.DataFrame,
    htf_df: pd.DataFrame | None = None,
    pair: str | None = None,
    tf: TF = "M5",
    swing_length: int = 5,
    lookback: int = 30,
    atr_mult: float = 1.5,
    atr_len: int = 20,
    sl_buffer_atr: float = 0.1,
    tz: str = DEFAULT_TZ,
    drop_off_session: bool = True,
    exhaustion_score: int | None = 10,
    smt_df: pd.DataFrame | None = None,
    smt_sign: int = 1,
    smt_partner: str | None = None,
    context_df: pd.DataFrame | None = None,
    dol_targets: bool = False,
) -> TradeDecision:
    """Score the latest bar of `ltf_df` for an ICT setup. `htf_df` drives bias
    (falls back to `ltf_df`).

    Two robustness guards (from in-sample tuning, forward-validated via shadow):
      drop_off_session   — veto signals during the OFF session (illiquid
                           rollover/weekend; the only net-losing session bucket).
      exhaustion_score   — veto when total confluence hits this score or above:
                           max-confluence setups reverse almost always
                           ("over-confluence = exhaustion"). None disables.
    """
    last_ts = ltf_df.index[-1].to_pydatetime()
    sess = session_of(pd.Timestamp(last_ts), tz)
    bias = htf_bias(htf_df if htf_df is not None else ltf_df, swing_length, atr_mult, atr_len)

    # liquidity + the triggering sweep
    pools = find_liquidity_pools(ltf_df, swing_length, atr_len=atr_len)
    sweeps = find_sweeps(ltf_df, pools=pools, swing_length=swing_length, atr_len=atr_len)
    cutoff = ltf_df.index[max(0, len(ltf_df) - lookback)]
    recent = [s for s in sweeps if pd.Timestamp(s.ts) >= cutoff]
    if not recent:
        return TradeDecision(ts=last_ts, direction="none", grade="skip", score=0,
                             htf_bias=bias.direction, pair=pair,
                             notes="no recent liquidity sweep")

    sweep = recent[-1]                       # most recent sweep defines direction
    direction = "long" if sweep.direction == "bull" else "short"
    want_bias = "bull" if direction == "long" else "bear"

    st = analyze_structure(ltf_df, swing_length, atr_mult, atr_len)
    dr = dealing_range(ltf_df, swing_length)
    fvgs = mark_mitigation(find_fvgs(ltf_df, tf), ltf_df)
    obs = mark_ob_mitigation(find_order_blocks(ltf_df, tf, atr_mult, atr_len), ltf_df)
    last_close = float(ltf_df["close"].iloc[-1])

    # standing liquidity (PDH/PDL/...) — opt-in only. DOL targeting validated
    # net-NEGATIVE as a default (distant levels = lower hit rate); kept behind a
    # flag for experiments. Equal-pools are the one real magnet and the baseline
    # nearest_unswept already captures those.
    levels = None
    sweep_major = None
    if dol_targets:
        ctx = context_df if context_df is not None else ltf_df
        levels = significant_levels(ctx, tz)
        sweep_major = sweep_significance(sweep.level, sweep.side, levels,
                                         tol=0.3 * _atr_ref(ltf_df, atr_len))

    score = 0
    confs: list[str] = []
    missing: list[str] = []

    # 1) HTF bias alignment
    if bias.direction == want_bias:
        score += 2; confs.append(f"HTF bias aligned ({want_bias})")
    elif bias.direction == "neutral":
        confs.append("HTF bias neutral")
    else:
        score -= 2; missing.append(f"HTF bias opposes ({bias.direction})")

    # 2) liquidity sweep (the trigger — always present here)
    score += 2; confs.append(f"{sweep.side}-side sweep @ {sweep.level:.5f}")
    if sweep.kind == "equal":
        score += 1; confs.append("swept equal-pool (strong liquidity)")
    if sweep_major is not None:
        # canonical Turtle Soup — ran a major standing level. Record-only for
        # now (no score): validated before it earns confluence weight.
        confs.append(f"swept {sweep_major} (major standing level)")

    # 2b) SMT divergence — does the correlated pair fail to confirm the raid?
    # Record-only for now (no score): validated as predictive before it earns
    # confluence weight. Ch.17 Intermarket Relationships.
    smt_note = None
    if smt_df is not None:
        _si = smt_df.index.searchsorted(ltf_df.index[-1], side="right")
        sdf = smt_df.iloc[:_si]                                 # guard lookahead
        res = smt_divergence(ltf_df, sdf, ref_ts=sweep.pool.created_ts,
                             sweep_ts=sweep.ts, direction=direction,
                             sign=smt_sign, partner=smt_partner)
        smt_note = "confirmed" if res.diverged else "none"
        (confs if res.diverged else missing).append(
            f"SMT {'divergence' if res.diverged else 'no-confirm'} vs {smt_partner or 'partner'}")

    # 3) structure shift after the sweep, in direction
    mss_ev = _structure_after(st.events, sweep.ts, want_bias)
    if mss_ev is not None and mss_ev.is_mss:
        score += 2; confs.append("MSS confirms reversal")
    elif mss_ev is not None:
        score += 1; confs.append(f"{mss_ev.event_type} in direction")
    else:
        missing.append("no structure shift after sweep")

    # 4) premium/discount location
    if dr is not None:
        in_half = dr.in_discount(last_close) if direction == "long" else dr.in_premium(last_close)
        if in_half:
            score += 1; confs.append("price in " + ("discount" if direction == "long" else "premium"))
        else:
            missing.append("price not in correct PD half")
        if dr.in_ote(last_close, want_bias):
            score += 1; confs.append("price in OTE")
    else:
        missing.append("no dealing range")

    # 5) PD array (entry mechanism)
    arr = _entry_array(fvgs, obs, dr, direction)
    if arr is not None:
        score += 2; confs.append(f"{arr[0]} in zone")
    else:
        missing.append("no unmitigated PD array in zone")

    # 6) killzone timing
    if sess in KILLZONES:
        score += 1; confs.append("killzone")
    else:
        missing.append("outside killzone")

    # grade
    grade = "skip"
    for g, thresh in GRADE_THRESHOLDS:
        if score >= thresh:
            grade = g
            break

    # robustness guards (tuning-derived; see docstring). Veto -> grade=skip so
    # both backtest and live shadow drop the signal with no extra wiring.
    if drop_off_session and sess == Session.OFF:
        grade = "skip"; missing.append("OFF session (illiquid) — vetoed")
    if exhaustion_score is not None and score >= exhaustion_score:
        grade = "skip"
        missing.append(f"exhaustion guard (score>={exhaustion_score}) — vetoed")

    # trade plan — target the draw on liquidity (nearest significant standing
    # level / equal-pool), falling back to nearest structural swing, then DR.
    entry = stop = target = rr = None
    target_kind = None
    buf = sl_buffer_atr * _atr_ref(ltf_df, atr_len)
    if direction == "long":
        entry = arr[1] if arr is not None else (dr.ote_sweet_spot("bull") if dr else last_close)
        floor = min(sweep.extreme, arr[2]) if arr is not None else sweep.extreme
        stop = floor - buf
        if dol_targets:
            tp_price, target_kind = best_target(pools, levels, "buy", entry)
        else:
            tp_pool = nearest_unswept(pools, "buy", entry)
            tp_price = tp_pool.price if tp_pool is not None else None
            target_kind = tp_pool.kind if tp_pool is not None else None
        target = tp_price if tp_price is not None else (dr.high if dr else None)
        if tp_price is None:
            target_kind = "DR" if dr else None
    else:
        entry = arr[2] if arr is not None else (dr.ote_sweet_spot("bear") if dr else last_close)
        ceil = max(sweep.extreme, arr[1]) if arr is not None else sweep.extreme
        stop = ceil + buf
        if dol_targets:
            tp_price, target_kind = best_target(pools, levels, "sell", entry)
        else:
            tp_pool = nearest_unswept(pools, "sell", entry)
            tp_price = tp_pool.price if tp_pool is not None else None
            target_kind = tp_pool.kind if tp_pool is not None else None
        target = tp_price if tp_price is not None else (dr.low if dr else None)
        if tp_price is None:
            target_kind = "DR" if dr else None

    if entry is not None and stop is not None and target is not None:
        risk = abs(entry - stop)
        reward = (target - entry) if direction == "long" else (entry - target)
        rr = (reward / risk) if risk > 0 else None
        # a setup pointing the wrong way (target behind entry) is not tradable
        if rr is not None and rr <= 0:
            target = None; rr = None

    return TradeDecision(
        ts=last_ts, direction=direction, grade=grade, score=score,
        htf_bias=bias.direction, entry=entry, stop=stop, target=target, rr=rr,
        pair=pair, confluences=confs, missing=missing, smt=smt_note,
        target_kind=target_kind, sweep_major=sweep_major,
    )
