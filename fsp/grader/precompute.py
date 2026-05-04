"""Precompute all structure + context over a full window, exposing as-of-T lookups.

Instead of re-running find_swings/find_fvgs/etc per decision bar, we detect once
over the whole backtest window and enrich each artefact with its "mitigation_ts"
or equivalent first-invalidation timestamp. At query time T we filter by
confirmation_ts <= T and treat an artefact as mitigated only if mitigation_ts <= T.

This keeps strict no-look-ahead semantics: a swing at bar i is only visible
from bar i+L onwards (after pivot-length confirmation), and its broken/mitigated
flags reflect only prices <= T.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from fsp.context.bias import BiasState
from fsp.context.cycle import CycleState
from fsp.context.levels import _resample_hl
from fsp.context.sessions import DEFAULT_TZ, session_of
from fsp.context.smt import SMTEvent
from fsp.data.types import Cycle, FVG, Grade, Level, OFBias, OrderBlock, Session, Swing
from fsp.structure.displacement import atr
from fsp.structure.fvg import find_fvgs
from fsp.structure.order_blocks import find_order_blocks
from fsp.structure.swings import find_swings


# -----------------------------------------------------------------------------
# Rich wrappers that carry an "invalidation timestamp" in addition to the flag.
# We keep the same attribute names as the originals so grader code still works.
# -----------------------------------------------------------------------------

@dataclass
class FVGRich:
    ts: datetime
    top: float
    bottom: float
    direction: str
    tf: str
    confirm_ts: datetime           # when this FVG became visible (= ts, the 3rd bar)
    mit_ts: datetime | None = None # first ts at which it was mitigated
    inv_ts: datetime | None = None

    @property
    def mitigated(self) -> bool:
        return self.mit_ts is not None

    @property
    def inverted(self) -> bool:
        return self.inv_ts is not None

    def mitigated_as_of(self, ts) -> bool:
        return self.mit_ts is not None and self.mit_ts <= ts


@dataclass
class OBRich:
    ts: datetime
    top: float
    bottom: float
    direction: str
    tf: str
    confirm_ts: datetime           # when the displacement that created it closed
    mit_ts: datetime | None = None

    @property
    def mitigated(self) -> bool:
        return self.mit_ts is not None

    def mitigated_as_of(self, ts) -> bool:
        return self.mit_ts is not None and self.mit_ts <= ts


@dataclass
class LevelRich:
    price: float
    label: str
    kind: str
    ts: datetime
    swept_ts: datetime | None = None

    @property
    def swept(self) -> bool:
        return self.swept_ts is not None

    def swept_as_of(self, ts) -> bool:
        return self.swept_ts is not None and self.swept_ts <= ts


# -----------------------------------------------------------------------------
# Precompute bundle
# -----------------------------------------------------------------------------

@dataclass
class Precomputed:
    pair: str
    ltf: pd.DataFrame
    h1: pd.DataFrame
    daily: pd.DataFrame

    # LTF arrays (numpy for hot path)
    ltf_low: np.ndarray = field(default=None)
    ltf_high: np.ndarray = field(default=None)
    ltf_close: np.ndarray = field(default=None)
    ltf_index: pd.DatetimeIndex = field(default=None)
    ltf_atr14: pd.Series = field(default=None)

    # Structures (static; filtered by confirm_ts at query time)
    fvgs: list[FVGRich] = field(default_factory=list)
    obs: list[OBRich] = field(default_factory=list)

    # Bias timeline — list of (event_ts, OFBias, last_event_str) sorted
    bias_events: list[tuple[pd.Timestamp, OFBias, str]] = field(default_factory=list)

    # Cycle precompute
    atr5_h1: pd.Series = field(default=None)
    atr50_h1: pd.Series = field(default=None)
    daily_range: pd.Series = field(default=None)   # indexed by local-date
    daily_range_local_idx: pd.DatetimeIndex = field(default=None)

    # HTF levels: per-local-date snapshot
    htf_snapshots: dict[pd.Timestamp, dict[str, LevelRich]] = field(default_factory=dict)
    monday_snapshots: dict[pd.Timestamp, dict[str, LevelRich]] = field(default_factory=dict)

    # SMT
    smt_events: list[SMTEvent] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _first_hit_ts(ts_start, tops_mask_arr, idx: pd.DatetimeIndex,
                   low: np.ndarray, high: np.ndarray,
                   top: float, bottom: float) -> datetime | None:
    """Find first bar after ts_start whose [low, high] overlaps [bottom, top]."""
    # locate insertion point
    start_i = idx.searchsorted(ts_start, side="right")
    if start_i >= len(idx):
        return None
    hit_mask = (low[start_i:] <= top) & (high[start_i:] >= bottom)
    if not hit_mask.any():
        return None
    first = int(np.argmax(hit_mask))
    return idx[start_i + first].to_pydatetime()


def _first_close_beyond_ts(ts_start, idx, close: np.ndarray, price: float,
                            direction: str) -> datetime | None:
    start_i = idx.searchsorted(ts_start, side="right")
    if start_i >= len(idx):
        return None
    if direction == "below":
        mask = close[start_i:] < price
    else:
        mask = close[start_i:] > price
    if not mask.any():
        return None
    first = int(np.argmax(mask))
    return idx[start_i + first].to_pydatetime()


def _first_break_ts(ts_start, idx, low: np.ndarray, high: np.ndarray,
                     price: float, kind: str) -> datetime | None:
    start_i = idx.searchsorted(ts_start, side="right")
    if start_i >= len(idx):
        return None
    if kind == "high":
        mask = high[start_i:] > price
    else:
        mask = low[start_i:] < price
    if not mask.any():
        return None
    first = int(np.argmax(mask))
    return idx[start_i + first].to_pydatetime()


# -----------------------------------------------------------------------------
# Bias timeline
# -----------------------------------------------------------------------------

def _build_bias_timeline(h1_df: pd.DataFrame, length: int = 5) -> list[tuple]:
    """Walk all H1 swings once and emit (event_ts, bias, last_event_str) events.

    Event ts = swing confirmation ts (swing.ts + length bars, since a pivot at bar
    i is only confirmed at i+L). For simplicity we use the swing's ts and assume
    it's visible immediately after — callers should guard with "confirm_ts <= T".

    Actually to preserve no-look-ahead semantics, event_ts = pivot_ts + L * timestep.
    """
    swings = find_swings(h1_df, length=length)
    if not swings:
        return [(h1_df.index[0] if len(h1_df) else pd.Timestamp("1970-01-01", tz="UTC"),
                 OFBias.NEUTRAL, "init")]

    events = []
    state = OFBias.NEUTRAL
    last_event = "init"
    prev_ph: float | None = None
    prev_pl: float | None = None

    # Pivot confirmation delay in timedelta
    confirm_delay = pd.Timedelta(hours=length)   # H1 bars

    for s in swings:
        event_ts = pd.Timestamp(s.ts) + confirm_delay
        if s.kind == "high":
            if prev_ph is not None:
                if s.price > prev_ph:
                    if state == OFBias.BEAR:
                        state, last_event = OFBias.BULL, "CHoCH-up"
                    else:
                        state, last_event = OFBias.BULL, "BOS-up"
                    events.append((event_ts, state, last_event))
            prev_ph = s.price
        else:
            if prev_pl is not None:
                if s.price < prev_pl:
                    if state == OFBias.BULL:
                        state, last_event = OFBias.BEAR, "CHoCH-dn"
                    else:
                        state, last_event = OFBias.BEAR, "BOS-dn"
                    events.append((event_ts, state, last_event))
            prev_pl = s.price

    if not events:
        events = [(h1_df.index[0], OFBias.NEUTRAL, "init")]
    return events


def bias_as_of(pc: Precomputed, ts: pd.Timestamp) -> BiasState:
    """Find the most recent bias event with event_ts <= ts."""
    events = pc.bias_events
    # binary search
    lo, hi = 0, len(events) - 1
    found = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if events[mid][0] <= ts:
            found = events[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    if found is None:
        return BiasState(OFBias.NEUTRAL, None, None, None, None, "init")
    return BiasState(found[1], None, None, None, None, found[2])


# -----------------------------------------------------------------------------
# Cycle
# -----------------------------------------------------------------------------

def cycle_as_of(pc: Precomputed, ts: pd.Timestamp) -> CycleState:
    """Look up ATR ratio at nearest H1 bar <= ts, combine with daily ADR."""
    idx = pc.atr5_h1.index
    i = idx.searchsorted(ts, side="right") - 1
    if i < 0 or len(idx) == 0:
        return CycleState(Cycle.NEUTRAL, 0, 0, 1.0, 0, 0, 0)
    af = float(pc.atr5_h1.iloc[i]) if not pd.isna(pc.atr5_h1.iloc[i]) else 0.0
    as_ = float(pc.atr50_h1.iloc[i]) if not pd.isna(pc.atr50_h1.iloc[i]) else 0.0
    ratio = (af / as_) if as_ else 1.0

    # Daily ADR / today's range
    local_ts = ts.tz_convert(DEFAULT_TZ)
    local_date = local_ts.normalize()
    adr5 = 0.0
    today_range = 0.0
    if pc.daily_range_local_idx is not None and len(pc.daily_range_local_idx):
        di = pc.daily_range_local_idx.searchsorted(local_date, side="right") - 1
        if di >= 0:
            today_range = float(pc.daily_range.iloc[di])
            lo = max(0, di - 5)
            window = pc.daily_range.iloc[lo:di]  # prior 5 days, not including today
            if len(window):
                adr5 = float(window.mean())
    adr_pct = (today_range / adr5 * 100) if adr5 else 0.0

    if ratio > 1.15:
        cyc = Cycle.EXPANSION
    elif ratio < 0.85:
        cyc = Cycle.CONSOLIDATION
    else:
        cyc = Cycle.NEUTRAL
    return CycleState(cyc, af, as_, ratio, adr5, today_range, adr_pct)


# -----------------------------------------------------------------------------
# HTF levels + Monday range
# -----------------------------------------------------------------------------

def _precompute_htf_by_date(h1: pd.DataFrame) -> dict:
    """For each local date where h1 has data, compute {PDH,PDL,PWH,PWL,PMH,PML,DO,WO}
    enriched with swept_ts (first time that level gets taken after its anchor ts).
    Returns {local_date (Timestamp) -> {label -> LevelRich}}.
    """
    if h1.empty:
        return {}
    idx_utc = h1.index
    local = h1.copy()
    local.index = idx_utc.tz_convert(DEFAULT_TZ)
    # cached daily & weekly & monthly resamples
    d_agg = _resample_hl(h1, "D", DEFAULT_TZ)
    w_agg = _resample_hl(h1, "W-MON", DEFAULT_TZ)
    m_agg = _resample_hl(h1, "MS", DEFAULT_TZ)

    highs = h1["high"].values
    lows = h1["low"].values

    # For sweep calculations we use h1 highs/lows after each level's ts
    def sweep_ts(lvl_ts, price, kind):
        return _first_break_ts(pd.Timestamp(lvl_ts), idx_utc, lows, highs, price, kind)

    out: dict = {}
    local_dates = pd.DatetimeIndex(sorted(set(local.index.normalize())))
    for d in local_dates:
        snap: dict[str, LevelRich] = {}
        for prefix, agg in [("PD", d_agg), ("PW", w_agg), ("PM", m_agg)]:
            # find the last agg bar that ended <= d (i.e. agg start < d)
            # d_agg is indexed by period start. We want the most recently COMPLETED
            # period: its next period start must be <= d.
            # Simplest: find the last agg row with index < d.
            completed = agg[agg.index < d]
            if len(completed) < 1:
                continue
            last = completed.iloc[-1]
            ts = last.name.to_pydatetime()
            ph = float(last["high"])
            pl = float(last["low"])
            snap[f"{prefix}H"] = LevelRich(ph, f"{prefix}H", "high", ts,
                                            swept_ts=sweep_ts(ts, ph, "high"))
            snap[f"{prefix}L"] = LevelRich(pl, f"{prefix}L", "low", ts,
                                            swept_ts=sweep_ts(ts, pl, "low"))
        # current-period opens
        cur_d = d_agg[d_agg.index <= d]
        if len(cur_d):
            last = cur_d.iloc[-1]
            snap["DO"] = LevelRich(float(last["open"]), "DO", "high",
                                    last.name.to_pydatetime(),
                                    swept_ts=sweep_ts(last.name, last["open"], "high"))
        cur_w = w_agg[w_agg.index <= d]
        if len(cur_w):
            last = cur_w.iloc[-1]
            snap["WO"] = LevelRich(float(last["open"]), "WO", "high",
                                    last.name.to_pydatetime(),
                                    swept_ts=sweep_ts(last.name, last["open"], "high"))
        out[d] = snap
    return out


def _precompute_monday_by_week(h1: pd.DataFrame) -> dict:
    if h1.empty:
        return {}
    idx_utc = h1.index
    local = h1.copy()
    local.index = idx_utc.tz_convert(DEFAULT_TZ)
    highs = h1["high"].values
    lows = h1["low"].values

    out = {}
    # Weekly grouping by Monday-start week in local tz
    local_sorted = local
    weeks = pd.DatetimeIndex(sorted(set((local.index - pd.to_timedelta(local.index.weekday, unit="D")).normalize())))
    for mon in weeks:
        mon_day = local[(local.index >= mon) & (local.index < mon + pd.Timedelta(days=1))]
        if mon_day.empty:
            continue
        mrh = float(mon_day["high"].max())
        mrl = float(mon_day["low"].min())
        # anchor ts = end of Monday (only visible after Monday closed)
        mon_end_local = mon + pd.Timedelta(days=1)
        mon_end_utc = mon_end_local.tz_convert("UTC")
        out[mon] = {
            "MRH": LevelRich(mrh, "MRH", "high", mon_end_utc.to_pydatetime(),
                              swept_ts=_first_break_ts(mon_end_utc, idx_utc, lows, highs, mrh, "high")),
            "MRL": LevelRich(mrl, "MRL", "low", mon_end_utc.to_pydatetime(),
                              swept_ts=_first_break_ts(mon_end_utc, idx_utc, lows, highs, mrl, "low")),
        }
    return out


def levels_as_of(pc: Precomputed, ts: pd.Timestamp) -> dict[str, LevelRich]:
    """Return {label -> LevelRich} visible at ts, with swept flag relative to ts."""
    local_ts = ts.tz_convert(DEFAULT_TZ)
    local_date = local_ts.normalize()
    # htf snapshot for this local date
    snap = pc.htf_snapshots.get(local_date)
    if snap is None:
        # find nearest prior date
        dates = sorted(pc.htf_snapshots.keys())
        idx = np.searchsorted([d.value for d in dates], local_date.value, side="right") - 1
        if idx >= 0:
            snap = pc.htf_snapshots[dates[idx]]
        else:
            snap = {}

    # monday range
    mon = (local_ts - pd.Timedelta(days=local_ts.weekday())).normalize()
    mon_snap = pc.monday_snapshots.get(mon, {})

    combined: dict[str, LevelRich] = {}
    for k, v in snap.items():
        # A level is visible only if its ts <= query ts
        if pd.Timestamp(v.ts) <= ts:
            combined[k] = v
    for k, v in mon_snap.items():
        if pd.Timestamp(v.ts) <= ts:
            combined[k] = v
    return combined


# -----------------------------------------------------------------------------
# Top-level precompute
# -----------------------------------------------------------------------------

def precompute(pair: str,
                ltf: pd.DataFrame, h1: pd.DataFrame, daily: pd.DataFrame,
                other: pd.DataFrame | None = None,
                other_pair: str | None = None,
                dxy: pd.DataFrame | None = None,
                verbose: bool = False) -> Precomputed:
    from fsp.context.smt import detect_smt_positive, detect_smt_negative

    pc = Precomputed(pair=pair, ltf=ltf, h1=h1, daily=daily)
    pc.ltf_index = ltf.index
    pc.ltf_low = ltf["low"].values.astype(float)
    pc.ltf_high = ltf["high"].values.astype(float)
    pc.ltf_close = ltf["close"].values.astype(float)
    pc.ltf_atr14 = atr(ltf, 14)

    if verbose:
        print(f"  [pc] LTF {len(ltf)} bars  H1 {len(h1)}  D {len(daily)}")

    # -- FVGs with mitigation_ts --
    raw_fvgs = find_fvgs(ltf, tf="M15")
    fvgs: list[FVGRich] = []
    for f in raw_fvgs:
        mit = _first_hit_ts(pd.Timestamp(f.ts), None, pc.ltf_index,
                              pc.ltf_low, pc.ltf_high, f.top, f.bottom)
        inv_ts = None
        if mit is not None:
            inv_ts = _first_close_beyond_ts(pd.Timestamp(f.ts), pc.ltf_index,
                                              pc.ltf_close,
                                              f.bottom if f.direction == "bull" else f.top,
                                              "below" if f.direction == "bull" else "above")
        fvgs.append(FVGRich(ts=f.ts, top=f.top, bottom=f.bottom, direction=f.direction,
                             tf=f.tf, confirm_ts=f.ts, mit_ts=mit, inv_ts=inv_ts))
    pc.fvgs = fvgs

    # -- OBs with mitigation_ts --
    raw_obs = find_order_blocks(ltf, tf="M15")
    obs: list[OBRich] = []
    for ob in raw_obs:
        # OB only visible after the displacement candle CLOSED. That's the bar after ob.ts
        # (ob.ts is the "source" candle, displacement is at ob.ts + 1 bar).
        # Approximation: confirm_ts = ob.ts + median bar delta
        # Simpler: shift confirm_ts by one bar in ltf index
        ci = pc.ltf_index.searchsorted(pd.Timestamp(ob.ts), side="left")
        confirm_ts = pc.ltf_index[min(ci + 1, len(pc.ltf_index) - 1)].to_pydatetime()
        mit = _first_hit_ts(pd.Timestamp(confirm_ts), None, pc.ltf_index,
                             pc.ltf_low, pc.ltf_high, ob.top, ob.bottom)
        obs.append(OBRich(ts=ob.ts, top=ob.top, bottom=ob.bottom, direction=ob.direction,
                           tf=ob.tf, confirm_ts=confirm_ts, mit_ts=mit))
    pc.obs = obs

    # -- Bias timeline (on H1) --
    pc.bias_events = _build_bias_timeline(h1, length=5)

    # -- Cycle precompute --
    pc.atr5_h1 = atr(h1, 5)
    pc.atr50_h1 = atr(h1, 50)
    if not daily.empty:
        d_range = (daily["high"] - daily["low"]).copy()
        # index by local-date-normalised
        local_idx = daily.index.tz_convert(DEFAULT_TZ).normalize()
        d_range.index = local_idx
        pc.daily_range = d_range
        pc.daily_range_local_idx = local_idx
    else:
        pc.daily_range = pd.Series(dtype=float)
        pc.daily_range_local_idx = pd.DatetimeIndex([])

    # -- HTF levels per-local-date --
    pc.htf_snapshots = _precompute_htf_by_date(h1)
    pc.monday_snapshots = _precompute_monday_by_week(h1)

    # -- SMT --
    smt: list[SMTEvent] = []
    if other is not None and other_pair is not None:
        smt += detect_smt_positive(h1, other, pair, other_pair)
    if dxy is not None and not dxy.empty:
        smt += detect_smt_negative(h1, dxy, pair, "DXY")
    # Shift event ts by confirmation delay (L=3 H1 bars) to prevent look-ahead
    confirm_delay = pd.Timedelta(hours=3)
    smt_shifted: list[SMTEvent] = []
    for ev in smt:
        smt_shifted.append(SMTEvent(
            ts=(pd.Timestamp(ev.ts) + confirm_delay).to_pydatetime(),
            pair_a=ev.pair_a, pair_b=ev.pair_b,
            kind=ev.kind, at=ev.at, note=ev.note))
    pc.smt_events = sorted(smt_shifted, key=lambda e: e.ts)

    if verbose:
        print(f"  [pc] fvgs={len(pc.fvgs)}  obs={len(pc.obs)}  "
              f"smt={len(pc.smt_events)}  bias_events={len(pc.bias_events)}  "
              f"htf_dates={len(pc.htf_snapshots)}")

    return pc


# -----------------------------------------------------------------------------
# FAST GRADER — mirrors grade_setup but reads from Precomputed
# -----------------------------------------------------------------------------

def _nearest_pda_fast(price: float, pc: Precomputed, ts, direction: str):
    """Mirror _nearest_in_direction but filter by confirm_ts + mit_ts <= ts."""
    candidates = []
    for f in pc.fvgs:
        if pd.Timestamp(f.confirm_ts) > ts:
            continue
        if f.mitigated_as_of(ts):
            continue
        if direction == "long" and f.direction == "bull" and f.top < price:
            candidates.append((f"FVG@{f.bottom:.5f}", f.top, f.bottom))
        if direction == "short" and f.direction == "bear" and f.bottom > price:
            candidates.append((f"FVG@{f.top:.5f}", f.top, f.bottom))
    for ob in pc.obs:
        if pd.Timestamp(ob.confirm_ts) > ts:
            continue
        if ob.mitigated_as_of(ts):
            continue
        if direction == "long" and ob.direction == "bull" and ob.top < price:
            candidates.append((f"OB@{ob.bottom:.5f}", ob.top, ob.bottom))
        if direction == "short" and ob.direction == "bear" and ob.bottom > price:
            candidates.append((f"OB@{ob.top:.5f}", ob.top, ob.bottom))
    if not candidates:
        return None
    best = min(candidates, key=lambda c: abs(price - (c[1] + c[2]) / 2))
    mid = (best[1] + best[2]) / 2
    return (best[0], best[1], best[2], mid)


def _nearest_opposing_fast(price: float, direction: str,
                            levels: dict[str, LevelRich], ts) -> LevelRich | None:
    best: LevelRich | None = None
    for lvl in levels.values():
        if lvl.swept_as_of(ts):
            continue
        if direction == "long" and lvl.kind == "high" and lvl.price > price:
            if best is None or lvl.price < best.price:
                best = lvl
        if direction == "short" and lvl.kind == "low" and lvl.price < price:
            if best is None or lvl.price > best.price:
                best = lvl
    return best


def grade_setup_fast(pc: Precomputed, ts: pd.Timestamp, li: int = None):
    """Precomputed-input variant of grade_setup. Drop-in replacement returning
    the same SetupCandidate structure.

    li : optional int index into pc.ltf_index for the decision bar. If None,
    we binary-search.
    """
    from fsp.grader.setup import ChecklistItem, SetupCandidate, _pip

    pair = pc.pair
    pip = _pip(pair)
    if li is None:
        li = int(pc.ltf_index.searchsorted(ts, side="right")) - 1
    if li < 0 or li >= len(pc.ltf_index):
        # Bad index; return SKIP
        return SetupCandidate(pair=pair, direction=None, grade=Grade.SKIP,
                               entry=None, sl=None, tp1=None, tp2=None,
                               invalidation_pips=None, rr_tp1=None, rr_tp2=None,
                               risk_r=0.0, key_level_ref="—", checklist=[],
                               context={"now": ts.isoformat(), "fast": True})

    price = float(pc.ltf_close[li])
    session = session_of(ts)
    cyc = cycle_as_of(pc, ts)
    bias = bias_as_of(pc, ts)
    levels = levels_as_of(pc, ts)

    direction = None
    if bias.bias == OFBias.BULL: direction = "long"
    elif bias.bias == OFBias.BEAR: direction = "short"

    pda = _nearest_pda_fast(price, pc, ts, direction) if direction else None

    cl: list[ChecklistItem] = []
    cl.append(ChecklistItem("Active execution session",
        passed=session in (Session.LONDON, Session.NY_AM, Session.NY_PM),
        note=f"session={session.value}"))
    cl.append(ChecklistItem("Not lunch / off-hours",
        passed=session not in (Session.LUNCH, Session.OFF),
        note=f"session={session.value}"))
    cl.append(ChecklistItem("Clear H1 OF bias (not neutral)",
        passed=bias.bias != OFBias.NEUTRAL,
        note=f"bias={bias.bias.value} ({bias.last_event})"))
    cl.append(ChecklistItem("Cycle = EXPANSION",
        passed=cyc.cycle.value == "EXPANSION",
        note=f"cycle={cyc.cycle.value} ratio={cyc.atr_ratio:.2f}"))
    cl.append(ChecklistItem("ADR% < 100%  (room to expand)",
        passed=cyc.adr_pct < 100, note=f"ADR%={cyc.adr_pct:.0f}"))

    # Recent sweep in last 20 LTF bars against any HTF level
    lo = max(0, li - 19)
    r_lows = pc.ltf_low[lo:li + 1]
    r_highs = pc.ltf_high[lo:li + 1]
    r_closes = pc.ltf_close[lo:li + 1]
    recent_sweep = False
    if direction == "long":
        for lvl in levels.values():
            if lvl.kind == "low":
                if ((r_lows < lvl.price) & (r_closes > lvl.price)).any():
                    recent_sweep = True; break
    elif direction == "short":
        for lvl in levels.values():
            if lvl.kind == "high":
                if ((r_highs > lvl.price) & (r_closes < lvl.price)).any():
                    recent_sweep = True; break
    cl.append(ChecklistItem("Recent liquidity sweep (≤20 LTF bars)", passed=recent_sweep))

    cl.append(ChecklistItem("Unmitigated PDA available in direction",
        passed=pda is not None, note=pda[0] if pda else "none"))

    # SMT in direction within 24h
    cutoff = ts - pd.Timedelta(days=1)
    smt_ok = False
    # binary search for events in (cutoff, ts]
    idx_hi = 0
    # linear scan backwards is fine because we expect few recent events
    for ev in reversed(pc.smt_events):
        if pd.Timestamp(ev.ts) > ts: continue
        if pd.Timestamp(ev.ts) < cutoff: break
        if direction == "long" and ev.kind == "bull": smt_ok = True; break
        if direction == "short" and ev.kind == "bear": smt_ok = True; break
    cl.append(ChecklistItem("SMT aligned (≤24h)", passed=smt_ok))

    # Entry, SL, TPs
    entry = sl = tp1 = tp2 = None
    inv_pips = rr1 = rr2 = None
    opposing = None
    if pda and direction:
        top, bot = pda[1], pda[2]
        entry = (top + bot) / 2
        atr_val = float(pc.ltf_atr14.iloc[li]) if not pd.isna(pc.ltf_atr14.iloc[li]) else 0.0
        buffer = atr_val * 0.3
        if direction == "long":
            sl = bot - buffer
        else:
            sl = top + buffer
        inv_pips = abs(entry - sl) / pip
        opposing = _nearest_opposing_fast(entry, direction, levels, ts)
        if opposing:
            risk = abs(entry - sl)
            tp1 = opposing.price
            rr1 = abs(tp1 - entry) / risk if risk else None
            # TP2 = next level beyond TP1
            others = [lvl for lvl in levels.values()
                       if not lvl.swept_as_of(ts) and
                       lvl.kind == ("high" if direction == "long" else "low") and
                       ((direction == "long" and lvl.price > opposing.price) or
                        (direction == "short" and lvl.price < opposing.price))]
            if others:
                t2 = min(others, key=lambda lv: abs(lv.price - opposing.price))
                tp2 = t2.price
                rr2 = abs(tp2 - entry) / risk if risk else None

    cl.append(ChecklistItem("Invalidation ≤ 30 pips",
        passed=inv_pips is not None and inv_pips <= 30,
        note=f"{inv_pips:.1f} pips" if inv_pips else "n/a"))
    cl.append(ChecklistItem("RR to TP1 ≥ 2.0",
        passed=rr1 is not None and rr1 >= 2.0,
        note=f"{rr1:.2f}R" if rr1 else "n/a"))

    # Grade gating — must mirror grade_setup
    passed_count = sum(1 for c in cl if c.passed)
    total = len(cl)
    cmap = {c.name: c for c in cl}
    is_choch = bias.last_event and "CHoCH" in bias.last_event
    cl.append(ChecklistItem("Entry is CHoCH (reversal, not BOS)",
                             passed=bool(is_choch), note=str(bias.last_event)))
    adr_ok = cyc.adr_pct < 100
    hard_blockers = [
        not cl[0].passed,
        cl[1].passed is False,
        direction is None,
        pda is None,
        inv_pips is None or inv_pips > 40,
        not cmap.get("Cycle = EXPANSION", ChecklistItem("x", True)).passed,
        not cmap.get("Recent liquidity sweep (≤20 LTF bars)", ChecklistItem("x", True)).passed,
        not is_choch,
        not adr_ok,
    ]
    if any(hard_blockers):
        grade = Grade.SKIP; risk_r = 0.0
    elif passed_count >= total - 1:
        grade = Grade.A_PLUS; risk_r = 1.5
    elif passed_count >= total - 3:
        grade = Grade.A; risk_r = 1.0
    elif passed_count >= total - 5:
        grade = Grade.B; risk_r = 0.5
    else:
        grade = Grade.SKIP; risk_r = 0.0

    return SetupCandidate(
        pair=pair, direction=direction, grade=grade,
        entry=entry, sl=sl, tp1=tp1, tp2=tp2,
        invalidation_pips=inv_pips, rr_tp1=rr1, rr_tp2=rr2,
        risk_r=risk_r, key_level_ref=pda[0] if pda else "—",
        checklist=cl,
        context={
            "price": price, "session": session.value,
            "cycle": cyc.cycle.value, "adr_pct": round(cyc.adr_pct, 1),
            "bias": bias.bias.value, "bias_event": bias.last_event,
            "opposing_target": opposing.label if opposing else None,
            "opposing_price": opposing.price if opposing else None,
            "now": ts.isoformat(),
        })
