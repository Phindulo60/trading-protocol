"""Time-based liquidity levels: PDH/PDL, PWH/PWL, PMH/PML, D/W opens, Monday range."""
from __future__ import annotations

import pandas as pd

from fsp.context.sessions import DEFAULT_TZ
from fsp.data.types import Level


def _resample_hl(df: pd.DataFrame, rule: str, tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """Resample UTC bars into a HTF (D/W/MS) keyed on the local calendar."""
    local = df.copy()
    local.index = local.index.tz_convert(tz)
    agg = local.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    return agg


def htf_levels(df: pd.DataFrame, tz: str = DEFAULT_TZ) -> dict[str, Level]:
    """Compute last-completed PDH/PDL/PWH/PWL/PMH/PML + current D/W opens.

    Returns a dict of label → Level, anchored to the most recent completed HTF candle.
    """
    out: dict[str, Level] = {}
    if df.empty:
        return out

    for rule, prefix in [("D", "PD"), ("W-MON", "PW"), ("MS", "PM")]:
        agg = _resample_hl(df, rule, tz)
        if len(agg) < 2:
            continue
        prev = agg.iloc[-2]
        out[f"{prefix}H"] = Level(price=float(prev["high"]), label=f"{prefix}H",
                                  kind="high", ts=prev.name.to_pydatetime())
        out[f"{prefix}L"] = Level(price=float(prev["low"]), label=f"{prefix}L",
                                  kind="low", ts=prev.name.to_pydatetime())

    # Current-period opens
    d_agg = _resample_hl(df, "D", tz)
    if len(d_agg):
        cur = d_agg.iloc[-1]
        out["DO"] = Level(price=float(cur["open"]), label="DO", kind="high",
                          ts=cur.name.to_pydatetime())
    w_agg = _resample_hl(df, "W-MON", tz)
    if len(w_agg):
        cur = w_agg.iloc[-1]
        out["WO"] = Level(price=float(cur["open"]), label="WO", kind="high",
                          ts=cur.name.to_pydatetime())
    return out


def monday_range(df: pd.DataFrame, tz: str = DEFAULT_TZ) -> Level | None:
    """Return the Monday-range high/low of the current week (or None if Monday hasn't happened)."""
    if df.empty:
        return None
    local = df.copy()
    local.index = local.index.tz_convert(tz)
    # Current week's Monday
    last_ts = local.index[-1]
    this_monday = (last_ts - pd.Timedelta(days=last_ts.weekday())).normalize()
    mon = local[(local.index >= this_monday) & (local.index < this_monday + pd.Timedelta(days=1))]
    if mon.empty:
        return None
    return {
        "MRH": Level(price=float(mon["high"].max()), label="MRH", kind="high",
                     ts=mon.index[0].to_pydatetime()),
        "MRL": Level(price=float(mon["low"].min()), label="MRL", kind="low",
                     ts=mon.index[0].to_pydatetime()),
    }


def mark_swept(levels: dict[str, Level], df: pd.DataFrame) -> dict[str, Level]:
    for lvl in levels.values():
        future = df[df.index > pd.Timestamp(lvl.ts)]
        if future.empty:
            continue
        if lvl.kind == "high":
            lvl.swept = bool((future["high"] > lvl.price).any())
        else:
            lvl.swept = bool((future["low"] < lvl.price).any())
    return levels
