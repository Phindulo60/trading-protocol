"""FX session definitions. Default times in America/New_York (NY clock)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal

import pandas as pd
import pytz

from fsp.data.types import Session

DEFAULT_TZ = "America/New_York"

# NY-clock definitions (matches TradingView Pine defaults we wrote earlier)
SESSION_WINDOWS: dict[Session, tuple[time, time]] = {
    Session.ASIA:   (time(18, 0), time(2, 0)),   # spans midnight
    Session.LONDON: (time(2, 0),  time(5, 0)),
    Session.NY_AM:  (time(7, 0),  time(12, 0)),
    Session.LUNCH:  (time(12, 0), time(13, 0)),
    Session.NY_PM:  (time(13, 0), time(16, 0)),
}


def _in_window(t: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= t < end
    return t >= start or t < end  # spans midnight


def session_of(ts: pd.Timestamp, tz: str = DEFAULT_TZ) -> Session:
    local = ts.tz_convert(tz) if ts.tzinfo else ts.tz_localize("UTC").tz_convert(tz)
    t = local.time()
    for sess, (s, e) in SESSION_WINDOWS.items():
        if _in_window(t, s, e):
            return sess
    return Session.OFF


def annotate_sessions(df: pd.DataFrame, tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """Add a 'session' column classifying each bar."""
    out = df.copy()
    out["session"] = [session_of(ts, tz).value for ts in df.index]
    return out


@dataclass
class SessionRange:
    session: Session
    date: pd.Timestamp       # the local date the session belongs to
    high: float
    low: float
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp


def session_ranges(df: pd.DataFrame, tz: str = DEFAULT_TZ) -> list[SessionRange]:
    """Group bars by (session, local-day) and return H/L per completed session block."""
    if df.empty:
        return []
    annotated = annotate_sessions(df, tz)
    # For Asia-spanning-midnight, attribute to the local date at the END of the session
    local_idx = annotated.index.tz_convert(tz)
    # Session-date key: use the bar's local date if session ∈ {LO, NY_AM, LUNCH, NY_PM}
    # For Asia, bucket 18:00–23:59 with next day's 00:00–02:00
    dates = []
    for ts, sess in zip(local_idx, annotated["session"]):
        d = ts.normalize()
        if sess == Session.ASIA.value and ts.time() >= time(18, 0):
            d = (ts + pd.Timedelta(days=1)).normalize()
        dates.append(d)
    annotated["local_date"] = dates

    blocks = annotated[annotated["session"] != Session.OFF.value].groupby(
        ["local_date", "session"], sort=True
    )
    out: list[SessionRange] = []
    for (d, s), g in blocks:
        out.append(SessionRange(
            session=Session(s), date=d,
            high=float(g["high"].max()), low=float(g["low"].min()),
            start_ts=g.index.min(), end_ts=g.index.max(),
        ))
    return sorted(out, key=lambda r: r.end_ts)
