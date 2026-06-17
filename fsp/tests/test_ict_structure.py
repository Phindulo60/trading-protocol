"""Tests for the ICT BOS/CHoCH/MSS state machine.

Fixtures are hand-crafted with swing_length=2 so a pivot needs only 2 bars of
confirmation, keeping the OHLC tables compact and the expected swings easy to
trace by hand.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fsp.ict.structure import analyze_structure, StructureEvent, StructureState


def make_df(bars: list[tuple[float, float, float, float]],
            start: str = "2026-06-01 00:00", freq: str = "5min") -> pd.DataFrame:
    """bars = list of (open, high, low, close) -> OHLC frame with tz-aware UTC index."""
    idx = pd.date_range(start, periods=len(bars), freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open": [b[0] for b in bars],
            "high": [b[1] for b in bars],
            "low": [b[2] for b in bars],
            "close": [b[3] for b in bars],
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# A — bullish BOS from a neutral start
# ---------------------------------------------------------------------------
def test_bullish_bos_from_neutral():
    # (o, h, l, c). Pivot-high prints at idx2 (h=15), confirmed at idx4; price
    # closes above it at idx5 -> the first break, so BOS (not CHoCH).
    bars = [
        (9.5, 10, 9, 9.8),
        (10.5, 11, 10, 10.8),
        (12.5, 15, 12, 14.8),
        (10.5, 11, 9, 9.8),
        (9.5, 10, 8, 8.8),
        (13.5, 16, 13, 16.0),
        (15.5, 17, 15, 16.8),
    ]
    st = analyze_structure(make_df(bars), swing_length=2)
    assert len(st.events) == 1
    ev = st.events[0]
    assert ev.event_type == "BOS"
    assert ev.direction == "bull"
    assert ev.broken_level == 15
    assert st.trend == 1
    assert st.bias == "bull"


# ---------------------------------------------------------------------------
# B — uptrend then reversal: bullish BOS followed by bearish CHoCH
# ---------------------------------------------------------------------------
def test_bos_then_choch_reversal():
    bars = [
        (9.2, 10.0, 9.0, 9.5),
        (10.2, 11.0, 10.0, 10.5),
        (12.5, 15.0, 12.0, 14.5),
        (11.5, 12.0, 10.0, 10.5),
        (11.0, 11.5, 9.5, 10.0),
        (16.0, 18.0, 15.5, 17.5),   # idx5: closes above 15 -> BOS bull
        (15.5, 16.0, 14.0, 15.0),
        (14.0, 17.0, 13.5, 16.5),
        (15.0, 15.5, 12.0, 13.0),
        (13.5, 14.0, 10.0, 10.5),
        (12.5, 13.0, 9.0, 9.5),
        (11.5, 12.0, 8.0, 8.5),     # idx11: closes below 9.5 low -> CHoCH bear
    ]
    st = analyze_structure(make_df(bars), swing_length=2)
    types = [(e.event_type, e.direction) for e in st.events]
    assert types == [("BOS", "bull"), ("CHoCH", "bear")]
    assert st.trend == -1
    assert st.bias == "bear"
    # default ATR(20) over 12 bars is all NaN -> no displacement, no MSS
    assert all(not e.displacement for e in st.events)
    assert all(not e.is_mss for e in st.events)
    assert st.last_mss is None


# ---------------------------------------------------------------------------
# C — bearish BOS from neutral (mirror of A)
# ---------------------------------------------------------------------------
def test_bearish_bos_from_neutral():
    bars = [
        (16.8, 17, 16, 16.5),
        (16.2, 16, 15, 15.5),
        (12.5, 13, 10, 10.5),   # idx2: pivot-low 10, confirmed idx4
        (14.5, 16, 14, 15.5),
        (15.5, 17, 15, 16.5),
        (11.0, 12, 8, 8.5),     # idx5: closes below 10 -> BOS bear
        (10.5, 11, 7, 7.5),
    ]
    st = analyze_structure(make_df(bars), swing_length=2)
    assert len(st.events) == 1
    ev = st.events[0]
    assert ev.event_type == "BOS"
    assert ev.direction == "bear"
    assert ev.broken_level == 10
    assert st.trend == -1


# ---------------------------------------------------------------------------
# D — MSS: a CHoCH delivered by a displacement candle
# ---------------------------------------------------------------------------
def test_mss_is_choch_with_displacement():
    bars = [
        (9.85, 10.0, 9.8, 9.9),
        (10.05, 10.4, 10.0, 10.3),
        (10.55, 11.0, 10.5, 10.8),   # idx2: pivot-high 11, conf idx4
        (10.55, 10.6, 10.2, 10.3),
        (10.45, 10.5, 10.1, 10.2),   # idx4: pivot-low 10.1, conf idx6
        (10.85, 11.2, 10.8, 11.1),   # idx5: close 11.1 > 11 -> BOS bull
        (10.95, 11.0, 10.6, 10.7),
        (10.75, 10.8, 10.4, 10.5),   # idx5 pivot-high 11.2 conf idx7
        (10.55, 10.6, 8.0, 8.2),     # idx8: big bearish body, close < 10.1 -> CHoCH+disp = MSS
    ]
    st = analyze_structure(make_df(bars), swing_length=2, atr_len=3, atr_mult=1.5)
    types = [(e.event_type, e.direction) for e in st.events]
    assert types == [("BOS", "bull"), ("CHoCH", "bear")]
    mss = st.events[1]
    assert mss.displacement is True
    assert mss.is_mss is True
    assert st.last_mss is mss
    # the BOS is not an MSS even if it had displacement
    assert st.events[0].is_mss is False
    assert st.trend == -1


# ---------------------------------------------------------------------------
# E — insufficient data
# ---------------------------------------------------------------------------
def test_insufficient_data_returns_neutral():
    bars = [(10, 11, 9, 10.5), (10.5, 11.5, 9.5, 11), (11, 12, 10, 11.5), (11.5, 12.5, 10.5, 12)]
    st = analyze_structure(make_df(bars), swing_length=2)  # needs >=5 bars
    assert st.events == []
    assert st.trend == 0
    assert st.bias == "neutral"
    assert st.last_event is None


# ---------------------------------------------------------------------------
# F — no swings (flat market) -> neutral
# ---------------------------------------------------------------------------
def test_flat_market_no_events():
    bars = [(10, 10.1, 9.9, 10.0)] * 12
    st = analyze_structure(make_df(bars), swing_length=2)
    assert st.events == []
    assert st.trend == 0


# ---------------------------------------------------------------------------
# G — event dataclass invariants
# ---------------------------------------------------------------------------
def test_event_is_mss_property():
    bos = StructureEvent(ts=None, event_type="BOS", direction="bull",
                         broken_level=1.0, broken_swing_ts=None, close=1.0, displacement=True)
    assert bos.is_mss is False  # BOS is never an MSS
    choch = StructureEvent(ts=None, event_type="CHoCH", direction="bear",
                           broken_level=1.0, broken_swing_ts=None, close=1.0, displacement=False)
    assert choch.is_mss is False  # CHoCH without displacement
    mss = StructureEvent(ts=None, event_type="CHoCH", direction="bear",
                         broken_level=1.0, broken_swing_ts=None, close=1.0, displacement=True)
    assert mss.is_mss is True


def test_empty_state_bias():
    st = StructureState()
    assert st.bias == "neutral"
    assert st.last_mss is None
