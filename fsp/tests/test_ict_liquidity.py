"""Tests for ICT liquidity pools + reclaim-sweep detection (swing_length=2)."""
from __future__ import annotations

import pandas as pd

from fsp.ict.liquidity import (
    find_liquidity_pools, find_sweeps, nearest_unswept,
    LiquidityPool, LiquiditySweep,
)


def make_df(bars, start="2026-06-01 00:00", freq="5min"):
    """bars = list of (open, high, low, close) -> OHLC frame, tz-aware UTC index."""
    idx = pd.date_range(start, periods=len(bars), freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": [b[0] for b in bars], "high": [b[1] for b in bars],
         "low": [b[2] for b in bars], "close": [b[3] for b in bars]},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Pools — single swings
# ---------------------------------------------------------------------------
def test_pools_from_single_swings():
    bars = [
        (9.5, 10, 9, 9.8), (10.5, 11, 10, 10.8), (12.5, 15, 12, 14.0),
        (10.5, 11, 9, 9.8), (9.5, 10, 8, 8.5), (10.5, 12, 11, 11.5),
        (10.5, 11, 10, 10.5),
    ]
    pools = find_liquidity_pools(make_df(bars), swing_length=2)
    buys = [p for p in pools if p.side == "buy"]
    sells = [p for p in pools if p.side == "sell"]
    assert len(buys) == 1 and buys[0].price == 15 and buys[0].kind == "swing"
    assert len(sells) == 1 and sells[0].price == 8 and sells[0].kind == "swing"
    assert buys[0].strength == 1


# ---------------------------------------------------------------------------
# Pools — relative-equal highs cluster into one pool
# ---------------------------------------------------------------------------
def test_equal_highs_cluster():
    bars = [
        (9.5, 10, 9, 9.8), (10.5, 11, 10, 10.8), (13.5, 15.00, 12, 14.0),
        (10.5, 11, 9, 9.8), (9.5, 10, 8, 8.5), (10.5, 11, 9, 9.8),
        (13.5, 15.05, 12, 14.0), (10.5, 11, 9, 9.8), (9.5, 10, 8, 8.5),
    ]
    pools = find_liquidity_pools(make_df(bars), swing_length=2)
    buys = [p for p in pools if p.side == "buy"]
    assert len(buys) == 1
    eq = buys[0]
    assert eq.kind == "equal"
    assert eq.strength == 2
    assert eq.price == 15.05  # level = highest of the equal highs


# ---------------------------------------------------------------------------
# Sell-side sweep -> bullish reclaim
# ---------------------------------------------------------------------------
def test_sellside_sweep_bullish():
    bars = [
        (11.8, 12, 11, 11.5), (11.5, 11, 10, 10.5), (12.5, 13, 9, 9.5),   # PL@2 = 9
        (11.5, 12, 10, 11.5), (12.5, 13, 11, 12.5),
        (9.0, 14, 8.5, 13.0),   # idx5: wicks to 8.5 (< 9), closes 13.0 -> sweep
        (11.5, 13, 11, 12.5), (12.5, 14, 12, 13.5),
    ]
    sweeps = find_sweeps(make_df(bars), swing_length=2)
    assert len(sweeps) >= 1
    sw = sweeps[0]
    assert sw.side == "sell"
    assert sw.direction == "bull"
    assert sw.level == 9
    assert sw.extreme == 8.5
    assert sw.close > sw.level
    assert sw.pool.swept is True


# ---------------------------------------------------------------------------
# Buy-side sweep -> bearish reclaim
# ---------------------------------------------------------------------------
def test_buyside_sweep_bearish():
    bars = [
        (8.2, 9, 8, 8.5), (9.2, 10, 9, 9.5), (8.5, 13, 7, 12.5),   # PH@2 = 13
        (9.5, 10, 8, 8.5), (8.5, 9, 7, 7.5),
        (13.5, 14, 6, 7.0),   # idx5: wicks to 14 (> 13), closes 7.0 -> sweep
        (8.5, 9, 7, 7.5), (7.5, 8, 6, 6.5),
    ]
    sweeps = find_sweeps(make_df(bars), swing_length=2)
    assert len(sweeps) >= 1
    sw = sweeps[0]
    assert sw.side == "buy"
    assert sw.direction == "bear"
    assert sw.level == 13
    assert sw.extreme == 14
    assert sw.close < sw.level


# ---------------------------------------------------------------------------
# A clean break-through (no reclaim) is NOT a sweep
# ---------------------------------------------------------------------------
def test_breakthrough_is_not_a_sweep():
    bars = [
        (12.8, 13, 12, 12.5), (12.5, 12, 11, 11.5), (13.5, 14, 9, 9.5),  # PL@2=9, PH@2=14
        (11.0, 12, 11, 11.5), (12.0, 13, 12, 12.5),
        (10.5, 11, 8, 8.5),   # closes 8.5 < 9 (break, not reclaim)
        (9.5, 10, 7, 7.5),    # closes 7.5 < 9
    ]
    sweeps = find_sweeps(make_df(bars), swing_length=2)
    assert sweeps == []


# ---------------------------------------------------------------------------
# nearest_unswept helper (TP targeting)
# ---------------------------------------------------------------------------
def test_nearest_unswept():
    ts = pd.Timestamp("2026-06-01", tz="UTC").to_pydatetime()
    pools = [
        LiquidityPool(price=1.1080, side="buy", kind="swing", members=[ts], created_ts=ts),
        LiquidityPool(price=1.1120, side="buy", kind="equal", members=[ts, ts], created_ts=ts),
        LiquidityPool(price=1.1050, side="buy", kind="swing", members=[ts], created_ts=ts, swept=True),
        LiquidityPool(price=1.0980, side="sell", kind="swing", members=[ts], created_ts=ts),
        LiquidityPool(price=1.0950, side="sell", kind="equal", members=[ts, ts], created_ts=ts),
    ]
    ref = 1.1000
    up = nearest_unswept(pools, "buy", ref)
    assert up is not None and up.price == 1.1080   # nearest unswept above (1.1050 is swept)
    down = nearest_unswept(pools, "sell", ref)
    assert down is not None and down.price == 1.0980  # nearest below
    # nothing above 1.13
    assert nearest_unswept(pools, "buy", 1.13) is None


# ---------------------------------------------------------------------------
# insufficient data
# ---------------------------------------------------------------------------
def test_insufficient_data():
    bars = [(10, 11, 9, 10.5), (10.5, 11.5, 9.5, 11), (11, 12, 10, 11.5)]
    df = make_df(bars)
    assert find_liquidity_pools(df, swing_length=2) == []
    assert find_sweeps(df, swing_length=2) == []
