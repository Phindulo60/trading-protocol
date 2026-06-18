"""Tests for ICT SMT divergence detection."""
from __future__ import annotations

import pandas as pd

from fsp.ict.smt import smt_divergence, partner_for, PARTNERS, SMTResult


def _ts(start, n, freq="5min"):
    return pd.date_range(start, periods=n, freq=freq, tz="UTC")


def _df(lows, highs, start="2026-06-15 11:00"):
    """Minimal df with lows and highs at given values."""
    n = len(lows)
    idx = _ts(start, n)
    return pd.DataFrame({
        "open": [(h + l) / 2 for h, l in zip(highs, lows)],
        "high": highs,
        "low": lows,
        "close": [(h + l) / 2 for h, l in zip(highs, lows)],
    }, index=idx)


def test_bullish_smt_positive_corr():
    """Primary sweeps lows (lower low at t2); partner makes higher low -> divergence."""
    # 10 bars, ref at bar 3, sweep at bar 8
    lows = [1.10, 1.10, 1.10, 1.095, 1.10, 1.10, 1.10, 1.10, 1.088, 1.10]
    highs = [1.11] * 10
    primary = _df(lows, highs)

    # partner (positive corr): makes a HIGHER low at bar 8 vs bar 3
    p_lows = [1.30, 1.30, 1.30, 1.295, 1.30, 1.30, 1.30, 1.30, 1.298, 1.30]
    partner = _df(p_lows, [1.32] * 10)

    ref_ts = primary.index[3]    # pool origin (the prior swing low)
    sweep_ts = primary.index[8]  # sweep bar

    res = smt_divergence(primary, partner, ref_ts=ref_ts, sweep_ts=sweep_ts,
                         direction="long", sign=1, partner="GBPUSD", window=1)
    assert res.diverged is True
    assert res.direction == "long"
    assert res.note == "divergence"
    # partner low at sweep (1.298) > partner low at ref (1.295)
    assert res.now > res.ref


def test_no_smt_when_partner_confirms():
    """Both primary and partner make lower lows -> no divergence."""
    lows = [1.10, 1.10, 1.10, 1.095, 1.10, 1.10, 1.10, 1.10, 1.088, 1.10]
    highs = [1.11] * 10
    primary = _df(lows, highs)

    # partner ALSO makes lower low at sweep
    p_lows = [1.30, 1.30, 1.30, 1.295, 1.30, 1.30, 1.30, 1.30, 1.290, 1.30]
    partner = _df(p_lows, [1.32] * 10)

    res = smt_divergence(primary, partner, ref_ts=primary.index[3],
                         sweep_ts=primary.index[8],
                         direction="long", sign=1, window=1)
    assert res.diverged is False
    assert res.note == "confirmation"


def test_bearish_smt_positive_corr():
    """Primary sweeps highs (higher high at t2); partner makes lower high -> divergence."""
    highs = [1.20, 1.20, 1.20, 1.210, 1.20, 1.20, 1.20, 1.20, 1.215, 1.20]
    lows = [1.19] * 10
    primary = _df(lows, highs)

    # partner lower high at sweep bar vs ref bar
    p_highs = [1.50, 1.50, 1.50, 1.510, 1.50, 1.50, 1.50, 1.50, 1.505, 1.50]
    partner = _df([1.49] * 10, p_highs)

    res = smt_divergence(primary, partner, ref_ts=primary.index[3],
                         sweep_ts=primary.index[8],
                         direction="short", sign=1, window=1)
    assert res.diverged is True
    assert res.now < res.ref   # partner high dropped


def test_inverse_corr_bullish():
    """Inverse partner (sign=-1): primary lower low, partner should make higher high.
    Divergence = partner fails to make higher high."""
    lows = [1.10, 1.10, 1.10, 1.095, 1.10, 1.10, 1.10, 1.10, 1.088, 1.10]
    highs = [1.11] * 10
    primary = _df(lows, highs)

    # inverse partner highs: lower at sweep vs ref -> divergence
    p_highs = [0.93, 0.93, 0.93, 0.940, 0.93, 0.93, 0.93, 0.93, 0.935, 0.93]
    partner = _df([0.92] * 10, p_highs)

    res = smt_divergence(primary, partner, ref_ts=primary.index[3],
                         sweep_ts=primary.index[8],
                         direction="long", sign=-1, window=1)
    assert res.diverged is True  # partner high dropped (failed to confirm inverse)


def test_insufficient_data():
    """Empty partner df -> not diverged, note about insufficient data."""
    primary = _df([1.10] * 5, [1.11] * 5)
    empty = pd.DataFrame({"open": [], "high": [], "low": [], "close": []},
                         index=pd.DatetimeIndex([], tz="UTC"))
    res = smt_divergence(primary, empty, ref_ts=primary.index[0],
                         sweep_ts=primary.index[4], direction="long")
    assert res.diverged is False
    assert "insufficient" in res.note


def test_partner_map_covers_all_pairs():
    """Our 7 pairs all have a partner defined."""
    for pair in ["EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDJPY", "EURJPY", "GBPJPY"]:
        p = partner_for(pair)
        assert p is not None, f"no partner for {pair}"
        assert p[0] in PARTNERS   # partner itself has a mapping
        assert p[1] in (1, -1)


def test_engine_records_smt():
    """decide() records smt field when partner df provided."""
    from fsp.tests.test_ict_engine import make_df, BULL_BARS
    from fsp.ict.engine import decide

    # partner that diverges: at the sweep bar (bar 8), partner makes higher low
    primary = make_df(BULL_BARS)
    # simple partner: all bars at 1.30 except around sweep timing
    n = len(BULL_BARS)
    p_lows = [1.30] * n
    p_highs = [1.32] * n
    partner = pd.DataFrame({
        "open": [1.31] * n, "high": p_highs, "low": p_lows, "close": [1.31] * n,
    }, index=primary.index)

    d = decide(primary, swing_length=2, atr_len=3, lookback=30,
               smt_df=partner, smt_sign=1, smt_partner="GBPUSD")
    assert d.smt in ("confirmed", "none")  # whatever the synthetic data yields
    # key: it's recorded and doesn't crash

    # without smt_df -> smt is None
    d2 = decide(primary, swing_length=2, atr_len=3, lookback=30)
    assert d2.smt is None
