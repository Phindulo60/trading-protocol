"""Tests for the ICT confluence engine."""
from __future__ import annotations

import pandas as pd

from fsp.ict.engine import decide, in_killzone, TradeDecision


# A bullish reversal: range -> sell-side sweep of 1.1000 -> displacement up
# (MSS/CHoCH) -> deep pullback into discount/OTE. Last bar in NY-AM killzone.
BULL_BARS = [
    (1.1020, 1.1030, 1.1010, 1.1025),
    (1.1025, 1.1050, 1.1020, 1.1045),
    (1.1045, 1.1048, 1.1030, 1.1035),
    (1.1035, 1.1040, 1.1000, 1.1005),
    (1.1005, 1.1030, 1.1002, 1.1028),
    (1.1028, 1.1052, 1.1025, 1.1030),
    (1.1030, 1.1035, 1.1015, 1.1020),
    (1.1020, 1.1025, 1.1008, 1.1012),
    (1.1012, 1.1018, 1.0990, 1.1015),   # sweep of 1.1000 (wick 1.0990, close back above)
    (1.1015, 1.1020, 1.0985, 1.0992),
    (1.0992, 1.0998, 1.0975, 1.1040),   # displacement up
    (1.1040, 1.1075, 1.1038, 1.1070),
    (1.1070, 1.1080, 1.1060, 1.1072),   # high 1.1080
    (1.1058, 1.1060, 1.1040, 1.1045),
    (1.1045, 1.1048, 1.1018, 1.1022),
    (1.1022, 1.1025, 1.1008, 1.1012),   # last: discount/OTE
]


def make_df(bars, start="2026-06-15 11:00", freq="5min"):
    idx = pd.date_range(start, periods=len(bars), freq=freq, tz="UTC")
    return pd.DataFrame({"open": [b[0] for b in bars], "high": [b[1] for b in bars],
                         "low": [b[2] for b in bars], "close": [b[3] for b in bars]}, index=idx)


def reflect(bars, k=2.30):
    """Vertical mirror: bull setup -> bear setup (high<->low swap)."""
    return [(k - o, k - l, k - h, k - c) for (o, h, l, c) in bars]


def test_long_setup():
    d = decide(make_df(BULL_BARS), swing_length=2, atr_len=3, lookback=30)
    assert d.direction == "long"
    assert d.grade in ("A+", "A")
    assert d.is_tradable
    assert d.htf_bias == "bull"
    assert d.stop < d.entry < d.target          # coherent long plan
    assert d.rr is not None and d.rr > 0
    assert any("sell-side sweep" in c for c in d.confluences)
    assert any("killzone" in c for c in d.confluences)


def test_short_setup_mirror():
    d = decide(make_df(reflect(BULL_BARS)), swing_length=2, atr_len=3, lookback=30)
    assert d.direction == "short"
    assert d.grade in ("A+", "A")
    assert d.is_tradable
    assert d.htf_bias == "bear"
    assert d.stop > d.entry > d.target          # coherent short plan
    assert d.rr is not None and d.rr > 0
    assert any("buy-side sweep" in c for c in d.confluences)


def test_no_sweep_is_skip():
    flat = make_df([(1.10, 1.1005, 1.0995, 1.10)] * 20)
    d = decide(flat, swing_length=2, atr_len=3)
    assert d.direction == "none"
    assert d.grade == "skip"
    assert not d.is_tradable
    assert "no recent liquidity sweep" in d.notes


def test_sweep_outside_lookback_is_skip():
    # the sweep happens early; a tiny lookback window excludes it
    d = decide(make_df(BULL_BARS), swing_length=2, atr_len=3, lookback=2)
    assert d.direction == "none"
    assert d.grade == "skip"


def test_in_killzone():
    # June EDT = UTC-4. London KZ = NY 02:00-05:00 = UTC 06:00-09:00.
    assert in_killzone(pd.Timestamp("2026-06-15 07:00", tz="UTC"))   # NY 03:00 London
    # NY-AM = NY 07:00-12:00 = UTC 11:00-16:00.
    assert in_killzone(pd.Timestamp("2026-06-15 13:00", tz="UTC"))   # NY 09:00 NY-AM
    assert not in_killzone(pd.Timestamp("2026-06-15 21:00", tz="UTC"))  # NY 17:00 off
    assert not in_killzone(pd.Timestamp("2026-06-15 00:00", tz="UTC"))  # NY 20:00 Asia


def test_summary_string():
    d = decide(make_df(BULL_BARS), swing_length=2, atr_len=3)
    s = d.summary()
    assert "LONG" in s and "entry=" in s and "target=" in s and "R)" in s
    skip = decide(make_df([(1.10, 1.1005, 1.0995, 1.10)] * 20), swing_length=2, atr_len=3)
    assert "no setup" in skip.summary()


def test_off_session_vetoed():
    # last bar lands at UTC 20:30 = NY 16:30 -> OFF (gap between NY_PM and ASIA)
    off = make_df(BULL_BARS, start="2026-06-15 19:15")
    base = decide(off, swing_length=2, atr_len=3, lookback=30,
                  drop_off_session=False, exhaustion_score=None)
    assert base.is_tradable                          # qualifies absent the guard
    vetoed = decide(off, swing_length=2, atr_len=3, lookback=30,
                    drop_off_session=True, exhaustion_score=None)
    assert vetoed.direction == "long"                # direction still detected
    assert vetoed.grade == "skip"
    assert not vetoed.is_tradable
    assert any("OFF session" in m for m in vetoed.missing)


def test_exhaustion_guard_vetoes_high_score():
    df = make_df(BULL_BARS)                           # NY-AM killzone, scores high
    keep = decide(df, swing_length=2, atr_len=3, exhaustion_score=None)
    assert keep.is_tradable
    # bite the guard at a threshold the setup clears
    vetoed = decide(df, swing_length=2, atr_len=3, exhaustion_score=keep.score)
    assert vetoed.grade == "skip"
    assert not vetoed.is_tradable
    assert any("exhaustion guard" in m for m in vetoed.missing)


def test_default_setup_passes_both_guards():
    # in-killzone, score < 10, not OFF -> defaults keep it tradable
    d = decide(make_df(BULL_BARS), swing_length=2, atr_len=3)
    assert d.is_tradable
    assert d.score < 10
