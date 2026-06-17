"""Tests for dealing range / premium-discount / OTE and HTF bias (swing_length=2)."""
from __future__ import annotations

import pandas as pd
import pytest

from fsp.ict.premium_discount import dealing_range, DealingRange, OTE_SWEET
from fsp.ict.bias import htf_bias, Bias


def make_df(bars, start="2026-06-01 00:00", freq="5min"):
    idx = pd.date_range(start, periods=len(bars), freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": [b[0] for b in bars], "high": [b[1] for b in bars],
         "low": [b[2] for b in bars], "close": [b[3] for b in bars]},
        index=idx,
    )


# High @ idx2 (15), Low @ idx4 (8) -> low more recent -> bearish leg
UPTHEN = [
    (9.5, 10, 9, 9.8), (10.5, 11, 10, 10.8), (12.5, 15, 12, 14.0),
    (10.5, 11, 9, 9.8), (9.5, 10, 8, 8.8), (13.5, 16, 13, 16.0),
    (15.5, 17, 15, 16.8),
]
# Low @ idx2 (10), High @ idx4 (17) -> high more recent -> bullish leg
DOWNTHEN = [
    (16.8, 17, 16, 16.5), (16.2, 16, 15, 15.5), (12.5, 13, 10, 10.5),
    (14.5, 16, 14, 15.5), (15.5, 17, 15, 16.5), (11.0, 12, 8, 8.5),
    (10.5, 11, 7, 7.5),
]


def test_dealing_range_basics():
    dr = dealing_range(make_df(UPTHEN), swing_length=2)
    assert dr is not None
    assert dr.high == 15 and dr.low == 8
    assert dr.size == 7
    assert dr.equilibrium == pytest.approx(11.5)
    assert dr.direction == "bear"   # low (idx4) formed after high (idx2)


def test_position_and_zone():
    dr = dealing_range(make_df(UPTHEN), swing_length=2)
    assert dr.position(11.5) == pytest.approx(0.5)
    assert dr.zone(11.5) == "equilibrium"
    assert dr.position(9) == pytest.approx(1 / 7)
    assert dr.zone(9) == "discount"
    assert dr.in_discount(9)
    assert dr.zone(14) == "premium"
    assert dr.in_premium(14)


def test_ote_bearish_leg():
    dr = dealing_range(make_df(UPTHEN), swing_length=2)  # bear leg, size 7
    lo, hi = dr.ote("bear")
    assert lo == pytest.approx(8 + 0.62 * 7)   # 12.34
    assert hi == pytest.approx(8 + 0.79 * 7)   # 13.53
    assert dr.in_ote(13, "bear")
    assert not dr.in_ote(11, "bear")
    assert dr.ote_sweet_spot("bear") == pytest.approx(8 + OTE_SWEET * 7)


def test_ote_bullish_leg():
    dr = dealing_range(make_df(DOWNTHEN), swing_length=2)  # bull leg, high 17 low 10 size 7
    assert dr.direction == "bull"
    lo, hi = dr.ote("bull")
    assert lo == pytest.approx(17 - 0.79 * 7)   # 11.47
    assert hi == pytest.approx(17 - 0.62 * 7)   # 12.66
    assert dr.in_ote(12, "bull")
    assert not dr.in_ote(16, "bull")


def test_no_range_on_flat_or_short():
    assert dealing_range(make_df([(10, 11, 9, 10.5)] * 12), swing_length=2) is None
    assert dealing_range(make_df([(10, 11, 9, 10.5)] * 3), swing_length=2) is None


# --------------------------------------------------------------------------
# bias
# --------------------------------------------------------------------------
def test_bias_bullish():
    b = htf_bias(make_df(UPTHEN), swing_length=2)  # single bullish BOS -> trend +1
    assert b.direction == "bull"
    assert b.score >= 2
    assert b.range_zone is not None
    assert any("bullish" in r for r in b.reasons)


def test_bias_bearish_with_mss():
    # structure module's MSS fixture: BOS bull then CHoCH-bear MSS (atr_len=3)
    bars = [
        (9.85, 10.0, 9.8, 9.9), (10.05, 10.4, 10.0, 10.3), (10.55, 11.0, 10.5, 10.8),
        (10.55, 10.6, 10.2, 10.3), (10.45, 10.5, 10.1, 10.2), (10.85, 11.2, 10.8, 11.1),
        (10.95, 11.0, 10.6, 10.7), (10.75, 10.8, 10.4, 10.5), (10.55, 10.6, 8.0, 8.2),
    ]
    b = htf_bias(make_df(bars), swing_length=2, atr_len=3, atr_mult=1.5)
    assert b.direction == "bear"
    assert b.score == -3   # -2 trend, -1 MSS
    assert any("MSS" in r for r in b.reasons)


def test_bias_neutral_flat():
    b = htf_bias(make_df([(10, 10.1, 9.9, 10.0)] * 12), swing_length=2)
    assert b.direction == "neutral"
    assert b.score == 0
    assert b.range_position is None
