"""Tests for the ICT shadow signal path."""
from __future__ import annotations

import pandas as pd

from fsp.ict.shadow import scan_ict_shadow, decision_to_signal, scan_batch_ict_shadow
from fsp.ict.engine import TradeDecision
from fsp.tests.test_ict_engine import BULL_BARS, make_df, reflect


def test_decision_to_signal_maps_fields():
    d = TradeDecision(ts=pd.Timestamp("2026-06-15 13:00", tz="UTC").to_pydatetime(),
                      direction="long", grade="A", score=8, htf_bias="bull",
                      entry=1.1000, stop=1.0980, target=1.1040, rr=2.0,
                      confluences=["sell-side sweep", "MSS", "OTE"])
    sig = decision_to_signal(d, "EURUSD")
    assert sig is not None
    assert sig.strategy == "ICT_SHADOW"
    assert sig.direction == "long" and sig.entry == 1.1000 and sig.sl == 1.0980
    assert sig.tp1 == 1.1040 and sig.rr_tp1 == 2.0
    assert sig.inv_pips == 20.0          # 0.0020 / 0.0001
    assert "ICT A" in sig.note
    assert sig.context["grade"] == "A"
    assert "|ICT_SHADOW|" in sig.dedup_key


def test_jpy_pip_scaling():
    d = TradeDecision(ts=pd.Timestamp("2026-06-15 13:00", tz="UTC").to_pydatetime(),
                      direction="short", grade="A+", score=10, htf_bias="bear",
                      entry=156.00, stop=156.30, target=155.10, rr=3.0)
    sig = decision_to_signal(d, "USDJPY")
    assert sig.inv_pips == 30.0          # 0.30 / 0.01


def test_not_tradable_returns_none():
    d = TradeDecision(ts=pd.Timestamp("2026-06-15 13:00", tz="UTC").to_pydatetime(),
                      direction="none", grade="skip", score=0, htf_bias="neutral")
    assert decision_to_signal(d, "EURUSD") is None


def test_scan_long_setup():
    sig = scan_ict_shadow("EURUSD", make_df(BULL_BARS), None,
                          window=16, swing_length=2, atr_len=3)
    assert sig is not None
    assert sig.strategy == "ICT_SHADOW" and sig.direction == "long"
    assert sig.sl < sig.entry < sig.tp1


def test_scan_short_setup():
    sig = scan_ict_shadow("EURUSD", make_df(reflect(BULL_BARS)), None,
                          window=16, swing_length=2, atr_len=3)
    assert sig is not None and sig.direction == "short"
    assert sig.sl > sig.entry > sig.tp1


def test_scan_grade_filter():
    # A-grade setup rejected when we demand A+
    sig = scan_ict_shadow("EURUSD", make_df(BULL_BARS), None,
                          window=16, swing_length=2, atr_len=3, min_grade="A+")
    assert sig is None


def test_scan_flat_returns_none():
    flat = make_df([(1.10, 1.1005, 1.0995, 1.10)] * 30)
    assert scan_ict_shadow("EURUSD", flat, None, window=20, swing_length=2, atr_len=3) is None


def test_scan_batch():
    batch = {
        "EURUSD": (make_df(BULL_BARS), None),
        "GBPUSD": (make_df([(1.10, 1.1005, 1.0995, 1.10)] * 30), None),
    }
    sigs = scan_batch_ict_shadow(batch, window=16, swing_length=2, atr_len=3)
    assert len(sigs) == 1 and sigs[0].pair == "EURUSD"
