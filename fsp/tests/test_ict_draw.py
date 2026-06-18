"""Tests for draw-on-liquidity targeting + sweep significance."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from fsp.data.types import Level
from fsp.ict.liquidity import LiquidityPool
from fsp.ict.draw import (significant_levels, draw_targets, best_target,
                          sweep_significance)


def _lvl(price, label, kind, swept=False):
    lv = Level(price=price, label=label, kind=kind, ts=datetime(2026, 6, 15))
    lv.swept = swept
    return lv


def _pool(price, side, kind="swing", swept=False):
    return LiquidityPool(price=price, side=side, kind=kind,
                         members=[datetime(2026, 6, 15)],
                         created_ts=datetime(2026, 6, 15), swept=swept)


def test_significant_levels_pdh_pdl():
    # 3 days of hourly bars; PDH/PDL should come from the prior completed day
    idx = pd.date_range("2026-06-15 00:00", periods=72, freq="1h", tz="UTC")
    base = 1.10
    highs = [base + 0.01 + (i % 24) * 0.0001 for i in range(72)]
    lows = [base - 0.01 - (i % 24) * 0.0001 for i in range(72)]
    df = pd.DataFrame({"open": base, "high": highs, "low": lows, "close": base}, index=idx)
    lv = significant_levels(df)
    assert "PDH" in lv and "PDL" in lv
    assert lv["PDH"].kind == "high" and lv["PDL"].kind == "low"
    # opens must be dropped
    assert "DO" not in lv and "WO" not in lv


def test_draw_targets_prefers_significant_nearest_first():
    levels = {"PDH": _lvl(1.1050, "PDH", "high"),
              "PWH": _lvl(1.1100, "PWH", "high")}
    pools = [_pool(1.1030, "buy", kind="equal"),     # equal-pool (significant)
             _pool(1.1020, "buy", kind="swing")]     # minor swing (excluded here)
    entry = 1.1000
    out = draw_targets(pools, levels, "buy", entry)
    prices = [p for p, _ in out]
    # nearest-first; minor swing 1.1020 excluded
    assert prices == sorted(prices)
    assert 1.1020 not in prices
    assert 1.1030 in prices and 1.1050 in prices and 1.1100 in prices
    assert out[0] == (1.1030, "EQH")


def test_draw_targets_skips_swept():
    levels = {"PDH": _lvl(1.1050, "PDH", "high", swept=True)}
    pools = [_pool(1.1030, "buy", kind="equal", swept=True)]
    assert draw_targets(pools, levels, "buy", 1.1000) == []


def test_best_target_falls_back_to_swing():
    # no significant levels -> nearest structural swing pool
    pools = [_pool(1.1025, "buy", kind="swing"), _pool(1.1040, "buy", kind="swing")]
    price, label = best_target(pools, {}, "buy", 1.1000)
    assert price == 1.1025 and label == "swing"


def test_best_target_prefers_significant_over_swing():
    levels = {"PDH": _lvl(1.1060, "PDH", "high")}
    pools = [_pool(1.1025, "buy", kind="swing")]   # nearer minor swing
    price, label = best_target(pools, levels, "buy", 1.1000)
    # PDH chosen as the draw even though farther than the minor swing
    assert price == 1.1060 and label == "PDH"


def test_sweep_significance_matches_within_tol():
    levels = {"PDL": _lvl(1.0950, "PDL", "low")}
    # sell-side sweep (ran a low) near PDL
    assert sweep_significance(1.09505, "sell", levels, tol=0.0002) == "PDL"
    # outside tolerance -> None
    assert sweep_significance(1.0980, "sell", levels, tol=0.0002) is None
    # wrong side (buy-side sweep looks at highs) -> None
    assert sweep_significance(1.09505, "buy", levels, tol=0.0002) is None


def test_sweep_significance_prefers_highest_rank():
    levels = {"PDH": _lvl(1.1050, "PDH", "high"),
              "PWH": _lvl(1.1050, "PWH", "high")}   # coincident -> weekly outranks daily
    assert sweep_significance(1.1050, "buy", levels, tol=0.0002) == "PWH"


def test_engine_records_target_and_sweep_major():
    from fsp.tests.test_ict_engine import make_df, BULL_BARS
    from fsp.ict.engine import decide
    d = decide(make_df(BULL_BARS), swing_length=2, atr_len=3, lookback=30)
    # target_kind is always set when a plan exists
    if d.is_tradable:
        assert d.target_kind in ("PDH", "PWH", "PMH", "PDL", "PWL", "PML",
                                 "EQH", "EQL", "swing", "DR")
    # sweep_major is None or a level label (no crash)
    assert d.sweep_major is None or isinstance(d.sweep_major, str)
