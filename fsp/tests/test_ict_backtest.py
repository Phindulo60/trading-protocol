"""Tests for the ICT backtester. The decision source is stubbed so the
fill / TP / SL / cancel / filter logic is verified in isolation."""
from __future__ import annotations

import pandas as pd

from fsp.ict.backtest import simulate_ict, aggregate, report
from fsp.ict.engine import TradeDecision
from fsp.backtest.engine import ExecConfig, BacktestResult


def make_df(bars, start="2026-06-15 11:00", freq="15min"):
    idx = pd.date_range(start, periods=len(bars), freq=freq, tz="UTC")
    return pd.DataFrame({"open": [b[0] for b in bars], "high": [b[1] for b in bars],
                         "low": [b[2] for b in bars], "close": [b[3] for b in bars]}, index=idx)


def fire_at(target_ts, direction, entry, stop, target, grade="A", score=8):
    """Decider stub: returns a tradable decision only when the window ends at target_ts."""
    rr = (abs(target - entry) / abs(entry - stop))

    def _decider(win, hwin=None, **kw):
        i = win.index[-1]
        if i == target_ts:
            return TradeDecision(ts=i.to_pydatetime(), direction=direction, grade=grade,
                                 score=score, htf_bias="bull", entry=entry, stop=stop,
                                 target=target, rr=rr, confluences=["stub"])
        return TradeDecision(ts=i.to_pydatetime(), direction="none", grade="skip",
                             score=0, htf_bias="neutral")
    return _decider


# flat baseline; we splice in the price action that fills/targets
def base(n=30, px=1.1000):
    return [(px, px + 0.0002, px - 0.0002, px) for _ in range(n)]


def test_long_win():
    bars = base(20)
    # signal at i=10 (entry 1.0990, stop 1.0980, target 1.1010 -> 2R)
    # after signal: dip to fill 1.0990, then rally to 1.1010
    bars[12] = (1.1000, 1.1001, 1.0989, 1.0995)   # fills entry 1.0990
    bars[14] = (1.0995, 1.1012, 1.0993, 1.1010)   # hits target 1.1010
    df = make_df(bars)
    dec = fire_at(df.index[10], "long", 1.0990, 1.0980, 1.1010)
    res = simulate_ict(df, "EURUSD", decider=dec, window=5, min_grade="A",
                       exec_cfg=ExecConfig(partial_pct=1.0, min_rr_tp1=1.5,
                                           spread_pips=0.0, sl_slippage_pips=0.0))
    closed = [t for t in res.trades if t.outcome not in ("eop",)]
    assert len(closed) == 1
    t = closed[0]
    assert t.direction == "long" and t.outcome == "win1"
    assert t.weighted_r > 1.9   # ~2R minus costs
    assert res.stats()["win_rate"] == 1.0


def test_long_loss():
    bars = base(20)
    bars[12] = (1.1000, 1.1001, 1.0989, 1.0995)   # fills entry 1.0990
    bars[14] = (1.0992, 1.0993, 1.0978, 1.0980)   # hits stop 1.0980
    df = make_df(bars)
    dec = fire_at(df.index[10], "long", 1.0990, 1.0980, 1.1010)
    res = simulate_ict(df, "EURUSD", decider=dec, window=5,
                       exec_cfg=ExecConfig(partial_pct=1.0, min_rr_tp1=1.5,
                                           spread_pips=0.0, sl_slippage_pips=0.0))
    closed = [t for t in res.trades if t.outcome not in ("eop",)]
    assert len(closed) == 1
    assert closed[0].outcome == "loss"
    assert closed[0].weighted_r == -1.0


def test_unfilled_is_cancelled():
    bars = base(20)               # price never dips to entry 1.0990
    df = make_df(bars)
    dec = fire_at(df.index[10], "long", 1.0990, 1.0980, 1.1010)
    res = simulate_ict(df, "EURUSD", decider=dec, window=5,
                       exec_cfg=ExecConfig(partial_pct=1.0, min_rr_tp1=1.5,
                                           max_pending_bars=4))
    # never filled -> dropped, no trades counted
    assert all(t.filled_ts is not None for t in res.trades)
    assert res.stats().get("total", 0) == 0


def test_grade_filter_blocks_b():
    bars = base(20)
    bars[12] = (1.1000, 1.1001, 1.0989, 1.0995)
    bars[14] = (1.0995, 1.1012, 1.0993, 1.1010)
    df = make_df(bars)
    dec = fire_at(df.index[10], "long", 1.0990, 1.0980, 1.1010, grade="B", score=5)
    res = simulate_ict(df, "EURUSD", decider=dec, window=5, min_grade="A")
    assert len(res.trades) == 0


def test_not_tradable_no_trade():
    df = make_df(base(20))
    def dec(win, hwin=None, **kw):
        return TradeDecision(ts=win.index[-1].to_pydatetime(), direction="none",
                             grade="skip", score=0, htf_bias="neutral")
    res = simulate_ict(df, "EURUSD", decider=dec, window=5)
    assert len(res.trades) == 0


def test_aggregate_and_report():
    bars = base(20)
    bars[12] = (1.1000, 1.1001, 1.0989, 1.0995)
    bars[14] = (1.0995, 1.1012, 1.0993, 1.1010)
    df = make_df(bars)
    dec = fire_at(df.index[10], "long", 1.0990, 1.0980, 1.1010)
    cfg = ExecConfig(partial_pct=1.0, min_rr_tp1=1.5, spread_pips=0.0, sl_slippage_pips=0.0)
    r1 = simulate_ict(df, "EURUSD", decider=dec, window=5, exec_cfg=cfg)
    r2 = simulate_ict(df, "GBPUSD", decider=dec, window=5, exec_cfg=cfg)
    merged = aggregate({"EURUSD": r1, "GBPUSD": r2})
    assert merged.stats()["total"] == 2
    txt = report({"EURUSD": r1, "GBPUSD": r2})
    assert "EURUSD" in txt and "ALL" in txt and "by grade" in txt
