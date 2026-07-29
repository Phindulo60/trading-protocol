"""Tests for the SCALP_MR backtest engine — cost accounting and fill logic."""
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from fsp.backtest.engine import Trade
from fsp.backtest.scalp_engine import (
    ScalpExecConfig, _close_trade, _enter, _update, pip_pnl,
    run_scalp_backtest,
)
from fsp.signals.base import Signal

PIP = 0.0001


def _bar(high, low, close):
    return pd.Series({"open": close, "high": high, "low": low,
                      "close": close, "volume": 100})


TS = pd.Timestamp("2026-07-28 10:00", tz="UTC")


def _long_trade(cfg):
    """entry 1.1000, sl 1.0990 (10 pip risk), tp 1.1010 (10 pip target)."""
    sig = Signal(
        strategy="SCALP_MR", pair="EURUSD", direction="long",
        entry=1.1000, sl=1.0990, tp1=1.1010, tp2=None,
        inv_pips=10.0, rr_tp1=1.0, rr_tp2=None, risk_r=1.0,
        note="", ts=TS.isoformat(), context={},
    )
    return _enter(sig, TS, 1.1000, cfg, PIP)


def test_zero_cost_tp_is_exactly_one_r():
    cfg = ScalpExecConfig(spread_pips=0.0, sl_slippage_pips=0.0)
    t = _long_trade(cfg)
    assert t.fill == pytest.approx(1.1000)
    _close_trade(t, "win1", 1.1010, TS, cfg, PIP)
    assert t.r_multiple == pytest.approx(1.0)
    assert pip_pnl(t, PIP) == pytest.approx(10.0)


def test_spread_is_paid_twice_on_a_round_trip():
    """1 pip spread on a 10-pip target/10-pip risk win => 0.9R, not 0.95R."""
    cfg = ScalpExecConfig(spread_pips=1.0, sl_slippage_pips=0.0)
    t = _long_trade(cfg)
    assert t.fill == pytest.approx(1.1000 + 0.5 * PIP)
    _close_trade(t, "win1", 1.1010, TS, cfg, PIP)
    assert t.r_multiple == pytest.approx(0.9)
    assert pip_pnl(t, PIP) == pytest.approx(9.0)


def test_loss_is_worse_than_minus_one_r_after_costs():
    """A stopped-out scalp loses risk + spread + slippage, not a flat -1R."""
    cfg = ScalpExecConfig(spread_pips=1.0, sl_slippage_pips=0.5)
    t = _long_trade(cfg)
    _update(t, _bar(1.1002, 1.0980, 1.0985), TS, cfg, PIP)
    assert t.outcome == "loss"
    # 10 risk + 1.0 spread (0.5 in, 0.5 out) + 0.5 stop slippage = 11.5 pips
    assert pip_pnl(t, PIP) == pytest.approx(-11.5)
    assert t.r_multiple == pytest.approx(-1.15)


def test_sl_wins_same_bar_conflict():
    cfg = ScalpExecConfig(spread_pips=0.0, sl_slippage_pips=0.0)
    t = _long_trade(cfg)
    _update(t, _bar(1.1015, 1.0985, 1.1000), TS, cfg, PIP)
    assert t.outcome == "loss"


def test_timeout_exits_at_close_after_max_hold():
    cfg = ScalpExecConfig(spread_pips=0.0, sl_slippage_pips=0.0, max_hold_bars=3)
    t = _long_trade(cfg)
    for _ in range(2):
        _update(t, _bar(1.1005, 1.0995, 1.1003), TS, cfg, PIP)
        assert t.outcome == "open"
    _update(t, _bar(1.1005, 1.0995, 1.1004), TS, cfg, PIP)
    assert t.outcome == "timeout"
    assert pip_pnl(t, PIP) == pytest.approx(4.0)


def test_short_direction_pays_costs_in_the_right_direction():
    cfg = ScalpExecConfig(spread_pips=1.0, sl_slippage_pips=0.0)
    sig = Signal(
        strategy="SCALP_MR", pair="EURUSD", direction="short",
        entry=1.1000, sl=1.1010, tp1=1.0990, tp2=None,
        inv_pips=10.0, rr_tp1=1.0, rr_tp2=None, risk_r=1.0,
        note="", ts=TS.isoformat(), context={},
    )
    t = _enter(sig, TS, 1.1000, cfg, PIP)
    assert t.fill == pytest.approx(1.1000 - 0.5 * PIP)
    _close_trade(t, "win1", 1.0990, TS, cfg, PIP)
    assert t.r_multiple == pytest.approx(0.9)


def test_jpy_pip_size_is_used():
    cfg = ScalpExecConfig(spread_pips=0.0, sl_slippage_pips=0.0)
    sig = Signal(
        strategy="SCALP_MR", pair="USDJPY", direction="long",
        entry=150.00, sl=149.90, tp1=150.10, tp2=None,
        inv_pips=10.0, rr_tp1=1.0, rr_tp2=None, risk_r=1.0,
        note="", ts=TS.isoformat(), context={},
    )
    t = _enter(sig, TS, 150.00, cfg, 0.01)
    _close_trade(t, "win1", 150.10, TS, cfg, 0.01)
    assert pip_pnl(t, 0.01) == pytest.approx(10.0)


# ── end-to-end replay ────────────────────────────────────────────────

def _spike_series(n: int = 2000, seed: int = 1,
                  vol_pips: float = 2.5) -> pd.DataFrame:
    """Seeded M5 random walk. Real band dynamics produce band-exit/re-entry
    setups naturally, so the replay exercises the same code path as live data
    instead of a hand-placed spike that has to thread every filter."""
    idx = pd.date_range("2026-07-27 07:00", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, vol_pips * PIP, n)
    close = 1.1000 + steps.cumsum()
    wick = rng.uniform(0.5, 2.0, n) * PIP
    return pd.DataFrame({"open": close - steps, "high": close + wick,
                         "low": close - wick, "close": close,
                         "volume": np.full(n, 200)}, index=idx)


def test_replay_produces_closed_trades_and_all_are_accounted():
    df = _spike_series()
    res = run_scalp_backtest(
        "EURUSD", df.index[0].to_pydatetime(), df.index[-1].to_pydatetime(),
        cfg=ScalpExecConfig(spread_pips=0.5), m5=df,
    )
    assert res.trades, "expected the synthetic spikes to fire signals"
    assert all(t.outcome in ("win1", "loss", "timeout", "eop") for t in res.trades)
    assert all(t.close_ts is not None for t in res.trades)
    assert all(t.grade == "SCALP_MR" for t in res.trades)


def test_wider_spread_never_improves_expectancy():
    df = _spike_series()
    s, e = df.index[0].to_pydatetime(), df.index[-1].to_pydatetime()
    tight = run_scalp_backtest("EURUSD", s, e, m5=df,
                               cfg=ScalpExecConfig(spread_pips=0.3))
    wide = run_scalp_backtest("EURUSD", s, e, m5=df,
                              cfg=ScalpExecConfig(spread_pips=3.0))
    assert tight.stats()["total"] > 0
    assert wide.stats()["expectancy"] < tight.stats()["expectancy"]


def test_entry_delay_defers_the_fill():
    df = _spike_series()
    s, e = df.index[0].to_pydatetime(), df.index[-1].to_pydatetime()
    now = run_scalp_backtest("EURUSD", s, e, m5=df,
                             cfg=ScalpExecConfig(entry_delay_bars=0))
    late = run_scalp_backtest("EURUSD", s, e, m5=df,
                              cfg=ScalpExecConfig(entry_delay_bars=3))
    assert now.trades and late.trades
    # No fill can land earlier than 3 bars after its signal, and some signals
    # go stale in the interval, so a delayed run never fills more trades.
    assert late.trades[0].open_ts >= now.trades[0].open_ts + timedelta(minutes=15)
    assert len(late.trades) <= len(now.trades)


def test_only_one_position_open_at_a_time():
    df = _spike_series()
    res = run_scalp_backtest("EURUSD", df.index[0].to_pydatetime(),
                             df.index[-1].to_pydatetime(), m5=df)
    spans = sorted((t.open_ts, t.close_ts) for t in res.trades)
    for (_, close_a), (open_b, _) in zip(spans, spans[1:]):
        assert open_b >= close_a


def test_empty_frame_returns_no_trades():
    res = run_scalp_backtest("EURUSD", pd.Timestamp("2026-01-01").to_pydatetime(),
                             pd.Timestamp("2026-01-02").to_pydatetime(),
                             m5=pd.DataFrame())
    assert res.trades == []
    assert res.stats()["total"] == 0


def test_stale_signal_beyond_sl_is_dropped_not_filled():
    """A delayed fill past the stop must be skipped. Filling it would book an
    instant 'stop out' at a price better than the fill -- a phantom profit."""
    from fsp.backtest.scalp_engine import _is_stale

    long_sig = Signal(
        strategy="SCALP_MR", pair="EURUSD", direction="long",
        entry=1.1000, sl=1.0990, tp1=1.1010, tp2=None,
        inv_pips=10.0, rr_tp1=1.0, rr_tp2=None, risk_r=1.0,
        note="", ts=TS.isoformat(), context={},
    )
    assert _is_stale(long_sig, 1.0985)        # already through the stop
    assert _is_stale(long_sig, 1.1015)        # already through the target
    assert not _is_stale(long_sig, 1.1002)

    short_sig = Signal(
        strategy="SCALP_MR", pair="EURUSD", direction="short",
        entry=1.1000, sl=1.1010, tp1=1.0990, tp2=None,
        inv_pips=10.0, rr_tp1=1.0, rr_tp2=None, risk_r=1.0,
        note="", ts=TS.isoformat(), context={},
    )
    assert _is_stale(short_sig, 1.1015)
    assert _is_stale(short_sig, 1.0985)
    assert not _is_stale(short_sig, 1.0998)

