"""Tests for the outcome resolver hold-window handling (incl ICT_SHADOW)."""
from __future__ import annotations

import pandas as pd

from fsp.journal.resolver import _DEFAULT_HOLD, _resolve_one


def _m15(n: int, *, tp_at: int | None = None, sl_at: int | None = None,
         start: str = "2026-06-15 13:15") -> pd.DataFrame:
    """Build n M15 bars that stay inside (sl, tp1) unless a touch is injected.

    Default body: high 1.1030 / low 1.0990 / close 1.1010 — never hits a
    1.0980 SL or 1.1040 TP. ``tp_at``/``sl_at`` (1-based) inject one touch bar.
    """
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    highs = [1.1030] * n
    lows = [1.0990] * n
    closes = [1.1010] * n
    if tp_at is not None:
        highs[tp_at - 1] = 1.1045          # >= tp1 1.1040
    if sl_at is not None:
        lows[sl_at - 1] = 1.0975           # <= sl 1.0980
    return pd.DataFrame({"high": highs, "low": lows, "close": closes}, index=idx)


def _sig(strategy: str, *, context: dict | None = None) -> dict:
    return {
        "id": 1, "pair": "EURUSD", "ts": "2026-06-15T13:00:00+00:00",
        "direction": "long", "entry": 1.1000, "sl": 1.0980, "tp1": 1.1040,
        "rr_tp1": 2.0, "strategy": strategy, "context": context or {},
    }


def test_default_hold_includes_ict_shadow():
    assert _DEFAULT_HOLD["ICT_SHADOW"] == 64


def test_ict_shadow_resolves_tp_beyond_legacy_window():
    # TP touched at bar 30 — within the 64-bar ICT window but past the legacy 16.
    m15 = _m15(50, tp_at=30)
    outcome, r, _ = _resolve_one(_sig("ICT_SHADOW"), m15)
    assert outcome == "win" and r == 2.0


def test_short_legacy_strategy_times_out_before_bar_30():
    # TREND_RSI (default hold 8) never reaches the bar-30 TP → not a win.
    m15 = _m15(50, tp_at=30)
    outcome, _, _ = _resolve_one(_sig("TREND_RSI"), m15)
    assert outcome == "timeout"


def test_context_max_hold_overrides_default():
    # Explicit context window (5) clips before the bar-30 TP → timeout.
    m15 = _m15(50, tp_at=30)
    outcome, _, _ = _resolve_one(_sig("ICT_SHADOW", context={"max_hold_bars": 5}), m15)
    assert outcome == "timeout"


def test_sl_hit_returns_loss():
    m15 = _m15(50, sl_at=10)
    outcome, r, _ = _resolve_one(_sig("ICT_SHADOW"), m15)
    assert outcome == "loss" and r == -1.0
