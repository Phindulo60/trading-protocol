"""Tests for daily report module."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fsp.notify.daily_report import (
    REPORT_HOUR_UTC,
    _stats,
    _outcome_icon,
    _fmt_trade_line,
    should_send_report,
    compose_report,
)


def _trade(outcome: str | None = None, r: float | None = None,
           sent: bool = True, strat: str = "TREND_RSI") -> dict:
    return {
        "id": 1,
        "ts": "2026-05-27T10:00:00+00:00",
        "pair": "USDCAD",
        "strategy": strat,
        "direction": "long",
        "entry": 1.38000,
        "sl": 1.37800,
        "tp1": 1.38500,
        "rr_tp1": 2.5,
        "risk_r": 1.0,
        "sent": sent,
        "outcome": outcome,
        "r_multiple": r,
        "exit_ts": None,
    }


# ── _stats ──────────────────────────────────────────────────────────────────

def test_stats_empty():
    s = _stats([])
    assert s["n"] == 0
    assert s["sent"] == 0


def test_stats_all_open():
    trades = [_trade(outcome=None) for _ in range(3)]
    s = _stats(trades)
    assert s["n"] == 3
    assert s["sent"] == 3
    assert s["open"] == 3
    assert s["closed"] == 0
    assert s["wr"] == 0.0  # no closed trades


def test_stats_mixed():
    trades = [
        _trade(outcome="win", r=2.5),
        _trade(outcome="loss", r=-1.0),
        _trade(outcome="win", r=2.5),
        _trade(outcome=None),  # open
    ]
    s = _stats(trades)
    assert s["sent"] == 4
    assert s["closed"] == 3
    assert s["wins"] == 2
    assert s["losses"] == 1
    assert s["open"] == 1
    assert abs(s["wr"] - 2/3) < 0.01
    assert s["total_r"] == 4.0  # 2.5 + (-1.0) + 2.5 + 0


def test_stats_only_unsent_excluded():
    trades = [
        _trade(outcome="win", r=2.5, sent=False),
        _trade(outcome="loss", r=-1.0, sent=True),
    ]
    s = _stats(trades)
    assert s["sent"] == 1
    assert s["closed"] == 1
    assert s["wins"] == 0


# ── _outcome_icon ───────────────────────────────────────────────────────────

def test_outcome_icon_win():
    assert _outcome_icon("win", 2.5) == "✅"


def test_outcome_icon_loss():
    assert _outcome_icon("loss", -1.0) == "❌"


def test_outcome_icon_open():
    assert _outcome_icon(None, None) == "🔄"


def test_outcome_icon_timeout_pos():
    assert _outcome_icon("timeout", 0.5) == "🟢"


def test_outcome_icon_timeout_neg():
    assert _outcome_icon("timeout", -0.3) == "🔴"


def test_outcome_icon_timeout_no_r():
    assert _outcome_icon("timeout", None) == "⏱"


# ── _fmt_trade_line ─────────────────────────────────────────────────────────

def test_fmt_trade_line_win():
    line = _fmt_trade_line(_trade(outcome="win", r=2.5))
    assert "✅" in line
    assert "TREND_RSI" in line
    assert "LONG" in line
    assert "+2.50R" in line


def test_fmt_trade_line_open():
    line = _fmt_trade_line(_trade(outcome=None))
    assert "🔄" in line
    assert "OPEN" in line


# ── should_send_report ──────────────────────────────────────────────────────

def test_should_send_report_too_early():
    now = datetime(2026, 5, 27, 10, 0, tzinfo=timezone.utc)
    assert should_send_report(now, last_report_date=None) is False


def test_should_send_report_at_target():
    now = datetime(2026, 5, 27, REPORT_HOUR_UTC, 0, tzinfo=timezone.utc)
    assert should_send_report(now, last_report_date=None) is True


def test_should_send_report_past_target():
    now = datetime(2026, 5, 27, REPORT_HOUR_UTC + 1, 0, tzinfo=timezone.utc)
    assert should_send_report(now, last_report_date=None) is True


def test_should_send_report_already_sent():
    now = datetime(2026, 5, 27, REPORT_HOUR_UTC, 0, tzinfo=timezone.utc)
    assert should_send_report(now, last_report_date="2026-05-27") is False


def test_should_send_report_new_day():
    now = datetime(2026, 5, 28, REPORT_HOUR_UTC, 0, tzinfo=timezone.utc)
    assert should_send_report(now, last_report_date="2026-05-27") is True


# ── compose_report ──────────────────────────────────────────────────────────

def test_compose_report_runs_without_data(tmp_path, monkeypatch):
    """compose_report should not crash when journal is empty."""
    monkeypatch.setenv("FSP_DATA_DIR", str(tmp_path))
    msg = compose_report(hours=24, pair="USDCAD")
    assert "FSP Daily Report" in msg
    assert "No signals" in msg


# ── escape_md ───────────────────────────────────────────────────────────────

def test_escape_md_underscores():
    from fsp.notify.telegram import escape_md
    assert escape_md("TREND_RSI signal") == r"TREND\_RSI signal"


def test_escape_md_asterisks():
    from fsp.notify.telegram import escape_md
    assert escape_md("Rule 5: 0% win rate") == r"Rule 5: 0% win rate"


def test_escape_md_combined():
    from fsp.notify.telegram import escape_md
    src = "TREND_RSI is *broken*"
    assert escape_md(src) == r"TREND\_RSI is \*broken\*"


def test_escape_md_empty():
    from fsp.notify.telegram import escape_md
    assert escape_md("") == ""
    assert escape_md(None) == ""
